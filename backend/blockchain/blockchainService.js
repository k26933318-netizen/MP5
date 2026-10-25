const localLedger = require('./localLedger');
const mongo = require('../data/mongoRepository');
const crypto = require('crypto');
const { withContract } = require('./fabricGateway');

class BlockchainAdapter {
  async recordVote(input) { throw new Error('Not implemented'); }
  async getVote(voteId) { throw new Error('Not implemented'); }
  async getElectionVotes(electionId) { throw new Error('Not implemented'); }
  async getElectionHistory(electionId) { throw new Error('Not implemented'); }
  async verifyLedgerIntegrity(electionId) { throw new Error('Not implemented'); }
}

class LocalLedgerAdapter extends BlockchainAdapter {
  async recordVote(input) { return localLedger.appendVote(input); }
  async getVote(voteId) { return localLedger.getVote(voteId); }
  async getElectionVotes(electionId) { return localLedger.getElectionVotes(electionId); }
  async getElectionHistory(electionId) { return localLedger.getElectionHistory(electionId); }
  async verifyLedgerIntegrity(electionId) { return localLedger.verifyLedgerIntegrity(electionId); }
}

class FabricAdapter extends BlockchainAdapter {
  async syncElection(election, candidates) {
    return withContract(async contract => {
      let stored = JSON.parse(await contract.evaluateTransaction('getElection', election.electionId));
      if (!stored) {
        stored = JSON.parse(await contract.submitTransaction(
          'registerElection',
          election.electionId,
          election.electionName,
          election.startDate,
          election.endDate,
          election.status
        ));
      } else if (stored.electionName !== election.electionName ||
          stored.startDate !== election.startDate || stored.endDate !== election.endDate) {
        throw new Error('Fabric election details do not match the application record.');
      }
      if (stored.status !== election.status) {
        const statusPath = {
          DRAFT: ['UPCOMING', 'ACTIVE', 'CLOSED', 'RESULTS_DECLARED'],
          UPCOMING: ['ACTIVE', 'CLOSED', 'RESULTS_DECLARED'],
          ACTIVE: ['CLOSED', 'RESULTS_DECLARED'],
          CLOSED: ['RESULTS_DECLARED']
        };
        let nextStatus = stored.status;
        for (const status of statusPath[stored.status] || []) {
          await contract.submitTransaction('updateElectionStatus', election.electionId, status);
          nextStatus = status;
          if (nextStatus === election.status) break;
        }
        if (nextStatus !== election.status) {
          throw new Error(`Cannot synchronize Fabric election status ${stored.status} to ${election.status}`);
        }
      }
      for (const candidate of candidates) {
        const storedCandidate = JSON.parse(await contract.evaluateTransaction('getCandidate', candidate.candidateId));
        if (!storedCandidate) {
          await contract.submitTransaction(
            'registerCandidate',
            candidate.candidateId,
            candidate.electionId,
            candidate.candidateName,
            candidate.party || ''
          );
        } else if (storedCandidate.electionId !== candidate.electionId ||
            storedCandidate.candidateName !== candidate.candidateName ||
            storedCandidate.party !== (candidate.party || '')) {
          throw new Error(`Fabric candidate ${candidate.candidateId} does not match the application record.`);
        }
      }
    });
  }

  async recordVote(input) {
    return withContract(async contract => JSON.parse(await contract.submitTransaction(
      'castVote',
      input.electionId,
      input.voterHash,
      input.candidateId
    )));
  }
  async getVote(voteId) {
    return withContract(async contract => JSON.parse(await contract.evaluateTransaction('getVote', voteId)));
  }
  async getElectionVotes(electionId) {
    return withContract(async contract => JSON.parse(await contract.evaluateTransaction('getElectionVotes', electionId)));
  }
  async getElectionHistory(electionId) {
    return withContract(async contract => JSON.parse(await contract.evaluateTransaction('getElectionHistory', electionId)));
  }
  async verifyLedgerIntegrity() {
    const error = new Error('Integrity verification is not implemented for the Fabric backend');
    error.code = 'FABRIC_INTEGRITY_UNSUPPORTED';
    error.statusCode = 501;
    throw error;
  }
  async getVoteByTransactionId(transactionId) {
    return withContract(async contract => JSON.parse(await contract.evaluateTransaction('getVoteByTransactionId', transactionId)));
  }
  async verifyVoteReceipt(transactionId) {
    const vote = await this.getVoteByTransactionId(transactionId);
    return {
      transactionId,
      electionId: vote.electionId,
      voteId: vote.voteId,
      status: vote.transactionId === transactionId ? 'VERIFIED' : 'FAILED',
      verifiedAt: new Date().toISOString()
    };
  }
  async health() {
    return withContract(async contract => {
      const result = JSON.parse(await contract.evaluateTransaction('ping'));
      return result.status === 'ok';
    });
  }
}

const backendName = process.env.BLOCKCHAIN_BACKEND || 'fabric';
if (!['local', 'fabric'].includes(backendName)) throw new Error(`Unsupported BLOCKCHAIN_BACKEND: ${backendName}`);
const adapter = backendName === 'fabric' ? new FabricAdapter() : new LocalLedgerAdapter();

class BlockchainService {
  constructor(repository = mongo, voteAdapter = adapter) {
    this.repository = repository;
    this.adapter = voteAdapter;
  }
  async listElections() { return this.repository.listElections(); }
  async syncElection(electionId) {
    if (typeof this.adapter.syncElection !== 'function') return;
    const election = await this.repository.getElection(electionId);
    if (!election) return;
    const candidates = await this.repository.getCandidates(electionId);
    await this.adapter.syncElection(election, candidates);
  }
  async getCandidates(electionId) {
    await this.syncElection(electionId);
    return this.repository.getCandidates(electionId);
  }
  async registerElection(input) {
    const election = await this.repository.createElection(input);
    await this.syncElection(election.electionId);
    return election;
  }
  async registerCandidate(input) {
    if (!await this.repository.getElection(input.electionId)) throw new Error('Election does not exist');
    const candidate = await this.repository.createCandidate(input);
    await this.syncElection(input.electionId);
    return candidate;
  }
  async updateElectionState(electionId, nextStatus, actor = 'SYSTEM') {
    const election = await this.repository.getElection(electionId);
    if (!election) throw new Error('Election does not exist');
    if (!this.repository.VALID_ELECTION_STATES.includes(nextStatus)) throw new Error('Invalid election status');
    this.repository.validateElectionStateTransition(election.status, nextStatus);
    const updated = await this.repository.updateElection(electionId, { status: nextStatus }, actor);
    await this.syncElection(electionId);
    return updated;
  }
  async castVote(input) {
    const voter = await this.repository.getVoterCredentials(input.voterId);
    if (!voter || voter.status !== 'APPROVED') {
      const error = new Error('Approved voter account required');
      error.code = 'VOTER_NOT_APPROVED';
      error.statusCode = 403;
      throw error;
    }
    const election = await this.repository.getElection(input.electionId);
    if (!election) throw new Error('Election does not exist');
    const now = Date.now();
    if (election.status !== 'ACTIVE' || now < Date.parse(election.startDate) || now > Date.parse(election.endDate)) {
      throw new Error('Voting is closed or election is not active');
    }
    await this.syncElection(input.electionId);
    const candidate = await this.repository.getCandidate(input.candidateId, input.electionId);
    if (!candidate) throw new Error('Candidate does not belong to election');
    const securitySettings = typeof this.repository.getVotingSettings === 'function'
      ? await this.repository.getVotingSettings()
      : { faceVerificationEnabled: true };
    if (securitySettings.faceVerificationEnabled) {
      const enrollment = await this.repository.getFaceEnrollment(input.voterId);
      if (!enrollment) {
        const error = new Error('Face enrollment is required before voting');
        error.code = 'FACE_NOT_ENROLLED';
        error.statusCode = 403;
        throw error;
      }
      const enrollmentUpdatedAt = enrollment.updatedAt instanceof Date
        ? enrollment.updatedAt
        : new Date(enrollment.updatedAt);
      const authorization = await this.repository.consumeFaceVerification({
        tokenHash: crypto.createHash('sha256').update(input.verificationToken || '').digest('hex'),
        voterId: input.voterId,
        electionId: input.electionId,
        sessionHash: input.sessionHash,
        enrollmentUpdatedAt
      });
      if (!authorization.valid) {
        await this.repository.logAudit('INVALID_VERIFICATION_TOKEN', {
          actorId: input.voterId,
          actorRole: 'VOTER',
          resourceType: 'ELECTION',
          resourceId: input.electionId,
          status: 'FAILURE',
          description: authorization.code
        });
        const error = new Error('Face verification authorization is invalid or expired');
        error.code = authorization.code;
        error.statusCode = 403;
        throw error;
      }
    } else {
      await this.repository.logAudit('FACE_VERIFICATION_BYPASSED_BY_ADMIN_SETTING', {
        actorId: input.voterId,
        actorRole: 'VOTER',
        resourceType: 'ELECTION',
        resourceId: input.electionId,
        status: 'SUCCESS',
        description: 'Face verification was disabled by an administrator for voting.'
      });
    }
    const voterHash = crypto.createHash('sha256')
      .update(`${input.electionId}:${String(input.voterId).trim().toLowerCase()}`)
      .digest('hex');
    try {
      await this.repository.createVoteClaim(input.electionId, voterHash);
    } catch (error) {
      if (error.code === 'VOTE_ALREADY_CAST') {
        await this.repository.logAudit('DUPLICATE_VOTE_BLOCKED', {
          actorId: input.voterId,
          actorRole: 'VOTER',
          resourceType: 'ELECTION',
          resourceId: input.electionId,
          status: 'FAILURE',
          description: 'Duplicate vote rejected by unique vote claim'
        });
      }
      throw error;
    }
    let vote;
    try {
      vote = await this.adapter.recordVote({
        electionId: input.electionId,
        candidateId: input.candidateId,
        voterReference: input.voterId,
        voterHash
      });
    } catch (error) {
      try {
        await this.repository.releaseVoteClaim(input.electionId, voterHash);
      } catch (releaseError) {
        throw Object.assign(new AggregateError(
          [error, releaseError],
          'Ledger submission failed and the pending vote claim could not be released.'
        ), { code: 'VOTE_CLAIM_CLEANUP_FAILED', statusCode: 500 });
      }
      if (/already voted/i.test(error.message)) {
        const duplicate = new Error('Voter has already voted in this election.');
        duplicate.code = 'VOTE_ALREADY_CAST';
        duplicate.statusCode = 409;
        throw duplicate;
      }
      if (typeof this.adapter.health === 'function' && !error.statusCode) {
        const unavailable = new Error('Hyperledger Fabric could not record the vote.');
        unavailable.code = 'FABRIC_SUBMISSION_FAILED';
        unavailable.statusCode = 503;
        unavailable.cause = error;
        throw unavailable;
      }
      throw error;
    }
    await this.repository.completeVoteClaim(input.electionId, voterHash, vote.transactionId);
    await this.repository.logAudit('VOTE_CAST', { actorId: input.voterId, actorRole: 'VOTER', resourceType: 'VOTE', resourceId: vote.voteId || vote.transactionId, status: 'SUCCESS', description: 'Vote recorded to ledger' });
    return vote;
  }
  async getVote(voteId) { return this.adapter.getVote(voteId); }
  async getElectionVotes(electionId) { return this.adapter.getElectionVotes(electionId); }
  async getElectionHistory(electionId) { return this.adapter.getElectionHistory(electionId); }
  async verifyLedgerIntegrity(electionId) { return this.adapter.verifyLedgerIntegrity(electionId); }
  async getVoteByTransactionId(transactionId) {
    return typeof this.adapter.getVoteByTransactionId === 'function'
      ? this.adapter.getVoteByTransactionId(transactionId)
      : localLedger.getVoteByTransactionId(transactionId);
  }
  async verifyVoteReceipt(transactionId) {
    return typeof this.adapter.verifyVoteReceipt === 'function'
      ? this.adapter.verifyVoteReceipt(transactionId)
      : localLedger.verifyVoteReceipt(transactionId);
  }
  async health() {
    if (typeof this.adapter.health === 'function') return (await this.adapter.health()) ? 'ok' : 'unavailable';
    return 'ok';
  }
  async generateResults(electionId) {
    const election = await this.repository.getElection(electionId).catch(() => null);
    const votes = await this.adapter.getElectionVotes(electionId);
    const totals = {};
    for (const vote of votes) totals[vote.candidateId] = (totals[vote.candidateId] || 0) + 1;
    const result = {
      electionId,
      electionName: election ? election.electionName : 'Election results',
      status: 'RESULTS_DECLARED',
      totalVotesCast: votes.length,
      votesByCandidate: totals,
      generatedAt: new Date().toISOString(),
      integrity: typeof this.adapter.health === 'function'
        ? { status: 'FABRIC_LEDGER', verifiedAt: new Date().toISOString() }
        : await this.adapter.verifyLedgerIntegrity(electionId)
    };
    return result;
  }
}

module.exports = { BlockchainService, BlockchainAdapter, LocalLedgerAdapter, FabricAdapter, adapter, backendName };
