const { Contract } = require('fabric-contract-api');

const json = value => JSON.stringify(value);
const read = bytes => JSON.parse(Buffer.from(bytes).toString('utf8'));
const assertId = (value, name) => {
  if (!value || typeof value !== 'string' || !/^[A-Za-z0-9_.:-]{1,128}$/.test(value)) throw new Error(`Invalid ${name}`);
};

class VotingContract extends Contract {
  async ping() { return json({ status: 'ok' }); }

  async _get(ctx, key) {
    const value = await ctx.stub.getState(key);
    if (!value || value.length === 0) return null;
    return read(value);
  }

  async registerElection(ctx, electionId, electionName, startDate, endDate, status = 'DRAFT') {
    assertId(electionId, 'electionId');
    if (!electionName || !Date.parse(startDate) || !Date.parse(endDate) || Date.parse(startDate) >= Date.parse(endDate)) throw new Error('Invalid election details');
    if (!['DRAFT', 'UPCOMING', 'ACTIVE', 'CLOSED', 'RESULTS_DECLARED'].includes(status)) throw new Error('Invalid election status');
    const existing = await this._get(ctx, `election~${electionId}`);
    if (existing) {
      if (existing.electionName !== electionName || existing.startDate !== startDate || existing.endDate !== endDate) {
        throw new Error('Election already exists with different details');
      }
      return json(existing);
    }
    const election = { electionId, electionName, startDate, endDate, status };
    await ctx.stub.putState(`election~${electionId}`, Buffer.from(json(election)));
    return json(election);
  }

  async updateElectionStatus(ctx, electionId, status) {
    assertId(electionId, 'electionId');
    const election = await this._get(ctx, `election~${electionId}`);
    if (!election) throw new Error('Election does not exist');
    const allowed = {
      DRAFT: ['UPCOMING'],
      UPCOMING: ['ACTIVE'],
      ACTIVE: ['CLOSED'],
      CLOSED: ['RESULTS_DECLARED']
    };
    if (!(allowed[election.status] || []).includes(status)) {
      throw new Error(`Invalid election state transition: ${election.status} -> ${status}`);
    }
    election.status = status;
    await ctx.stub.putState(`election~${electionId}`, Buffer.from(json(election)));
    return json(election);
  }

  async getElection(ctx, electionId) {
    assertId(electionId, 'electionId');
    const election = await this._get(ctx, `election~${electionId}`);
    return json(election);
  }

  async registerCandidate(ctx, candidateId, electionId, candidateName, party = '') {
    assertId(candidateId, 'candidateId'); assertId(electionId, 'electionId');
    if (!candidateName) throw new Error('Invalid candidateName');
    if (!await this._get(ctx, `election~${electionId}`)) throw new Error('Election does not exist');
    const candidate = { candidateId, electionId, candidateName, party };
    const existing = await this._get(ctx, `candidate~${candidateId}`);
    if (existing) {
      if (existing.electionId !== electionId || existing.candidateName !== candidateName || existing.party !== party) {
        throw new Error('Candidate already exists with different details');
      }
      return json(existing);
    }
    await ctx.stub.putState(`candidate~${candidateId}`, Buffer.from(json(candidate)));
    return json(candidate);
  }

  async getCandidate(ctx, candidateId) {
    assertId(candidateId, 'candidateId');
    return json(await this._get(ctx, `candidate~${candidateId}`));
  }

  async castVote(ctx, electionId, voterReference, candidateId) {
    assertId(electionId, 'electionId'); assertId(voterReference, 'voterReference'); assertId(candidateId, 'candidateId');
    if (!/^[a-f0-9]{64}$/i.test(voterReference)) throw new Error('Voter reference must be a SHA-256 hash');
    const election = await this._get(ctx, `election~${electionId}`);
    if (!election) throw new Error('Election does not exist');
    const now = Date.now();
    if (election.status !== 'ACTIVE' || now < Date.parse(election.startDate) || now > Date.parse(election.endDate)) throw new Error('Election is inactive or has ended');
    const candidate = await this._get(ctx, `candidate~${candidateId}`);
    if (!candidate) throw new Error('Candidate does not exist');
    if (candidate.electionId !== electionId) throw new Error('Candidate does not belong to election');
    const voterKey = `voted~${electionId}~${voterReference}`;
    if (await this._get(ctx, voterKey)) throw new Error('Voter has already voted');
    const transactionId = ctx.stub.getTxID();
    const timestamp = new Date(ctx.stub.getTxTimestamp().seconds.low * 1000).toISOString();
    const vote = { voteId: `vote_${transactionId}`, electionId, voterReferenceHash: voterReference, candidateId, timestamp, transactionId };
    await ctx.stub.putState(`vote~${vote.voteId}`, Buffer.from(json(vote)));
    await ctx.stub.putState(voterKey, Buffer.from(json({ electionId, voterReference, voteId: vote.voteId, transactionId })));
    await ctx.stub.putState(`transaction~${transactionId}`, Buffer.from(json({ voteId: vote.voteId })));
    ctx.stub.setEvent('VoteCast', Buffer.from(json({ voteId: vote.voteId, electionId, candidateId, transactionId })));
    return json(vote);
  }

  async getVote(ctx, voteId) { assertId(voteId, 'voteId'); const vote = await this._get(ctx, `vote~${voteId}`); if (!vote) throw new Error('Vote does not exist'); return json(vote); }

  async getVoteByTransactionId(ctx, transactionId) {
    assertId(transactionId, 'transactionId');
    const index = await this._get(ctx, `transaction~${transactionId}`);
    if (!index) throw new Error('Transaction does not exist');
    return this.getVote(ctx, index.voteId);
  }

  async getElectionVotes(ctx, electionId) {
    assertId(electionId, 'electionId'); const iterator = await ctx.stub.getStateByRange(`vote~vote_`, `vote~vote_\uffff`); const result = [];
    for await (const item of iterator) { const vote = read(item.value); if (vote.electionId === electionId) result.push(vote); }
    return json(result);
  }

  async getElectionHistory(ctx, electionId) {
    const votes = read(await this.getElectionVotes(ctx, electionId)); const result = [];
    for (const vote of votes) { const history = await ctx.stub.getHistoryForKey(`vote~${vote.voteId}`); const entries = []; for await (const item of history) entries.push({ transactionId: item.txId, timestamp: item.timestamp, isDelete: item.is_delete, value: item.value.length ? read(item.value) : null }); result.push({ voteId: vote.voteId, history: entries }); }
    return json(result);
  }
}
module.exports = VotingContract;
