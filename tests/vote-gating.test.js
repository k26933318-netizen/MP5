const test = require('node:test');
const assert = require('node:assert/strict');
const { BlockchainService } = require('../backend/blockchain/blockchainService');

function setup({ enrolled = true, tokenValid = true, candidateExists = true, failFirstLedgerWrite = false } = {}) {
  const claims = new Set();
  const recorded = [];
  const audits = [];
  let shouldFailLedgerWrite = failFirstLedgerWrite;
  const repository = {
    async getVoterCredentials(voterId) {
      return voterId === 'VOTER-1' ? { voterId, status: 'APPROVED' } : null;
    },
    async getElection(electionId) {
      return {
        electionId,
        status: 'ACTIVE',
        startDate: new Date(Date.now() - 60_000).toISOString(),
        endDate: new Date(Date.now() + 60_000).toISOString()
      };
    },
    async getCandidate(candidateId, electionId) {
      return candidateExists ? { candidateId, electionId } : null;
    },
    async getFaceEnrollment() {
      return enrolled ? { updatedAt: new Date() } : null;
    },
    async consumeFaceVerification(input) {
      assert.equal(input.voterId, 'VOTER-1');
      assert.equal(input.electionId, 'ELECTION-1');
      assert.equal(input.sessionHash, 'session-fingerprint');
      assert.match(input.tokenHash, /^[a-f0-9]{64}$/);
      return { valid: tokenValid, code: 'VERIFICATION_TOKEN_INVALID' };
    },
    async createVoteClaim(electionId, voterHash) {
      const key = `${electionId}:${voterHash}`;
      if (claims.has(key)) {
        const error = new Error('Voter has already voted in this election');
        error.code = 'VOTE_ALREADY_CAST';
        error.statusCode = 409;
        throw error;
      }
      claims.add(key);
    },
    async releaseVoteClaim(electionId, voterHash) {
      claims.delete(`${electionId}:${voterHash}`);
    },
    async completeVoteClaim() {},
    async logAudit(...args) { audits.push(args); }
  };
  const adapter = {
    async recordVote(input) {
      if (shouldFailLedgerWrite) {
        shouldFailLedgerWrite = false;
        throw new Error('Fabric peer is unavailable');
      }
      const vote = {
        voteId: `vote_${recorded.length + 1}`,
        transactionId: `tx_${recorded.length + 1}`,
        status: 'CONFIRMED'
      };
      recorded.push({ input, vote });
      return vote;
    }
  };
  return { service: new BlockchainService(repository, adapter), recorded, audits };
}

const validRequest = {
  voterId: 'VOTER-1',
  electionId: 'ELECTION-1',
  candidateId: 'CANDIDATE-1',
  verificationToken: 'one-time-token',
  sessionHash: 'session-fingerprint',
  voterReference: 'ATTACKER-CONTROLLED-ID'
};

test('a vote without an enrolled face is rejected before ledger submission', async () => {
  const { service, recorded } = setup({ enrolled: false });
  await assert.rejects(() => service.castVote(validRequest), { code: 'FACE_NOT_ENROLLED' });
  assert.equal(recorded.length, 0);
});

test('a vote with an invalid verification token is rejected before ledger submission', async () => {
  const { service, recorded } = setup({ tokenValid: false });
  await assert.rejects(() => service.castVote(validRequest), { code: 'VERIFICATION_TOKEN_INVALID' });
  assert.equal(recorded.length, 0);
});

test('a valid one-time verification authorization permits a vote', async () => {
  const { service, recorded } = setup();
  const vote = await service.castVote(validRequest);
  assert.equal(vote.transactionId, 'tx_1');
  assert.equal(recorded.length, 1);
  assert.equal(recorded[0].input.electionId, validRequest.electionId);
  assert.equal(recorded[0].input.candidateId, validRequest.candidateId);
  assert.equal(recorded[0].input.voterReference, validRequest.voterId);
  assert.match(recorded[0].input.voterHash, /^[a-f0-9]{64}$/);
});

test('a failed ledger write releases the pending claim so the voter can retry', async () => {
  const { service, recorded } = setup({ failFirstLedgerWrite: true });
  await assert.rejects(() => service.castVote(validRequest), /Fabric peer is unavailable/);
  const vote = await service.castVote(validRequest);
  assert.equal(vote.transactionId, 'tx_1');
  assert.equal(recorded.length, 1);
});

test('simultaneous attempts for the same voter and election can record only one vote', async () => {
  const { service, recorded } = setup();
  const results = await Promise.allSettled([
    service.castVote({ ...validRequest, verificationToken: 'token-1' }),
    service.castVote({ ...validRequest, verificationToken: 'token-2' })
  ]);
  assert.equal(results.filter(result => result.status === 'fulfilled').length, 1);
  assert.equal(results.filter(result => result.status === 'rejected' && result.reason.code === 'VOTE_ALREADY_CAST').length, 1);
  assert.equal(recorded.length, 1);
});

test('an invalid candidate cannot consume a vote authorization into a ledger transaction', async () => {
  const { service, recorded } = setup({ candidateExists: false });
  await assert.rejects(() => service.castVote(validRequest), /Candidate does not belong/);
  assert.equal(recorded.length, 0);
});
