const test = require('node:test');
const assert = require('node:assert/strict');
const VotingContract = require('../lib/votingContract');

function context() {
  const state = new Map(); let tx = 0;
  const stub = {
    getState: async key => state.get(key) || Buffer.alloc(0),
    putState: async (key, value) => state.set(key, value),
    getTxID: () => `tx${++tx}`,
    getTxTimestamp: () => ({ seconds: { low: Math.floor(Date.now() / 1000) } }),
    setEvent: () => {},
    getStateByRange: async function* () { for (const [key, value] of state) if (key.startsWith('vote~')) yield { key, value }; },
    getHistoryForKey: async function* () {}
  };
  return { stub };
}
const dates = () => { const start = new Date(Date.now() - 60_000).toISOString(); const end = new Date(Date.now() + 60_000).toISOString(); return [start, end]; };

const voterHash = 'a'.repeat(64);
 test('accepts a vote and rejects a duplicate', async () => { const contract = new VotingContract(); const ctx = context(); const [start, end] = dates(); await contract.registerElection(ctx, 'E1', 'Election', start, end, 'ACTIVE'); await contract.registerCandidate(ctx, 'C1', 'E1', 'Candidate'); const vote = JSON.parse(await contract.castVote(ctx, 'E1', voterHash, 'C1')); assert.equal(vote.candidateId, 'C1'); assert.equal(vote.voterReferenceHash, voterHash); await assert.rejects(() => contract.castVote(ctx, 'E1', voterHash, 'C1'), /already voted/); });
test('rejects raw voter identifiers on the Fabric ledger', async () => { const contract = new VotingContract(); const ctx = context(); const [start, end] = dates(); await contract.registerElection(ctx, 'E1', 'Election', start, end, 'ACTIVE'); await contract.registerCandidate(ctx, 'C1', 'E1', 'Candidate'); await assert.rejects(() => contract.castVote(ctx, 'E1', 'VOTER001', 'C1'), /SHA-256 hash/); });
test('rejects a candidate from another election', async () => { const contract = new VotingContract(); const ctx = context(); const [start, end] = dates(); await contract.registerElection(ctx, 'E1', 'Election', start, end, 'ACTIVE'); await contract.registerElection(ctx, 'E2', 'Election 2', start, end, 'ACTIVE'); await contract.registerCandidate(ctx, 'C2', 'E2', 'Candidate'); await assert.rejects(() => contract.castVote(ctx, 'E1', voterHash, 'C2'), /does not belong/); });
test('rejects an inactive election', async () => { const contract = new VotingContract(); const ctx = context(); await contract.registerElection(ctx, 'E1', 'Election', '2020-01-01T00:00:00.000Z', '2020-01-02T00:00:00.000Z'); await contract.registerCandidate(ctx, 'C1', 'E1', 'Candidate'); await assert.rejects(() => contract.castVote(ctx, 'E1', voterHash, 'C1'), /inactive/); });
test('returns a Fabric transaction-derived vote ID and transaction ID', async () => { const contract = new VotingContract(); const ctx = context(); const [start, end] = dates(); await contract.registerElection(ctx, 'E1', 'Election', start, end, 'ACTIVE'); await contract.registerCandidate(ctx, 'C1', 'E1', 'Candidate'); const vote = JSON.parse(await contract.castVote(ctx, 'E1', voterHash, 'C1')); assert.equal(vote.transactionId, 'tx1'); assert.equal(vote.voteId, 'vote_tx1'); });
test('rejects an unknown candidate', async () => { const contract = new VotingContract(); const ctx = context(); const [start, end] = dates(); await contract.registerElection(ctx, 'E1', 'Election', start, end, 'ACTIVE'); await assert.rejects(() => contract.castVote(ctx, 'E1', voterHash, 'UNKNOWN'), /Candidate does not exist/); });
test('returns election vote records from the ledger state', async () => { const contract = new VotingContract(); const ctx = context(); const [start, end] = dates(); await contract.registerElection(ctx, 'E1', 'Election', start, end, 'ACTIVE'); await contract.registerCandidate(ctx, 'C1', 'E1', 'Candidate'); await contract.castVote(ctx, 'E1', voterHash, 'C1'); assert.equal(JSON.parse(await contract.getElectionVotes(ctx, 'E1')).length, 1); });
test('looks up a vote by its Fabric transaction ID', async () => { const contract = new VotingContract(); const ctx = context(); const [start, end] = dates(); await contract.registerElection(ctx, 'E1', 'Election', start, end, 'ACTIVE'); await contract.registerCandidate(ctx, 'C1', 'E1', 'Candidate'); await contract.castVote(ctx, 'E1', voterHash, 'C1'); const vote = JSON.parse(await contract.getVoteByTransactionId(ctx, 'tx1')); assert.equal(vote.transactionId, 'tx1'); });
test('election and candidate synchronization is idempotent', async () => { const contract = new VotingContract(); const ctx = context(); const [start, end] = dates(); const firstElection = JSON.parse(await contract.registerElection(ctx, 'E1', 'Election', start, end, 'ACTIVE')); const repeatedElection = JSON.parse(await contract.registerElection(ctx, 'E1', 'Election', start, end, 'ACTIVE')); assert.deepEqual(repeatedElection, firstElection); const firstCandidate = JSON.parse(await contract.registerCandidate(ctx, 'C1', 'E1', 'Candidate', 'Party')); const repeatedCandidate = JSON.parse(await contract.registerCandidate(ctx, 'C1', 'E1', 'Candidate', 'Party')); assert.deepEqual(repeatedCandidate, firstCandidate); });
test('election statuses follow the allowed lifecycle', async () => { const contract = new VotingContract(); const ctx = context(); const [start, end] = dates(); await contract.registerElection(ctx, 'E1', 'Election', start, end, 'DRAFT'); const updated = JSON.parse(await contract.updateElectionStatus(ctx, 'E1', 'UPCOMING')); assert.equal(updated.status, 'UPCOMING'); await assert.rejects(() => contract.updateElectionStatus(ctx, 'E1', 'CLOSED'), /Invalid election state transition/); assert.equal(JSON.parse(await contract.getElection(ctx, 'E1')).status, 'UPCOMING'); });
