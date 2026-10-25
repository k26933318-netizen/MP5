const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const auth = require('../backend/auth');
const mongo = require('../backend/data/mongoRepository');
const { appendVote, verifyLedgerIntegrity, getElectionVotes, verifyVoteReceipt } = require('../backend/blockchain/localLedger');

const ledgerPath = path.resolve(__dirname, '../backend/data/test-votes.ndjson');

test.after(async () => {
  await mongo.closeConnection();
});

async function withTempLedger(fn) {
  const original = process.env.VOTES_LEDGER_PATH;
  process.env.VOTES_LEDGER_PATH = ledgerPath;
  if (fs.existsSync(ledgerPath)) fs.unlinkSync(ledgerPath);
  try {
    return await fn();
  } finally {
    if (original === undefined) delete process.env.VOTES_LEDGER_PATH;
    else process.env.VOTES_LEDGER_PATH = original;
    if (fs.existsSync(ledgerPath)) fs.unlinkSync(ledgerPath);
  }
}

test('password hashing uses bcrypt and can verify a valid password', async () => {
  const hash = await auth.hashPassword('StrongPass!123');
  assert.match(hash, /^\$2[aby]\$\d{2}\$/);
  assert.equal(await auth.verifyPassword('StrongPass!123', hash), true);
  assert.equal(await auth.verifyPassword('WrongPass', hash), false);
});

test('duplicate vote is rejected before ledger append', () => {
  withTempLedger(() => {
    const first = appendVote({ electionId: 'ELEC-1', voterReference: 'Voter-10', candidateId: 'CAND-1' });
    assert.ok(first.transactionId);
    assert.throws(() => appendVote({ electionId: 'ELEC-1', voterReference: 'voter-10', candidateId: 'CAND-2' }), /already (voted|recorded)/i);
  });
});

test('ledger integrity returns valid when chain is intact', () => {
  withTempLedger(() => {
    appendVote({ electionId: 'ELEC-2', voterReference: 'User-1', candidateId: 'A' });
    appendVote({ electionId: 'ELEC-2', voterReference: 'User-2', candidateId: 'B' });
    const result = verifyLedgerIntegrity('ELEC-2');
    assert.equal(result.status, 'VALID');
    assert.equal(result.invalidHashes, 0);
    assert.equal(result.brokenLinks, 0);
  });
});

test('ledger integrity detects tampering', () => {
  withTempLedger(() => {
    const vote = appendVote({ electionId: 'ELEC-3', voterReference: 'User-1', candidateId: 'A' });
    const filePath = path.resolve(__dirname, '../backend/data/test-votes.ndjson');
    const lines = fs.readFileSync(filePath, 'utf8').trim().split(/\r?\n/);
    const tampered = JSON.parse(lines[0]);
    tampered.candidateId = 'Tampered';
    tampered.recordHash = 'bad-hash';
    fs.writeFileSync(filePath, JSON.stringify(tampered) + '\n', 'utf8');
    const result = verifyLedgerIntegrity('ELEC-3');
    assert.equal(result.status, 'FAILED');
    assert.ok(result.invalidHashes >= 1 || result.brokenLinks >= 1);
    assert.equal(vote.electionId, 'ELEC-3');
  });
});

test('election state transitions reject invalid status changes', async () => {
  const { BlockchainService } = require('../backend/blockchain/blockchainService');
  const service = new BlockchainService({ getElection: async () => null }, {});
  await assert.rejects(
    () => service.updateElectionState('ELEC-INVALID', 'ACTIVE'),
    /Election does not exist/i
  );
});

test('receipt verification succeeds for a valid transaction', () => {
  withTempLedger(() => {
    const vote = appendVote({ electionId: 'ELEC-RESULTS', voterReference: 'ReceiptUser', candidateId: 'C1' });
    const result = verifyVoteReceipt(vote.transactionId);
    assert.equal(result.status, 'VERIFIED');
    assert.equal(result.electionId, 'ELEC-RESULTS');
  });
});

test('results generation totals votes across candidates', async () => {
  const { BlockchainService, LocalLedgerAdapter } = require('../backend/blockchain/blockchainService');
  const service = new BlockchainService(undefined, new LocalLedgerAdapter());
  await withTempLedger(async () => {
    appendVote({ electionId: 'ELEC-COUNTS', voterReference: 'A', candidateId: 'CAND-1' });
    appendVote({ electionId: 'ELEC-COUNTS', voterReference: 'B', candidateId: 'CAND-1' });
    appendVote({ electionId: 'ELEC-COUNTS', voterReference: 'C', candidateId: 'CAND-2' });
    const results = await service.generateResults('ELEC-COUNTS');
    assert.equal(results.totalVotesCast, 3);
    assert.equal(results.votesByCandidate['CAND-1'], 2);
    assert.equal(results.votesByCandidate['CAND-2'], 1);
  });
});
