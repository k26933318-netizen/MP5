const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const defaultLedgerPath = path.resolve(__dirname, '../data/votes.ndjson');
const getLedgerPath = () => process.env.VOTES_LEDGER_PATH || defaultLedgerPath;

function ensureLedger() {
  const ledgerPath = getLedgerPath();
  fs.mkdirSync(path.dirname(ledgerPath), { recursive: true });
  if (!fs.existsSync(ledgerPath)) fs.writeFileSync(ledgerPath, '', 'utf8');
}

function readVotes() {
  const ledgerPath = getLedgerPath();
  ensureLedger();
  return fs.readFileSync(ledgerPath, 'utf8')
    .split(/\r?\n/)
    .filter(Boolean)
    .map(line => JSON.parse(line));
}

function computeRecordHash(payload) {
  return crypto.createHash('sha256').update(JSON.stringify(payload)).digest('hex');
}

function normalizeVoterReference(value) {
  return String(value || '').trim().toLowerCase();
}

function appendVote(input) {
  const votes = readVotes();
  const voterHash = crypto.createHash('sha256').update(`${input.electionId}:${normalizeVoterReference(input.voterReference)}`).digest('hex');
  if (votes.some(vote => vote.electionId === input.electionId && vote.voterHash === voterHash)) {
    const error = new Error('Vote already recorded for this election.');
    error.code = 'DUPLICATE_VOTE';
    throw error;
  }

  const transactionId = `local_${crypto.randomBytes(12).toString('hex')}`;
  const timestamp = new Date().toISOString();
  const electionVotes = votes.filter(vote => vote.electionId === input.electionId);
  const previousHash = electionVotes.length ? electionVotes[electionVotes.length - 1].recordHash : 'GENESIS';
  const payload = { transactionId, electionId: input.electionId, candidateId: input.candidateId, voterHash, timestamp, previousHash, status: 'CONFIRMED' };
  const recordHash = computeRecordHash(payload);
  const vote = { voteId: `vote_${transactionId}`, ...payload, recordHash };
  fs.appendFileSync(getLedgerPath(), `${JSON.stringify(vote)}\n`, 'utf8');
  return vote;
}

function getVote(voteId) {
  const vote = readVotes().find(item => item.voteId === voteId);
  if (!vote) throw new Error('Vote does not exist');
  return vote;
}

function getVoteByTransactionId(transactionId) {
  const vote = readVotes().find(item => item.transactionId === transactionId);
  if (!vote) throw new Error('Transaction does not exist');
  return vote;
}

function getElectionVotes(electionId) {
  return readVotes().filter(vote => vote.electionId === electionId);
}

function getElectionHistory(electionId) {
  return getElectionVotes(electionId).map(vote => ({
    voteId: vote.voteId,
    history: [{ transactionId: vote.transactionId, timestamp: vote.timestamp, value: vote, recordHash: vote.recordHash }]
  }));
}

function verifyLedgerIntegrity(electionId) {
  const votes = getElectionVotes(electionId);
  let invalidHashes = 0;
  let brokenLinks = 0;
  let invalidTransactions = 0;
  let checked = 0;
  const seenTxIds = new Set();
  let previousHash = null;

  for (const vote of votes) {
    checked += 1;
    if (!vote.transactionId || !vote.recordHash || !vote.timestamp || !vote.electionId || !vote.candidateId || !vote.voterHash) {
      invalidTransactions += 1;
      continue;
    }
    if (seenTxIds.has(vote.transactionId)) {
      invalidTransactions += 1;
    }
    seenTxIds.add(vote.transactionId);
    const expectedPreviousHash = previousHash || 'GENESIS';
    if (vote.previousHash !== expectedPreviousHash) {
      brokenLinks += 1;
    }
    const payload = { transactionId: vote.transactionId, electionId: vote.electionId, candidateId: vote.candidateId, voterHash: vote.voterHash, timestamp: vote.timestamp, previousHash: vote.previousHash, status: vote.status || 'CONFIRMED' };
    const hash = computeRecordHash(payload);
    if (hash !== vote.recordHash) {
      invalidHashes += 1;
    }
    previousHash = vote.recordHash;
  }

  const status = invalidHashes === 0 && brokenLinks === 0 && invalidTransactions === 0 ? 'VALID' : 'FAILED';
  return { status, electionId, recordsChecked: checked, validRecords: checked - (invalidHashes + brokenLinks + invalidTransactions), invalidHashes, brokenLinks, invalidTransactions, verifiedAt: new Date().toISOString() };
}

function verifyVoteReceipt(transactionId) {
  const vote = getVoteByTransactionId(transactionId);
  const integrity = verifyLedgerIntegrity(vote.electionId);
  const valid = integrity.status === 'VALID' && vote.recordHash && vote.transactionId === transactionId;
  return {
    transactionId,
    electionId: vote.electionId,
    voteId: vote.voteId,
    recordHash: vote.recordHash,
    status: valid ? 'VERIFIED' : 'FAILED',
    integrity,
    verifiedAt: new Date().toISOString()
  };
}

module.exports = { appendVote, getVote, getVoteByTransactionId, getElectionVotes, getElectionHistory, verifyLedgerIntegrity, verifyVoteReceipt, computeRecordHash };