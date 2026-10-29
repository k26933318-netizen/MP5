const test = require('node:test');
const assert = require('node:assert/strict');
const bcrypt = require('bcrypt');
const auth = require('../backend/auth');
const mongo = require('../backend/data/mongoRepository');

test('voter, admin, officer, and candidate credentials create the expected sessions', async () => {
  const methods = {
    voter: mongo.getVoterCredentials,
    officer: mongo.getElectionOfficerCredentials,
    candidate: mongo.getCandidateCredentials
  };
  const previousAdminUsername = process.env.ADMIN_USERNAME;
  const previousAdminPassword = process.env.ADMIN_PASSWORD;
  const passwordHash = await bcrypt.hash('TestPass!123', 4);
  process.env.ADMIN_USERNAME = 'test-admin';
  process.env.ADMIN_PASSWORD = 'AdminPass!123';
  mongo.getVoterCredentials = async username => username === 'voter-test'
    ? { voterId: 'voter-test', name: 'Voter Test', status: 'APPROVED', passwordHash }
    : null;
  mongo.getElectionOfficerCredentials = async username => username === 'officer-test'
    ? { officerId: 'OFF-TEST', username, name: 'Officer Test', status: 'APPROVED', passwordHash }
    : null;
  mongo.getCandidateCredentials = async username => username === 'candidate-test'
    ? { candidateId: 'CAND-TEST', username, name: 'Candidate Test', status: 'APPROVED', passwordHash, partyId: 'PARTY-TEST' }
    : null;

  try {
    const logins = [
      ['voter', 'voter-test', 'voter'],
      ['admin', 'test-admin', 'admin'],
      ['election_officer', 'officer-test', 'election_officer'],
      ['candidate', 'candidate-test', 'candidate']
    ];
    for (const [role, username, expectedRole] of logins) {
      const session = await auth.login({
        role,
        username,
        password: role === 'admin' ? 'AdminPass!123' : 'TestPass!123'
      });
      assert.equal(session.user.role, expectedRole);
      assert.ok(session.token);
      auth.sessions.delete(session.token);
    }
  } finally {
    mongo.getVoterCredentials = methods.voter;
    mongo.getElectionOfficerCredentials = methods.officer;
    mongo.getCandidateCredentials = methods.candidate;
    if (previousAdminUsername === undefined) delete process.env.ADMIN_USERNAME;
    else process.env.ADMIN_USERNAME = previousAdminUsername;
    if (previousAdminPassword === undefined) delete process.env.ADMIN_PASSWORD;
    else process.env.ADMIN_PASSWORD = previousAdminPassword;
  }
});

test('suspended election officers and candidates cannot sign in', async () => {
  const originalGetOfficer = mongo.getElectionOfficerCredentials;
  const originalGetCandidate = mongo.getCandidateCredentials;
  const passwordHash = await bcrypt.hash('TestPass!123', 4);
  mongo.getElectionOfficerCredentials = async () => ({
    officerId: 'OFF-SUSPENDED', username: 'officer-suspended', status: 'SUSPENDED', passwordHash
  });
  mongo.getCandidateCredentials = async () => ({
    candidateId: 'CAND-SUSPENDED', username: 'candidate-suspended', status: 'SUSPENDED', passwordHash
  });

  try {
    await assert.rejects(() => auth.login({ role: 'election_officer', username: 'officer-suspended', password: 'TestPass!123' }), /Invalid election officer credentials/);
    await assert.rejects(() => auth.login({ role: 'candidate', username: 'candidate-suspended', password: 'TestPass!123' }), /Invalid candidate credentials/);
  } finally {
    mongo.getElectionOfficerCredentials = originalGetOfficer;
    mongo.getCandidateCredentials = originalGetCandidate;
  }
});

test('pending voters, invalid passwords, and unknown roles cannot sign in', async () => {
  const originalGetVoter = mongo.getVoterCredentials;
  const originalGetOfficer = mongo.getElectionOfficerCredentials;
  const previousAdminUsername = process.env.ADMIN_USERNAME;
  const previousAdminPassword = process.env.ADMIN_PASSWORD;
  const passwordHash = await bcrypt.hash('TestPass!123', 4);
  process.env.ADMIN_USERNAME = 'test-admin-denied';
  process.env.ADMIN_PASSWORD = 'AdminPass!123';
  mongo.getVoterCredentials = async username => username === 'pending-test'
    ? { voterId: username, status: 'PENDING', passwordHash }
    : null;
  mongo.getElectionOfficerCredentials = async () => null;

  try {
    await assert.rejects(() => auth.login({ role: 'voter', username: 'pending-test', password: 'TestPass!123' }));
    await assert.rejects(() => auth.login({ role: 'election_officer', username: 'missing-officer', password: 'TestPass!123' }));
    await assert.rejects(() => auth.login({ role: 'admin', username: 'test-admin-denied', password: 'incorrect' }));
    await assert.rejects(() => auth.login({ role: 'unknown', username: 'someone', password: 'TestPass!123' }), /Invalid account type/);
  } finally {
    mongo.getVoterCredentials = originalGetVoter;
    mongo.getElectionOfficerCredentials = originalGetOfficer;
    if (previousAdminUsername === undefined) delete process.env.ADMIN_USERNAME;
    else process.env.ADMIN_USERNAME = previousAdminUsername;
    if (previousAdminPassword === undefined) delete process.env.ADMIN_PASSWORD;
    else process.env.ADMIN_PASSWORD = previousAdminPassword;
  }
});
