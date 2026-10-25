const { MongoClient } = require('mongodb');
const crypto = require('crypto');
const bcrypt = require('bcrypt');

const uri = process.env.MONGODB_URI || 'mongodb://127.0.0.1:27017';
const databaseName = process.env.MONGODB_DATABASE || 'VotingSystem';
const FACE_ENROLLMENT_LOCK_ID = 'face-enrollment';
const FACE_ENROLLMENT_LOCK_LEASE_MS = 2 * 60 * 1000;
const FACE_ENROLLMENT_LOCK_WAIT_MS = 30 * 1000;
let client;
let database;
let databasePromise;

const VALID_ELECTION_STATES = ['DRAFT', 'UPCOMING', 'ACTIVE', 'CLOSED', 'RESULTS_DECLARED'];
const VALID_ELECTION_TRANSITIONS = {
  DRAFT: ['UPCOMING'],
  UPCOMING: ['ACTIVE'],
  ACTIVE: ['CLOSED'],
  CLOSED: ['RESULTS_DECLARED']
};

async function db() {
  if (database) return database;
  if (!databasePromise) {
    databasePromise = initializeDatabase();
  }
  const connectionAttempt = databasePromise;
  try {
    return await connectionAttempt;
  } catch (error) {
    if (databasePromise === connectionAttempt) databasePromise = null;
    throw error;
  }
}

async function initializeDatabase() {
  const mongoClient = new MongoClient(uri, {
    serverSelectionTimeoutMS: Number(process.env.MONGODB_SERVER_SELECTION_TIMEOUT_MS || 5000)
  });
  try {
    await mongoClient.connect();
    const connectedDatabase = mongoClient.db(databaseName);
    try {
      await connectedDatabase.collection('faceEnrollmentLocks').insertOne({
        _id: FACE_ENROLLMENT_LOCK_ID,
        owner: null,
        expiresAt: new Date(0)
      });
    } catch (error) {
      if (error.code !== 11000) throw error;
    }
    await connectedDatabase.collection('voters').createIndex({ voterId: 1 }, { unique: true });
    await connectedDatabase.collection('faceEncodings').createIndex({ voterId: 1 }, { unique: true });
    await connectedDatabase.collection('faceChallenges').createIndex({ challengeId: 1 }, { unique: true });
    await connectedDatabase.collection('faceChallenges').createIndex({ expiresAt: 1 }, { expireAfterSeconds: 0 });
    await connectedDatabase.collection('faceEncodings').updateMany(
      { enrolled: { $exists: false } },
      {
        $set: { enrolled: false, updatedAt: new Date() },
        $unset: { encryptedEmbedding: '' }
      }
    );
    await connectedDatabase.collection('voteClaims').createIndex({ electionId: 1, voterHash: 1 }, { unique: true });
    await connectedDatabase.collection('voteClaims').createIndex({ createdAt: -1 });
    await connectedDatabase.collection('elections').createIndex({ electionId: 1 }, { unique: true });
    await connectedDatabase.collection('candidates').createIndex({ candidateId: 1 }, { unique: true });
    await connectedDatabase.collection('auditLogs').createIndex({ timestamp: -1 });
    await connectedDatabase.collection('auditLogs').createIndex({ eventType: 1 });
    await seed(connectedDatabase);
    client = mongoClient;
    database = connectedDatabase;
    return connectedDatabase;
  } catch (error) {
    try {
      await mongoClient.close();
    } catch (closeError) {
      console.error('Failed to close MongoDB client after initialization error:', closeError.message);
    }
    throw error;
  }
}

async function seed(database) {
  const elections = database.collection('elections');
  const candidates = database.collection('candidates');
  const parties = database.collection('parties');
  const alliances = database.collection('alliances');
  const voters = database.collection('voters');
  const officers = database.collection('electionOfficers');
  await elections.updateOne({ electionId: 'TN-ASSEMBLY-2026' }, { $setOnInsert: { electionId: 'TN-ASSEMBLY-2026', electionName: 'Tamil Nadu Assembly Election 2026', startDate: '2026-01-01T00:00:00.000Z', endDate: '2026-12-31T23:59:59.000Z', status: 'ACTIVE', createdAt: new Date() } }, { upsert: true });
  const defaults = [
    ['ARJUN', 'Arjun', 'Progress Alliance'],
    ['VIDYUT', 'Vidyut Vel Pandian', 'People First'],
    ['RAVI', 'Ravi', 'New Tamil Front']
  ];
  for (const [candidateId, candidateName, party] of defaults) await candidates.updateOne({ candidateId }, { $setOnInsert: { candidateId, electionId: 'TN-ASSEMBLY-2026', candidateName, party, createdAt: new Date() } }, { upsert: true });

  const seededParties = [
    { partyId: 'PARTY-001', partyName: 'Progress Alliance', partySymbol: 'Rising Sun', presidentName: 'Meena Krishnan', foundedYear: 1988 },
    { partyId: 'PARTY-002', partyName: 'People First', partySymbol: 'Lamp', presidentName: 'Arun Prakash', foundedYear: 1996 },
    { partyId: 'PARTY-003', partyName: 'New Tamil Front', partySymbol: 'Tree', presidentName: 'Ravi Subramaniam', foundedYear: 2011 }
  ];
  for (const party of seededParties) await parties.updateOne({ partyId: party.partyId }, { $setOnInsert: { ...party, createdAt: new Date() } }, { upsert: true });

  await alliances.updateOne({ allianceId: 'ALLIANCE-001' }, { $setOnInsert: { allianceId: 'ALLIANCE-001', allianceName: 'United State Front', allianceSymbol: 'Unity', leaderPartyId: 'PARTY-001', formedYear: 2024, createdAt: new Date() } }, { upsert: true });

  const demoPasswordHash = await bcrypt.hash('Vote@123', 12);
  await voters.updateOne({ voterId: 'VOTER001' }, { $setOnInsert: { voterId: 'VOTER001', name: 'Demo Voter', passwordHash: demoPasswordHash, dob: '1990-01-15', phone: '+91 90000 00001', email: 'voter001@example.com', address: { city: 'Chennai', district: 'Chennai', pincode: '600001' }, aadhaarRef: 'DEMO-REF-001', role: 'VOTER', status: 'APPROVED', createdAt: new Date() } }, { upsert: true });

  const officerPasswordHash = await bcrypt.hash('Officer@123', 12);
  await officers.updateOne({ officerId: 'OFF-001' }, { $setOnInsert: { officerId: 'OFF-001', name: 'Kavitha Raman', username: 'officer1', email: 'officer1@example.com', contactNumber: '+91 90000 00002', passwordHash: officerPasswordHash, role: 'ELECTION_OFFICER', status: 'APPROVED', createdAt: new Date() } }, { upsert: true });

  const candidatePasswordHash = await bcrypt.hash('Candidate@123', 12);
  await candidates.updateOne({ candidateId: 'CAND-001' }, { $setOnInsert: { candidateId: 'CAND-001', name: 'Arjun Kumar', candidateName: 'Arjun Kumar', username: 'candidate1', email: 'candidate1@example.com', passwordHash: candidatePasswordHash, partyId: 'PARTY-001', partyName: 'Progress Alliance', electionId: 'TN-ASSEMBLY-2026', status: 'APPROVED', role: 'CANDIDATE', createdAt: new Date() } }, { upsert: true });
}

function validateElectionStateTransition(currentStatus, nextStatus) {
  const allowed = VALID_ELECTION_TRANSITIONS[currentStatus] || [];
  if (!allowed.includes(nextStatus)) throw new Error(`Invalid election state transition: ${currentStatus} -> ${nextStatus}`);
}

async function listElections() { return (await db()).collection('elections').find({}, { projection: { _id: 0 } }).sort({ startDate: 1 }).toArray(); }
async function getElection(electionId) { return (await db()).collection('elections').findOne({ electionId }, { projection: { _id: 0 } }); }
async function getCandidates(electionId) { return (await db()).collection('candidates').find({ electionId, role: { $exists: false } }, { projection: { _id: 0 } }).sort({ candidateName: 1 }).toArray(); }
async function getCandidate(candidateId, electionId) { return (await db()).collection('candidates').findOne({ candidateId, electionId }, { projection: { _id: 0 } }); }

async function createElection(input) {
  const election = {
    ...input,
    status: input.status || 'DRAFT',
    createdAt: new Date(),
    updatedAt: new Date()
  };
  if (!VALID_ELECTION_STATES.includes(election.status)) throw new Error('Invalid election status');
  await (await db()).collection('elections').insertOne(election);
  return { ...election, _id: undefined };
}

async function updateElection(electionId, updates, actor = null) {
  const election = await getElection(electionId);
  if (!election) throw new Error('Election does not exist');
  const nextStatus = updates.status || election.status;
  validateElectionStateTransition(election.status, nextStatus);
  const merged = { ...election, ...updates, updatedAt: new Date() };
  await (await db()).collection('elections').updateOne({ electionId }, { $set: merged });
  await logAudit('ELECTION_UPDATED', {
    actorId: actor || 'SYSTEM',
    actorRole: actor ? 'ADMIN' : 'SYSTEM',
    resourceType: 'ELECTION',
    resourceId: electionId,
    status: 'SUCCESS',
    description: `Election ${electionId} updated to ${nextStatus}`
  });
  return merged;
}

async function createCandidate(input) {
  const candidate = { ...input, party: input.party || '', createdAt: new Date() };
  await (await db()).collection('candidates').insertOne(candidate);
  return { ...candidate, _id: undefined };
}

async function registerParty(input) {
  const party = {
    partyId: input.partyId || `PARTY-${Date.now()}`,
    partyName: input.partyName,
    partySymbol: input.partySymbol || '',
    imageData: input.imageData || null,
    presidentName: input.presidentName || '',
    foundedYear: input.foundedYear || new Date().getFullYear(),
    createdAt: new Date()
  };
  await (await db()).collection('parties').insertOne(party);
  return { ...party, _id: undefined };
}

async function updatePartyLogo(partyId, imageData) {
  const result = await (await db()).collection('parties').updateOne(
    { partyId },
    { $set: { imageData, updatedAt: new Date() } }
  );
  if (!result.matchedCount) {
    const error = new Error('Party does not exist.');
    error.code = 'PARTY_NOT_FOUND';
    error.statusCode = 404;
    throw error;
  }
  return getParty(partyId);
}

async function listParties() { return (await db()).collection('parties').find({}, { projection: { _id: 0 } }).sort({ partyName: 1 }).toArray(); }
async function getParty(partyId) { return (await db()).collection('parties').findOne({ partyId }, { projection: { _id: 0 } }); }

async function registerAlliance(input) {
  const alliance = {
    allianceId: input.allianceId || `ALLIANCE-${Date.now()}`,
    allianceName: input.allianceName,
    allianceSymbol: input.allianceSymbol || '',
    leaderPartyId: input.leaderPartyId || null,
    formedYear: input.formedYear || new Date().getFullYear(),
    createdAt: new Date()
  };
  await (await db()).collection('alliances').insertOne(alliance);
  return { ...alliance, _id: undefined };
}

async function listAlliances() { return (await db()).collection('alliances').find({}, { projection: { _id: 0 } }).sort({ allianceName: 1 }).toArray(); }

async function registerElectionOfficer(input) {
  const passwordHash = await bcrypt.hash(String(input.password), 12);
  const officer = {
    officerId: input.officerId || `OFF-${Date.now()}`,
    name: input.name,
    username: input.username,
    email: input.email || null,
    contactNumber: input.contactNumber || null,
    passwordHash,
    role: 'ELECTION_OFFICER',
    status: 'APPROVED',
    createdAt: new Date()
  };
  await (await db()).collection('electionOfficers').insertOne(officer);
  await logAudit('ELECTION_OFFICER_REGISTERED', { actorId: officer.officerId, actorRole: 'ADMIN', resourceType: 'ELECTION_OFFICER', resourceId: officer.officerId, status: 'SUCCESS', description: 'Election officer registered' });
  return { ...officer, _id: undefined, passwordHash: undefined };
}

async function listElectionOfficers() { return (await db()).collection('electionOfficers').find({}, { projection: { _id: 0, passwordHash: 0 } }).sort({ createdAt: -1 }).toArray(); }
async function getElectionOfficerCredentials(username) { return (await db()).collection('electionOfficers').findOne({ $or: [{ username }, { officerId: username }, { email: username }] }, { projection: { _id: 0 } }); }

async function registerCandidateAccount(input) {
  const party = await getParty(input.partyId);
  const passwordHash = await bcrypt.hash(String(input.password), 12);
  const candidate = {
    candidateId: input.candidateId || `CAND-${Date.now()}`,
    name: input.name || input.candidateName,
    candidateName: input.candidateName || input.name,
    username: input.username || input.email || input.candidateId,
    email: input.email || null,
    phone: input.phone || null,
    passwordHash,
    partyId: input.partyId || null,
    partyName: party ? party.partyName : (input.partyName || ''),
    electionId: input.electionId || null,
    status: 'APPROVED',
    role: 'CANDIDATE',
    createdAt: new Date()
  };
  await (await db()).collection('candidates').insertOne(candidate);
  return { ...candidate, _id: undefined, passwordHash: undefined };
}

async function listCandidateAccounts() { return (await db()).collection('candidates').find({ $or: [{ role: 'CANDIDATE' }, { username: { $exists: true } }] }, { projection: { _id: 0, passwordHash: 0 } }).sort({ createdAt: -1 }).toArray(); }
async function getCandidateCredentials(username) { return (await db()).collection('candidates').findOne({ $or: [{ username }, { candidateId: username }, { email: username }] }, { projection: { _id: 0 } }); }

async function registerVoter(input, encryptedEmbedding, modelVersion) {
  const database = await db();
  const passwordHash = await bcrypt.hash(String(input.password), 12);
  const createdAt = new Date();
  const voter = {
    voterId: input.voterId,
    name: input.name,
    passwordHash,
    dob: input.dob || null,
    gender: input.gender || null,
    phone: input.phone,
    email: input.email || null,
    address: input.address || null,
    aadhaarRef: input.aadhaarRef || null,
    role: 'VOTER',
    status: 'PENDING',
    faceConsentAt: createdAt,
    createdAt
  };
  const enrolledAt = new Date();
  try {
    await database.collection('voters').insertOne(voter);
    await database.collection('faceEncodings').insertOne({
      voterId: voter.voterId,
      ...encryptedEmbedding,
      enrolled: true,
      modelVersion,
      embeddingDimension: 512,
      enrolledAt,
      updatedAt: enrolledAt,
      authorizations: []
    });
    await logAudit('VOTER_REGISTERED', { actorId: voter.voterId, actorRole: 'VOTER', resourceType: 'VOTER', resourceId: voter.voterId, status: 'SUCCESS', description: 'New voter registered with face enrollment and awaiting approval' });
  } catch (error) {
    const rollbackErrors = [];
    try {
      await database.collection('faceEncodings').deleteOne({ voterId: voter.voterId, enrolledAt });
    } catch (rollbackError) {
      rollbackErrors.push(rollbackError);
    }
    try {
      await database.collection('voters').deleteOne({ voterId: voter.voterId, createdAt });
    } catch (rollbackError) {
      rollbackErrors.push(rollbackError);
    }
    if (rollbackErrors.length) {
      throw Object.assign(new AggregateError(
        [error, ...rollbackErrors],
        'Voter registration failed and its partial records could not be removed.'
      ), {
        code: 'VOTER_REGISTRATION_ROLLBACK_FAILED',
        statusCode: 500
      });
    }
    throw error;
  }
  return { voterId: voter.voterId, name: voter.name, status: voter.status };
}

async function getVoter(voterId) { return (await db()).collection('voters').findOne({ voterId }, { projection: { _id: 0, passwordHash: 0 } }); }
async function getVoterCredentials(voterId) { return (await db()).collection('voters').findOne({ voterId }, { projection: { _id: 0 } }); }
async function faceMetadata(voterId) {
  const enrollment = await (await db()).collection('faceEncodings').findOne(
    { voterId },
    { projection: { _id: 0, enrolled: 1, modelVersion: 1, embeddingDimension: 1, enrolledAt: 1 } }
  );
  return {
    enrolled: Boolean(enrollment && enrollment.enrolled),
    modelVersion: enrollment?.modelVersion || null,
    embeddingDimension: enrollment?.embeddingDimension || null,
    enrolledAt: enrollment?.enrolledAt || null
  };
}

async function getVoterWithFaceStatus(voterId) {
  const voter = await getVoter(voterId);
  if (!voter) return null;
  return { ...voter, faceEnrollment: await faceMetadata(voterId) };
}

async function listVoters() {
  const database = await db();
  const voters = await database.collection('voters')
    .find({}, { projection: { _id: 0, passwordHash: 0 } })
    .sort({ createdAt: -1 })
    .toArray();
  const voterIds = voters.map(voter => voter.voterId);
  const enrollments = await database.collection('faceEncodings')
    .find({ voterId: { $in: voterIds } }, {
      projection: { _id: 0, voterId: 1, enrolled: 1, modelVersion: 1, embeddingDimension: 1, enrolledAt: 1 }
    })
    .toArray();
  const metadataByVoter = new Map(enrollments.map(enrollment => [enrollment.voterId, enrollment]));
  return voters.map(voter => {
    const face = metadataByVoter.get(voter.voterId);
    return {
      ...voter,
      faceEnrollment: {
        enrolled: Boolean(face && face.enrolled),
        modelVersion: face?.modelVersion || null,
        embeddingDimension: face?.embeddingDimension || null,
        enrolledAt: face?.enrolledAt || null
      }
    };
  });
}

async function approveVoter(voterId, status, actor = 'SYSTEM') {
  const result = await (await db()).collection('voters').updateOne({ voterId }, { $set: { status } });
  if (!result.matchedCount) throw new Error('Voter does not exist');
  await logAudit(status === 'APPROVED' ? 'VOTER_APPROVED' : 'VOTER_REJECTED', {
    actorId: actor,
    actorRole: 'ADMIN',
    resourceType: 'VOTER',
    resourceId: voterId,
    status: 'SUCCESS',
    description: `Voter ${voterId} marked ${status}`
  });
  return getVoter(voterId);
}

async function saveFaceEnrollment(voterId, encryptedEmbedding, modelVersion) {
  const timestamp = new Date();
  await (await db()).collection('faceEncodings').updateOne(
    { voterId },
    {
      $set: {
        voterId,
        ...encryptedEmbedding,
        enrolled: true,
        modelVersion,
        embeddingDimension: 512,
        enrolledAt: timestamp,
        updatedAt: timestamp,
        authorizations: []
      },
      $unset: { encryptedEmbedding: '' }
    },
    { upsert: true }
  );
}

async function withFaceEnrollmentLock(action) {
  const collection = (await db()).collection('faceEnrollmentLocks');
  const owner = crypto.randomUUID();
  const deadline = Date.now() + FACE_ENROLLMENT_LOCK_WAIT_MS;

  while (true) {
    const now = new Date();
    const lock = await collection.updateOne(
      { _id: FACE_ENROLLMENT_LOCK_ID, expiresAt: { $lte: now } },
      {
        $set: {
          owner,
          expiresAt: new Date(now.getTime() + FACE_ENROLLMENT_LOCK_LEASE_MS)
        }
      }
    );
    if (lock.modifiedCount === 1) break;
    if (Date.now() >= deadline) {
      const error = new Error('Face enrollment is busy. Please try again.');
      error.code = 'FACE_ENROLLMENT_BUSY';
      error.statusCode = 503;
      throw error;
    }
    await new Promise(resolve => setTimeout(resolve, 100));
  }

  let renewal = Promise.resolve();
  let renewalError;
  const renew = setInterval(() => {
    renewal = renewal.then(async () => {
      const now = new Date();
      const result = await collection.updateOne(
        {
          _id: FACE_ENROLLMENT_LOCK_ID,
          owner,
          expiresAt: { $gt: now }
        },
        { $set: { expiresAt: new Date(now.getTime() + FACE_ENROLLMENT_LOCK_LEASE_MS) } }
      );
      if (result.modifiedCount !== 1) {
        const error = new Error('Face enrollment lock was lost.');
        error.code = 'FACE_ENROLLMENT_LOCK_LOST';
        renewalError = error;
      }
    }).catch(error => {
      renewalError = error;
    });
  }, FACE_ENROLLMENT_LOCK_LEASE_MS / 3);
  renew.unref();

  try {
    const result = await action();
    await renewal;
    if (renewalError) {
      const error = new Error('Face enrollment could not be completed safely.');
      error.code = 'FACE_ENROLLMENT_LOCK_LOST';
      error.statusCode = 503;
      error.cause = renewalError;
      throw error;
    }
    return result;
  } finally {
    clearInterval(renew);
    await renewal;
    await collection.updateOne(
      { _id: FACE_ENROLLMENT_LOCK_ID, owner },
      { $set: { owner: null, expiresAt: new Date(0) } }
    );
  }
}

async function getFaceEnrollment(voterId) {
  return (await db()).collection('faceEncodings').findOne({ voterId, enrolled: true }, { projection: { _id: 0 } });
}

async function listFaceEnrollments() {
  return (await db()).collection('faceEncodings').find(
    { enrolled: true },
    { projection: { _id: 0, voterId: 1, modelVersion: 1, ciphertext: 1, iv: 1, authTag: 1, keyVersion: 1 } }
  ).toArray();
}

async function createFaceVerification(record) {
  const collection = (await db()).collection('faceEncodings');
  await collection.updateOne(
    { voterId: record.voterId, enrolled: true, updatedAt: record.enrollmentUpdatedAt },
    { $pull: { authorizations: { expiresAt: { $lte: new Date() } } } }
  );
  const result = await collection.updateOne(
    { voterId: record.voterId, enrolled: true, updatedAt: record.enrollmentUpdatedAt },
    {
      $push: {
        authorizations: {
          tokenHash: record.tokenHash,
          electionId: record.electionId,
          sessionHash: record.sessionHash,
          expiresAt: record.expiresAt
        }
      }
    }
  );
  if (!result.matchedCount) {
    const error = new Error('Face enrollment changed during verification');
    error.code = 'FACE_ENROLLMENT_CHANGED';
    error.statusCode = 409;
    throw error;
  }
}

async function consumeFaceVerification({ tokenHash, voterId, electionId, sessionHash, enrollmentUpdatedAt }) {
  const collection = (await db()).collection('faceEncodings');
  const now = new Date();
  const voterRecord = await collection.findOne(
    { voterId, 'authorizations.tokenHash': tokenHash },
    { projection: { _id: 0, authorizations: { $elemMatch: { tokenHash } } } }
  );
  const authorization = voterRecord?.authorizations?.[0];
  if (!authorization) {
    return { valid: false, code: 'VERIFICATION_TOKEN_INVALID' };
  }
  if (authorization.expiresAt <= now) {
    await collection.updateOne({ voterId }, { $pull: { authorizations: { tokenHash } } });
    return { valid: false, code: 'VERIFICATION_TOKEN_EXPIRED' };
  }
  if (authorization.electionId !== electionId || authorization.sessionHash !== sessionHash) {
    return { valid: false, code: 'VERIFICATION_TOKEN_INVALID' };
  }
  const consumed = await collection.updateOne({
    voterId,
    enrolled: true,
    updatedAt: enrollmentUpdatedAt,
    authorizations: {
      $elemMatch: { tokenHash, electionId, sessionHash, expiresAt: { $gt: now } }
    }
  }, { $pull: { authorizations: { tokenHash } } });
  return consumed.matchedCount ? { valid: true } : { valid: false, code: 'VERIFICATION_TOKEN_INVALID' };
}

async function createFaceChallenge({ challengeId, voterId, electionId, sessionHash, action, expiresAt }) {
  await (await db()).collection('faceChallenges').insertOne({
    challengeId,
    voterId,
    electionId,
    sessionHash,
    action,
    createdAt: new Date(),
    expiresAt,
    consumedAt: null
  });
}

async function consumeFaceChallenge({ challengeId, voterId, electionId, sessionHash }) {
  const collection = (await db()).collection('faceChallenges');
  const now = new Date();
  const challenge = await collection.findOne({
    challengeId,
    voterId,
    electionId,
    sessionHash,
    consumedAt: null,
    expiresAt: { $gt: now }
  }, { projection: { _id: 0, action: 1, createdAt: 1 } });
  if (!challenge) return { valid: false, code: 'LIVENESS_CHALLENGE_INVALID' };
  const consumed = await collection.updateOne({
    challengeId,
    voterId,
    electionId,
    sessionHash,
    action: challenge.action,
    consumedAt: null,
    expiresAt: { $gt: now }
  }, { $set: { consumedAt: now } });
  if (!consumed.modifiedCount) return { valid: false, code: 'LIVENESS_CHALLENGE_INVALID' };
  const minimumDurationMs = Number(process.env.FACE_CHALLENGE_MIN_DURATION_MS || 1500);
  if (!Number.isFinite(minimumDurationMs) || minimumDurationMs < 500 || minimumDurationMs > 10000 ||
      now.getTime() - challenge.createdAt.getTime() < minimumDurationMs) {
    return { valid: false, code: 'LIVENESS_CHALLENGE_TOO_FAST' };
  }
  return { valid: true, action: challenge.action };
}

async function createVoteClaim(electionId, voterHash) {
  try {
    await (await db()).collection('voteClaims').insertOne({
      electionId,
      voterHash,
      status: 'PENDING',
      createdAt: new Date(),
      transactionId: null
    });
  } catch (error) {
    if (error.code === 11000) {
      const duplicate = new Error('Voter has already voted in this election');
      duplicate.code = 'VOTE_ALREADY_CAST';
      duplicate.statusCode = 409;
      throw duplicate;
    }
    throw error;
  }
}

async function completeVoteClaim(electionId, voterHash, transactionId) {
  const result = await (await db()).collection('voteClaims').updateOne(
    { electionId, voterHash, status: 'PENDING' },
    { $set: { status: 'RECORDED', transactionId, recordedAt: new Date() } }
  );
  if (!result.matchedCount) throw new Error('Vote claim could not be finalized');
}

async function releaseVoteClaim(electionId, voterHash) {
  await (await db()).collection('voteClaims').deleteOne({
    electionId,
    voterHash,
    status: 'PENDING'
  });
}

async function healthCheck() {
  await (await db()).command({ ping: 1 });
  return 'ok';
}
async function closeConnection() {
  if (databasePromise) await databasePromise;
  const mongoClient = client;
  client = null;
  database = null;
  databasePromise = null;
  if (mongoClient) await mongoClient.close();
}
async function logAudit(action, fields = {}) {
  const allowed = ['eventId', 'eventType', 'actorId', 'actorRole', 'timestamp', 'resourceType', 'resourceId', 'status', 'description'];
  const clean = {};
  for (const key of allowed) {
    if (fields[key] !== undefined) clean[key] = fields[key];
  }
  await (await db()).collection('auditLogs').insertOne({
    eventId: clean.eventId || `evt_${Date.now()}_${Math.random().toString(16).slice(2, 8)}`,
    eventType: action,
    actorId: clean.actorId || 'SYSTEM',
    actorRole: clean.actorRole || 'SYSTEM',
    timestamp: clean.timestamp || new Date(),
    resourceType: clean.resourceType || 'SYSTEM',
    resourceId: clean.resourceId || null,
    status: clean.status || 'SUCCESS',
    description: clean.description || action,
  });
}

module.exports = {
  VALID_ELECTION_STATES,
  VALID_ELECTION_TRANSITIONS,
  listElections,
  getElection,
  getCandidates,
  getCandidate,
  createElection,
  updateElection,
  createCandidate,
  registerParty,
  updatePartyLogo,
  listParties,
  getParty,
  registerAlliance,
  listAlliances,
  registerElectionOfficer,
  listElectionOfficers,
  getElectionOfficerCredentials,
  registerCandidateAccount,
  listCandidateAccounts,
  getCandidateCredentials,
  registerVoter,
  getVoter,
  getVoterWithFaceStatus,
  getVoterCredentials,
  listVoters,
  approveVoter,
  saveFaceEnrollment,
  withFaceEnrollmentLock,
  getFaceEnrollment,
  listFaceEnrollments,
  createFaceChallenge,
  consumeFaceChallenge,
  createFaceVerification,
  consumeFaceVerification,
  createVoteClaim,
  completeVoteClaim,
  releaseVoteClaim,
  logAudit,
  validateElectionStateTransition,
  healthCheck,
  closeConnection
};