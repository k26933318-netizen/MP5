const test = require('node:test');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');

const { encryptEmbedding, decryptEmbedding } = require('../backend/face/embeddingCrypto');
const { cosineSimilarity, faceSimilarityThreshold, findMatchingEmbedding } = require('../backend/face/similarity');

function withEncryptionKey(callback) {
  const previous = process.env.FACE_ENCRYPTION_KEY;
  process.env.FACE_ENCRYPTION_KEY = crypto.randomBytes(32).toString('base64');
  try {
    return callback();
  } finally {
    if (previous === undefined) delete process.env.FACE_ENCRYPTION_KEY;
    else process.env.FACE_ENCRYPTION_KEY = previous;
  }
}

test('face embeddings are encrypted with authenticated encryption and round-trip', () => {
  withEncryptionKey(() => {
    const embedding = Array.from({ length: 512 }, (_, index) => index / 512);
    const encrypted = encryptEmbedding(embedding);
    assert.equal(encrypted.ciphertext.includes(JSON.stringify(embedding)), false);
    assert.deepEqual(decryptEmbedding(encrypted), embedding);
  });
});

test('modified encrypted face embeddings fail authentication', () => {
  withEncryptionKey(() => {
    const encrypted = encryptEmbedding(Array(512).fill(0.25));
    encrypted.authTag = `${encrypted.authTag[0] === 'A' ? 'B' : 'A'}${encrypted.authTag.slice(1)}`;
    assert.throws(() => decryptEmbedding(encrypted), { code: 'FACE_ENROLLMENT_DECRYPTION_FAILED' });
  });
});

test('face encryption requires a configured 32-byte key', () => {
  const previous = process.env.FACE_ENCRYPTION_KEY;
  delete process.env.FACE_ENCRYPTION_KEY;
  try {
    assert.throws(() => encryptEmbedding(Array(512).fill(1)), { code: 'FACE_ENCRYPTION_NOT_CONFIGURED' });
  } finally {
    if (previous !== undefined) process.env.FACE_ENCRYPTION_KEY = previous;
  }
});

test('cosine similarity compares equal and orthogonal embeddings', () => {
  const same = Array(512).fill(0);
  same[0] = 1;
  const orthogonal = Array(512).fill(0);
  orthogonal[1] = 1;
  assert.equal(cosineSimilarity(same, same), 1);
  assert.equal(cosineSimilarity(same, orthogonal), 0);
});

test('duplicate enrollment matching selects only embeddings at or above the threshold', () => {
  const candidate = Array(512).fill(0);
  candidate[0] = 1;
  const unrelated = Array(512).fill(0);
  unrelated[1] = 1;
  const match = Array(512).fill(0);
  match[0] = 0.8;
  match[1] = 0.6;

  assert.deepEqual(findMatchingEmbedding(candidate, [unrelated, match], 0.79), { index: 1, score: 0.8 });
  assert.equal(findMatchingEmbedding(candidate, [unrelated, match], 0.81), null);
});

test('background estimation failures expose only a generic face validation error', async () => {
  const faceClient = require('../backend/face/faceServiceClient');
  const originalFetch = global.fetch;
  const previousKey = process.env.FACE_SERVICE_API_KEY;
  process.env.FACE_SERVICE_API_KEY = 'a'.repeat(48);
  global.fetch = async () => ({
    ok: false,
    status: 422,
    json: async () => ({
      detail: {
        code: 'BACKGROUND_ESTIMATION_FAILED',
        message: 'Internal segmentation exception',
        near_white_ratio: 0.12
      }
    })
  });
  try {
    await assert.rejects(
      faceClient.enrollFrames([{ buffer: Buffer.from('image'), mimetype: 'image/jpeg' }]),
      { code: 'FACE_VALIDATION_FAILED', message: 'Unable to verify your face. Please retry.' }
    );
  } finally {
    global.fetch = originalFetch;
    if (previousKey === undefined) delete process.env.FACE_SERVICE_API_KEY;
    else process.env.FACE_SERVICE_API_KEY = previousKey;
  }
});

async function withRegistrationStubs(enrolledEmbeddings, options = {}) {
  const mongo = require('../backend/data/mongoRepository');
  const faceClient = require('../backend/face/faceServiceClient');
  const controller = require('../backend/blockchain/blockchainController');
  const previousKey = process.env.FACE_ENCRYPTION_KEY;
  const originalMethods = {
    listFaceEnrollments: mongo.listFaceEnrollments,
    registerVoter: mongo.registerVoter,
    withFaceEnrollmentLock: mongo.withFaceEnrollmentLock,
    enrollFrames: faceClient.enrollFrames
  };
  const candidate = Array(512).fill(0);
  candidate[0] = 1;
  process.env.FACE_ENCRYPTION_KEY = crypto.randomBytes(32).toString('base64');
  const enrollments = enrolledEmbeddings.map((entry, index) => ({
    voterId: entry.voterId || `VOTER-${index}`,
    modelVersion: 'arcface-test',
    ...encryptEmbedding(entry.embedding || entry)
  }));
  let registrationArgs;
  let registerCalls = 0;
  let lockQueue = Promise.resolve();
  mongo.listFaceEnrollments = async () => enrollments.slice();
  mongo.registerVoter = async (...args) => {
    registerCalls += 1;
    if (options.registerErrorOnCall === registerCalls) throw options.registerError;
    registrationArgs = args;
    enrollments.push({
      voterId: args[0].voterId,
      modelVersion: args[2],
      ...args[1]
    });
    return { voterId: args[0].voterId, status: 'PENDING' };
  };
  mongo.withFaceEnrollmentLock = action => {
    const operation = lockQueue.then(action, action);
    lockQueue = operation.then(() => undefined, () => undefined);
    return operation;
  };
  faceClient.enrollFrames = async images => {
    assert.equal(images.length, 3);
    if (options.faceError) throw options.faceError;
    return ({
      embedding: options.embedding || candidate,
      dimension: 512,
      model_version: 'arcface-test',
      checks: options.checks || { position: true, background: true, quality: true, stable: true, presentation_attack: true }
    });
  };
  const invoke = async index => {
    const voterId = index === 0 ? 'NEW-VOTER' : `NEW-VOTER-${index}`;
    const request = {
      body: {
        registration: JSON.stringify({ voterId, name: 'New User', phone: '123', password: 'password', biometricConsent: true })
      },
      files: Array.from({ length: 3 }, () => ({ buffer: Buffer.from('image'), mimetype: 'image/jpeg' }))
    };
    const response = {
      statusCode: 200,
      status(code) { this.statusCode = code; return this; },
      json(payload) { this.payload = payload; return this; }
    };
    let error;
    await controller.registerVoter(request, response, caught => { error = caught; });
    return { response, error };
  };
  try {
    const results = await Promise.all(
      Array.from({ length: options.registrationCount || 1 }, (_value, index) => invoke(index))
    );
    return { ...results[0], results, registrationArgs, enrollments, registerCalls };
  } finally {
    mongo.listFaceEnrollments = originalMethods.listFaceEnrollments;
    mongo.registerVoter = originalMethods.registerVoter;
    mongo.withFaceEnrollmentLock = originalMethods.withFaceEnrollmentLock;
    faceClient.enrollFrames = originalMethods.enrollFrames;
    if (previousKey === undefined) delete process.env.FACE_ENCRYPTION_KEY;
    else process.env.FACE_ENCRYPTION_KEY = previousKey;
  }
}

test('voter registration rejects a matching face before creating an account', async () => {
  const enrolled = Array(512).fill(0);
  enrolled[0] = 1;
  const { response, error } = await withRegistrationStubs([enrolled]);
  assert.equal(error.code, 'DUPLICATE_FACE');
  assert.equal(response.payload, undefined);
});

test('voter registration rejects an already enrolled face under the same voter ID', async () => {
  const enrolled = Array(512).fill(0);
  enrolled[0] = 1;
  const { error, registerCalls } = await withRegistrationStubs([
    { voterId: 'NEW-VOTER', embedding: enrolled }
  ]);
  assert.equal(error.code, 'DUPLICATE_FACE');
  assert.equal(registerCalls, 0);
});

test('face re-enrollment rejects the enrolled voter face without replacing the template', async () => {
  const mongo = require('../backend/data/mongoRepository');
  const faceClient = require('../backend/face/faceServiceClient');
  const controller = require('../backend/blockchain/blockchainController');
  const previousKey = process.env.FACE_ENCRYPTION_KEY;
  const originalMethods = {
    listFaceEnrollments: mongo.listFaceEnrollments,
    withFaceEnrollmentLock: mongo.withFaceEnrollmentLock,
    getVoterCredentials: mongo.getVoterCredentials,
    saveFaceEnrollment: mongo.saveFaceEnrollment,
    logAudit: mongo.logAudit,
    enrollFrames: faceClient.enrollFrames
  };
  const embedding = Array(512).fill(0);
  embedding[0] = 1;
  let saved = false;
  process.env.FACE_ENCRYPTION_KEY = crypto.randomBytes(32).toString('base64');
  mongo.listFaceEnrollments = async () => [{
    voterId: 'EXISTING-VOTER',
    modelVersion: 'arcface-test',
    ...encryptEmbedding(embedding)
  }];
  mongo.withFaceEnrollmentLock = action => action();
  mongo.getVoterCredentials = async () => ({ voterId: 'EXISTING-VOTER', status: 'APPROVED' });
  mongo.saveFaceEnrollment = async () => { saved = true; };
  mongo.logAudit = async () => {};
  faceClient.enrollFrames = async () => ({
    embedding,
    dimension: 512,
    model_version: 'arcface-test',
    checks: { position: true, background: true, quality: true, stable: true, presentation_attack: true }
  });
  const response = { json(payload) { this.payload = payload; return this; } };
  let error;
  try {
    await controller.enrollFace({
      user: { voterId: 'EXISTING-VOTER' },
      files: Array.from({ length: 3 }, () => ({ buffer: Buffer.from('image'), mimetype: 'image/jpeg' }))
    }, response, caught => { error = caught; });
    assert.equal(error.code, 'DUPLICATE_FACE');
    assert.equal(error.message, 'This face is already registered.');
    assert.equal(saved, false);
    assert.equal(response.payload, undefined);
  } finally {
    mongo.listFaceEnrollments = originalMethods.listFaceEnrollments;
    mongo.withFaceEnrollmentLock = originalMethods.withFaceEnrollmentLock;
    mongo.getVoterCredentials = originalMethods.getVoterCredentials;
    mongo.saveFaceEnrollment = originalMethods.saveFaceEnrollment;
    mongo.logAudit = originalMethods.logAudit;
    faceClient.enrollFrames = originalMethods.enrollFrames;
    if (previousKey === undefined) delete process.env.FACE_ENCRYPTION_KEY;
    else process.env.FACE_ENCRYPTION_KEY = previousKey;
  }
});

test('voter registration stores a face enrollment when no duplicate matches', async () => {
  const unrelated = Array(512).fill(0);
  unrelated[1] = 1;
  const { response, error, registrationArgs } = await withRegistrationStubs([unrelated]);
  assert.equal(error, undefined);
  assert.equal(response.statusCode, 201);
  assert.equal(response.payload.success, true);
  assert.equal(response.payload.faceEnrollment.enrolled, true);
  assert.equal(registrationArgs[0].voterId, 'NEW-VOTER');
  assert.ok(registrationArgs[1].ciphertext);
  assert.ok(registrationArgs[1].iv);
  assert.ok(registrationArgs[1].authTag);
  assert.ok(registrationArgs[1].keyVersion);
  assert.equal(registrationArgs[2], 'arcface-test');
});

test('enrollment does not require a separate liveness challenge or passive anti-spoof result', async () => {
  const unrelated = Array(512).fill(0);
  unrelated[1] = 1;
  const { response, error } = await withRegistrationStubs([unrelated], {
    checks: { position: true, quality: true, stable: true, head_movement: false, presentation_attack: false }
  });
  assert.equal(error, undefined);
  assert.equal(response.statusCode, 201);
  assert.equal(response.payload.faceEnrollment.enrolled, true);
});

test('enrollment still rejects a face that fails position, quality, or stability checks', async () => {
  const unrelated = Array(512).fill(0);
  unrelated[1] = 1;
  const { error, registerCalls } = await withRegistrationStubs([unrelated], {
    checks: { position: true, quality: false, stable: true, head_movement: false, presentation_attack: false }
  });
  assert.equal(error.code, 'FACE_VALIDATION_FAILED');
  assert.equal(registerCalls, 0);
});

test('concurrent registrations cannot enroll the same face twice', async () => {
  const { results, enrollments } = await withRegistrationStubs([], { registrationCount: 2 });
  assert.equal(results.filter(result => !result.error && result.response.payload?.success).length, 1);
  assert.equal(results.filter(result => result.error?.code === 'DUPLICATE_FACE').length, 1);
  assert.equal(enrollments.length, 1);
});

test('face-service and image-quality failures do not create an enrollment', async () => {
  for (const faceError of [
    Object.assign(new Error('Unavailable'), { code: 'FACE_SERVICE_UNAVAILABLE', statusCode: 503 }),
    Object.assign(new Error('Image quality rejected'), { code: 'LOW_IMAGE_QUALITY', statusCode: 422 })
  ]) {
    const { error, registerCalls, enrollments } = await withRegistrationStubs([], { faceError });
    assert.equal(error, faceError);
    assert.equal(registerCalls, 0);
    assert.equal(enrollments.length, 0);
  }
});

test('database failure is not reported as success and a later retry can enroll', async () => {
  const databaseError = Object.assign(new Error('Database unavailable'), {
    code: 'DATABASE_UNAVAILABLE',
    statusCode: 503
  });
  const { results, enrollments } = await withRegistrationStubs([], {
    registrationCount: 2,
    registerError: databaseError,
    registerErrorOnCall: 1
  });
  assert.equal(results[0].error, databaseError);
  assert.equal(results[0].response.payload, undefined);
  assert.equal(results[1].error, undefined);
  assert.equal(results[1].response.payload.success, true);
  assert.equal(enrollments.length, 1);
});

test('registration frontend requires backend-confirmed enrollment and protects in-flight state', () => {
  const source = fs.readFileSync(path.join(__dirname, '..', 'public', 'app.js'), 'utf8');
  assert.match(source, /result\.success !== true/);
  assert.match(source, /result\.faceEnrollment\?\.enrolled !== true/);
  assert.match(source, /activeCameraStream !== assessmentStream/);
  assert.match(source, /if \(submitting \|\| currentStage === 'submitting'\) return/);
  assert.match(source, /while \(assessing\) await new Promise\(resolve => setTimeout\(resolve, 50\)\)/);
  assert.match(source, /message\.textContent = error\.message/);
  assert.match(source, /mode === 'enroll' \|\| mode === 'registration'/);
  assert.match(source, /const challengeMode = mode === 'verify'/);
  assert.match(source, /requiredStableFrames = enrollmentMode \|\| challengeMode \? 3 : 2/);
  assert.match(source, /signal: AbortSignal\.timeout\(12000\)/);
  assert.match(source, /Face check is taking too long/);
  assert.match(source, /face service did not respond\|waking up; retry in a minute\|running in render/i);
  assert.match(source, /actionButton\.disabled = true;\s*setCameraMessage\(result\.message \|\| 'Center your whole face inside the guide and hold still\.'/);
  assert.match(source, /activeCameraStream && !faceAssessmentTimer/);
  assert.match(source, /const checkedFrames = assessmentFrames\.slice\(-requiredStableFrames\)/);
  assert.match(source, /readyCount < requiredStableFrames/);
  assert.match(source, /workflow', enrollmentMode \? 'enroll' : 'verify'/);
});

test('missing enrollment image uses a clear message without exposing frame-count bounds', async () => {
  const controller = require('../backend/blockchain/blockchainController');
  let error;
  await controller.registerVoter({
    body: {
      registration: JSON.stringify({
        voterId: 'NEW-VOTER', name: 'New User', phone: '123', password: 'password', biometricConsent: true
      })
    },
    files: []
  }, {}, caught => { error = caught; });
  assert.equal(error.code, 'INVALID_FRAME_COUNT');
  assert.equal(error.message, 'The face check is incomplete. Please try again.');
  assert.doesNotMatch(error.message, /between \d+ and \d+|frames/i);
});

async function withFaceVerificationStubs({
  matchingFace,
  challengeValid = true,
  headMovementValid = true,
  antiSpoofValid = true
}) {
  const mongo = require('../backend/data/mongoRepository');
  const faceClient = require('../backend/face/faceServiceClient');
  const auth = require('../backend/auth');
  const controller = require('../backend/blockchain/blockchainController');
  const previousKey = process.env.FACE_ENCRYPTION_KEY;
  const originalMethods = {
    getVoterCredentials: mongo.getVoterCredentials,
    getElection: mongo.getElection,
    getFaceEnrollment: mongo.getFaceEnrollment,
    consumeFaceChallenge: mongo.consumeFaceChallenge,
    createFaceVerification: mongo.createFaceVerification,
    logAudit: mongo.logAudit,
    verifyFrames: faceClient.verifyFrames,
    parseSessionToken: auth.parseSessionToken
  };
  const embedding = Array(512).fill(0);
  embedding[0] = 1;
  let authorization;
  process.env.FACE_ENCRYPTION_KEY = crypto.randomBytes(32).toString('base64');
  mongo.getVoterCredentials = async voterId => ({ voterId, status: 'APPROVED' });
  mongo.getElection = async electionId => ({
    electionId,
    status: 'ACTIVE',
    startDate: new Date(Date.now() - 60_000).toISOString(),
    endDate: new Date(Date.now() + 60_000).toISOString()
  });
  mongo.getFaceEnrollment = async () => ({
    ...encryptEmbedding(embedding),
    modelVersion: 'arcface-test',
    updatedAt: new Date()
  });
  mongo.consumeFaceChallenge = async input => {
    assert.equal(input.voterId, 'VOTER-1');
    assert.equal(input.electionId, 'ELECTION-1');
    assert.equal(input.sessionHash, crypto.createHash('sha256').update('test-session').digest('hex'));
    return challengeValid
      ? { valid: true, action: 'turn_left' }
      : { valid: false, code: 'LIVENESS_CHALLENGE_INVALID' };
  };
  mongo.createFaceVerification = async value => { authorization = value; };
  mongo.logAudit = async () => {};
  let faceAnalysisCalls = 0;
  faceClient.verifyFrames = async (frames, action) => {
    faceAnalysisCalls += 1;
    assert.equal(frames.length, 9);
    assert.equal(action, 'turn_left');
    return {
      embedding: matchingFace ? embedding : [...embedding.slice(1), 1],
      dimension: 512,
      model_version: 'arcface-test',
      checks: {
        position: true,
        background: true,
        quality: true,
        stable: true,
        head_movement: headMovementValid,
        presentation_attack: antiSpoofValid
      }
    };
  };
  auth.parseSessionToken = () => 'test-session';
  const request = {
    user: { voterId: 'VOTER-1' },
    body: {
      electionId: 'ELECTION-1',
      challengeId: 'server-issued-challenge',
      livenessPassed: true,
      verified: true
    },
    files: Array.from({ length: 9 }, () => ({ buffer: Buffer.from('image'), mimetype: 'image/jpeg' }))
  };
  const response = {
    payload: null,
    json(payload) { this.payload = payload; return this; }
  };
  let error;
  try {
    await controller.verifyFace(request, response, caught => { error = caught; });
    return { response, error, authorization, faceAnalysisCalls };
  } finally {
    mongo.getVoterCredentials = originalMethods.getVoterCredentials;
    mongo.getElection = originalMethods.getElection;
    mongo.getFaceEnrollment = originalMethods.getFaceEnrollment;
    mongo.consumeFaceChallenge = originalMethods.consumeFaceChallenge;
    mongo.createFaceVerification = originalMethods.createFaceVerification;
    mongo.logAudit = originalMethods.logAudit;
    faceClient.verifyFrames = originalMethods.verifyFrames;
    auth.parseSessionToken = originalMethods.parseSessionToken;
    if (previousKey === undefined) delete process.env.FACE_ENCRYPTION_KEY;
    else process.env.FACE_ENCRYPTION_KEY = previousKey;
  }
}

test('voting face verification issues a one-time authorization only for the enrolled face', async () => {
  const { response, error, authorization } = await withFaceVerificationStubs({ matchingFace: true });
  assert.equal(error, undefined);
  assert.equal(response.payload.verified, true);
  assert.match(response.payload.verificationToken, /^[a-f0-9]{64}$/);
  assert.match(authorization.tokenHash, /^[a-f0-9]{64}$/);
  assert.equal(authorization.voterId, 'VOTER-1');
  assert.equal(authorization.electionId, 'ELECTION-1');
  assert.ok(authorization.expiresAt > authorization.createdAt);
});

test('voting face verification rejects a different face and issues no authorization', async () => {
  const { response, error, authorization } = await withFaceVerificationStubs({ matchingFace: false });
  assert.equal(error.code, 'FACE_VERIFICATION_FAILED');
  assert.equal(response.payload, null);
  assert.equal(authorization, undefined);
});

test('frontend verification flags cannot bypass an invalid, expired, or reused challenge', async () => {
  const { response, error, authorization, faceAnalysisCalls } = await withFaceVerificationStubs({
    matchingFace: true,
    challengeValid: false
  });
  assert.equal(error.code, 'LIVENESS_CHALLENGE_INVALID');
  assert.equal(response.payload, null);
  assert.equal(authorization, undefined);
  assert.equal(faceAnalysisCalls, 0);
});

test('missing server-side head movement or PAD checks cannot issue authorization', async () => {
  for (const options of [
    { headMovementValid: false },
    { antiSpoofValid: false }
  ]) {
    const { response, error, authorization } = await withFaceVerificationStubs({
      matchingFace: true,
      ...options
    });
    assert.equal(error.code, options.antiSpoofValid === false
      ? 'FACE_VALIDATION_FAILED'
      : 'LIVENESS_CHALLENGE_FAILED');
    assert.equal(response.payload, null);
    assert.equal(authorization, undefined);
  }
});

async function invokeFaceChallenge({ voterStatus = 'APPROVED' } = {}) {
  const mongo = require('../backend/data/mongoRepository');
  const auth = require('../backend/auth');
  const controller = require('../backend/blockchain/blockchainController');
  const previous = {
    getVoterCredentials: mongo.getVoterCredentials,
    getElection: mongo.getElection,
    getFaceEnrollment: mongo.getFaceEnrollment,
    createFaceChallenge: mongo.createFaceChallenge,
    parseSessionToken: auth.parseSessionToken
  };
  let savedChallenge;
  mongo.getVoterCredentials = async voterId => ({ voterId, status: voterStatus });
  mongo.getElection = async electionId => ({
    electionId,
    status: 'ACTIVE',
    startDate: new Date(Date.now() - 60_000).toISOString(),
    endDate: new Date(Date.now() + 60_000).toISOString()
  });
  mongo.getFaceEnrollment = async () => ({ enrolled: true });
  mongo.createFaceChallenge = async challenge => { savedChallenge = challenge; };
  auth.parseSessionToken = () => 'challenge-test-session';
  const response = {
    payload: null,
    json(payload) { this.payload = payload; return this; }
  };
  let error;
  try {
    await controller.createFaceChallenge({
      user: { voterId: 'VOTER-1' },
      body: { electionId: 'ELECTION-1' }
    }, response, caught => { error = caught; });
    return { response, error, savedChallenge };
  } finally {
    mongo.getVoterCredentials = previous.getVoterCredentials;
    mongo.getElection = previous.getElection;
    mongo.getFaceEnrollment = previous.getFaceEnrollment;
    mongo.createFaceChallenge = previous.createFaceChallenge;
    auth.parseSessionToken = previous.parseSessionToken;
  }
}

test('liveness challenge is random, expires, and binds to voter, election, and session', async () => {
  const { response, error, savedChallenge } = await invokeFaceChallenge();
  assert.equal(error, undefined);
  assert.ok(['turn_left', 'turn_right'].includes(response.payload.action));
  assert.equal(savedChallenge.action, response.payload.action);
  assert.equal(savedChallenge.voterId, 'VOTER-1');
  assert.equal(savedChallenge.electionId, 'ELECTION-1');
  assert.equal(savedChallenge.sessionHash, crypto.createHash('sha256').update('challenge-test-session').digest('hex'));
  assert.ok(savedChallenge.challengeId);
  assert.ok(savedChallenge.expiresAt > new Date());
});

test('unapproved voter cannot obtain a liveness challenge', async () => {
  const { response, error, savedChallenge } = await invokeFaceChallenge({ voterStatus: 'PENDING' });
  assert.equal(error.code, 'VOTER_NOT_APPROVED');
  assert.equal(response.payload, null);
  assert.equal(savedChallenge, undefined);
});

test('face similarity threshold is configurable and rejects invalid values', () => {
  const previous = process.env.FACE_SIMILARITY_THRESHOLD;
  process.env.FACE_SIMILARITY_THRESHOLD = '0.61';
  try {
    assert.equal(faceSimilarityThreshold(), 0.61);
    process.env.FACE_SIMILARITY_THRESHOLD = '2';
    assert.throws(() => faceSimilarityThreshold(), /between 0 and 1/);
  } finally {
    if (previous === undefined) delete process.env.FACE_SIMILARITY_THRESHOLD;
    else process.env.FACE_SIMILARITY_THRESHOLD = previous;
  }
});
