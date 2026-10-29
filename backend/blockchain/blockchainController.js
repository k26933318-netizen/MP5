const { BlockchainService } = require('./blockchainService');
const mongo = require('../data/mongoRepository');
const auth = require('../auth');
const crypto = require('crypto');
const { encryptEmbedding, decryptEmbedding } = require('../face/embeddingCrypto');
const faceClient = require('../face/faceServiceClient');
const { cosineSimilarity, faceSimilarityThreshold, findMatchingEmbedding } = require('../face/similarity');
const service = new BlockchainService();
let voterRegistrationQueue = Promise.resolve();

function asyncRoute(handler) { return (request, response, next) => Promise.resolve(handler(request, response)).catch(next); }
function body(request) { return request.body || {}; }
function validate(input, fields) {
  for (const field of fields) {
    if (input[field] === undefined || input[field] === null || String(input[field]).trim() === '') {
      const error = new Error(`Invalid ${field}`);
      error.statusCode = 400;
      throw error;
    }
  }
}
function send(fields, action) { return asyncRoute(async (request, response) => { const input = body(request); validate(input, fields); return response.json({ success: true, ...(await action(input)) }); }); }

function mapVoteForClient(vote) {
  if (!vote) return null;
  return {
    voteId: vote.voteId,
    transactionId: vote.transactionId,
    electionId: vote.electionId,
    candidateId: vote.candidateId,
    timestamp: vote.timestamp,
    previousHash: vote.previousHash,
    recordHash: vote.recordHash,
    status: vote.status || 'CONFIRMED'
  };
}

function sessionHash(request) {
  return crypto.createHash('sha256').update(auth.parseSessionToken(request)).digest('hex');
}

function voterAudit(action, request, electionId, status, description) {
  return mongo.logAudit(action, {
    actorId: request.user.voterId,
    actorRole: 'VOTER',
    resourceType: electionId ? 'ELECTION' : 'VOTER',
    resourceId: electionId || request.user.voterId,
    status,
    description
  });
}

function requireFaceImage(request) {
  if (!request.file) {
    const error = new Error('A face image is required.');
    error.code = 'INVALID_IMAGE';
    error.statusCode = 400;
    throw error;
  }
  return request.file;
}

function requireFaceImages(request, minimum = 5, maximum = 15) {
  const files = request.files;
  if (!Array.isArray(files) || files.length < minimum || files.length > maximum) {
    const error = new Error(minimum === 1 && maximum === 1
      ? 'Capture one clear face image before continuing.'
      : 'The face check is incomplete. Please try again.');
    error.code = 'INVALID_FRAME_COUNT';
    error.statusCode = 400;
    throw error;
  }
  const totalBytes = files.reduce((total, file) => total + file.buffer.length, 0);
  const maximumBytes = Number(process.env.FACE_SEQUENCE_MAX_UPLOAD_BYTES || 6 * 1024 * 1024);
  if (!Number.isSafeInteger(maximumBytes) || maximumBytes <= 0) {
    const error = new Error('Face sequence upload limit configuration is invalid.');
    error.code = 'FACE_CONFIGURATION_INVALID';
    error.statusCode = 500;
    throw error;
  }
  if (totalBytes > maximumBytes) {
    const error = new Error('Captured face frames exceed the allowed total upload size.');
    error.code = 'IMAGE_TOO_LARGE';
    error.statusCode = 413;
    throw error;
  }
  return files;
}

function faceChallengeFramesPerStage() {
  const framesPerStage = Number(process.env.FACE_CHALLENGE_FRAMES_PER_STAGE || 3);
  if (!Number.isInteger(framesPerStage) || framesPerStage < 3 || framesPerStage > 5) {
    const error = new Error('Face challenge frame configuration is invalid.');
    error.code = 'FACE_CONFIGURATION_INVALID';
    error.statusCode = 500;
    throw error;
  }
  return framesPerStage;
}

function requireValidFaceAnalysis(result, {
  requireBackground = true,
  requireStability = true,
  requireLiveness = true
} = {}) {
  const livenessPassed = result?.checks?.presentation_attack === true;
  if (!result || result.dimension !== 512 || !Array.isArray(result.embedding) ||
      typeof result.model_version !== 'string' || !result.model_version ||
      !result.checks || result.checks.position !== true ||
      (requireBackground && result.checks.background !== true) ||
      result.checks.quality !== true || (requireStability && result.checks.stable !== true) ||
      (requireLiveness && !livenessPassed)) {
    const error = new Error('Face analysis did not confirm every required server-side check.');
    error.code = 'FACE_VALIDATION_FAILED';
    error.statusCode = 403;
    throw error;
  }
}

function requireValidPartyImage(image) {
  if (!image) return null;
  const signature = image.buffer.subarray(0, 12);
  const valid = image.mimetype === 'image/jpeg'
    ? signature[0] === 0xff && signature[1] === 0xd8 && signature[2] === 0xff
    : image.mimetype === 'image/png'
      ? signature.length >= 8 && signature.subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]))
      : image.mimetype === 'image/webp'
        ? signature.toString('ascii', 0, 4) === 'RIFF' && signature.toString('ascii', 8, 12) === 'WEBP'
        : false;
  if (!valid) {
    const error = new Error('Party logo content does not match its image type.');
    error.code = 'INVALID_PARTY_IMAGE';
    error.statusCode = 415;
    throw error;
  }
  return image;
}

function serializeFaceEnrollment(action) {
  const result = voterRegistrationQueue.then(action, action);
  voterRegistrationQueue = result.then(() => undefined, () => undefined);
  return result;
}

async function ensureFaceIsNotRegistered(embedding, modelVersion) {
  const enrollments = await mongo.listFaceEnrollments();
  if (enrollments.some(enrollment => enrollment.modelVersion !== modelVersion)) {
    const error = new Error('Existing face enrollments use a different recognition model and must be renewed before face enrollment.');
    error.code = 'FACE_MODEL_VERSION_MISMATCH';
    error.statusCode = 409;
    throw error;
  }
  const embeddings = enrollments.map(enrollment => decryptEmbedding(enrollment));
  if (findMatchingEmbedding(embedding, embeddings, faceSimilarityThreshold())) {
    const error = new Error('This face is already registered.');
    error.code = 'DUPLICATE_FACE';
    error.statusCode = 409;
    throw error;
  }
}

module.exports = {
  login: asyncRoute(async (request, response) => { const input = body(request); validate(input, ['role', 'username', 'password']); return response.json({ success: true, ...(await auth.login(input)) }); }),
  logout: asyncRoute(async (request, response) => { auth.logout(request); return response.json({ success: true }); }),
  assessFace: asyncRoute(async (request, response) => {
    const image = requireFaceImage(request);
    const workflow = request.body?.workflow === 'enroll' ? 'enroll' : 'verify';
    const result = await faceClient.assessPosition(image, request.body?.stage || 'position', workflow);
    return response.json({
      ready: result.ready === true,
      state: ['red', 'yellow'].includes(result.state) ? result.state : 'red',
      face: result.face || null,
      faceCount: Number.isInteger(result.faceCount) ? result.faceCount : result.face ? 1 : 0,
      message: result.message || (result.ready ? 'Hold still while your face position is checked.' : 'Adjust your position and try again.')
    });
  }),
  registerVoter: asyncRoute(async (request, response) => {
    let input;
    try {
      input = JSON.parse(request.body?.registration || '{}');
    } catch {
      const error = new Error('Registration details are invalid.');
      error.code = 'INVALID_REGISTRATION';
      error.statusCode = 400;
      throw error;
    }
    validate(input, ['voterId', 'name', 'phone', 'password']);
    if (input.biometricConsent !== true) {
      const error = new Error('Consent to face-template processing is required to register.');
      error.code = 'FACE_CONSENT_REQUIRED';
      error.statusCode = 400;
      throw error;
    }
    const images = requireFaceImages(request, 3, 3);
    const result = await faceClient.enrollFrames(images);
    requireValidFaceAnalysis(result, { requireBackground: false, requireLiveness: false });

    const encrypted = encryptEmbedding(result.embedding);
    const voter = await serializeFaceEnrollment(() => mongo.withFaceEnrollmentLock(async () => {
      await ensureFaceIsNotRegistered(result.embedding, result.model_version);
      return mongo.registerVoter(input, encrypted, result.model_version);
    }));
    return response.status(201).json({
      success: true,
      voter,
      faceEnrollment: { enrolled: true, modelVersion: result.model_version, embeddingDimension: 512 }
    });
  }),
  listVoters: asyncRoute(async (_request, response) => response.json({ success: true, voters: await mongo.listVoters() })),
  updateVoter: asyncRoute(async (request, response) => response.json({ success: true, voter: await mongo.updateVoter(request.params.voterId, request.body, request.user?.username) })),
  approveVoter: asyncRoute(async (request, response) => response.json({ success: true, voter: await mongo.approveVoter(request.params.voterId, request.body.status || 'APPROVED', request.user?.username || 'ADMIN', request.user?.role === 'election_officer' ? 'ELECTION_OFFICER' : 'ADMIN') })),
  getVoter: asyncRoute(async (request, response) => {
    if (request.user.role !== 'admin' && request.user.voterId !== request.params.voterId) {
      return response.status(403).json({ success: false, code: 'FORBIDDEN', message: 'You may only view your own voter record.' });
    }
    return response.json({ success: true, voter: await mongo.getVoterWithFaceStatus(request.params.voterId) });
  }),
  faceStatus: asyncRoute(async (request, response) => {
    const voter = await mongo.getVoterWithFaceStatus(request.user.voterId);
    if (!voter || voter.status !== 'APPROVED') {
      return response.status(403).json({ success: false, code: 'VOTER_NOT_APPROVED', message: 'An approved voter account is required.' });
    }
    return response.json({ success: true, faceEnrollment: voter.faceEnrollment });
  }),
  enrollFace: asyncRoute(async (request, response) => {
    const voterId = request.user.voterId;
    try {
      const images = requireFaceImages(request, 3, 3);
      const voter = await mongo.getVoterCredentials(voterId);
      if (!voter || voter.status !== 'APPROVED') {
        const error = new Error('An approved voter account is required to enroll a face.');
        error.code = 'VOTER_NOT_APPROVED';
        error.statusCode = 403;
        throw error;
      }
      const result = await faceClient.enrollFrames(images);
      requireValidFaceAnalysis(result, { requireBackground: false, requireLiveness: false });
      const encrypted = encryptEmbedding(result.embedding);
      await serializeFaceEnrollment(() => mongo.withFaceEnrollmentLock(async () => {
        await ensureFaceIsNotRegistered(result.embedding, result.model_version);
        await mongo.saveFaceEnrollment(voterId, encrypted, result.model_version);
      }));
      await voterAudit('FACE_ENROLLMENT_SUCCESS', request, null, 'SUCCESS', 'Face enrollment completed');
      return response.json({ success: true, message: 'Face enrolled successfully', faceEnrollment: { enrolled: true, modelVersion: result.model_version, embeddingDimension: 512 } });
    } catch (error) {
      await voterAudit('FACE_ENROLLMENT_FAILED', request, null, 'FAILURE', error.code || 'ENROLLMENT_FAILED');
      throw error;
    }
  }),
  verifyFace: asyncRoute(async (request, response) => {
    const voterId = request.user.voterId;
    const electionId = request.body?.electionId;
    try {
      const framesPerStage = faceChallengeFramesPerStage();
      const images = requireFaceImages(request, framesPerStage * 3, framesPerStage * 3);
      if (typeof electionId !== 'string' || !electionId.trim()) {
        const error = new Error('Election ID is required.');
        error.code = 'INVALID_ELECTION';
        error.statusCode = 400;
        throw error;
      }
      const voter = await mongo.getVoterCredentials(voterId);
      if (!voter || voter.status !== 'APPROVED') {
        const error = new Error('An approved voter account is required.');
        error.code = 'VOTER_NOT_APPROVED';
        error.statusCode = 403;
        throw error;
      }
      const election = await mongo.getElection(electionId);
      const now = Date.now();
      if (!election || election.status !== 'ACTIVE' || now < Date.parse(election.startDate) || now > Date.parse(election.endDate)) {
        const error = new Error('Election is not currently active.');
        error.code = 'ELECTION_NOT_ACTIVE';
        error.statusCode = 400;
        throw error;
      }
      const enrolled = await mongo.getFaceEnrollment(voterId);
      if (!enrolled) {
        const error = new Error('Face enrollment is required before voting.');
        error.code = 'FACE_NOT_ENROLLED';
        error.statusCode = 403;
        throw error;
      }
      const challengeId = request.body?.challengeId;
      if (typeof challengeId !== 'string' || !challengeId) {
        const error = new Error('Complete the current liveness challenge before verification.');
        error.code = 'LIVENESS_CHALLENGE_INVALID';
        error.statusCode = 403;
        throw error;
      }
      const challenge = await mongo.consumeFaceChallenge({
        challengeId,
        voterId,
        electionId,
        sessionHash: sessionHash(request)
      });
      if (!challenge.valid) {
        const error = new Error(challenge.code === 'LIVENESS_CHALLENGE_TOO_FAST'
          ? 'The liveness challenge response was too fast. Please complete the movement and retry.'
          : 'The liveness challenge is expired, already used, or invalid.');
        error.code = challenge.code;
        error.statusCode = 403;
        throw error;
      }
      const enrollmentUpdatedAt = enrolled.updatedAt instanceof Date ? enrolled.updatedAt : new Date(enrolled.updatedAt);
      const storedEmbedding = decryptEmbedding(enrolled);
      const generated = await faceClient.verifyFrames(images, challenge.action);
      requireValidFaceAnalysis(generated);
      if (generated.checks.head_movement !== true) {
        const error = new Error('The requested head movement was not validated.');
        error.code = 'LIVENESS_CHALLENGE_FAILED';
        error.statusCode = 403;
        throw error;
      }
      if (generated.model_version !== enrolled.modelVersion) {
        const error = new Error('Face enrollment must be renewed for the current recognition model.');
        error.code = 'FACE_MODEL_VERSION_MISMATCH';
        error.statusCode = 409;
        throw error;
      }
      const score = cosineSimilarity(generated.embedding, storedEmbedding);
      if (score < faceSimilarityThreshold()) {
        const error = new Error('Face verification failed. Please try again with good lighting and only one face in view.');
        error.code = 'FACE_VERIFICATION_FAILED';
        error.statusCode = 403;
        throw error;
      }

      const token = crypto.randomBytes(32).toString('hex');
      const ttlMs = Number(process.env.FACE_VERIFICATION_TTL_MS || 3 * 60 * 1000);
      if (!Number.isFinite(ttlMs) || ttlMs <= 0 || ttlMs > 15 * 60 * 1000) {
        const error = new Error('Face verification token lifetime configuration is invalid');
        error.code = 'FACE_CONFIGURATION_INVALID';
        error.statusCode = 500;
        throw error;
      }
      await mongo.createFaceVerification({
        tokenHash: crypto.createHash('sha256').update(token).digest('hex'),
        voterId,
        electionId,
        sessionHash: sessionHash(request),
        enrollmentUpdatedAt,
        createdAt: new Date(),
        expiresAt: new Date(Date.now() + ttlMs)
      });
      await voterAudit('FACE_VERIFICATION_SUCCESS', request, electionId, 'SUCCESS', 'Face verification completed');
      return response.json({ success: true, verified: true, verificationToken: token });
    } catch (error) {
      await voterAudit('FACE_VERIFICATION_FAILED', request, electionId || null, 'FAILURE', error.code || 'VERIFICATION_FAILED');
      throw error;
    }
  }),
  createFaceChallenge: asyncRoute(async (request, response) => {
    const voterId = request.user.voterId;
    const electionId = request.body?.electionId;
    if (typeof electionId !== 'string' || !electionId.trim()) {
      const error = new Error('Election ID is required.');
      error.code = 'INVALID_ELECTION';
      error.statusCode = 400;
      throw error;
    }
    const [voter, election, enrollment] = await Promise.all([
      mongo.getVoterCredentials(voterId),
      mongo.getElection(electionId),
      mongo.getFaceEnrollment(voterId)
    ]);
    const now = Date.now();
    if (!voter || voter.status !== 'APPROVED') {
      const error = new Error('An approved voter account is required.');
      error.code = 'VOTER_NOT_APPROVED';
      error.statusCode = 403;
      throw error;
    }
    if (!election || election.status !== 'ACTIVE' ||
        now < Date.parse(election.startDate) || now > Date.parse(election.endDate)) {
      const error = new Error('Election is not currently active.');
      error.code = 'ELECTION_NOT_ACTIVE';
      error.statusCode = 400;
      throw error;
    }
    if (!enrollment) {
      const error = new Error('Face enrollment is required before voting.');
      error.code = 'FACE_NOT_ENROLLED';
      error.statusCode = 403;
      throw error;
    }
    const ttlMs = Number(process.env.FACE_CHALLENGE_TTL_SECONDS || 90) * 1000;
    const framesPerStage = faceChallengeFramesPerStage();
    if (!Number.isInteger(ttlMs) || ttlMs < 15_000 || ttlMs > 5 * 60_000 ||
        framesPerStage < 3 || framesPerStage > 5) {
      const error = new Error('Face challenge configuration is invalid.');
      error.code = 'FACE_CONFIGURATION_INVALID';
      error.statusCode = 500;
      throw error;
    }
    const actions = ['turn_left', 'turn_right'];
    const action = actions[crypto.randomInt(actions.length)];
    const challengeId = crypto.randomBytes(32).toString('hex');
    const expiresAt = new Date(Date.now() + ttlMs);
    await mongo.createFaceChallenge({
      challengeId,
      voterId,
      electionId,
      sessionHash: sessionHash(request),
      action,
      expiresAt
    });
    return response.json({
      success: true,
      challengeId,
      action,
      expiresAt: expiresAt.toISOString(),
      framesPerStage,
      stages: ['center', action, 'center']
    });
  }),
  listElections: asyncRoute(async (_request, response) => response.json({ success: true, elections: await service.listElections() })),
  getCandidates: asyncRoute(async (request, response) => {
    const [candidates, parties] = await Promise.all([
      service.getCandidates(request.params.electionId),
      mongo.listParties()
    ]);
    const partiesByName = new Map(parties.map(party => [party.partyName.trim().toLowerCase(), party]));
    return response.json({
      success: true,
      candidates: candidates.map(candidate => {
        const party = partiesByName.get(String(candidate.party || '').trim().toLowerCase());
        return {
          ...candidate,
          partyImage: party?.imageData || null,
          partySymbol: party?.partySymbol || null
        };
      })
    });
  }),
  registerElection: send(['electionId', 'electionName', 'startDate', 'endDate'], input => service.registerElection(input)),
  updateElection: asyncRoute(async (request, response) => {
    const input = body(request);
    return response.json({ success: true, election: await service.updateElectionState(request.params.electionId, input.status || 'UPCOMING', request.user?.username || 'ADMIN') });
  }),
  getElectionStats: asyncRoute(async (request, response) => {
    const election = await mongo.getElection(request.params.electionId);
    if (!election) throw Object.assign(new Error('Election does not exist'), { statusCode: 404 });
    const votes = await service.getElectionVotes(request.params.electionId);
    return response.json({ success: true, stats: { electionId: election.electionId, status: election.status, votesCast: votes.length } });
  }),
  registerCandidate: send(['candidateId', 'electionId', 'candidateName'], input => service.registerCandidate(input)),
  registerParty: asyncRoute(async (request, response) => {
    const input = body(request);
    validate(input, ['partyId', 'partyName']);
    const image = requireValidPartyImage(request.file);
    const imageData = image
      ? `data:${image.mimetype};base64,${image.buffer.toString('base64')}`
      : null;
    return response.status(201).json({
      success: true,
      party: await mongo.registerParty({ ...input, imageData })
    });
  }),
  updatePartyLogo: asyncRoute(async (request, response) => {
    const image = requireValidPartyImage(request.file);
    if (!image) {
      const error = new Error('Upload a party logo image.');
      error.code = 'INVALID_PARTY_IMAGE';
      error.statusCode = 400;
      throw error;
    }
    const imageData = `data:${image.mimetype};base64,${image.buffer.toString('base64')}`;
    return response.json({
      success: true,
      party: await mongo.updatePartyLogo(request.params.partyId, imageData)
    });
  }),
  listParties: asyncRoute(async (_request, response) => response.json({ success: true, parties: await mongo.listParties() })),
  registerAlliance: send(['allianceId', 'allianceName'], input => mongo.registerAlliance(input)),
  listAlliances: asyncRoute(async (_request, response) => response.json({ success: true, alliances: await mongo.listAlliances() })),
  registerElectionOfficer: send(['officerId', 'name', 'username', 'password'], input => mongo.registerElectionOfficer(input)),
  listElectionOfficers: asyncRoute(async (_request, response) => response.json({ success: true, officers: await mongo.listElectionOfficers() })),
  updateElectionOfficer: asyncRoute(async (request, response) => response.json({ success: true, officer: await mongo.updateElectionOfficer(request.params.officerId, request.body, request.user?.username) })),
  getFaceVerificationSetting: asyncRoute(async (_request, response) => {
    const settings = await mongo.getVotingSettings();
    return response.json({ success: true, enabled: settings.faceVerificationEnabled });
  }),
  updateFaceVerificationSetting: asyncRoute(async (request, response) => {
    if (typeof request.body?.enabled !== 'boolean') {
      const error = new Error('The enabled value must be true or false.');
      error.statusCode = 400;
      throw error;
    }
    const settings = await mongo.setFaceVerificationEnabled(request.body.enabled, request.user.username);
    await mongo.logAudit('FACE_VERIFICATION_SETTING_UPDATED', {
      actorId: request.user.username,
      actorRole: 'ADMIN',
      resourceType: 'SYSTEM_SETTING',
      resourceId: 'voting',
      status: 'SUCCESS',
      description: `Face verification ${settings.faceVerificationEnabled ? 'enabled' : 'disabled'} for voting.`
    });
    return response.json({ success: true, enabled: settings.faceVerificationEnabled });
  }),
  registerCandidateAccount: send(['candidateId', 'name', 'username', 'password', 'partyId'], input => mongo.registerCandidateAccount(input)),
  listCandidateAccounts: asyncRoute(async (_request, response) => response.json({ success: true, candidates: await mongo.listCandidateAccounts() })),
  updateCandidateAccount: asyncRoute(async (request, response) => response.json({ success: true, candidate: await mongo.updateCandidateAccount(request.params.candidateId, request.body, request.user?.username) })),
  castVote: asyncRoute(async (request, response) => {
    const input = body(request);
    validate(input, ['electionId', 'candidateId']);
    const vote = await service.castVote({
      electionId: input.electionId,
      candidateId: input.candidateId,
      verificationToken: input.verificationToken,
      voterId: request.user.voterId,
      sessionHash: sessionHash(request)
    });
    return response.json({ success: true, vote });
  }),
  getVote: asyncRoute(async (request, response) => response.json({ success: true, vote: mapVoteForClient(await service.getVote(request.params.voteId)) })),
  getElectionVotes: asyncRoute(async (request, response) => response.json({ success: true, votes: (await service.getElectionVotes(request.params.electionId)).map(mapVoteForClient) })),
  getElectionHistory: asyncRoute(async (request, response) => response.json({ success: true, history: await service.getElectionHistory(request.params.electionId) })),
  getLedgerIntegrity: asyncRoute(async (request, response) => response.json({ success: true, integrity: await service.verifyLedgerIntegrity(request.params.electionId) })),
  verifyReceipt: asyncRoute(async (request, response) => {
    const input = body(request);
    const transactionId = input.transactionId || request.params.transactionId;
    if (!transactionId) throw Object.assign(new Error('Transaction ID is required'), { statusCode: 400 });
    return response.json({ success: true, receipt: await service.verifyVoteReceipt(transactionId) });
  }),
  getTransactionExplorer: asyncRoute(async (request, response) => {
    const electionId = request.query.electionId || 'TN-ASSEMBLY-2026';
    const votes = (await service.getElectionVotes(electionId)).map(vote => ({
      transactionId: vote.transactionId,
      electionId: vote.electionId,
      candidateId: vote.candidateId,
      timestamp: vote.timestamp,
      previousHash: vote.previousHash,
      recordHash: vote.recordHash,
      status: vote.status || 'CONFIRMED'
    }));
    return response.json({ success: true, transactions: votes });
  }),
  getResults: asyncRoute(async (request, response) => {
    const results = await service.generateResults(request.params.electionId);
    return response.json({ success: true, results });
  }),
  getAdminDashboard: asyncRoute(async (_request, response) => {
    const elections = await mongo.listElections();
    const voters = await mongo.listVoters();
    const votes = await service.getElectionVotes('TN-ASSEMBLY-2026');
    const integrity = typeof service.adapter.health === 'function'
      ? { status: 'FABRIC_PEER_MANAGED' }
      : await service.verifyLedgerIntegrity('TN-ASSEMBLY-2026');
    return response.json({ success: true, dashboard: { totalElections: elections.length, totalVoters: voters.length, approvedVoters: voters.filter(v => v.status === 'APPROVED').length, votesCast: votes.length, integrity } });
  })
};
