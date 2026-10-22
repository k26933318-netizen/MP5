const express = require('express');
const multer = require('multer');
const rateLimit = require('express-rate-limit');
const controller = require('../blockchain/blockchainController');
const auth = require('../auth');

const router = express.Router();
const faceUpload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: Number(process.env.FACE_MAX_UPLOAD_BYTES || 5 * 1024 * 1024), files: 1 },
  fileFilter: (_request, file, callback) => {
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.mimetype)) {
      const error = new Error('Upload a JPEG, PNG, or WebP image.');
      error.code = 'UNSUPPORTED_IMAGE_TYPE';
      error.statusCode = 415;
      return callback(error);
    }
    callback(null, true);
  }
});
const partyImageUpload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 2 * 1024 * 1024, files: 1 },
  fileFilter: (_request, file, callback) => {
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.mimetype)) {
      const error = new Error('Party logo must be a JPEG, PNG, or WebP image.');
      error.code = 'UNSUPPORTED_IMAGE_TYPE';
      error.statusCode = 415;
      return callback(error);
    }
    callback(null, true);
  }
});
const faceSequenceUpload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: Number(process.env.FACE_FRAME_MAX_UPLOAD_BYTES || 1024 * 1024), files: 15 },
  fileFilter: (_request, file, callback) => {
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.mimetype)) {
      const error = new Error('Upload JPEG, PNG, or WebP face frames.');
      error.code = 'UNSUPPORTED_IMAGE_TYPE';
      error.statusCode = 415;
      return callback(error);
    }
    callback(null, true);
  }
});
const faceLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  limit: Number(process.env.FACE_RATE_LIMIT || 10),
  standardHeaders: true,
  legacyHeaders: false,
  message: { success: false, code: 'FACE_RATE_LIMITED', message: 'Too many face requests. Please try again later.' }
});
const faceAssessmentLimiter = rateLimit({
  windowMs: 60 * 1000,
  limit: Number(process.env.FACE_ASSESS_RATE_LIMIT || 100),
  standardHeaders: true,
  legacyHeaders: false,
  message: { success: false, code: 'FACE_RATE_LIMITED', message: 'Too many face assessment requests. Please wait and retry.' }
});
router.use((request, response, next) => {
  const expected = process.env.BLOCKCHAIN_API_KEY;
  if (process.env.NODE_ENV === 'production' && expected && request.get('x-blockchain-api-key') !== expected) return response.status(401).json({ success: false, message: 'Unauthorized blockchain access' });
  next();
});
router.post('/auth/login', controller.login);
router.post('/auth/logout', auth.requireAuth, controller.logout);
router.post('/voters', faceLimiter, faceSequenceUpload.array('images', 15), controller.registerVoter);
router.get('/voters', auth.requireAuth, (request, response, next) => {
  if (!['admin', 'election_officer'].includes(request.user.role)) {
    return response.status(403).json({ success: false, code: 'FORBIDDEN', message: 'Administrator or election officer access required.' });
  }
  next();
}, controller.listVoters);
router.patch('/voters/:voterId/approve', auth.requireRole('admin'), controller.approveVoter);
router.get('/voters/:voterId', auth.requireAuth, controller.getVoter);
router.post('/voters/:voterId/face', auth.requireRole('voter'), faceLimiter, faceSequenceUpload.array('images', 15), controller.enrollFace);
router.get('/face/status', auth.requireRole('voter'), controller.faceStatus);
router.post('/face/assess', faceAssessmentLimiter, faceUpload.single('image'), controller.assessFace);
router.post('/face/enroll', auth.requireRole('voter'), faceLimiter, faceSequenceUpload.array('images', 15), controller.enrollFace);
router.post('/face/challenge', auth.requireRole('voter'), faceLimiter, controller.createFaceChallenge);
router.post('/face/verify', auth.requireRole('voter'), faceLimiter, faceSequenceUpload.array('images', 15), controller.verifyFace);
router.post('/elections', auth.requireRole('admin'), controller.registerElection);
router.patch('/elections/:electionId', auth.requireRole('admin'), controller.updateElection);
router.get('/elections', auth.requireAuth, controller.listElections);
router.get('/elections/:electionId/stats', auth.requireRole('admin'), controller.getElectionStats);
router.post('/candidates', auth.requireRole('admin'), controller.registerCandidate);
router.get('/election/:electionId/candidates', auth.requireAuth, controller.getCandidates);
router.post('/parties', auth.requireRole('admin'), partyImageUpload.single('image'), controller.registerParty);
router.patch('/parties/:partyId/logo', auth.requireRole('admin'), partyImageUpload.single('image'), controller.updatePartyLogo);
router.get('/parties', auth.requireAuth, controller.listParties);
router.post('/alliances', auth.requireRole('admin'), controller.registerAlliance);
router.get('/alliances', auth.requireRole('admin'), controller.listAlliances);
router.post('/officers', auth.requireRole('admin'), controller.registerElectionOfficer);
router.get('/officers', auth.requireRole('admin'), controller.listElectionOfficers);
router.post('/candidate-accounts', auth.requireRole('admin'), controller.registerCandidateAccount);
router.get('/candidate-accounts', auth.requireRole('admin'), controller.listCandidateAccounts);
router.post('/vote', auth.requireRole('voter'), controller.castVote);
router.get('/vote/:voteId', auth.requireAuth, controller.getVote);
router.get('/election/:electionId/votes', auth.requireRole('admin'), controller.getElectionVotes);
router.get('/election/:electionId/history', auth.requireRole('admin'), controller.getElectionHistory);
router.get('/ledger/:electionId/integrity', auth.requireRole('admin'), controller.getLedgerIntegrity);
router.post('/receipt/verify', auth.requireAuth, controller.verifyReceipt);
router.get('/transactions', auth.requireRole('admin'), controller.getTransactionExplorer);
router.get('/results/:electionId', auth.requireRole('admin'), controller.getResults);
router.get('/admin/dashboard', auth.requireRole('admin'), controller.getAdminDashboard);
module.exports = router;
