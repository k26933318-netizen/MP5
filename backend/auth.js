const crypto = require('crypto');
const bcrypt = require('bcrypt');
const mongo = require('./data/mongoRepository');

const sessions = new Map();
const failedLogins = new Map();
const SESSION_TTL_MS = 30 * 60 * 1000;
const MAX_LOGIN_ATTEMPTS = Number(process.env.MAX_LOGIN_ATTEMPTS || 5);
const LOGIN_COOLDOWN_MS = Number(process.env.LOGIN_COOLDOWN_MS || 5 * 60 * 1000);

const hash = value => crypto.createHash('sha256').update(String(value)).digest('hex');

async function hashPassword(password) {
  return bcrypt.hash(String(password), 12);
}

async function verifyPassword(password, storedHash) {
  if (!storedHash) return false;
  if (storedHash.startsWith('$2')) {
    return bcrypt.compare(String(password), storedHash);
  }
  const legacy = hash(password);
  return crypto.timingSafeEqual(Buffer.from(legacy, 'hex'), Buffer.from(storedHash, 'hex'));
}

function parseSessionToken(request) {
  const authHeader = request.get('authorization') || '';
  const bearerToken = authHeader.replace(/^Bearer\s+/i, '');
  if (bearerToken) return bearerToken;
  const cookieHeader = request.headers.cookie || '';
  const cookieValue = cookieHeader.split(';').map(part => part.trim()).find(part => part.startsWith('sessionToken='));
  return cookieValue ? decodeURIComponent(cookieValue.split('=').slice(1).join('=')) : '';
}

function markFailedLogin(key) {
  const existing = failedLogins.get(key) || { count: 0, lockUntil: 0 };
  const count = existing.count + 1;
  const now = Date.now();
  failedLogins.set(key, {
    count,
    lockUntil: count >= MAX_LOGIN_ATTEMPTS ? now + LOGIN_COOLDOWN_MS : existing.lockUntil || 0
  });
  return failedLogins.get(key);
}

function clearFailedLogin(key) {
  failedLogins.delete(key);
}

async function login(input) {
  const key = `${input.role || 'unknown'}:${(input.username || '').trim().toLowerCase()}`;
  const failedAttempt = failedLogins.get(key);
  if (failedAttempt && failedAttempt.lockUntil > Date.now()) {
    const waitSeconds = Math.ceil((failedAttempt.lockUntil - Date.now()) / 1000);
    throw new Error(`Too many failed login attempts. Please wait ${waitSeconds} seconds and try again.`);
  }

  if (input.role === 'admin') {
    const validUsername = (input.username || '').trim();
    const validPassword = input.password || '';
    const configuredUsername = process.env.ADMIN_USERNAME || '';
    const configuredPassword = process.env.ADMIN_PASSWORD || '';
    const isValid = Boolean(configuredUsername && configuredPassword &&
      !configuredPassword.startsWith('replace-with-') &&
      validUsername === configuredUsername && validPassword === configuredPassword);
    if (!isValid) {
      markFailedLogin(key);
      throw new Error('Invalid admin credentials');
    }
    clearFailedLogin(key);
    const user = { role: 'admin', username: validUsername };
    return createSession(user);
  }

  if (input.role === 'election_officer') {
    const officer = await mongo.getElectionOfficerCredentials(input.username);
    if (!officer) {
      markFailedLogin(key);
      throw new Error('Invalid election officer credentials');
    }
    const passwordMatches = await verifyPassword(input.password, officer.passwordHash);
    if (!passwordMatches) {
      markFailedLogin(key);
      throw new Error('Invalid election officer credentials');
    }
    clearFailedLogin(key);
    return createSession({ role: 'election_officer', username: officer.username, officerId: officer.officerId, name: officer.name });
  }

  if (input.role === 'candidate') {
    const candidate = await mongo.getCandidateCredentials(input.username);
    if (!candidate) {
      markFailedLogin(key);
      throw new Error('Invalid candidate credentials');
    }
    const passwordMatches = await verifyPassword(input.password, candidate.passwordHash);
    if (!passwordMatches) {
      markFailedLogin(key);
      throw new Error('Invalid candidate credentials');
    }
    clearFailedLogin(key);
    return createSession({ role: 'candidate', candidateId: candidate.candidateId, username: candidate.username, name: candidate.name || candidate.candidateName, partyId: candidate.partyId || null });
  }

  if (input.role !== 'voter') {
    markFailedLogin(key);
    throw new Error('Invalid account type');
  }

  const voter = await mongo.getVoterCredentials(input.username);
  if (!voter || voter.status !== 'APPROVED') {
    markFailedLogin(key);
    throw new Error('Invalid voter credentials or voter is not approved');
  }

  const passwordMatches = await verifyPassword(input.password, voter.passwordHash);
  if (!passwordMatches) {
    markFailedLogin(key);
    throw new Error('Invalid voter credentials or voter is not approved');
  }

  clearFailedLogin(key);
  return createSession({ role: 'voter', voterId: voter.voterId, name: voter.name });
}

function createSession(user) {
  const token = crypto.randomBytes(32).toString('hex');
  sessions.set(token, { ...user, expiresAt: Date.now() + SESSION_TTL_MS, createdAt: new Date().toISOString() });
  return { token, user };
}

function userFrom(request) {
  const token = parseSessionToken(request);
  const session = sessions.get(token);
  if (!session || session.expiresAt < Date.now()) {
    if (token) sessions.delete(token);
    return null;
  }
  return session;
}

function requireAuth(request, response, next) {
  const user = userFrom(request);
  if (!user) return response.status(401).json({ success: false, message: 'Login required' });
  request.user = user;
  next();
}

function requireRole(role) {
  return (request, response, next) => {
    requireAuth(request, response, () => {
      if (request.user.role !== role) return response.status(403).json({ success: false, message: `${role} access required` });
      next();
    });
  };
}

function logout(request) {
  const token = parseSessionToken(request);
  sessions.delete(token);
}

module.exports = { hash, hashPassword, verifyPassword, login, logout, requireAuth, requireRole, userFrom, parseSessionToken, sessions };