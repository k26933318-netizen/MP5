const crypto = require('crypto');

function encryptionKey() {
  const configured = process.env.FACE_ENCRYPTION_KEY || '';
  const key = /^[a-f0-9]{64}$/i.test(configured)
    ? Buffer.from(configured, 'hex')
    : Buffer.from(configured, 'base64');
  if (key.length !== 32) {
    const error = new Error('Face encryption is not configured');
    error.code = 'FACE_ENCRYPTION_NOT_CONFIGURED';
    error.statusCode = 503;
    throw error;
  }
  return key;
}

function encryptEmbedding(embedding) {
  if (!Array.isArray(embedding) || embedding.length !== 512 || embedding.some(value => !Number.isFinite(value))) {
    const error = new Error('Face service returned an invalid embedding');
    error.code = 'INVALID_FACE_EMBEDDING';
    error.statusCode = 502;
    throw error;
  }
  const iv = crypto.randomBytes(12);
  const cipher = crypto.createCipheriv('aes-256-gcm', encryptionKey(), iv);
  const ciphertext = Buffer.concat([
    cipher.update(JSON.stringify(embedding), 'utf8'),
    cipher.final()
  ]);
  return {
    ciphertext: ciphertext.toString('base64'),
    iv: iv.toString('base64'),
    authTag: cipher.getAuthTag().toString('base64'),
    keyVersion: process.env.FACE_ENCRYPTION_KEY_VERSION || '1'
  };
}

function decryptEmbedding(encrypted) {
  if (!encrypted || typeof encrypted.ciphertext !== 'string' ||
      typeof encrypted.iv !== 'string' || typeof encrypted.authTag !== 'string') {
    const error = new Error('Face enrollment is invalid and must be reset');
    error.code = 'FACE_ENROLLMENT_INVALID';
    error.statusCode = 409;
    throw error;
  }
  try {
    const decipher = crypto.createDecipheriv(
      'aes-256-gcm',
      encryptionKey(),
      Buffer.from(encrypted.iv, 'base64')
    );
    decipher.setAuthTag(Buffer.from(encrypted.authTag, 'base64'));
    const plaintext = Buffer.concat([
      decipher.update(Buffer.from(encrypted.ciphertext, 'base64')),
      decipher.final()
    ]).toString('utf8');
    const embedding = JSON.parse(plaintext);
    if (!Array.isArray(embedding) || embedding.length !== 512 ||
        embedding.some(value => !Number.isFinite(value))) {
      throw new Error('Invalid embedding payload');
    }
    return embedding;
  } catch (cause) {
    if (cause.code === 'FACE_ENCRYPTION_NOT_CONFIGURED') throw cause;
    const error = new Error('Stored face enrollment could not be decrypted');
    error.code = 'FACE_ENROLLMENT_DECRYPTION_FAILED';
    error.statusCode = 500;
    error.cause = cause;
    throw error;
  }
}

module.exports = { encryptEmbedding, decryptEmbedding };
