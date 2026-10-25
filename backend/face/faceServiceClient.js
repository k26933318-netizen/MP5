const FACE_REQUEST_TIMEOUT_MS = Number(process.env.FACE_REQUEST_TIMEOUT_MS || 15000);
// Render's free instance may need extra time to wake and load the face models.
const FACE_ASSESSMENT_TIMEOUT_MS = Number(process.env.FACE_ASSESSMENT_TIMEOUT_MS || 60000);
const FACE_SEQUENCE_TIMEOUT_MS = Number(process.env.FACE_SEQUENCE_TIMEOUT_MS || 45000);

function faceError(code, message, statusCode = 503) {
  const error = new Error(message);
  error.code = code;
  error.statusCode = statusCode;
  return error;
}

function baseUrl() {
  return (process.env.FACE_SERVICE_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '');
}

function serviceHeaders() {
  const apiKey = process.env.FACE_SERVICE_API_KEY;
  if (!apiKey || apiKey.startsWith('replace-with-') || apiKey.length < 32) {
    throw faceError('FACE_SERVICE_UNAVAILABLE', 'Face recognition service is not configured.');
  }
  return { 'x-face-service-key': apiKey };
}

async function health() {
  try {
    const response = await fetch(`${baseUrl()}/health`, {
      headers: serviceHeaders(),
      signal: AbortSignal.timeout(FACE_REQUEST_TIMEOUT_MS)
    });
    if (!response.ok) return { status: 'unavailable' };
    const data = await response.json();
    return data.status === 'ok' ? { status: 'ok', modelVersion: data.modelVersion } : { status: 'unavailable' };
  } catch {
    return { status: 'unavailable' };
  }
}

async function upload(path, file, extraFields = {}, timeoutMs = FACE_REQUEST_TIMEOUT_MS) {
  if (!file || !Buffer.isBuffer(file.buffer) || file.buffer.length === 0) {
    throw faceError('INVALID_IMAGE', 'Please capture an image before continuing.', 400);
  }
  const form = new FormData();
  form.append('image', new Blob([file.buffer], { type: file.mimetype }), file.originalname || 'capture.jpg');
  for (const [name, value] of Object.entries(extraFields)) form.append(name, String(value));
  return requestForm(path, form, timeoutMs);
}

function generateEmbedding(file) {
  return upload('/embedding', file);
}

function assessPosition(file, stage = 'position', workflow = 'verify') {
  return upload('/assess', file, { stage, workflow }, FACE_ASSESSMENT_TIMEOUT_MS);
}

function uploadFrames(path, files, fields = {}) {
  if (!Array.isArray(files) || files.length === 0) {
    throw faceError('INVALID_IMAGE', 'Capture the required face frames before continuing.', 400);
  }
  const form = new FormData();
  for (const file of files) {
    if (!file || !Buffer.isBuffer(file.buffer) || file.buffer.length === 0) {
      throw faceError('INVALID_IMAGE', 'A captured face frame is invalid.', 400);
    }
    form.append('images', new Blob([file.buffer], { type: file.mimetype }), file.originalname || 'capture.jpg');
  }
  for (const [name, value] of Object.entries(fields)) form.append(name, String(value));
  return requestForm(path, form, FACE_SEQUENCE_TIMEOUT_MS);
}

async function requestForm(path, form, timeoutMs) {
  let response;
  try {
    response = await fetch(`${baseUrl()}${path}`, {
      method: 'POST',
      headers: serviceHeaders(),
      body: form,
      signal: AbortSignal.timeout(timeoutMs)
    });
  } catch (cause) {
    throw faceError('FACE_SERVICE_UNAVAILABLE', 'Face recognition service is unavailable.', 503);
  }
  let result;
  try {
    result = await response.json();
  } catch {
    throw faceError('FACE_SERVICE_UNAVAILABLE', 'Face recognition service returned an invalid response.', 503);
  }
  if (!response.ok) {
    const detail = result.detail || result;
    const backgroundEstimationFailed = detail.code === 'BACKGROUND_ESTIMATION_FAILED';
    throw faceError(
      backgroundEstimationFailed ? 'FACE_VALIDATION_FAILED' : detail.code || 'FACE_SERVICE_UNAVAILABLE',
      backgroundEstimationFailed ? 'Unable to verify your face. Please retry.' : detail.message || 'Face recognition failed.',
      response.status >= 500 ? 503 : response.status
    );
  }
  return result;
}

function enrollFrames(files) {
  return uploadFrames('/frames/enroll', files);
}

function verifyFrames(files, challengeAction) {
  return uploadFrames('/frames/verify', files, { challenge_action: challengeAction });
}

module.exports = { health, generateEmbedding, assessPosition, enrollFrames, verifyFrames, faceError };
