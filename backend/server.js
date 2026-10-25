require('dotenv').config({ path: require('path').resolve(__dirname, '../.env') });

const express = require('express');
const path = require('path');
const rateLimit = require('express-rate-limit');
const blockchainRoutes = require('./routes/blockchainRoutes');
const mongo = require('./data/mongoRepository');
const { BlockchainService, backendName } = require('./blockchain/blockchainService');
const faceClient = require('./face/faceServiceClient');
const blockchainService = new BlockchainService();

const app = express();
app.disable('x-powered-by');
app.use((request, response, next) => {
  response.setHeader('X-Content-Type-Options', 'nosniff');
  response.setHeader('X-Frame-Options', 'DENY');
  response.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');
  response.setHeader('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; connect-src 'self'; img-src 'self' data:;" );
  next();
});
app.use(express.json({ limit: '1mb' }));
app.use(express.urlencoded({ extended: false }));
app.use(rateLimit({ windowMs: 60 * 1000, max: 120, standardHeaders: true, legacyHeaders: false, message: { success: false, error: { code: 'RATE_LIMITED', message: 'Too many requests. Please try again later.' } } }));
app.use((request, response, next) => {
  const origin = request.headers.origin;
  if (!origin) return next();
const allowedOrigins = process.env.ALLOWED_ORIGINS ? process.env.ALLOWED_ORIGINS.split(',') : ['http://localhost:3000', 'http://127.0.0.1:3000'];
    if (allowedOrigins.includes(origin)) response.setHeader('Access-Control-Allow-Origin', origin);
  response.setHeader('Access-Control-Allow-Methods', 'GET,POST,PUT,PATCH,DELETE,OPTIONS');
  response.setHeader('Access-Control-Allow-Headers', 'Content-Type,Authorization,x-blockchain-api-key');
  response.setHeader('Access-Control-Allow-Credentials', 'true');
  if (request.method === 'OPTIONS') return response.sendStatus(204);
  next();
});
app.use(express.static(path.resolve(__dirname, '../public')));
app.use('/api/blockchain', blockchainRoutes);
app.get('/api/health', async (_request, response) => {
  const [database, faceService, blockchain] = await Promise.all([
    mongo.healthCheck().then(() => 'ok').catch(() => 'unavailable'),
    faceClient.health().then(result => result.status).catch(() => 'unavailable'),
    blockchainService.health().then(status => status).catch(() => 'unavailable')
  ]);
  const backend = database === 'ok' ? 'ok' : 'degraded';
  const healthy = database === 'ok' && faceService === 'ok' && blockchain === 'ok';
  return response.status(healthy ? 200 : 503).json({
    backend,
    database,
    faceService,
    blockchain: backendName,
    blockchainStatus: blockchain
  });
});
app.use((error, _request, response, _next) => {
  const message = error.message || 'Request failed';
  const status = error.statusCode || (
    error.code === 'LIMIT_FILE_SIZE' ? 413 :
      error.code === 'LIMIT_UNEXPECTED_FILE' ? 400 :
        error.code === 11000 ? 409 :
          /(already voted|already recorded|already exists)/i.test(message) ? 409 :
            /(does not exist|not belong|not found)/i.test(message) ? 404 :
              /(network|endorsement|commit|unavailable)/i.test(message) ? 503 :
                /(invalid|denied|forbidden|not active|closed)/i.test(message) ? 400 : 400
  );
  const code = error.code === 'LIMIT_FILE_SIZE' ? 'IMAGE_TOO_LARGE' :
    error.code === 11000 ? 'DUPLICATE_RECORD' :
      error.code || (status === 503 ? 'SERVICE_UNAVAILABLE' : 'VALIDATION_ERROR');
  if (status >= 500) console.error(JSON.stringify({ event: 'REQUEST_FAILED', code, status }));
  const responseMessage = status >= 500
    ? code === 'FACE_SERVICE_UNAVAILABLE'
      ? 'The face service did not respond. It may be waking up; retry in a minute. If this continues, check that the face service is running in Render.'
      : 'The server could not complete the request.'
    : error.code === 11000
      ? 'This record already exists.'
      : message;
  response.status(status).json({
    success: false,
    error: {
      code,
      message: responseMessage
    },
    code,
    message: responseMessage
  });
});

if (require.main === module) {
  app.listen(process.env.PORT || 3000, () => console.log(`Blockchain API listening on port ${process.env.PORT || 3000}`));
}

module.exports = app;
