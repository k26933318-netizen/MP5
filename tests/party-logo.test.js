const test = require('node:test');
const assert = require('node:assert/strict');

const controller = require('../backend/blockchain/blockchainController');
const mongo = require('../backend/data/mongoRepository');

function invokeUpdatePartyLogo(file) {
  const response = {
    statusCode: 200,
    status(code) { this.statusCode = code; return this; },
    json(payload) { this.payload = payload; return this; }
  };
  let error;
  return controller.updatePartyLogo(
    { file, params: { partyId: 'PARTY-001' } },
    response,
    caught => { error = caught; }
  ).then(() => ({ response, error }));
}

test('existing party logos accept signature-validated images', async () => {
  const originalUpdate = mongo.updatePartyLogo;
  let saved;
  mongo.updatePartyLogo = async (partyId, imageData) => {
    saved = { partyId, imageData };
    return { partyId, imageData };
  };
  try {
    const png = Buffer.concat([
      Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
      Buffer.from('image data')
    ]);
    const { response, error } = await invokeUpdatePartyLogo({ mimetype: 'image/png', buffer: png });
    assert.equal(error, undefined);
    assert.equal(saved.partyId, 'PARTY-001');
    assert.equal(saved.imageData, `data:image/png;base64,${png.toString('base64')}`);
    assert.equal(response.payload.success, true);
  } finally {
    mongo.updatePartyLogo = originalUpdate;
  }
});

test('party logo updates reject mismatched image content', async () => {
  const { response, error } = await invokeUpdatePartyLogo({
    mimetype: 'image/png',
    buffer: Buffer.from('not a png')
  });
  assert.equal(response.payload, undefined);
  assert.equal(error.code, 'INVALID_PARTY_IMAGE');
  assert.equal(error.statusCode, 415);
});

test('party logo updates require an uploaded image', async () => {
  const { response, error } = await invokeUpdatePartyLogo(undefined);
  assert.equal(response.payload, undefined);
  assert.equal(error.code, 'INVALID_PARTY_IMAGE');
  assert.equal(error.statusCode, 400);
});
