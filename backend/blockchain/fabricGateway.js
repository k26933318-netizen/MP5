const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const grpc = require('@grpc/grpc-js');
const { connect, signers } = require('@hyperledger/fabric-gateway');

function required(name) {
  const value = process.env[name];
  if (!value) {
    const error = new Error(`Missing Fabric configuration: ${name}`);
    error.code = 'FABRIC_CONFIGURATION_INVALID';
    error.statusCode = 503;
    throw error;
  }
  return value;
}

function newGateway() {
  const tlsRootCert = fs.readFileSync(required('FABRIC_TLS_CERT_PATH'));
  const identity = {
    mspId: required('FABRIC_MSP_ID'),
    credentials: fs.readFileSync(required('FABRIC_CERT_PATH'))
  };
  const keyDirectory = required('FABRIC_KEY_DIR');
  const keyFile = fs.readdirSync(keyDirectory).find(file => fs.statSync(path.join(keyDirectory, file)).isFile());
  if (!keyFile) {
    const error = new Error('Fabric signing identity has no private key file.');
    error.code = 'FABRIC_CONFIGURATION_INVALID';
    error.statusCode = 503;
    throw error;
  }
  const privateKey = fs.readFileSync(path.join(keyDirectory, keyFile));
  const client = new grpc.Client(required('FABRIC_PEER_ENDPOINT'), grpc.credentials.createSsl(tlsRootCert), {
    'grpc.ssl_target_name_override': required('FABRIC_PEER_HOST_ALIAS')
  });
  return { gateway: connect({ identity, signer: signers.newPrivateKeySigner(crypto.createPrivateKey(privateKey)), client }), client };
}

async function withContract(callback) {
  let gateway;
  let client;
  try {
    ({ gateway, client } = newGateway());
  } catch (cause) {
    if (cause.code === 'FABRIC_CONFIGURATION_INVALID') throw cause;
    const error = new Error('Hyperledger Fabric credentials or connection configuration are unavailable.');
    error.code = 'FABRIC_CONFIGURATION_INVALID';
    error.statusCode = 503;
    error.cause = cause;
    throw error;
  }
  try {
    const network = gateway.getNetwork(process.env.FABRIC_CHANNEL || 'votingchannel');
    const contract = network.getContract(process.env.FABRIC_CHAINCODE || 'voting-chaincode');
    try {
      return await callback(contract);
    } catch (cause) {
      if (!['ECONNREFUSED', 'ETIMEDOUT', 'EHOSTUNREACH', 'ENOTFOUND', 'UNAVAILABLE', 'DEADLINE_EXCEEDED', 4, 14].includes(cause.code)) {
        throw cause;
      }
      const error = new Error('Hyperledger Fabric peer is unavailable.');
      error.code = 'FABRIC_UNAVAILABLE';
      error.statusCode = 503;
      error.cause = cause;
      throw error;
    }
  } finally {
    gateway.close();
    client.close();
  }
}

module.exports = { withContract };
