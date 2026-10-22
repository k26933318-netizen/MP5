param(
  [string]$FabricSamples = $env:FABRIC_SAMPLES
)
if (-not $FabricSamples) { throw 'Set FABRIC_SAMPLES to the fabric-samples directory.' }
Push-Location "$FabricSamples\test-network"
try {
  .\network.sh down
  .\network.sh up createChannel -c votingchannel -ca
  .\network.sh deployCC -ccn voting-chaincode -ccp "$PSScriptRoot\..\..\blockchain\chaincode\voting-chaincode" -ccl javascript -c votingchannel
} finally { Pop-Location }
