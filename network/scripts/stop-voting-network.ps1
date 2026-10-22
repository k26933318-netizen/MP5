param([string]$FabricSamples = $env:FABRIC_SAMPLES)
if (-not $FabricSamples) { throw 'Set FABRIC_SAMPLES to the fabric-samples directory.' }
Push-Location "$FabricSamples\test-network"
try { .\network.sh down } finally { Pop-Location }
