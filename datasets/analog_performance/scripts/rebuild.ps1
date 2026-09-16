param(
    [Parameter(Mandatory=$true)][string]$Destination,
    [Parameter(Mandatory=$true)][string]$Source,
    [string]$Matlab = 'matlab',
    [string]$Python = 'python'
)
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $Destination) { throw 'Destination must be a new directory.' }
$datasetDir = New-Item -ItemType Directory -Path $Destination
New-Item -ItemType Directory -Path (Join-Path $datasetDir.FullName '_work') | Out-Null
Copy-Item -LiteralPath $PSScriptRoot -Destination (Join-Path $datasetDir.FullName 'scripts') -Recurse
$env:ANALOG_SOURCE = (Resolve-Path -LiteralPath $Source).Path
$env:ANALOG_OUTPUT = $datasetDir.FullName
& $Matlab -batch "run(fullfile(getenv('ANALOG_OUTPUT'),'scripts','export_mat_tables.m'))"
if ($LASTEXITCODE -ne 0) { throw 'MATLAB export failed.' }
& $Python (Join-Path $datasetDir.FullName 'scripts\build_dataset.py') --source $env:ANALOG_SOURCE --output $datasetDir.FullName
if ($LASTEXITCODE -ne 0) { throw 'Dataset build failed.' }
& $Python (Join-Path $datasetDir.FullName 'scripts\verify_dataset.py') --output $datasetDir.FullName
if ($LASTEXITCODE -ne 0) { throw 'Dataset verification failed.' }
& $Python (Join-Path $datasetDir.FullName 'scripts\finalize_package.py') --output $datasetDir.FullName
if ($LASTEXITCODE -ne 0) { throw 'Dataset packaging failed.' }
