$env:PYTHONUTF8 = "1"
$BACKEND = "$PSScriptRoot\backend_services"
$PYTHON  = "$PSScriptRoot\venv\Scripts\python.exe"
$UVICORN = "$PSScriptRoot\venv\Scripts\uvicorn.exe"

Write-Host "=== IR System Local Startup ===" -ForegroundColor Cyan

# Start all 7 microservices in background
$services = @(
    @{ Name="Preprocessing"; Module="preprocessing_service.main:app"; Port=8001 },
    @{ Name="Indexing";       Module="indexing_service.main:app";       Port=8002 },
    @{ Name="Retrieval";      Module="retrieval_service.main:app";       Port=8003 },
    @{ Name="Evaluation";     Module="ranking_eval_service.main:app";    Port=8004 },
    @{ Name="Refinement";     Module="query_refinement_service.main:app"; Port=8005 },
    @{ Name="Topic";          Module="topic_service.main:app";           Port=8006 },
    @{ Name="API Gateway";    Module="api_gateway.main:app";             Port=8000 }
)

$jobs = @()
foreach ($svc in $services) {
    Write-Host "[*] Starting $($svc.Name) on port $($svc.Port)..." -ForegroundColor Yellow
    $job = Start-Process -FilePath $UVICORN `
        -ArgumentList "$($svc.Module) --host 0.0.0.0 --port $($svc.Port)" `
        -WorkingDirectory $BACKEND `
        -PassThru -WindowStyle Minimized
    $jobs += $job
    Start-Sleep -Seconds 1
}

Write-Host ""
Write-Host "=== Waiting for services to start (Models take time to load) ===" -ForegroundColor Cyan

# Check ports
function Check-Port {
    param($port)
    return Test-NetConnection -ComputerName localhost -Port $port -InformationLevel Quiet -WarningAction SilentlyContinue
}

$maxWaitSeconds = 180
$waited = 0
$all_up = $false

while ($waited -lt $maxWaitSeconds -and -not $all_up) {
    $all_up = $true
    foreach ($svc in $services) {
        if (-not (Check-Port $svc.Port)) {
            $all_up = $false
            break
        }
    }
    
    if (-not $all_up) {
        Start-Sleep -Seconds 5
        $waited += 5
        Write-Host "Still waiting for models to load... ($waited/$maxWaitSeconds seconds)" -ForegroundColor Yellow
    }
}

Write-Host "`n=== Service Health Check ===" -ForegroundColor Cyan
foreach ($svc in $services) {
    if (Check-Port $svc.Port) {
        Write-Host "  OK   $($svc.Port) ($($svc.Name))" -ForegroundColor Green
    } else {
        Write-Host "  DOWN $($svc.Port) ($($svc.Name))" -ForegroundColor Red
        $all_up = $false
    }
}

Write-Host "`n=== Loading Datasets into Memory ===" -ForegroundColor Cyan
Write-Host "Loading Quora..." -NoNewline
Invoke-RestMethod -Method Post -Uri "http://localhost:8001/dataset/load?dataset_name=quora" -ErrorAction SilentlyContinue | Out-Null
Write-Host " OK" -ForegroundColor Green
Write-Host "Loading MS MARCO..." -NoNewline
Invoke-RestMethod -Method Post -Uri "http://localhost:8001/dataset/load?dataset_name=msmarco" -ErrorAction SilentlyContinue | Out-Null
Write-Host " OK" -ForegroundColor Green

Write-Host "`n=== Backend is running at http://localhost:8000 ===" -ForegroundColor Yellow
Write-Host "=== API docs: http://localhost:8000/docs ===`n" -ForegroundColor Yellow

Write-Host "Press ENTER to stop all services..." -ForegroundColor Magenta
Read-Host

# Cleanup
foreach ($job in $jobs) {
    Stop-Process -Id $job.Id -ErrorAction SilentlyContinue
}
Write-Host "All services stopped." -ForegroundColor Cyan
