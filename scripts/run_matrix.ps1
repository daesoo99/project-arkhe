# Project Arkhē 매트릭스 벤치마크 (개선 버전)
$ErrorActionPreference = "Continue"  # 오류가 있어도 계속 진행
$env:OLLAMA_HOST = "http://127.0.0.1:11434"

Write-Host "🔬 Project Arkhē 매트릭스 벤치마크" -ForegroundColor Cyan
Write-Host "=" * 60

# 사용 가능한 모델 확인
Write-Host "`n🤖 사용 가능한 모델 확인..."
try {
    $modelList = ollama list | Select-String -Pattern "^(\w+:\w+|\w+)" | ForEach-Object { $_.Matches.Groups[1].Value }
    $availableModels = $modelList | Where-Object { $_ -and $_ -ne "NAME" }
    
    if ($availableModels.Count -eq 0) {
        Write-Host "❌ 설치된 모델이 없습니다" -ForegroundColor Red
        Write-Host "해결방법: .\scripts\setup.ps1을 먼저 실행하세요" -ForegroundColor Yellow
        exit 1
    }
    
    Write-Host "✅ 발견된 모델: $($availableModels -join ', ')" -ForegroundColor Green
} catch {
    Write-Host "❌ Ollama 연결 실패: $_" -ForegroundColor Red
    exit 1
}

# 벤치마크 옵션
$taskLimit = 5  # 태스크 수 제한 (빠른 테스트용)
$modelString = $availableModels -join ","

Write-Host "`n📊 벤치마크 설정:"
Write-Host "  모델: $modelString"
Write-Host "  태스크 수: $taskLimit"
Write-Host "  예상 소요시간: $($availableModels.Count * $taskLimit * 2)초"

# 결과 파일 준비
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$resultDir = "results"
$resultFile = "$resultDir\matrix_$timestamp.csv"

if (-not (Test-Path $resultDir)) {
    New-Item -ItemType Directory -Path $resultDir | Out-Null
}

# 매트릭스 벤치마크 실행
Write-Host "`n🚀 벤치마크 실행 시작..."

try {
    python .\experiments\bench_simple.py --models $modelString --limit $taskLimit --outdir $resultDir
    
    # 결과 분석
    Write-Host "`n📈 결과 분석..." -ForegroundColor Green
    
    # CSV 파일에서 결과 읽기
    $csvFiles = Get-ChildItem -Path $resultDir -Filter "bench_*.csv" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    
    if ($csvFiles) {
        $results = Import-Csv $csvFiles.FullName
        
        Write-Host "`n🏆 모델별 성능 순위:" -ForegroundColor Yellow
        
        # 모델별 집계
        $modelStats = $results | Group-Object model | ForEach-Object {
            $accuracies = $_.Group | Where-Object { $_.accuracy -ne "" -and $_.accuracy -ne "None" } | ForEach-Object { [double]$_.accuracy }
            $times = $_.Group | Where-Object { $_.total_ms -ne "" -and [int]$_.total_ms -gt 0 } | ForEach-Object { [int]$_.total_ms }
            
            [PSCustomObject]@{
                Model = $_.Name
                AvgAccuracy = if ($accuracies.Count -gt 0) { ($accuracies | Measure-Object -Average).Average } else { 0 }
                AvgTime = if ($times.Count -gt 0) { ($times | Measure-Object -Average).Average } else { 0 }
                TaskCount = $_.Count
            }
        }
        
        # 정확도 순 정렬
        $modelStats | Sort-Object AvgAccuracy -Descending | Format-Table -AutoSize
        
        # 최고 성능 모델
        $bestAccuracy = $modelStats | Sort-Object AvgAccuracy -Descending | Select-Object -First 1
        $fastestModel = $modelStats | Sort-Object AvgTime | Select-Object -First 1
        
        Write-Host "🥇 최고 정확도: $($bestAccuracy.Model) ($($bestAccuracy.AvgAccuracy.ToString('P1')))" -ForegroundColor Green
        Write-Host "⚡ 최고 속도: $($fastestModel.Model) ($([math]::Round($fastestModel.AvgTime))ms)" -ForegroundColor Green
        
        Write-Host "`n💾 상세 결과: $($csvFiles.FullName)" -ForegroundColor Cyan
        
    } else {
        Write-Host "⚠️ 결과 파일을 찾을 수 없습니다" -ForegroundColor Yellow
    }
    
} catch {
    Write-Host "❌ 벤치마크 실행 실패: $_" -ForegroundColor Red
    exit 1
}

Write-Host "`n🎉 매트릭스 벤치마크 완료!" -ForegroundColor Green
Write-Host "추가 분석을 원하면 CSV 파일을 Excel이나 Python으로 분석하세요" -ForegroundColor White