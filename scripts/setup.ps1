# Project Arkhē 자동 설정 스크립트 (개선 버전)
$ErrorActionPreference = "Stop"
$models = @("gemma:2b","qwen2:0.5b","phi3:mini","mistral:7b","llama3:8b","codellama:7b")
$env:OLLAMA_HOST = "http://127.0.0.1:11434"

Write-Host "🚀 Project Arkhē 환경 자동 설정" -ForegroundColor Green
Write-Host "=" * 50

# Python 패키지 설치
Write-Host "`n📦 Python 패키지 설치..."
try {
    pip install -r requirements.txt
    Write-Host "✅ 패키지 설치 완료" -ForegroundColor Green
} catch {
    Write-Host "⚠️ 패키지 설치 실패: $_" -ForegroundColor Yellow
}

# 모델 설치
Write-Host "`n🤖 모델 설치 시작..."
Write-Host "예상 총 용량: ~25GB (선택적 설치 가능)"

foreach ($m in $models) {
    Write-Host "  🔽 $m 설치 중..." -NoNewline
    try {
        ollama pull $m | Out-Null
        Write-Host " ✅" -ForegroundColor Green
    } catch {
        Write-Host " ❌ ($($_))" -ForegroundColor Red
        Write-Warning "모델 $m 설치 실패 - 수동으로 재시도하세요"
    }
}

# 설치된 모델 확인
Write-Host "`n🎯 설치된 모델 목록:"
try {
    ollama list
} catch {
    Write-Host "❌ Ollama 서비스가 실행되지 않습니다" -ForegroundColor Red
    Write-Host "해결방법: Ollama를 다운로드하여 설치하세요 (https://ollama.ai)" -ForegroundColor Yellow
}

# .env 파일 설정
if (Test-Path ".\.env" -PathType Leaf) {
    Write-Host "`n✅ .env 파일이 이미 존재합니다"
} else {
    @"
# 클라우드 키 (비우면 로컬만 사용됨)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=

# 로컬 LLM (Ollama)
OLLAMA_HOST=http://127.0.0.1:11434
LOCAL_MODEL=gemma:2b

# 공통 옵션
TEMP=0.2
MAX_TOKENS=512
TIMEOUT_S=120
"@ | Out-File -Encoding UTF8 .\.env
    Write-Host "`n✅ .env 파일 생성 완료" -ForegroundColor Green
}

# 빠른 테스트
Write-Host "`n🧪 빠른 동작 테스트..."
try {
    python experiments/bench_simple.py --limit 1 | Out-Null
    Write-Host "✅ 시스템 정상 동작" -ForegroundColor Green
} catch {
    Write-Host "⚠️ 테스트 실패: $_" -ForegroundColor Yellow
}

Write-Host "`n🎉 설정 완료!" -ForegroundColor Green
Write-Host "실행 명령어:" -ForegroundColor Cyan
Write-Host "  기본 벤치마크: python experiments/bench_simple.py" -ForegroundColor White
Write-Host "  매트릭스 테스트: .\scripts\run_matrix.ps1" -ForegroundColor White
Write-Host "  빠른 테스트: python experiments/quick_test.py" -ForegroundColor White