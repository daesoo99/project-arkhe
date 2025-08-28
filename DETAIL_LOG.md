# Project Arkhē - Detail Log

## [20250101-0000_initial-multi-agent] Multi-Agent 기본 성능 검증
### Command
`python experiments/run_baseline_comparison.py`
### Parameters
| key | value |
|-----|-------|
| models_draft | qwen2:0.5b |
| models_review | qwen2:0.5b |  
| models_judge | llama3:8b |
| k_samples | 3/2/1 |
| temperature | 0.4-0.8 |
| test_questions | 15 (math, knowledge, coding) |
### Environment
- python: 3.x
- libs: tiktoken, ollama clients
- platform: Windows/Ollama local
### Logs / Errors
Multi-Agent NONE: 50.2% accuracy, 1,766 tokens, efficiency 0.028
Single llama3:8b: 87.7% accuracy, 152 tokens, efficiency 0.577
토큰 계산 로직: src/orchestrator/isolation_pipeline.py:119-148
### Git / Diff
- commit: 39a2c60 (feat: comprehensive multi-agent research framework)
- dirty: no
### Artifacts
- outputs: experiments/results/baseline_comparison_*.json
- figures: 없음
### Decision Evidence
- metrics: 11배 토큰 비용 차이, 42.8% 정확도 격차
- 비교표: Multi-Agent 완전 실패로 판정

## [20250101-0001_information-asymmetry] 정보 비대칭 효과 분석
### Command
`python src/orchestrator/isolation_pipeline.py`
### Parameters
| key | value |
|-----|-------|
| isolation_levels | NONE, PARTIAL, COMPLETE |
| test_questions | 12 (standard benchmarks) |
| models | qwen2:0.5b + llama3:8b |
| k_samples | 3 |
### Environment
- python: 3.x
- libs: 동일
- platform: Windows/Ollama local
### Logs / Errors
NONE: 80.0% accuracy, 101 tokens
PARTIAL: 60.0% accuracy, 56 tokens (worst)
COMPLETE: 80.0% accuracy, 82 tokens
### Git / Diff
- commit: 동일
- dirty: no
### Artifacts
- outputs: experiments/results/isolation_experiment_*.json
- figures: 없음
### Decision Evidence
- metrics: PARTIAL이 예상과 달리 최악 성능
- Counter-intuitive 결과로 "Goldilocks zone" 가설 반박

## [20250810-1947_token-calculation-fix] 사고과정 중심 아키텍처 재설계
### Command
`분석 완료, 구현 대기 중`
### Parameters
| key | value |
|-----|-------|
| 현재_토큰_계산 | 누적 방식 (기하급수적 증가) |
| 예시_차이 | 275 vs 35 토큰 (8배) |
| 목표_개선 | 50% 토큰 감소, 70%+ 정확도 |
| 구현_방안 | A(Aggregator) + B(Prompt) |
### Environment
- python: 3.x
- analysis_tools: 수동 토큰 계산, 로직 분석
- target: src/orchestrator/ 디렉토리
### Logs / Errors
토큰 계산 분석:
- Draft: 15토큰 × 3 + 10토큰 출력 = 55토큰
- Review: 50토큰 × 2 + 30토큰 출력 = 130토큰  
- Judge: 80토큰 × 1 + 10토큰 출력 = 90토큰
- 총합: 275토큰 vs Single 35토큰
### Git / Diff
- commit: 현재 작업 중
- dirty: yes (README.md Protocol 업데이트, 로그 파일 정리)
### Artifacts
- outputs: 구현 예정
- figures: 구현 예정
### Decision Evidence
- metrics: 토큰 계산 로직 분석 완료
- 설계 방안: ThoughtAggregator(공통 추출) + ContextCompressor(사고 압축)

## [20250811-1100_ab-comparison] A/B 방안 성능 비교 실패
### Command
`python test_b_approach.py`
`python test_thought_transfer.py`
### Parameters  
| key | value |
|-----|-------|
| models | qwen2:0.5b (모든 단계) |
| test_questions | 4개 (Seoul, 2+2, seasons, renewable) |
| pipeline_stages | Draft(3) → Review(2) → Judge(1) |
| approaches | A안(ThoughtAggregator) vs B안(프롬프트개선) |
### Environment
- python: 3.x
- models: qwen2:0.5b via Ollama
- platform: Windows/RTX 4060
### Logs / Errors
B안 구조화 실패: "[모든 Draft가 동의하는 내용]" 헤더만 출력
A안 압축 실패: "압축 실패 감지: 6.58 > 1.0, 원본 사용"
Draft 품질 저하: "수요양, 수요양, 수요양" 반복 출력
### Git / Diff
- commit: 현재 작업 중
- dirty: yes (test_b_approach.py, test_thought_transfer.py 추가)
### Artifacts
- outputs: 실패한 구조화 텍스트들
- codes: test_b_approach.py, test_thought_transfer.py
### Decision Evidence
- metrics: Single 180ms vs Multi 3000ms+ (17배 차이)
- 핵심 문제: 0.5B 모델이 구조화된 프롬프트를 이해하지 못함

## [20250811-1200_pipeline-integration] 전체 파이프라인 Multi-Agent 완전 실패
### Command
`python test_full_pipeline.py`
`python test_improved_judge.py`
### Parameters
| key | value |
|-----|-------|
| models | qwen2:0.5b (모든 Agent) |
| test_cases | Seoul, 2+2, Jupiter, Shakespeare |
| judge_prompt | 개선된 버전 (Draft 원본 + Review 종합) |
| approaches | A안 + B안 + Single 3방향 비교 |
### Environment  
- python: 3.x
- models: qwen2:0.5b via Ollama
- platform: Windows/RTX 4060
### Logs / Errors
심각한 성능 저하:
- Single: 효율성 0.0375, 정확도 75%, 토큰 20개
- A안: 효율성 0.000845, 정확도 50%, 토큰 650개
- B안: 효율성 0.000668, 정확도 50%, 토큰 750개
이상한 답변들: "Mercury가 가장 큰 행성", "Romeo는 러시아 소설"
### Git / Diff
- commit: 현재 작업 중
- dirty: yes (test_full_pipeline.py, test_improved_judge.py 추가)
### Artifacts
- outputs: results/full_pipeline_comparison_*.json
- figures: 없음
### Decision Evidence
- metrics: Single 대비 44-56배 효율성 차이
- 비교표: Multi-Agent 모든 지표에서 참패
- 핵심 발견: 모델 지식 한계가 구조적 개선보다 중요

## [20250811-1300_model-upgrade] 7B 모델 업그레이드 성공
### Command
`python test_7b_pipeline.py`
### Parameters
| key | value |  
|-----|-------|
| models_upgrade | qwen2:0.5b → qwen2:7b |
| all_agents | Draft/Review/Judge 모든 단계 7B 통일 |
| test_cases | 5개 (Seoul, 2+2, Jupiter, Shakespeare, 광속) |
| monitoring | 각 단계별 진행상황 출력 |
| approaches | A안(ThoughtAggregator) vs B안(프롬프트개선) vs Single |
### Environment
- python: 3.x
- models: qwen2:7b (약 4.1GB) 
- platform: Windows/RTX 4060 (8GB VRAM)
### Logs / Errors
놀라운 성능 개선:
- B안 7B: 정확도 80% (1위), 효율성 0.000556
- A안 7B: 정확도 60% (3위), 효율성 0.000533  
- Single 7B: 정확도 60% (3위), 효율성 0.076923
실행 시간 증가: A안 76초, B안 34초, Single 0.5초
A안 지속 문제: Jupiter → "토성" 압축 왜곡
### Git / Diff  
- commit: 현재 작업 중
- dirty: yes (test_7b_pipeline.py 완료)
### Artifacts
- outputs: results/7b_pipeline_comparison_1754904720.json
- codes: test_7b_pipeline.py
### Decision Evidence  
- 핵심 성과: **Multi-Agent가 Single 최초 역전** (B안 80% vs Single 60%)
- 방식별 차이: B안(직접 읽기) > A안(압축) > Single
- 계층 구조 필요성: 더 큰 Judge 모델로 원래 설계 의도 구현

## [20250811-1400_hierarchical-experiment] 계층적 Multi-Agent 실험
### Command
`python simple_hierarchical_test.py`
### Parameters
| key | value |
|-----|-------|
| draft_model | qwen2:0.5b |
| review_model | qwen2:7b |
| judge_model | llama3:8b |
| test_question | "What is the capital of South Korea?" |
| expected_answer | "Seoul" |
| approach | B안 베이스 (직접 사고과정 전달) |
### Environment
- python: 3.x
- models: 계층적 구조 (0.5B → 7B → 8B)
- platform: Windows/RTX 4060
### Logs / Errors
Option 1 (Draft→Review→Judge): 100% 정확, 1,687토큰, 35,274ms, 효율성 0.000593
Option 2 (Draft→Judge): 100% 정확, 753토큰, 10,867ms, 효율성 0.001328
Single 8B Model: 100% 정확, 16토큰, 666ms, 효율성 0.062500
Review 단계 비용: +934토큰 (+124%), +24,407ms (+224%)
### Git / Diff
- commit: 현재 작업 중  
- dirty: yes (simple_hierarchical_test.py 추가)
### Artifacts
- codes: simple_hierarchical_test.py
- outputs: results/simple_hierarchical_results.json
### Decision Evidence
- 핵심 발견: Review 단계가 정확도 개선 없이 비용만 2배 증가
- 계층적 구조 실패: Single 모델이 53-105배 더 효율적
- 결론: 간단한 사실적 질문에서는 Multi-Agent 가치 부재

## [아카이브] 이전 구현 기록 (docs/CLAUDE.md에서 이관)
### 주요 구현 완료 내역
- **정답 판별 로직**: 6가지 전문 채점기 (`src/utils/scorers.py`)
- **파이프라인 시스템**: 3가지 패턴 지원 (`src/orchestrator/pipeline.py`) 
- **LLM 통합**: 자동 프로바이더 감지 (`src/llm/simple_llm.py`)
- **경제적 지능 메트릭**: 비용 점수 α*지연+β*계산비용 (α=0.3, β=0.7)
- **AB 테스트 결과**: Single vs Multi 성능 비교 완료