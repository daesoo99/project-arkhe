# Project Arkhē - Experiment Log

## 히스토리 (ARCHIVED)

### [20250101-0000_initial-multi-agent] Multi-Agent 기본 성능 검증 (ARCHIVED - DIAGNOSED)

- **가설**: Multi-Agent 시스템(Draft→Review→Judge)이 단일 모델보다 더 높은 정확도와 효율성을 제공할 것이다.
- **실험**:
  - 데이터/모델/파라미터/커맨드:
    - data: 표준 벤치마크 15개 질문 (math, knowledge, coding)
    - models: qwen2:0.5b (Draft/Review) + llama3:8b (Judge)
    - params: temperature=0.4-0.8, k_samples=3/2/1
    - cmd: `python experiments/run_baseline_comparison.py`
- **결과**: 
  - Multi-Agent NONE: 50.2% 정확도, 1,766 토큰, 효율성 0.028
  - Single llama3:8b: 87.7% 정확도, 152 토큰, 효율성 0.577
  - **실패**: Single 모델이 42.8% 더 높은 정확도, 11배 낮은 비용
- **원인 분석**: 
  1. 토큰 계산 방식 - 누적 프롬프트로 기하급수적 증가
  2. 작은 모델(qwen2:0.5b)의 품질 한계
  3. 단순 결과 나열 방식으로 사고과정 손실
- **[DECISION]**
  - 선택: Multi-Agent 아키텍처 근본적 재설계 필요
  - 근거: 토큰 비효율성이 핵심 원인으로 판단됨
  - 영향: 정보 비대칭 실험 추가 후 아키텍처 수정 방향 결정
- **향후 계획(분기 가능)**:
  1) [20250101-0001_information-asymmetry] 정보 공유 수준별 성능 비교
- **실행 상태**:
  - [20250101-0001_information-asymmetry]: 완료 → 결과: 세션 [20250101-0001_information-asymmetry] 참조
- **관련**:
  - DETAIL_LOG.md#[20250101-0000_initial-multi-agent]
  - 실패 코드: failed_hypotheses/20250101-0000_initial-multi-agent_cumulative-prompts.py

### [20250101-0001_information-asymmetry] 정보 비대칭 효과 분석 (ARCHIVED - PARTIAL SUCCESS)

- **가설**: 정보 공유 수준(NONE/PARTIAL/COMPLETE)에 따라 Multi-Agent 성능이 달라질 것이다.
- **실험**:
  - 데이터/모델/파라미터/커맨드:
    - data: 표준 벤치마크 12개 질문
    - models: 동일한 모델 구성 (qwen2:0.5b + llama3:8b)
    - params: 3가지 격리 수준 비교
    - cmd: `python src/orchestrator/isolation_pipeline.py`
- **결과**:
  - NONE (완전 공유): 80.0% 정확도, 101 토큰
  - PARTIAL (제한 공유): 60.0% 정확도, 56 토큰 (최악)
  - COMPLETE (완전 독립): 80.0% 정확도, 82 토큰
- **원인 분석**: 
  - Counter-intuitive: 부분 공유가 가장 나쁨
  - "Goldilocks zone" 가설 반박
  - 완전 공유 or 완전 독립이 더 효과적
- **[DECISION]**
  - 선택: 토큰 계산 방식 문제가 핵심으로 확인, 아키텍처 재설계 진행
  - 근거: 정보 공유 최적화로는 근본 문제 해결 불가
  - 영향: 사고과정 중심 압축 아키텍처 개발 착수
- **향후 계획(분기 가능)**:
  1) [20250810-1947_token-calculation-fix] 토큰 계산 문제 해결 및 새 아키텍처 설계
- **실행 상태**:
  - [20250810-1947_token-calculation-fix]: 진행중
- **관련**:
  - DETAIL_LOG.md#[20250101-0001_information-asymmetry]

## 현재 진행 중 (ACTIVE)

### [20250810-1947_token-calculation-fix] 사고과정 중심 아키텍처 재설계

- **가설**: Multi-Agent 시스템의 비효율성은 토큰 계산 방식과 누적 프롬프트 길이 때문이다. 사고과정 중심의 압축 아키텍처로 전환하면 성능 역전이 가능하다.
- **실험**:
  - 데이터/모델/파라미터/커맨드:
    - data: 표준 벤치마크 + 복잡한 문제 세트
    - models: 기존 모델 구성 유지
    - params: A방안(Aggregator/Compressor) vs B방안(프롬프트 개선)
    - cmd: `python experiments/run_thought_compression_experiment.py` (구현 예정)
- **결과**: (실험 진행 예정)
- **원인 분석**: 
  - 현재 구현: Draft→Review→Judge 단계마다 이전 결과 모두 누적
  - 토큰 예시: 간단한 "2+3=?" 질문에도 275 토큰 vs Single 35 토큰 (8배 차이)
  - 사고과정 손실: 단순 결과 나열로 창의적 아이디어 제거
- **[DECISION]**
  - 선택: A방안(ThoughtAggregator)과 B방안(프롬프트 개선) 병행 구현
  - 근거: 공통 요소 추출 + 개별 특징 보존이 핵심
  - 영향: 토큰 50% 감소 + 정확도 70% 이상 달성 목표
- **[구현 방안]**
  - **A방안: ThoughtAggregator 컴포넌트**
    - 별도 LLM 모델을 사용하는 ThoughtAggregator 클래스를 Draft↔Review, Review↔Judge 사이에 추가. 이전 단계 결과들을 분석하여 공통 핵심 아이디어는 압축하고 독창적 접근법은 별도 보존한 후 "공통 핵심 + 개별 특징" 형태로 다음 단계에 전달. 새로운 모델 추가로 정보 처리 전문화하는 아키텍처 접근법.
  - **B방안: 사고과정 분석 프롬프트**  
    - 기존 Agent들의 프롬프트를 "단순 결과 나열 → 사고과정 분석" 모드로 변경. 이전 단계 결과를 그대로 나열하는 대신 "1. 공통 아이디어 추출, 2. 독특한 접근 분석, 3. 통합 답변 생성" 구조로 프롬프트를 재설계. 기존 파이프라인 구조는 유지하면서 정보 처리 방식만 개선하는 점진적 접근법.
- **향후 계획(분기 가능)**:
  1) [20250811-1000_thought-aggregator] ThoughtAggregator 컴포넌트 구현
  2) [20250811-1030_prompt-improvement] 사고과정 분석 프롬프트 개선
  3) [20250811-1100_ab-comparison] A/B 방안 성능 비교 실험
- **실행 상태**:
  - [20250811-1000_thought-aggregator]: 완료 → 결과: ThoughtAggregator 구현 완료, 81% 토큰 압축 달성
  - [20250811-1030_prompt-improvement]: 대기  
  - [20250811-1100_ab-comparison]: 대기
- **관련**:
  - DETAIL_LOG.md#[20250810-1947_token-calculation-fix]
  - 기존 문제 코드: failed_hypotheses/20250101-0000_initial-multi-agent_cumulative-prompts.py

### [20250811-1000_thought-aggregator] ThoughtAggregator 컴포넌트 구현 (ARCHIVED - COMPLETED)

- **가설**: 별도 LLM 모델(qwen2:0.5b)을 사용하는 ThoughtAggregator 클래스가 다중 응답을 분석하여 공통 핵심과 개별 특징을 추출한 후 압축된 컨텍스트를 생성할 수 있다.
- **실험**:
  - 데이터/모델/파라미터/커맨드:
    - data: 간단한 테스트 응답 3개 ("Seoul is the capital..." 시리즈)
    - models: qwen2:0.5b (분석용), tiktoken (토큰 계산)
    - params: temperature=0.3/0.4/0.2 (단계별), max_tokens=200/300/400
    - cmd: `python test_thought_aggregator.py`
- **결과**: 
  - 압축률: 0.19 (81% 토큰 절약)
  - 원본 토큰: 53개 → 압축 후: 10개
  - 공통 핵심: "Seoul is the capital of South Korea."
  - 개별 특징: 3개 독창적 접근법 성공적 추출
- **원인 분석**: 
  - LLM 기반 압축이 효과적으로 중복 제거
  - 공통 요소와 개별 특징 분리가 성공적
  - 폴백 메커니즘으로 안정성 확보
- **[DECISION]**
  - 선택: A방안(ThoughtAggregator) 1차 구현 완료, 통합 파이프라인 구현 진행
  - 근거: 목표 대비 우수한 압축 성능 확인 (50% 목표 대비 81% 달성)
  - 영향: B방안과의 성능 비교 실험 준비 완료
- **향후 계획(분기 가능)**:
  1) [20250811-1100_ab-comparison] A/B 방안 성능 비교 실험
  2) [20250811-1200_pipeline-integration] 전체 파이프라인 통합 테스트
- **실행 상태**:
  - [20250811-1100_ab-comparison]: 완료 → 결과: 세션 [20250811-1100_ab-comparison] 참조
  - [20250811-1200_pipeline-integration]: 완료 → 결과: 세션 [20250811-1200_pipeline-integration] 참조
- **[핵심 설계 아이디어]**:
  - **사고과정 전달**: Draft들의 사고과정과 답을 압축하여 Review에게 전달할 때, a,b,c의 공통된 부분 + 각 모델의 독특한 사고과정까지 전달되기를 원함. 단순 결론 압축이 아닌 추론 방식의 다양성 보존이 목표.
  - **Review 단계 설계**: Review들은 3개 Draft의 공통된부분 + 3개의 다른 의견을 2명의 reviewer가 보고 추가적으로 피드백. 맞는 의견 + 더다양한 의견일 수 있음. Review → Judge 사이 압축 적용 여부는 고민 중 (2개뿐이라 효과 적을 수 있음).
  - **정보 손실 위험**: (1) LLM의 잘못된 판단 - "AI는 유용하다" vs "AI는 위험하다"를 둘 다 "AI 의견"으로 잘못 분류할 위험, (2) 미묘한 뉘앙스 손실 - "최선" vs "실용적" 차이가 "좋다"로 뭉개질 수 있음, 이 경우 압축하면 토큰이 오히려 늘어날 수 있어 B방안이 더 나을 수도.
- **관련**:
  - DETAIL_LOG.md#[20250811-1000_thought-aggregator]
  - 구현 파일: src/orchestrator/thought_aggregator.py
  - 통합 파이프라인: src/orchestrator/thought_compression_pipeline.py

## 현재 진행 중 (ACTIVE)

### [20250811-1100_ab-comparison] A/B 방안 성능 비교 (ARCHIVED - FAILED)

- **가설**: ThoughtAggregator(A안) vs 프롬프트개선(B안) 방식을 비교하여 사고과정 전달에서 더 효과적인 방법을 찾을 수 있다.
- **실험**:
  - 데이터/모델/파라미터/커맨드:
    - data: 4개 테스트 질문 (Seoul, 2+2, seasons, renewable energy)
    - models: qwen2:0.5b (모든 단계)
    - params: Draft-Review-Judge 3단계 파이프라인
    - cmd: `python test_b_approach.py`, `python test_thought_transfer.py`
- **결과**: 
  - A안: 평균 정확도 미측정, 압축 실패 다수 발생
  - B안: 프롬프트 구조화 실패, 헤더만 출력
  - 모든 접근법이 Single Model 대비 심각한 성능 저하
- **원인 분석**: 
  - qwen2:0.5b 모델의 근본적 한계 확인
  - 복잡한 사고과정에서 압축 실패 (압축률 > 1.0)
  - 구조화된 프롬프트를 모델이 제대로 따르지 못함
- **[DECISION]**
  - 선택: 모델 크기 업그레이드가 필수적, 0.5B → 7B 전환 결정
  - 근거: 구조/프롬프트 개선으로는 모델 지식 한계 극복 불가
  - 영향: 전체 파이프라인을 7B 모델로 재구성 필요
- **향후 계획(분기 가능)**:
  1) [20250811-1200_pipeline-integration] 전체 파이프라인 통합 테스트 (0.5B 마지막 검증)
  2) [20250811-1300_model-upgrade] 7B 모델로 업그레이드
- **실행 상태**:
  - [20250811-1200_pipeline-integration]: 완료 → 결과: 세션 [20250811-1200_pipeline-integration] 참조
  - [20250811-1300_model-upgrade]: 진행중
- **관련**:
  - DETAIL_LOG.md#[20250811-1100_ab-comparison]
  - 실패 코드: test_b_approach.py, test_thought_transfer.py

### [20250811-1200_pipeline-integration] 전체 파이프라인 통합 테스트 (ARCHIVED - FAILED)

- **가설**: Draft→Review→Judge 전체 파이프라인에서 A안/B안/Single 모델의 성능을 정확히 비교할 수 있다.
- **실험**:
  - 데이터/모델/파라미터/커맨드:
    - data: 4개 표준 질문 (Seoul, 2+2, Jupiter, Shakespeare)
    - models: qwen2:0.5b (모든 Agent)
    - params: 개선된 Judge 프롬프트 적용
    - cmd: `python test_full_pipeline.py`, `python test_improved_judge.py`
- **결과**: 
  - **효율성**: Single(0.0375) vs A안(0.000845) vs B안(0.000668) = **44-56배 차이**
  - **정확도**: Single(75%) vs Multi-Agent(50%) 
  - **토큰 효율**: Single(20토큰) vs Multi-Agent(600-800토큰) = **30-40배 차이**
  - **응답 시간**: Single(138ms) vs Multi-Agent(3000-6000ms) = **20-50배 차이**
- **원인 분석**: 
  - **Multi-Agent의 구조적 결함**: 작은 모델들의 오류가 누적되면서 Judge도 잘못된 판단
  - **토큰 폭증**: 간단한 질문도 수백 토큰 소모
  - **지식 한계**: Single Model도 Jupiter → "Mercury" 틀림 (qwen2:0.5b 한계)
  - **이상한 답변들**: "Romeo는 러시아 소설", "Mercury가 가장 큰 행성" 등
- **[DECISION]**
  - 선택: 0.5B 모델로는 Multi-Agent 가치 입증 불가, 7B 모델 전환 결정
  - 근거: 구조적 개선보다는 모델 성능 자체가 핵심 제약
  - 영향: 전면적인 모델 업그레이드와 재실험 필요
- **향후 계획(분기 가능)**:
  1) [20250811-1300_model-upgrade] 7B 모델로 업그레이드
  2) [20250811-1400_7b-pipeline-test] 7B 파이프라인 재실험
- **실행 상태**:
  - [20250811-1300_model-upgrade]: 진행중
  - [20250811-1400_7b-pipeline-test]: 대기
- **관련**:
  - DETAIL_LOG.md#[20250811-1200_pipeline-integration]
  - 결과 파일: results/full_pipeline_comparison_*.json
  - 개선 시도: test_improved_judge.py

### [20250811-1300_model-upgrade] 7B 모델 업그레이드 (ARCHIVED - COMPLETED)

- **가설**: qwen2:0.5b → qwen2:7b 모델 업그레이드로 Multi-Agent 시스템의 진정한 가치를 확인할 수 있다.
- **실험**:
  - 데이터/모델/파라미터/커맨드:
    - data: 5개 테스트 케이스 (Seoul, 2+2, Jupiter, Shakespeare, 광속)
    - models: qwen2:7b (Draft/Review/Judge 모든 단계)
    - params: A안(ThoughtAggregator) + B안(프롬프트개선) + Single 비교
    - cmd: `python test_7b_pipeline.py`
- **결과**: 
  - **B안 7B**: 80% 정확도 (1위) - **Multi-Agent가 Single 역전 달성!**
  - **A안 7B**: 60% 정확도 (3위) - 압축 과정에서 정보 왜곡 지속
  - **Single 7B**: 60% 정확도 (3위) - 예상 외 낮은 성능
- **원인 분석**: 
  - **모델 크기가 게임 체인저**: 7B vs 0.5B의 압도적 차이
  - **B안의 우수성**: 사고과정 직접 전달로 정보 손실 최소화
  - **A안의 한계**: Jupiter → "토성" 등 압축 과정 정보 왜곡 지속
  - **토큰 효율성**: 여전히 Single이 68-144배 우수
- **[DECISION]**
  - 선택: B안을 베이스로 Multi-Agent 추가 개선, 계층적 구조 도입
  - 근거: Multi-Agent가 정확도에서 Single을 능가함이 최초 입증
  - 영향: 더 큰 Judge 모델로 계층적 Multi-Agent 실험 필요
- **향후 계획(분기 가능)**:
  1) [20250811-1400_hierarchical-experiment] 계층적 Multi-Agent (Draft 7B → Judge 14B)
  2) [20250811-1500_b-approach-optimization] B안 베이스 최적화 (Review 제거 등)
- **실행 상태**:
  - [20250811-1400_hierarchical-experiment]: 대기 (계층 구조 설계 중)
  - [20250811-1500_b-approach-optimization]: 대기
- **관련**:
  - DETAIL_LOG.md#[20250811-1300_model-upgrade]
  - 구현 파일: test_7b_pipeline.py
  - 결과 파일: results/7b_pipeline_comparison_1754904720.json

## 현재 진행 중 (ACTIVE)

### [20250811-1400_hierarchical-experiment] 계층적 Multi-Agent 실험 (ARCHIVED - COMPLETED)

- **가설**: Draft(qwen2:0.5b) → Review(qwen2:7b) → Judge(llama3:8b) 계층적 구조로 Multi-Agent의 원래 설계 의도를 구현하면 Single 대비 정확도와 효율성을 모두 개선할 수 있다.
- **실험**:
  - 데이터/모델/파라미터/커맨드:
    - data: 1개 테스트 케이스 (Seoul 수도 질문)
    - models: Draft(qwen2:0.5b) + Review(qwen2:7b) + Judge(llama3:8b)
    - params: B안 베이스, Judge가 Draft 사고과정 + Review 분석 종합
    - cmd: `python simple_hierarchical_test.py`
- **결과**:
  - **정확도**: Option 1(1.00) = Option 2(1.00) = Single(1.00) - 모두 100% 정확
  - **효율성**: Option 1(0.000593) < Option 2(0.001328) << Single(0.062500) - Single이 압도적
  - **토큰 비용**: Option 1(1,687) vs Option 2(753) vs Single(16) - 105배 vs 47배 차이
  - **실행 시간**: Option 1(35,274ms) vs Option 2(10,867ms) vs Single(666ms) - 53배 vs 16배 차이
- **원인 분석**:
  - **Review 단계 무용성**: 정확도 개선 없이 토큰 124% 증가, 시간 224% 증가
  - **계층적 구조 실패**: 큰 Judge 모델도 효율성 차이 극복 불가
  - **간단한 문제의 한계**: 사실적 질문에서는 Multi-Agent 가치 부재
- **[DECISION]**
  - 선택: 계층적 Multi-Agent 실험 완료, 단순 사실적 질문에서는 Single이 최적
  - 근거: Review 단계가 비용만 증가시키고 정확도 개선 효과 없음
  - 영향: Multi-Agent는 복잡한 추론 문제에서만 가치가 있을 것으로 판단
- **향후 계획(분기 가능)**:
  1) [20250811-1500_complex-problem-test] 복잡한 추론 문제에서 Multi-Agent 재검증
  2) [20250811-1600_final-conclusion] 전체 실험 결과 종합 및 최종 결론 도출
- **실행 상태**:
  - [20250811-1500_complex-problem-test]: 대기
  - [20250811-1600_final-conclusion]: 대기
- **관련**:
  - DETAIL_LOG.md#[20250811-1400_hierarchical-experiment]
  - 구현 파일: simple_hierarchical_test.py
  - 결과 파일: results/simple_hierarchical_results.json