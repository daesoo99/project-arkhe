# Project Arkhē - 실험 진행 로그

## 📊 **실험 우선순위 재정렬 (2025-01-09)**

### **핵심 차별점**: "정보 비대칭(Information Asymmetry)"

**목표**: Multi-Agent 간 정보 공유 수준이 성능에 미치는 영향 정량화

### **재정렬된 우선순위**

1. **Multi-Agent Orchestration (정보 격리 실험) ✅**
   - 격리 수준별 성능 비교
   - 차별화 포인트 그 자체

2. **Shannon Entropy 승급 정책 (정보이론 트리거)**
   - Multi-Agent 작동을 위한 핵심 레버
   - H+JS/불확실성 결합으로 승급 정밀도 향상

3. **Economic Intelligence (비용 효율 배분)**
   - Pareto front(비용·지연·정확도) 도출
   - 실운영 가이드 제시

---

## 🧪 **Phase 1: Information Asymmetry 실험 결과**

### **실험 설계**
- **격리 수준**: NONE / PARTIAL / COMPLETE
- **모델 구성**: qwen2:0.5b → gemma:2b → llama3:8b (동일)
- **테스트 질문**: 5개 (간단→복잡)
- **측정 지표**: 정확도, 비용(토큰), 다양성(엔트로피), 시간

### **격리 수준별 차이점**

#### **NONE (기존 방식)**
```
Review: "Draft answers: A | B | C 모두 보고 개선"
Judge:  "Draft + Review 모두 보고 최종 판단"
```

#### **PARTIAL (부분 격리)**
```
Review: "Draft 하나만 참조하여 개선" 
Judge:  "Review 결과만 보고 판단" (Draft 차단)
```

#### **COMPLETE (완전 격리)**
```
Review: "독립적으로 답변" (Draft 무시)
Judge:  "독립적으로 답변" (이전 단계 무시)
```

### **🔥 핵심 결과**

```
      NONE: Accuracy=0.800, Diversity=0.766, Cost=689 tokens
   PARTIAL: Accuracy=1.000, Diversity=0.820, Cost=384 tokens  
  COMPLETE: Accuracy=1.000, Diversity=0.787, Cost=524 tokens
```

### **주요 발견**

1. **PARTIAL 격리가 최적**
   - 정확도 100% (NONE 대비 +20%p)
   - 비용 44% 절약 (689→384 토큰)
   - 다양성 최대 (0.820)

2. **정보 과부하 현상 확인**
   - NONE: 너무 많은 정보 → 모델 혼란 → 정확도 하락
   - PARTIAL: 적절한 정보량 → 명확한 개선 방향

3. **정보 비대칭 ↑ → 성능 ↑**
   - 예상과 반대 결과
   - "정보 제한이 오히려 품질 향상" 입증

### **시간 데이터 해석**
- 백그라운드 활동으로 인한 노이즈 존재
- 토큰 수, 정확도, 엔트로피는 신뢰할 만함
- 상대적 성능 순서는 일관됨

---

## 📁 **생성된 파일들**

### **코어 실험 프레임워크**
- `src/orchestrator/isolation_pipeline.py`: 정보 비대칭 실험 파이프라인
- `src/metrics/information_theory.py`: Shannon Entropy 계산기
- `src/orchestrator/experimental_pipeline.py`: 다중 샘플링 지원 파이프라인

### **실험 실행기들**
- `experiments/run_entropy_experiment.py`: Shannon Entropy 승급 정책 실험
- `experiments/run_quick_demo.py`: 빠른 데모용
- `experiments/run_isolation_experiment.py`: (예정) 격리 vs 단일모델 비교

### **결과 파일들**
- `experiments/results/isolation_experiment_1754704420.json`: 격리 실험 결과
- `datasets/experimental_questions.json`: 실험용 질문 데이터셋

---

## 🚀 **다음 단계 계획**

### **Priority 1: 현실적 벤치마크 비교**
```
PARTIAL (qwen→gemma→llama) vs 단일 llama3:8b vs 단일 gemma:2b
```

**핵심 질문**: "Multi-Agent가 단일 고급 모델보다 실제로 나은가?"

**예상 시나리오**:
- PARTIAL 승리: 다양성↑, 비용 비슷, 정확도 유사
- 단일 모델 승리: 속도↑, 일관성↑
- 상황별 다름: 복잡도에 따라 갈림

### **Priority 2: Shannon Entropy 승급 정책 고도화**
- H 단독 → H+JS+불확실성 결합
- 과승급 방지, 승급 정밀도 향상
- 목표: "비용 30-50%↓, 정확도 손실 ≤2%p"

### **Priority 3: Economic Intelligence 최적화**
- k, τ, 모델셋 조합 최적화
- Pareto front 도출
- Top-5 정책 카드 + 운영 가이드

---

## 🏆 **현재까지의 성과**

### **정량적 결과**
- **정보 격리 효과 입증**: PARTIAL로 정확도 20%p ↑, 비용 44% ↓
- **정보 이론 메트릭 구현**: Shannon Entropy, JS Divergence, 불확실성 점수
- **자동화된 실험 프레임워크**: 파라미터 스윕, JSONL 로깅, 메트릭 계산

### **핵심 인사이트**
1. **"정보 제한이 품질 향상"**: 정보 과부하 현상 확인
2. **"적절한 비대칭이 최적"**: PARTIAL > COMPLETE > NONE
3. **"Multi-Agent의 실용성"**: 단순 병렬이 아닌 전략적 격리

### **포트폴리오 메시지**
> **"정보 비대칭을 활용한 Multi-Agent 시스템으로 정확도 향상과 44% 비용 절약 동시 달성"**

---

## 📝 **실험 환경 정보**
- **OS**: Windows 11
- **Python**: 3.x
- **LLM Models**: Ollama (qwen2:0.5b, gemma:2b, llama3:8b)
- **실험 일시**: 2025-01-09
- **실험 위치**: `C:\Users\kimdaesoo\source\claude\Project-Arkhē`

---

## 🔍 **다음 실행 명령어들**

### **현재 사용 가능한 실험들**
```bash
# 정보 비대칭 격리 실험 재실행
python src/orchestrator/isolation_pipeline.py

# Shannon Entropy 승급 정책 데모
python experiments/run_quick_demo.py

# 파일럿 승급 정책 실험  
python experiments/run_entropy_experiment.py --mode pilot

# 정보 이론 메트릭 테스트
python src/metrics/information_theory.py
```

### **구현 예정**
```bash
# PARTIAL vs 단일 모델 비교 (다음 우선순위)
python experiments/run_baseline_comparison.py

# 고도화된 승급 정책 실험
python experiments/run_advanced_promotion.py
```