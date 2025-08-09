# -*- coding: utf-8 -*-
"""
Project Arkhē - Economic Intelligence Agent (Dual Mode)
경제적 지능 에이전트 - 듀얼 모드 지원 (strict/lite/auto)

변경 요약:
- execute(query, llm_factory=None, mode="auto", **kwargs) 시그니처
- ARKHE_EI_MODE 환경변수 지원  
- JS Divergence 듀얼 계산 (scipy 사용 vs 순수 Python 근사)
- 의존성 없이도 동작하는 경량 모드
"""

import os
import math
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import Counter

def _mean(values):
    """numpy 선택적 사용 평균 계산"""
    try:
        import numpy as np
        return float(np.mean(values))
    except Exception:
        return sum(values) / max(1, len(values))

def js_divergence(p, q, mode="auto"):
    """Jensen-Shannon Divergence with dual mode support"""
    mode_env = os.getenv("ARKHE_EI_MODE")
    if mode_env: 
        mode = mode_env.lower()
    
    have_np = have_scipy = False
    try:
        import numpy as np
        have_np = True
    except Exception:
        pass
    
    try:
        from scipy.spatial.distance import jensenshannon as jsd
        have_scipy = True
    except Exception:
        pass

    if mode == "strict":
        if not (have_np and have_scipy):
            raise RuntimeError("Strict mode requires numpy + scipy.")
        return float(jsd(p, q, base=2.0) ** 2)

    if (mode == "lite") or (mode == "auto" and not (have_np and have_scipy)):
        # 파이썬 근사: 대칭 KL
        def _norm(v):
            s = sum(v) or 1.0
            return [max(1e-12, x/s) for x in v]
        
        P, Q = _norm(p), _norm(q)
        M = [(x+y)/2.0 for x,y in zip(P,Q)]
        
        def _kl(A, B):
            return sum(a*math.log2(a/b) for a,b in zip(A,B) if a > 0 and b > 0)
        
        return 0.5*_kl(P, M) + 0.5*_kl(Q, M)

    # auto + (have_np and have_scipy)
    return float(jsd(p, q, base=2.0) ** 2)

@dataclass
class StageMetrics:
    """단계별 실측 메트릭"""
    stage_name: str
    model: str
    start_time: float
    end_time: float
    tokens_used: int
    cost_factor: float
    real_cost: float
    first_token_latency: float  # ms
    total_latency: float        # ms
    tokens_per_second: float
    success: bool
    error: Optional[str] = None

@dataclass 
class SampleResult:
    """샘플링 결과"""
    text: str
    tokens: int
    confidence: float
    embedding: Optional[List[float]] = None

@dataclass
class EconomicIntelligenceResult:
    """경제적 지능 최종 결과"""
    query: str
    final_answer: str
    total_stages: int
    executed_stages: int
    stage_metrics: List[StageMetrics]
    entropy_progression: List[float]
    promotion_decisions: List[bool]
    total_cost: float
    total_time: float
    economic_efficiency: float  # (Utility - λ·Cost) / Latency
    cost_saved_ratio: float     # vs naive execution

class InformationTheory:
    """정보 이론 계산"""
    
    @staticmethod
    def shannon_entropy(samples: List[str]) -> float:
        """Shannon Entropy 계산"""
        if not samples:
            return 0.0
        
        # 텍스트 유사성 기반 클러스터링 (간단한 근사)
        word_counts = []
        for sample in samples:
            words = sample.lower().split()
            word_counts.extend(words)
        
        if not word_counts:
            return 0.0
            
        counter = Counter(word_counts)
        total = sum(counter.values())
        
        entropy = 0.0
        for count in counter.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    @staticmethod
    def js_divergence_from_samples(samples1: List[str], samples2: List[str], mode="auto") -> float:
        """샘플에서 JS Divergence 계산"""
        if not samples1 or not samples2:
            return 1.0
        
        # 단어 분포로 근사
        def get_word_dist(samples):
            all_words = []
            for sample in samples:
                all_words.extend(sample.lower().split())
            counter = Counter(all_words)
            total = sum(counter.values())
            return {word: count/total for word, count in counter.items()}
        
        dist1 = get_word_dist(samples1)
        dist2 = get_word_dist(samples2)
        
        # 공통 단어 집합
        all_words = set(dist1.keys()) | set(dist2.keys())
        if not all_words:
            return 1.0
            
        p = [dist1.get(word, 0) for word in all_words]
        q = [dist2.get(word, 0) for word in all_words]
        
        return js_divergence(p, q, mode)

class PromotionPolicy:
    """승격 정책"""
    
    def __init__(self, entropy_threshold: float = 2.0, 
                 confidence_threshold: float = 0.6,
                 cost_sensitivity: float = 0.3):
        self.entropy_threshold = entropy_threshold
        self.confidence_threshold = confidence_threshold
        self.cost_sensitivity = cost_sensitivity
    
    def should_promote(self, samples: List[SampleResult], 
                      current_stage_cost: float,
                      next_stage_cost: float) -> Tuple[bool, Dict[str, float]]:
        """승급 여부 결정"""
        if not samples:
            return True, {"reason": "no_samples", "entropy": 0, "confidence": 0}
        
        texts = [s.text for s in samples]
        entropy = InformationTheory.shannon_entropy(texts)
        avg_confidence = _mean([s.confidence for s in samples])
        
        # 경제적 분석
        cost_ratio = next_stage_cost / current_stage_cost
        expected_utility_gain = entropy * (1 - avg_confidence)
        cost_adjusted_gain = expected_utility_gain - (self.cost_sensitivity * cost_ratio)
        
        should_promote = (
            entropy > self.entropy_threshold or
            avg_confidence < self.confidence_threshold or
            cost_adjusted_gain > 0.5
        )
        
        return should_promote, {
            "entropy": entropy,
            "confidence": avg_confidence,
            "cost_ratio": cost_ratio,
            "utility_gain": expected_utility_gain,
            "cost_adjusted_gain": cost_adjusted_gain
        }

class EconomicIntelligenceAgent:
    """경제적 지능 에이전트 - 듀얼 모드 지원"""
    
    def __init__(self, cost_sensitivity: float = 0.3, utility_weight: float = 0.7):
        self.cost_sensitivity = cost_sensitivity
        self.utility_weight = utility_weight
        self.promotion_policy = PromotionPolicy(cost_sensitivity=cost_sensitivity)
        
        # 모델별 비용 계수
        self.cost_factors = {
            "qwen2:0.5b": 0.8,
            "gemma:2b": 1.0, 
            "llama3:8b": 4.0
        }
        
    def _sample_stage(self, llm, prompt: str, model: str, 
                     n_samples: int = 3, temperature: float = 0.7) -> Tuple[List[SampleResult], StageMetrics]:
        """단계별 다중 샘플링"""
        start_time = time.time()
        samples = []
        total_tokens = 0
        
        for i in range(n_samples):
            try:
                sample_start = time.time()
                response = llm.generate(prompt, temperature=temperature, max_tokens=300)
                sample_end = time.time()
                
                if isinstance(response, dict):
                    text = response.get('response', str(response))
                    tokens = response.get('tokens', len(text.split()))
                else:
                    text = str(response)
                    tokens = len(text.split())
                
                first_token_lat = (sample_end - sample_start) * 1000  # ms
                confidence = 1.0 / (1.0 + temperature)  # 온도 기반 신뢰도 근사
                
                samples.append(SampleResult(
                    text=text,
                    tokens=tokens, 
                    confidence=confidence
                ))
                total_tokens += tokens
                
            except Exception as e:
                print(f"  Sample {i+1} failed: {e}")
                continue
        
        end_time = time.time()
        total_latency = (end_time - start_time) * 1000
        
        metrics = StageMetrics(
            stage_name=f"{model}_sampling",
            model=model,
            start_time=start_time,
            end_time=end_time,
            tokens_used=total_tokens,
            cost_factor=self.cost_factors.get(model, 1.0),
            real_cost=total_tokens * self.cost_factors.get(model, 1.0) / 1000,
            first_token_latency=total_latency / max(n_samples, 1),
            total_latency=total_latency,
            tokens_per_second=total_tokens / max((end_time - start_time), 0.001),
            success=len(samples) > 0
        )
        
        return samples, metrics
    
    def execute(self, query: str, llm_factory=None, mode="auto", **kwargs) -> EconomicIntelligenceResult:
        """경제적 지능 파이프라인 실행"""
        
        # 모드 설정 (환경변수 우선)
        mode_env = os.getenv("ARKHE_EI_MODE")
        if mode_env:
            mode = mode_env.lower()
        
        # LLM Factory 기본값 설정
        if llm_factory is None:
            from src.llm.simple_llm import create_llm_auto
            llm_factory = create_llm_auto
            
        print(f"\n=== Economic Intelligence Agent: {query[:50]}... ===")
        print(f"Mode: {mode.upper()}")
        
        # 의존성 상태 체크
        have_numpy = have_scipy = False
        try:
            import numpy as np
            have_numpy = True
        except ImportError:
            pass
        try:
            from scipy.spatial.distance import jensenshannon
            have_scipy = True
        except ImportError:
            pass
        
        deps = []
        if have_numpy: deps.append("NumPy")
        if have_scipy: deps.append("SciPy")
        print(f"Dependencies: {', '.join(deps) if deps else 'None (Lite mode)'}")
        
        # Strict 모드 검증
        if mode == "strict" and not (have_numpy and have_scipy):
            raise RuntimeError("Strict mode requires numpy + scipy, but they are not available")
        
        stage_metrics = []
        entropy_progression = []
        promotion_decisions = []
        all_samples = []
        
        # Stage 1: Draft (qwen2:0.5b) - 다중 샘플링
        print("Stage 1 (Draft): qwen2:0.5b - Multi-sampling...")
        llm_draft = llm_factory("qwen2:0.5b")
        draft_prompt = f"Answer concisely: {query}"
        
        draft_samples, draft_metrics = self._sample_stage(
            llm_draft, draft_prompt, "qwen2:0.5b", n_samples=5, temperature=0.8
        )
        stage_metrics.append(draft_metrics)
        all_samples.extend(draft_samples)
        
        if not draft_samples:
            return self._create_error_result(query, "Draft stage failed", stage_metrics)
        
        draft_entropy = InformationTheory.shannon_entropy([s.text for s in draft_samples])
        entropy_progression.append(draft_entropy)
        print(f"  Draft entropy: {draft_entropy:.3f}, samples: {len(draft_samples)}")
        
        # Promotion Decision 1: Draft -> Review
        should_promote_1, promo_1_info = self.promotion_policy.should_promote(
            draft_samples, self.cost_factors["qwen2:0.5b"], self.cost_factors["gemma:2b"]
        )
        promotion_decisions.append(should_promote_1)
        
        print(f"  Promotion 1: {should_promote_1} (entropy: {promo_1_info['entropy']:.3f}, confidence: {promo_1_info['confidence']:.3f})")
        
        if not should_promote_1:
            # 승급 없음 - Draft 결과로 종료
            best_draft = max(draft_samples, key=lambda x: x.confidence)
            return self._create_result(query, best_draft.text, 1, 1, stage_metrics, 
                                     entropy_progression, promotion_decisions, all_samples)
        
        # Stage 2: Review (gemma:2b) - 승급됨
        print("Stage 2 (Review): gemma:2b - Improving drafts...")
        llm_review = llm_factory("gemma:2b")
        best_drafts = sorted(draft_samples, key=lambda x: x.confidence, reverse=True)[:3]
        review_prompt = f"Improve these draft answers for: {query}\\n\\nDrafts:\\n" + \\\n                       "\\n".join([f"- {d.text}" for d in best_drafts]) + "\\n\\nImproved answer:"
        
        review_samples, review_metrics = self._sample_stage(
            llm_review, review_prompt, "gemma:2b", n_samples=3, temperature=0.5
        )
        stage_metrics.append(review_metrics)
        all_samples.extend(review_samples)
        
        if not review_samples:
            # Review 실패 - Draft 결과 사용
            best_draft = max(draft_samples, key=lambda x: x.confidence)
            return self._create_result(query, best_draft.text, 2, 1, stage_metrics,
                                     entropy_progression, promotion_decisions, all_samples)
        
        review_entropy = InformationTheory.shannon_entropy([s.text for s in review_samples])
        entropy_progression.append(review_entropy)
        print(f"  Review entropy: {review_entropy:.3f}, samples: {len(review_samples)}")
        
        # Promotion Decision 2: Review -> Judge
        should_promote_2, promo_2_info = self.promotion_policy.should_promote(
            review_samples, self.cost_factors["gemma:2b"], self.cost_factors["llama3:8b"]
        )
        promotion_decisions.append(should_promote_2)
        
        print(f"  Promotion 2: {should_promote_2} (entropy: {promo_2_info['entropy']:.3f}, cost_gain: {promo_2_info['cost_adjusted_gain']:.3f})")
        
        if not should_promote_2:
            # 승급 없음 - Review 결과로 종료
            best_review = max(review_samples, key=lambda x: x.confidence)
            return self._create_result(query, best_review.text, 2, 2, stage_metrics,
                                     entropy_progression, promotion_decisions, all_samples)
        
        # Stage 3: Judge (llama3:8b) - 최종 승급
        print("Stage 3 (Judge): llama3:8b - Final judgment...")
        llm_judge = llm_factory("llama3:8b")
        best_reviews = sorted(review_samples, key=lambda x: x.confidence, reverse=True)[:2]
        judge_prompt = f"""Provide the highest quality final answer by analyzing these previous attempts:

Question: {query}

Draft attempts: {[d.text for d in best_drafts[:2]]}

Reviewed attempts: {[r.text for r in best_reviews]}

Final answer:"""
        
        judge_samples, judge_metrics = self._sample_stage(
            llm_judge, judge_prompt, "llama3:8b", n_samples=2, temperature=0.2
        )
        stage_metrics.append(judge_metrics)
        all_samples.extend(judge_samples)
        
        if not judge_samples:
            # Judge 실패 - Review 결과 사용  
            best_review = max(review_samples, key=lambda x: x.confidence)
            return self._create_result(query, best_review.text, 3, 2, stage_metrics,
                                     entropy_progression, promotion_decisions, all_samples)
        
        judge_entropy = InformationTheory.shannon_entropy([s.text for s in judge_samples])
        entropy_progression.append(judge_entropy)
        print(f"  Judge entropy: {judge_entropy:.3f}, samples: {len(judge_samples)}")
        
        # 최종 결과 - 최고 신뢰도 답변 선택
        best_judge = max(judge_samples, key=lambda x: x.confidence)
        return self._create_result(query, best_judge.text, 3, 3, stage_metrics,
                                 entropy_progression, promotion_decisions, all_samples)
    
    def _create_result(self, query: str, final_answer: str, total_stages: int, 
                      executed_stages: int, stage_metrics: List[StageMetrics],
                      entropy_progression: List[float], promotion_decisions: List[bool],
                      all_samples: List[SampleResult]) -> EconomicIntelligenceResult:
        """결과 객체 생성"""
        total_cost = sum(m.real_cost for m in stage_metrics)
        total_time = sum(m.total_latency for m in stage_metrics)
        
        # 경제적 효율성 계산
        utility = len(final_answer) / 100  # 답변 품질 근사 (길이 기반)
        economic_efficiency = (utility - self.cost_sensitivity * total_cost) / max(total_time/1000, 0.001)
        
        # 비용 절약 비율 (vs 무조건 3단계)
        naive_cost = sum(self.cost_factors.values())  # 0.8 + 1.0 + 4.0
        cost_saved_ratio = 1 - (total_cost / naive_cost) if naive_cost > 0 else 0
        
        return EconomicIntelligenceResult(
            query=query,
            final_answer=final_answer,
            total_stages=total_stages,
            executed_stages=executed_stages,
            stage_metrics=stage_metrics,
            entropy_progression=entropy_progression,
            promotion_decisions=promotion_decisions,
            total_cost=total_cost,
            total_time=total_time,
            economic_efficiency=economic_efficiency,
            cost_saved_ratio=cost_saved_ratio
        )
    
    def _create_error_result(self, query: str, error: str, 
                           stage_metrics: List[StageMetrics]) -> EconomicIntelligenceResult:
        """에러 결과 생성"""
        return EconomicIntelligenceResult(
            query=query,
            final_answer=f"ERROR: {error}",
            total_stages=3,
            executed_stages=0,
            stage_metrics=stage_metrics,
            entropy_progression=[],
            promotion_decisions=[],
            total_cost=0,
            total_time=0,
            economic_efficiency=0,
            cost_saved_ratio=0
        )

# 하위 호환성을 위한 별칭
EconomicIntelligencePipeline = EconomicIntelligenceAgent