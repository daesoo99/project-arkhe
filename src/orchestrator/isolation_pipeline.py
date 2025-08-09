# -*- coding: utf-8 -*-
"""
Project Arkhē - Information Asymmetry Pipeline
정보 비대칭 기반 Multi-Agent Orchestration

핵심 연구: 에이전트간 정보 격리 수준이 성능에 미치는 영향
"""

import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from datetime import datetime

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from metrics.information_theory import InformationTheoryCalculator

class IsolationLevel(Enum):
    """정보 격리 수준"""
    NONE = "none"           # 모든 이전 결과 공유 (기존 방식)
    PARTIAL = "partial"     # 일부만 공유 (직전 단계만)
    COMPLETE = "complete"   # 완전 격리 (독립 실행)

@dataclass
class IsolationExperimentResult:
    """격리 실험 결과"""
    isolation_level: IsolationLevel
    question_id: str
    query: str
    expected_answer: str
    
    # 각 단계별 결과
    draft_samples: List[str]
    review_samples: List[str] 
    judge_samples: List[str]
    
    # 최종 집계 결과
    final_answer: str
    consensus_score: float  # 답변 간 일치도
    
    # 정보 이론 메트릭
    draft_entropy: float
    review_entropy: float
    judge_entropy: float
    cross_stage_diversity: float  # 단계 간 다양성
    
    # 비용 및 성능
    total_time_ms: int
    total_tokens: int
    accuracy_score: float  # 정답과의 일치도
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "isolation_level": self.isolation_level.value,
            "question_id": self.question_id,
            "query": self.query,
            "expected_answer": self.expected_answer,
            "draft_samples": self.draft_samples,
            "review_samples": self.review_samples,
            "judge_samples": self.judge_samples,
            "final_answer": self.final_answer,
            "consensus_score": float(self.consensus_score),
            "draft_entropy": float(self.draft_entropy),
            "review_entropy": float(self.review_entropy), 
            "judge_entropy": float(self.judge_entropy),
            "cross_stage_diversity": float(self.cross_stage_diversity),
            "total_time_ms": self.total_time_ms,
            "total_tokens": self.total_tokens,
            "accuracy_score": float(self.accuracy_score)
        }

class IsolationPipeline:
    """정보 비대칭 기반 Multi-Agent 파이프라인"""
    
    def __init__(self, isolation_level: IsolationLevel, k_samples: int = 3):
        self.isolation_level = isolation_level
        self.k_samples = k_samples
        self.calculator = InformationTheoryCalculator()
        self.context = {}
        self.step_results = []
        
    def run_experiment(self, query: str, expected_answer: str, 
                      question_id: str, llm_factory) -> IsolationExperimentResult:
        """격리 실험 실행"""
        
        print(f"\n*** ISOLATION EXPERIMENT: {self.isolation_level.value.upper()} ***")
        print(f"Query: {query}")
        
        start_time = time.time()
        self.context = {"query": query}
        self.step_results = []
        
        # 각 단계별 실행
        draft_samples = self._run_draft_stage(llm_factory)
        review_samples = self._run_review_stage(llm_factory, draft_samples)  
        judge_samples = self._run_judge_stage(llm_factory, draft_samples, review_samples)
        
        # 최종 답변 결정 및 메트릭 계산
        final_answer = self._determine_final_answer(draft_samples, review_samples, judge_samples)
        consensus_score = self._calculate_consensus_score(draft_samples, review_samples, judge_samples)
        accuracy_score = self._calculate_accuracy_score(final_answer, expected_answer)
        
        # 정보 이론 메트릭
        draft_entropy = self.calculator.shannon_entropy(draft_samples)
        review_entropy = self.calculator.shannon_entropy(review_samples) 
        judge_entropy = self.calculator.shannon_entropy(judge_samples)
        cross_stage_diversity = self._calculate_cross_stage_diversity(
            draft_samples, review_samples, judge_samples
        )
        
        total_time = int((time.time() - start_time) * 1000)
        
        # 정확한 토큰 계산
        try:
            import tiktoken
            encoder = tiktoken.encoding_for_model("gpt-4")
            
            # 각 단계별 입력/출력 토큰 계산
            total_tokens = 0
            
            # Draft 단계
            draft_input = f"Answer this question concisely: {query}"
            total_tokens += len(encoder.encode(draft_input)) * 3  # 3개 샘플
            for sample in draft_samples:
                total_tokens += len(encoder.encode(sample))
            
            # Review 단계  
            review_input = f"Improve this draft answer: {query} Draft answers: {' | '.join(draft_samples)}"
            total_tokens += len(encoder.encode(review_input)) * 2  # 2개 샘플
            for sample in review_samples:
                total_tokens += len(encoder.encode(sample))
            
            # Judge 단계
            judge_input = f"Provide the final, highest quality answer: {query} Draft answers: {' | '.join(draft_samples)} Review answers: {' | '.join(review_samples)}"
            total_tokens += len(encoder.encode(judge_input)) * 1  # 1개 샘플
            for sample in judge_samples:
                total_tokens += len(encoder.encode(sample))
                
        except ImportError:
            # 폴백: 단어 수 기반
            total_tokens = sum(len(sample.split()) for samples in [draft_samples, review_samples, judge_samples] 
                              for sample in samples)
        
        result = IsolationExperimentResult(
            isolation_level=self.isolation_level,
            question_id=question_id,
            query=query,
            expected_answer=expected_answer,
            draft_samples=draft_samples,
            review_samples=review_samples,
            judge_samples=judge_samples,
            final_answer=final_answer,
            consensus_score=consensus_score,
            draft_entropy=draft_entropy,
            review_entropy=review_entropy,
            judge_entropy=judge_entropy,
            cross_stage_diversity=cross_stage_diversity,
            total_time_ms=total_time,
            total_tokens=total_tokens,
            accuracy_score=accuracy_score
        )
        
        print(f"  Draft Entropy: {draft_entropy:.3f}")
        print(f"  Review Entropy: {review_entropy:.3f}")
        print(f"  Judge Entropy: {judge_entropy:.3f}")
        print(f"  Cross-stage Diversity: {cross_stage_diversity:.3f}")
        print(f"  Consensus: {consensus_score:.3f}")
        print(f"  Accuracy: {accuracy_score:.3f}")
        print(f"  Time: {total_time}ms")
        
        return result
    
    def _run_draft_stage(self, llm_factory) -> List[str]:
        """초안 단계 실행 (항상 독립)"""
        print(f"  [1/3] Draft stage - 3 samples")
        
        llm = llm_factory("qwen2:0.5b")
        prompt = f"Answer this question concisely: {self.context['query']}"
        
        samples = []
        for i in range(3):
            response = llm.generate(prompt, temperature=0.8)
            text = response.get("response", str(response)) if isinstance(response, dict) else str(response)
            samples.append(text)
        
        return samples
    
    def _run_review_stage(self, llm_factory, draft_samples: List[str]) -> List[str]:
        """검토 단계 실행"""
        print(f"  [2/3] Review stage - 2 samples - {self.isolation_level.value}")
        
        llm = llm_factory("qwen2:0.5b")
        samples = []
        
        for i in range(2):
            if self.isolation_level == IsolationLevel.NONE:
                # 모든 draft 정보 공유
                prompt = f"""Improve this draft answer:

Question: {self.context['query']}
Draft answers: {' | '.join(draft_samples)}

Improved answer:"""
            elif self.isolation_level == IsolationLevel.PARTIAL:  
                # 하나의 draft만 참조
                draft_ref = draft_samples[i % len(draft_samples)]
                prompt = f"""Improve this draft answer:

Question: {self.context['query']}
Draft: {draft_ref}

Improved answer:"""
            else:  # COMPLETE
                # 완전 독립 - draft 정보 차단
                prompt = f"Answer this question with high quality: {self.context['query']}"
            
            response = llm.generate(prompt, temperature=0.6)
            text = response.get("response", str(response)) if isinstance(response, dict) else str(response)
            samples.append(text)
        
        return samples
    
    def _run_judge_stage(self, llm_factory, draft_samples: List[str], 
                        review_samples: List[str]) -> List[str]:
        """판정 단계 실행"""
        print(f"  [3/3] Judge stage - 1 sample - {self.isolation_level.value}")
        
        llm = llm_factory("llama3:8b")
        samples = []
        
        for i in range(1):
            if self.isolation_level == IsolationLevel.NONE:
                # 모든 이전 정보 공유
                prompt = f"""Provide the final, highest quality answer:

Question: {self.context['query']}
Draft answers: {' | '.join(draft_samples)}
Review answers: {' | '.join(review_samples)}

Final answer:"""
            elif self.isolation_level == IsolationLevel.PARTIAL:
                # 직전 단계(review)만 참조
                review_ref = review_samples[i % len(review_samples)]
                prompt = f"""Provide the final, highest quality answer:

Question: {self.context['query']}
Previous answer: {review_ref}

Final answer:"""
            else:  # COMPLETE
                # 완전 독립
                prompt = f"Provide the definitive answer to this question: {self.context['query']}"
            
            response = llm.generate(prompt, temperature=0.4)
            text = response.get("response", str(response)) if isinstance(response, dict) else str(response)
            samples.append(text)
        
        return samples
    
    def _determine_final_answer(self, draft_samples: List[str], 
                              review_samples: List[str], 
                              judge_samples: List[str]) -> str:
        """최종 답변 결정"""
        # 가장 긴 judge 답변을 선택 (품질 휴리스틱)
        if judge_samples:
            return max(judge_samples, key=len)
        elif review_samples:
            return max(review_samples, key=len)
        elif draft_samples:
            return max(draft_samples, key=len)
        return "No answer generated"
    
    def _calculate_consensus_score(self, draft_samples: List[str], 
                                  review_samples: List[str], 
                                  judge_samples: List[str]) -> float:
        """답변 간 일치도 계산"""
        all_samples = draft_samples + review_samples + judge_samples
        if len(all_samples) < 2:
            return 1.0
        
        # 단순화: 첫 50자 기준 유사도
        normalized = [s.lower().strip()[:50] for s in all_samples]
        unique_answers = set(normalized)
        
        # 가장 빈번한 답변의 비율
        from collections import Counter
        counter = Counter(normalized)
        most_common_count = counter.most_common(1)[0][1]
        consensus = most_common_count / len(all_samples)
        
        return consensus
    
    def _calculate_accuracy_score(self, final_answer: str, expected_answer: str) -> float:
        """정답과의 일치도 계산 (단순 포함 관계)"""
        if not expected_answer or not final_answer:
            return 0.0
        
        final_lower = final_answer.lower()
        expected_lower = expected_answer.lower()
        
        # 완전 일치
        if expected_lower in final_lower:
            return 1.0
        
        # 부분 일치 (단어 단위)
        expected_words = set(expected_lower.split())
        final_words = set(final_lower.split())
        
        if expected_words and expected_words.intersection(final_words):
            return len(expected_words.intersection(final_words)) / len(expected_words)
        
        return 0.0
    
    def _calculate_cross_stage_diversity(self, draft_samples: List[str], 
                                       review_samples: List[str], 
                                       judge_samples: List[str]) -> float:
        """단계 간 다양성 측정"""
        if not all([draft_samples, review_samples, judge_samples]):
            return 0.0
        
        # 각 단계의 대표 답변
        draft_rep = draft_samples[0] if draft_samples else ""
        review_rep = review_samples[0] if review_samples else ""
        judge_rep = judge_samples[0] if judge_samples else ""
        
        # 단계 간 JS divergence 평균
        js1 = self.calculator.jensen_shannon_divergence([draft_rep], [review_rep])
        js2 = self.calculator.jensen_shannon_divergence([review_rep], [judge_rep]) 
        js3 = self.calculator.jensen_shannon_divergence([draft_rep], [judge_rep])
        
        return (js1 + js2 + js3) / 3.0

def create_isolation_experiment_suite():
    """표준 벤치마크 기반 격리 실험 스위트"""
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent / "datasets"))
    
    try:
        from standard_benchmarks import StandardBenchmarkLoader
        loader = StandardBenchmarkLoader()
        
        # 각 카테고리에서 2개씩 선택
        questions = loader.get_questions(count=12, categories=["math", "knowledge", "coding"])
        
        # 기존 포맷으로 변환
        test_questions = []
        for q in questions:
            test_questions.append({
                "id": q.id,
                "query": q.query,
                "expected": q.expected_answer
            })
        
        print(f"[+] Loaded {len(test_questions)} standard benchmark questions")
        return test_questions
        
    except ImportError:
        # 폴백: 기존 간단한 질문들
        print("[!] Using fallback questions (standard benchmarks not available)")
        return [
            {"id": "simple_geo", "query": "대한민국의 수도는?", "expected": "서울"},
            {"id": "simple_math", "query": "What is 2 + 2?", "expected": "4"},
            {"id": "medium_prog", "query": "Python에서 리스트를 정렬하는 메서드는?", "expected": "sort"},
            {"id": "medium_net", "query": "HTTP의 기본 포트 번호는?", "expected": "80"},
            {"id": "hard_algo", "query": "시간 복잡도 O(n log n)인 정렬 알고리즘은?", "expected": "merge sort"}
        ]

def run_isolation_comparison_experiment(llm_factory):
    """격리 수준 비교 실험 실행"""
    print("=" * 70)
    print("*** INFORMATION ASYMMETRY EXPERIMENT ***")
    print("격리 수준별 Multi-Agent 성능 비교")
    print("=" * 70)
    
    test_questions = create_isolation_experiment_suite()
    isolation_levels = [IsolationLevel.NONE, IsolationLevel.PARTIAL, IsolationLevel.COMPLETE]
    
    all_results = []
    
    for isolation_level in isolation_levels:
        print(f"\n[*] Testing {isolation_level.value.upper()} isolation...")
        
        pipeline = IsolationPipeline(isolation_level, k_samples=3)
        level_results = []
        
        for question in test_questions:
            try:
                result = pipeline.run_experiment(
                    query=question["query"],
                    expected_answer=question["expected"], 
                    question_id=question["id"],
                    llm_factory=llm_factory
                )
                level_results.append(result)
                all_results.append(result)
                
            except Exception as e:
                print(f"  ERROR on {question['id']}: {e}")
        
        # 수준별 요약
        if level_results:
            avg_accuracy = sum(r.accuracy_score for r in level_results) / len(level_results)
            avg_entropy = sum(r.judge_entropy for r in level_results) / len(level_results) 
            avg_consensus = sum(r.consensus_score for r in level_results) / len(level_results)
            avg_time = sum(r.total_time_ms for r in level_results) / len(level_results)
            
            print(f"\n  [+] {isolation_level.value.upper()} Summary:")
            print(f"    Accuracy: {avg_accuracy:.3f}")
            print(f"    Judge Entropy: {avg_entropy:.3f}")
            print(f"    Consensus: {avg_consensus:.3f}")
            print(f"    Avg Time: {avg_time:.0f}ms")
    
    # 결과 저장
    results_file = f"experiments/results/isolation_experiment_{int(time.time())}.json"
    Path(results_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump([r.to_dict() for r in all_results], f, ensure_ascii=False, indent=2)
    
    print(f"\n[+] Results saved: {results_file}")
    
    # 핵심 비교 분석
    print("\n" + "=" * 70)
    print("*** COMPARATIVE ANALYSIS ***")
    
    by_isolation = {}
    for result in all_results:
        level = result.isolation_level.value
        if level not in by_isolation:
            by_isolation[level] = []
        by_isolation[level].append(result)
    
    for level, results in by_isolation.items():
        if results:
            avg_accuracy = sum(r.accuracy_score for r in results) / len(results)
            avg_diversity = sum(r.cross_stage_diversity for r in results) / len(results)
            avg_cost = sum(r.total_tokens for r in results) / len(results)
            
            print(f"{level.upper():>10}: Accuracy={avg_accuracy:.3f}, "
                  f"Diversity={avg_diversity:.3f}, Cost={avg_cost:.0f} tokens")
    
    return all_results, results_file

if __name__ == "__main__":
    from llm.simple_llm import create_llm_auto
    try:
        run_isolation_comparison_experiment(create_llm_auto)
    except KeyboardInterrupt:
        print("\n[!] Experiment interrupted by user")
    except Exception as e:
        print(f"\n[!] Error: {e}")
        import traceback
        traceback.print_exc()