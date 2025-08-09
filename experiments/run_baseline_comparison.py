#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project Arkhē - Baseline Comparison Experiment
PARTIAL Multi-Agent vs Single Model 성능 비교

핵심 질문: "Multi-Agent PARTIAL이 단일 고급 모델보다 실제로 나은가?"
"""

import sys
import os
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from orchestrator.isolation_pipeline import IsolationPipeline, IsolationLevel
from llm.simple_llm import create_llm_auto
from metrics.information_theory import InformationTheoryCalculator

@dataclass
class BaselineResult:
    """베이스라인 비교 결과"""
    method_name: str
    question_id: str
    query: str
    expected_answer: str
    
    final_answer: str
    accuracy_score: float
    total_tokens: int
    total_time_ms: int
    
    # Multi-Agent 전용
    entropy: float = 0.0
    diversity: float = 0.0
    steps_executed: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_name": self.method_name,
            "question_id": self.question_id,
            "query": self.query,
            "expected_answer": self.expected_answer,
            "final_answer": self.final_answer,
            "accuracy_score": float(self.accuracy_score),
            "total_tokens": self.total_tokens,
            "total_time_ms": self.total_time_ms,
            "entropy": float(self.entropy),
            "diversity": float(self.diversity),
            "steps_executed": self.steps_executed
        }

class SingleModelRunner:
    """단일 모델 실행기"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.calculator = InformationTheoryCalculator()
    
    def run_single_model(self, query: str, expected_answer: str, 
                        question_id: str, llm_factory) -> BaselineResult:
        """단일 모델 실행"""
        
        print(f"  Testing {self.model_name}: {question_id}")
        
        start_time = time.time()
        llm = llm_factory(self.model_name)
        
        # 고품질 프롬프트 (Multi-Agent와 공정 비교를 위해)
        prompt = f"""Answer this question with high quality and accuracy:

Question: {query}

Provide a clear, concise, and accurate answer:"""
        
        try:
            response = llm.generate(prompt, temperature=0.4, max_tokens=512)
            
            if isinstance(response, dict):
                final_answer = response.get("response", str(response))
            else:
                final_answer = str(response)
            
            # 정확한 토큰 계산
            try:
                import tiktoken
                encoder = tiktoken.encoding_for_model("gpt-4")
                tokens = len(encoder.encode(prompt)) + len(encoder.encode(final_answer))
            except ImportError:
                tokens = len(prompt.split()) + len(final_answer.split())  # 폴백
            
            total_time = int((time.time() - start_time) * 1000)
            
            # 정확도 계산
            accuracy = self._calculate_accuracy(final_answer, expected_answer)
            
            return BaselineResult(
                method_name=f"Single-{self.model_name}",
                question_id=question_id,
                query=query,
                expected_answer=expected_answer,
                final_answer=final_answer,
                accuracy_score=accuracy,
                total_tokens=tokens,
                total_time_ms=total_time,
                steps_executed=1  # 단일 단계
            )
            
        except Exception as e:
            print(f"    ERROR: {e}")
            return BaselineResult(
                method_name=f"Single-{self.model_name}",
                question_id=question_id,
                query=query,
                expected_answer=expected_answer,
                final_answer=f"ERROR: {e}",
                accuracy_score=0.0,
                total_tokens=0,
                total_time_ms=0
            )
    
    def _calculate_accuracy(self, final_answer: str, expected_answer: str) -> float:
        """정답과의 일치도 계산"""
        if not expected_answer or not final_answer:
            return 0.0
        
        final_lower = final_answer.lower()
        expected_lower = expected_answer.lower()
        
        # 완전 포함
        if expected_lower in final_lower:
            return 1.0
        
        # 단어별 부분 일치
        expected_words = set(expected_lower.split())
        final_words = set(final_lower.split())
        
        if expected_words and expected_words.intersection(final_words):
            return len(expected_words.intersection(final_words)) / len(expected_words)
        
        return 0.0

class MultiAgentRunner:
    """Multi-Agent NONE 실행기"""
    
    def __init__(self):
        self.pipeline = IsolationPipeline(IsolationLevel.NONE, k_samples=3)
    
    def run_multi_agent(self, query: str, expected_answer: str, 
                       question_id: str, llm_factory) -> BaselineResult:
        """Multi-Agent NONE 실행"""
        
        print(f"  Testing Multi-Agent NONE: {question_id}")
        
        try:
            isolation_result = self.pipeline.run_experiment(
                query=query,
                expected_answer=expected_answer,
                question_id=question_id,
                llm_factory=llm_factory
            )
            
            return BaselineResult(
                method_name="Multi-Agent-NONE",
                question_id=question_id,
                query=query,
                expected_answer=expected_answer,
                final_answer=isolation_result.final_answer,
                accuracy_score=isolation_result.accuracy_score,
                total_tokens=isolation_result.total_tokens,
                total_time_ms=isolation_result.total_time_ms,
                entropy=isolation_result.judge_entropy,
                diversity=isolation_result.cross_stage_diversity,
                steps_executed=3  # Draft + Review + Judge
            )
            
        except Exception as e:
            print(f"    ERROR: {e}")
            return BaselineResult(
                method_name="Multi-Agent-NONE",
                question_id=question_id,
                query=query,
                expected_answer=expected_answer,
                final_answer=f"ERROR: {e}",
                accuracy_score=0.0,
                total_tokens=0,
                total_time_ms=0
            )

def load_test_questions() -> List[Dict[str, str]]:
    """표준 벤치마크 질문 로드"""
    sys.path.append(str(Path(__file__).parent.parent / "datasets"))
    from standard_benchmarks import StandardBenchmarkLoader
    
    loader = StandardBenchmarkLoader()
    
    # 균등한 혼합: 수학 + 지식 + 코딩
    benchmark_questions = loader.get_questions(
        count=15,
        categories=["math", "knowledge", "coding"],
        difficulties=["easy", "medium"]
    )
    
    # 기존 포맷으로 변환
    questions = []
    for q in benchmark_questions:
        questions.append({
            "id": q.id,
            "query": q.query,
            "expected_answer": q.expected_answer
        })
    
    return questions

def run_baseline_comparison():
    """베이스라인 비교 실험 실행"""
    
    print("=" * 70)
    print("*** BASELINE COMPARISON EXPERIMENT ***")
    print("NONE Multi-Agent vs High-Performance Single Models")
    print("=" * 70)
    
    # 테스트 질문 로드
    test_questions = load_test_questions()
    
    # 실행기들 초기화 (고성능 모델들 추가)
    runners = [
        ("Multi-Agent-NONE", MultiAgentRunner()),
        ("Single-llama3:8b", SingleModelRunner("llama3:8b")),
        ("Single-claude-3-haiku", SingleModelRunner("claude-3-haiku")),
        ("Single-gpt-4o-mini", SingleModelRunner("gpt-4o-mini"))
    ]
    
    all_results = []
    
    for method_name, runner in runners:
        print(f"\n[*] Testing {method_name}...")
        
        method_results = []
        
        for question in test_questions:
            try:
                if isinstance(runner, MultiAgentRunner):
                    result = runner.run_multi_agent(
                        query=question["query"],
                        expected_answer=question["expected_answer"],
                        question_id=question["id"],
                        llm_factory=create_llm_auto
                    )
                else:
                    result = runner.run_single_model(
                        query=question["query"],
                        expected_answer=question["expected_answer"],
                        question_id=question["id"],
                        llm_factory=create_llm_auto
                    )
                
                method_results.append(result)
                all_results.append(result)
                
                print(f"    → Accuracy: {result.accuracy_score:.3f}, "
                      f"Tokens: {result.total_tokens}, Time: {result.total_time_ms}ms")
                
            except Exception as e:
                print(f"    → ERROR: {e}")
        
        # 메서드별 요약
        if method_results:
            avg_accuracy = sum(r.accuracy_score for r in method_results) / len(method_results)
            avg_tokens = sum(r.total_tokens for r in method_results) / len(method_results)
            avg_time = sum(r.total_time_ms for r in method_results) / len(method_results)
            
            print(f"  [+] {method_name} Summary:")
            print(f"    Avg Accuracy: {avg_accuracy:.3f}")
            print(f"    Avg Tokens: {avg_tokens:.0f}")
            print(f"    Avg Time: {avg_time:.0f}ms")
    
    # 결과 저장
    results_file = f"experiments/results/baseline_comparison_{int(time.time())}.json"
    Path(results_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump([r.to_dict() for r in all_results], f, ensure_ascii=False, indent=2)
    
    # 최종 비교 분석
    print("\n" + "=" * 70)
    print("*** COMPARATIVE ANALYSIS ***")
    
    by_method = {}
    for result in all_results:
        method = result.method_name
        if method not in by_method:
            by_method[method] = []
        by_method[method].append(result)
    
    print(f"{'Method':<20} {'Accuracy':<10} {'Tokens':<10} {'Time(ms)':<10} {'Efficiency':<10}")
    print("-" * 70)
    
    for method, results in by_method.items():
        if results:
            avg_accuracy = sum(r.accuracy_score for r in results) / len(results)
            avg_tokens = sum(r.total_tokens for r in results) / len(results)
            avg_time = sum(r.total_time_ms for r in results) / len(results)
            
            # 효율성 = 정확도 / 토큰 수 (높을수록 좋음)
            efficiency = avg_accuracy / (avg_tokens / 100) if avg_tokens > 0 else 0
            
            print(f"{method:<20} {avg_accuracy:<10.3f} {avg_tokens:<10.0f} "
                  f"{avg_time:<10.0f} {efficiency:<10.3f}")
    
    print(f"\n[+] Results saved: {results_file}")
    
    # 핵심 메시지
    multi_agent_results = by_method.get("Multi-Agent-NONE", [])
    single_llama_results = by_method.get("Single-llama3:8b", [])
    
    if multi_agent_results and single_llama_results:
        ma_accuracy = sum(r.accuracy_score for r in multi_agent_results) / len(multi_agent_results)
        ma_tokens = sum(r.total_tokens for r in multi_agent_results) / len(multi_agent_results)
        
        sl_accuracy = sum(r.accuracy_score for r in single_llama_results) / len(single_llama_results)
        sl_tokens = sum(r.total_tokens for r in single_llama_results) / len(single_llama_results)
        
        accuracy_diff = ((ma_accuracy - sl_accuracy) / sl_accuracy) * 100 if sl_accuracy > 0 else 0
        cost_diff = ((ma_tokens - sl_tokens) / sl_tokens) * 100 if sl_tokens > 0 else 0
        
        print("\n" + "🎯" * 20)
        print("*** KEY FINDINGS ***")
        print(f"Multi-Agent vs Single llama3:8b:")
        print(f"  Accuracy: {accuracy_diff:+.1f}%")
        print(f"  Cost: {cost_diff:+.1f}%")
        
        if accuracy_diff > 0 and cost_diff < 0:
            print("  → Multi-Agent WINS: Better accuracy + Lower cost! 🏆")
        elif accuracy_diff > 5:
            print("  → Multi-Agent WINS: Significantly better accuracy! 🏆")
        elif cost_diff < -20:
            print("  → Multi-Agent WINS: Major cost savings! 💰")
        else:
            print("  → Mixed results: Context-dependent optimization needed 🤔")
    
    return all_results, results_file

if __name__ == "__main__":
    try:
        run_baseline_comparison()
    except KeyboardInterrupt:
        print("\n[!] Experiment interrupted by user")
    except Exception as e:
        print(f"\n[!] Error: {e}")
        import traceback
        traceback.print_exc()