#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project Arkhē - 2-Stage Pipeline Comparison Experiment
2단계 파이프라인 (Draft → Judge) 성능 비교

목적: Review 단계 제거가 효율성 개선 없이 품질 유지 가능한지 검증
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

@dataclass
class TwoStageResult:
    """2단계 실험 결과"""
    method_name: str
    problem_id: str
    query: str
    expected_answer: str
    
    final_answer: str
    total_tokens: int
    total_time_ms: int
    accuracy_score: float
    
    # 단계별 정보
    draft_samples: List[str]
    judge_samples: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_name": self.method_name,
            "problem_id": self.problem_id,
            "query": self.query,
            "expected_answer": self.expected_answer,
            "final_answer": self.final_answer,
            "total_tokens": self.total_tokens,
            "total_time_ms": self.total_time_ms,
            "accuracy_score": float(self.accuracy_score),
            "draft_samples": self.draft_samples,
            "judge_samples": self.judge_samples
        }

class TwoStagePipeline:
    """2단계 파이프라인: Draft → Judge"""
    
    def __init__(self):
        pass
    
    def run_experiment(self, query: str, expected_answer: str, 
                      question_id: str, llm_factory) -> TwoStageResult:
        """2단계 실험 실행"""
        
        print(f"\n*** 2-STAGE PIPELINE EXPERIMENT ***")
        print(f"Query: {query[:100]}...")
        
        start_time = time.time()
        
        # Stage 1: Draft Generation (qwen2:0.5b × 3)
        draft_samples = self._run_draft_stage(llm_factory, query)
        
        # Stage 2: Final Judge (llama3:8b × 1) - Skip Review
        judge_samples = self._run_judge_stage(llm_factory, query, draft_samples)
        
        # 최종 답변
        final_answer = judge_samples[0] if judge_samples else "No answer"
        accuracy_score = self._calculate_accuracy_score(final_answer, expected_answer)
        
        # 토큰 계산
        total_tokens = self._calculate_tokens(query, draft_samples, judge_samples)
        total_time = int((time.time() - start_time) * 1000)
        
        return TwoStageResult(
            method_name="2-Stage-Pipeline",
            problem_id=question_id,
            query=query,
            expected_answer=expected_answer,
            final_answer=final_answer,
            total_tokens=total_tokens,
            total_time_ms=total_time,
            accuracy_score=accuracy_score,
            draft_samples=draft_samples,
            judge_samples=judge_samples
        )
    
    def _run_draft_stage(self, llm_factory, query: str) -> List[str]:
        """Draft 단계 실행"""
        print(f"  [1/2] Draft stage - 3 samples (qwen2:0.5b)")
        
        llm = llm_factory("qwen2:0.5b")
        prompt = f"Answer this question concisely: {query}"
        
        samples = []
        for i in range(3):
            response = llm.generate(prompt, temperature=0.8)
            text = response.get("response", str(response)) if isinstance(response, dict) else str(response)
            samples.append(text)
        
        return samples
    
    def _run_judge_stage(self, llm_factory, query: str, draft_samples: List[str]) -> List[str]:
        """Judge 단계 실행 (Review 건너뛰기)"""
        print(f"  [2/2] Judge stage - 1 sample (llama3:8b) - DIRECT FROM DRAFT")
        
        llm = llm_factory("llama3:8b")
        
        # Draft 정보를 직접 Judge에 전달
        prompt = f"""Provide the final, highest quality answer by synthesizing these draft responses:

Question: {query}
Draft answers: {' | '.join(draft_samples)}

Final answer:"""
        
        response = llm.generate(prompt, temperature=0.4)
        text = response.get("response", str(response)) if isinstance(response, dict) else str(response)
        
        return [text]
    
    def _calculate_tokens(self, query: str, draft_samples: List[str], 
                         judge_samples: List[str]) -> int:
        """토큰 계산"""
        try:
            import tiktoken
            encoder = tiktoken.encoding_for_model("gpt-4")
            
            total_tokens = 0
            
            # Draft 단계
            draft_input = f"Answer this question concisely: {query}"
            total_tokens += len(encoder.encode(draft_input)) * 3  # 3개 샘플
            for sample in draft_samples:
                total_tokens += len(encoder.encode(sample))
            
            # Judge 단계 (Review 건너뛰기)
            judge_input = f"Provide the final, highest quality answer by synthesizing these draft responses: Question: {query} Draft answers: {' | '.join(draft_samples)} Final answer:"
            total_tokens += len(encoder.encode(judge_input)) * 1  # 1개 샘플
            for sample in judge_samples:
                total_tokens += len(encoder.encode(sample))
            
            return total_tokens
            
        except ImportError:
            # 폴백: 단어 수 기반
            return sum(len(sample.split()) for samples in [draft_samples, judge_samples] 
                      for sample in samples)
    
    def _calculate_accuracy_score(self, final_answer: str, expected_answer: str) -> float:
        """정답과의 일치도 계산"""
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

class TwoStageComparisonRunner:
    """2단계 파이프라인 비교 실험 러너"""
    
    def __init__(self):
        self.load_test_questions()
    
    def load_test_questions(self):
        """표준 벤치마크 문제 로드"""
        try:
            from datasets.standard_benchmarks import StandardBenchmarkLoader
            loader = StandardBenchmarkLoader()
            questions = loader.get_questions(count=20, categories=["math", "knowledge", "coding"])
            
            self.problems = []
            for q in questions:
                self.problems.append({
                    "id": q.id,
                    "query": q.query,
                    "expected": q.expected_answer
                })
            
            print(f"[+] Loaded {len(self.problems)} standard benchmark questions")
            
        except Exception as e:
            # 폴백: 간단한 질문들
            print(f"[!] Using fallback questions: {e}")
            self.problems = [
                {"id": "simple_geo", "query": "대한민국의 수도는?", "expected": "서울"},
                {"id": "simple_math", "query": "What is 2 + 2?", "expected": "4"},
                {"id": "medium_prog", "query": "Python에서 리스트를 정렬하는 메서드는?", "expected": "sort"},
                {"id": "medium_net", "query": "HTTP의 기본 포트 번호는?", "expected": "80"},
                {"id": "hard_algo", "query": "시간 복잡도 O(n log n)인 정렬 알고리즘은?", "expected": "merge sort"}
            ]
    
    def run_full_comparison(self) -> Dict[str, List[TwoStageResult]]:
        """전체 비교 실험"""
        
        print("=" * 80)
        print("*** 2-STAGE vs 3-STAGE vs SINGLE COMPARISON ***")
        print("Review 단계 제거 효과 검증")
        print("=" * 80)
        
        # 테스트할 방법들
        methods = [
            ("2-Stage-Pipeline", self._run_2stage),
            ("3-Stage-NONE", self._run_3stage_none),
            ("Single-llama3:8b", lambda p: self._run_single_model(p, "llama3:8b"))
        ]
        
        all_results = {}
        
        for method_name, method_func in methods:
            print(f"\n[*] Testing {method_name}...")
            all_results[method_name] = []
            
            for i, problem in enumerate(self.problems, 1):
                print(f"\n  [{i}/{len(self.problems)}] Problem: {problem['id']}")
                print(f"  Query: {problem['query'][:100]}...")
                
                try:
                    result = method_func(problem)
                    all_results[method_name].append(result)
                    
                    print(f"    → Tokens: {result.total_tokens}, Time: {result.total_time_ms}ms")
                    print(f"    → Accuracy: {result.accuracy_score:.3f}")
                    
                except Exception as e:
                    print(f"    → ERROR: {e}")
                
                time.sleep(1)  # 모델 과부하 방지
        
        # 결과 저장
        self._save_results(all_results)
        
        # 최종 분석
        self._print_final_analysis(all_results)
        
        return all_results
    
    def _run_2stage(self, problem: Dict[str, str]) -> TwoStageResult:
        """2단계 파이프라인 실행"""
        pipeline = TwoStagePipeline()
        
        result = pipeline.run_experiment(
            query=problem['query'],
            expected_answer=problem['expected'],
            question_id=problem['id'],
            llm_factory=create_llm_auto
        )
        
        return result
    
    def _run_3stage_none(self, problem: Dict[str, str]) -> TwoStageResult:
        """3단계 파이프라인 (NONE) 실행"""
        pipeline = IsolationPipeline(IsolationLevel.NONE, k_samples=3)
        
        isolation_result = pipeline.run_experiment(
            query=problem['query'],
            expected_answer=problem['expected'],
            question_id=problem['id'],
            llm_factory=create_llm_auto
        )
        
        # TwoStageResult 포맷으로 변환
        return TwoStageResult(
            method_name="3-Stage-NONE",
            problem_id=problem['id'],
            query=problem['query'],
            expected_answer=problem['expected'],
            final_answer=isolation_result.final_answer,
            total_tokens=isolation_result.total_tokens,
            total_time_ms=isolation_result.total_time_ms,
            accuracy_score=isolation_result.accuracy_score,
            draft_samples=isolation_result.draft_samples,
            judge_samples=isolation_result.judge_samples
        )
    
    def _run_single_model(self, problem: Dict[str, str], model_name: str) -> TwoStageResult:
        """단일 모델 실행"""
        start_time = time.time()
        
        llm = create_llm_auto(model_name)
        prompt = f"Answer this question: {problem['query']}"
        
        try:
            response = llm.generate(prompt, temperature=0.3, max_tokens=1024)
            
            if isinstance(response, dict):
                final_answer = response.get("response", str(response))
            else:
                final_answer = str(response)
            
            # 토큰 계산
            try:
                import tiktoken
                encoder = tiktoken.encoding_for_model("gpt-4")
                tokens = len(encoder.encode(prompt)) + len(encoder.encode(final_answer))
            except ImportError:
                tokens = len(prompt.split()) + len(final_answer.split())
            
            total_time = int((time.time() - start_time) * 1000)
            
            # 정확도 계산
            final_lower = final_answer.lower()
            expected_lower = problem['expected'].lower()
            
            if expected_lower in final_lower:
                accuracy = 1.0
            else:
                expected_words = set(expected_lower.split())
                final_words = set(final_lower.split())
                if expected_words:
                    accuracy = len(expected_words.intersection(final_words)) / len(expected_words)
                else:
                    accuracy = 0.0
            
            return TwoStageResult(
                method_name=f"Single-{model_name}",
                problem_id=problem['id'],
                query=problem['query'],
                expected_answer=problem['expected'],
                final_answer=final_answer,
                total_tokens=tokens,
                total_time_ms=total_time,
                accuracy_score=accuracy,
                draft_samples=[],
                judge_samples=[final_answer]
            )
            
        except Exception as e:
            raise e
    
    def _save_results(self, results: Dict[str, List[TwoStageResult]]):
        """결과 저장"""
        results_file = f"experiments/results/2stage_comparison_{int(time.time())}.json"
        Path(results_file).parent.mkdir(parents=True, exist_ok=True)
        
        # JSON 직렬화
        json_results = {}
        for method, result_list in results.items():
            json_results[method] = [r.to_dict() for r in result_list]
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n[+] Results saved: {results_file}")
        return results_file
    
    def _print_final_analysis(self, results: Dict[str, List[TwoStageResult]]):
        """최종 분석 출력"""
        print("\n" + "=" * 80)
        print("*** 2-STAGE PIPELINE ANALYSIS ***")
        print("=" * 80)
        
        # 전체 요약
        print(f"\n{'Method':<20} {'Avg Accuracy':<12} {'Avg Tokens':<11} {'Avg Time(ms)':<12} {'Efficiency':<10}")
        print("-" * 70)
        
        method_summaries = {}
        
        for method, result_list in results.items():
            if not result_list:
                continue
                
            valid_results = [r for r in result_list if r.total_tokens > 0]
            if not valid_results:
                continue
            
            avg_accuracy = sum(r.accuracy_score for r in valid_results) / len(valid_results)
            avg_tokens = sum(r.total_tokens for r in valid_results) / len(valid_results)
            avg_time = sum(r.total_time_ms for r in valid_results) / len(valid_results)
            
            # 효율성 = 정확도 / (토큰 / 100)
            efficiency = avg_accuracy / (avg_tokens / 100) if avg_tokens > 0 else 0
            
            method_summaries[method] = {
                "accuracy": avg_accuracy,
                "tokens": avg_tokens,
                "time": avg_time,
                "efficiency": efficiency
            }
            
            print(f"{method:<20} {avg_accuracy:<12.3f} {avg_tokens:<11.0f} {avg_time:<12.0f} {efficiency:<10.3f}")
        
        # 핵심 비교
        if "2-Stage-Pipeline" in method_summaries and "3-Stage-NONE" in method_summaries:
            two_stage = method_summaries["2-Stage-Pipeline"]
            three_stage = method_summaries["3-Stage-NONE"]
            
            print(f"\n{'='*20} 2-STAGE vs 3-STAGE {'='*20}")
            
            accuracy_diff = ((two_stage["accuracy"] - three_stage["accuracy"]) / three_stage["accuracy"]) * 100
            token_diff = ((two_stage["tokens"] - three_stage["tokens"]) / three_stage["tokens"]) * 100
            efficiency_diff = ((two_stage["efficiency"] - three_stage["efficiency"]) / three_stage["efficiency"]) * 100
            
            print(f"Accuracy Change (2-stage vs 3-stage): {accuracy_diff:+.1f}%")
            print(f"Token Cost Change: {token_diff:+.1f}%")
            print(f"Efficiency Change: {efficiency_diff:+.1f}%")
            
            # 결론
            if efficiency_diff > 5:
                print("🏆 2-Stage WINS: Better efficiency with similar quality!")
            elif efficiency_diff > -5:
                print("🤔 MIXED RESULTS: Similar performance")
            else:
                print("😰 3-Stage WINS: Quality loss not worth the savings")

def main():
    """메인 실행 함수"""
    try:
        runner = TwoStageComparisonRunner()
        results = runner.run_full_comparison()
        
        print("\n" + "🎯" * 30)
        print("*** 2-STAGE PIPELINE EXPERIMENT COMPLETED ***")
        print("Check detailed results in the saved JSON file.")
        
    except KeyboardInterrupt:
        print("\n[!] Experiment interrupted by user")
    except Exception as e:
        print(f"\n[!] Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()