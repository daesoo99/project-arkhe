#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project Arkhē - Category-wise Comparison Experiment
카테고리별 Multi-Agent vs Single Model 성능 비교

평가 대상:
- 4가지 데이터셋: GSM8K(수학), MMLU(지식), HumanEval(코딩), Mixed(혼합)
- 4가지 방법: Multi-Agent-NONE, Single-llama3:8b, Single-claude-3-haiku, Single-gpt-4o-mini
"""

import sys
import os
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'datasets'))

from orchestrator.isolation_pipeline import IsolationPipeline, IsolationLevel
from llm.simple_llm import create_llm_auto
from standard_benchmarks import StandardBenchmarkLoader
from metrics.information_theory import InformationTheoryCalculator

@dataclass
class CategoryResult:
    """카테고리별 비교 결과"""
    method_name: str
    category: str  # "math", "knowledge", "coding", "mixed"
    question_count: int
    
    avg_accuracy: float
    avg_tokens: int
    avg_time_ms: int
    efficiency: float  # accuracy/tokens
    
    # 세부 결과들
    individual_results: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_name": self.method_name,
            "category": self.category,
            "question_count": self.question_count,
            "avg_accuracy": float(self.avg_accuracy),
            "avg_tokens": self.avg_tokens,
            "avg_time_ms": self.avg_time_ms,
            "efficiency": float(self.efficiency),
            "individual_results": self.individual_results
        }

class CategoryComparisonRunner:
    """카테고리별 비교 실험 러너"""
    
    def __init__(self):
        self.loader = StandardBenchmarkLoader()
        self.calculator = InformationTheoryCalculator()
        
        # 테스트할 방법들
        self.methods = [
            ("Multi-Agent-NONE", self._run_multi_agent_none),
            ("Single-llama3:8b", lambda q, qid: self._run_single_model(q, qid, "llama3:8b")),
            ("Single-claude-3-haiku", lambda q, qid: self._run_single_model(q, qid, "claude-3-haiku")),
            ("Single-gpt-4o-mini", lambda q, qid: self._run_single_model(q, qid, "gpt-4o-mini"))
        ]
        
        # 카테고리별 질문 수
        self.questions_per_category = 8
    
    def run_full_comparison(self) -> Dict[str, List[CategoryResult]]:
        """전체 카테고리 비교 실험"""
        
        print("=" * 80)
        print("*** CATEGORY-WISE MULTI-AGENT COMPARISON ***")
        print("4 Methods × 4 Categories = 16 Experiments")
        print("=" * 80)
        
        # 카테고리별 질문 준비
        categories = {
            "math": self.loader.get_questions(count=self.questions_per_category, categories=["math"]),
            "knowledge": self.loader.get_questions(count=self.questions_per_category, categories=["knowledge"]),
            "coding": self.loader.get_questions(count=self.questions_per_category, categories=["coding"]),
            "mixed": self.loader.get_questions(count=self.questions_per_category, categories=["math", "knowledge", "coding"])
        }
        
        all_results = {}
        
        # 각 방법별로 실험
        for method_name, method_func in self.methods:
            print(f"\n[*] Testing {method_name}...")
            all_results[method_name] = []
            
            # 각 카테고리별로 테스트
            for category, questions in categories.items():
                print(f"  [+] Category: {category.upper()} ({len(questions)} questions)")
                
                category_results = []
                category_start_time = time.time()
                
                for i, question in enumerate(questions, 1):
                    print(f"    {i}/{len(questions)}: {question.id}")
                    
                    try:
                        result = method_func(question, question.id)
                        category_results.append(result)
                        
                        print(f"      → Acc: {result['accuracy']:.3f}, Tokens: {result['tokens']}, Time: {result['time_ms']}ms")
                        
                    except Exception as e:
                        print(f"      → ERROR: {e}")
                        category_results.append({
                            "question_id": question.id,
                            "accuracy": 0.0,
                            "tokens": 0,
                            "time_ms": 0,
                            "error": str(e)
                        })
                
                # 카테고리 요약
                if category_results:
                    avg_accuracy = sum(r["accuracy"] for r in category_results) / len(category_results)
                    avg_tokens = sum(r["tokens"] for r in category_results) / len(category_results)
                    avg_time = sum(r["time_ms"] for r in category_results) / len(category_results)
                    efficiency = avg_accuracy / (avg_tokens / 100) if avg_tokens > 0 else 0
                    
                    category_result = CategoryResult(
                        method_name=method_name,
                        category=category,
                        question_count=len(questions),
                        avg_accuracy=avg_accuracy,
                        avg_tokens=int(avg_tokens),
                        avg_time_ms=int(avg_time),
                        efficiency=efficiency,
                        individual_results=category_results
                    )
                    
                    all_results[method_name].append(category_result)
                    
                    print(f"    [+] {category.upper()} Summary: Acc={avg_accuracy:.3f}, Tokens={avg_tokens:.0f}, Time={avg_time:.0f}ms, Eff={efficiency:.3f}")
                
                time.sleep(1)  # 모델 과부하 방지
        
        # 결과 저장
        self._save_results(all_results)
        
        # 최종 분석
        self._print_final_analysis(all_results)
        
        return all_results
    
    def _run_multi_agent_none(self, question, question_id) -> Dict[str, Any]:
        """Multi-Agent NONE 모드 실행"""
        pipeline = IsolationPipeline(IsolationLevel.NONE, k_samples=3)
        
        start_time = time.time()
        result = pipeline.run_experiment(
            query=question.query,
            expected_answer=question.expected_answer,
            question_id=question_id,
            llm_factory=create_llm_auto
        )
        
        return {
            "question_id": question_id,
            "accuracy": result.accuracy_score,
            "tokens": result.total_tokens,
            "time_ms": result.total_time_ms,
            "final_answer": result.final_answer,
            "entropy": result.judge_entropy,
            "consensus": result.consensus_score
        }
    
    def _run_single_model(self, question, question_id, model_name: str) -> Dict[str, Any]:
        """단일 모델 실행"""
        start_time = time.time()
        
        try:
            llm = create_llm_auto(model_name)
            
            # 카테고리별 프롬프트 최적화
            if question.category == "math":
                prompt = f"""Solve this math problem step by step:

{question.query}

Show your work and provide the final numerical answer:"""
            elif question.category == "coding":
                prompt = f"""Write clean, working code for this problem:

{question.query}

Provide only the code solution:"""
            else:  # knowledge
                prompt = f"""Answer this question accurately and concisely:

{question.query}

Provide a clear, factual answer:"""
            
            response = llm.generate(prompt, temperature=0.3, max_tokens=512)
            
            if isinstance(response, dict):
                final_answer = response.get("response", str(response))
            else:
                final_answer = str(response)
            
            # 정확한 토큰 계산
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
            from utils.token_counter import UnifiedTokenCounter
            token_counter = UnifiedTokenCounter()
            
            token_usage = token_counter.calculate_usage(prompt, final_answer)
            tokens = token_usage.total_tokens
            
            # 정확도 계산
            accuracy = self._calculate_accuracy(final_answer, question.expected_answer)
            
            total_time = int((time.time() - start_time) * 1000)
            
            return {
                "question_id": question_id,
                "accuracy": accuracy,
                "tokens": tokens,
                "time_ms": total_time,
                "final_answer": final_answer
            }
            
        except Exception as e:
            return {
                "question_id": question_id,
                "accuracy": 0.0,
                "tokens": 0,
                "time_ms": 0,
                "error": str(e)
            }
    
    def _calculate_accuracy(self, final_answer: str, expected_answer: str) -> float:
        """정확도 계산 (카테고리별 최적화)"""
        if not expected_answer or not final_answer:
            return 0.0
        
        final_lower = final_answer.lower().strip()
        expected_lower = expected_answer.lower().strip()
        
        # 완전 포함
        if expected_lower in final_lower:
            return 1.0
        
        # 코딩: 핵심 키워드 체크
        if "def " in expected_lower and "def " in final_lower:
            expected_words = set(expected_lower.replace("def ", "").split())
            final_words = set(final_lower.replace("def ", "").split())
            if expected_words.intersection(final_words):
                return 0.7
        
        # 수학: 숫자 추출해서 비교
        import re
        expected_nums = re.findall(r'\d+', expected_lower)
        final_nums = re.findall(r'\d+', final_lower)
        if expected_nums and expected_nums[0] in final_nums:
            return 1.0
        
        # 일반적인 단어 매칭
        expected_words = set(expected_lower.split())
        final_words = set(final_lower.split())
        
        if expected_words and expected_words.intersection(final_words):
            return len(expected_words.intersection(final_words)) / len(expected_words)
        
        return 0.0
    
    def _save_results(self, results: Dict[str, List[CategoryResult]]):
        """결과 저장"""
        results_file = f"experiments/results/category_comparison_{int(time.time())}.json"
        Path(results_file).parent.mkdir(parents=True, exist_ok=True)
        
        # JSON 직렬화 가능한 형태로 변환
        json_results = {}
        for method, category_results in results.items():
            json_results[method] = [cr.to_dict() for cr in category_results]
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n[+] Results saved: {results_file}")
        return results_file
    
    def _print_final_analysis(self, results: Dict[str, List[CategoryResult]]):
        """최종 분석 출력"""
        print("\n" + "=" * 80)
        print("*** FINAL CATEGORY ANALYSIS ***")
        print("=" * 80)
        
        # 카테고리별 성능 비교 테이블
        categories = ["math", "knowledge", "coding", "mixed"]
        
        for category in categories:
            print(f"\n[{category.upper()}] Category Results:")
            print(f"{'Method':<25} {'Accuracy':<10} {'Tokens':<8} {'Time(ms)':<10} {'Efficiency':<10}")
            print("-" * 70)
            
            category_data = []
            for method, category_results in results.items():
                for cr in category_results:
                    if cr.category == category:
                        category_data.append((method, cr))
                        print(f"{method:<25} {cr.avg_accuracy:<10.3f} {cr.avg_tokens:<8} "
                              f"{cr.avg_time_ms:<10} {cr.efficiency:<10.3f}")
            
            # 카테고리별 승자
            if category_data:
                best_accuracy = max(category_data, key=lambda x: x[1].avg_accuracy)
                best_efficiency = max(category_data, key=lambda x: x[1].efficiency)
                print(f"  🏆 Best Accuracy: {best_accuracy[0]} ({best_accuracy[1].avg_accuracy:.3f})")
                print(f"  💰 Best Efficiency: {best_efficiency[0]} ({best_efficiency[1].efficiency:.3f})")
        
        # 전체 종합 분석
        print(f"\n{'='*20} OVERALL SUMMARY {'='*20}")
        method_totals = {}
        
        for method, category_results in results.items():
            total_acc = sum(cr.avg_accuracy for cr in category_results) / len(category_results)
            total_tokens = sum(cr.avg_tokens for cr in category_results) / len(category_results)
            total_eff = sum(cr.efficiency for cr in category_results) / len(category_results)
            
            method_totals[method] = {
                "accuracy": total_acc,
                "tokens": total_tokens,
                "efficiency": total_eff
            }
            
            print(f"{method:<25}: Acc={total_acc:.3f}, Tokens={total_tokens:.0f}, Eff={total_eff:.3f}")
        
        # 최종 승자
        if method_totals:
            best_overall = max(method_totals.items(), key=lambda x: x[1]["accuracy"])
            most_efficient = max(method_totals.items(), key=lambda x: x[1]["efficiency"])
            
            print(f"\n🏆 OVERALL WINNER (Accuracy): {best_overall[0]} ({best_overall[1]['accuracy']:.3f})")
            print(f"💰 MOST EFFICIENT: {most_efficient[0]} ({most_efficient[1]['efficiency']:.3f})")

def main():
    """메인 실행 함수"""
    try:
        runner = CategoryComparisonRunner()
        results = runner.run_full_comparison()
        
        print("\n" + "🎯" * 30)
        print("*** EXPERIMENT COMPLETED SUCCESSFULLY ***")
        print("Check the results file for detailed data.")
        
    except KeyboardInterrupt:
        print("\n[!] Experiment interrupted by user")
    except Exception as e:
        print(f"\n[!] Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()