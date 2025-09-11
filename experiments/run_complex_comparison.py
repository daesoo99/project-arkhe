#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project Arkhē - Complex Problem Comparison Experiment
복잡한 문제에서 Multi-Agent vs Single Model 성능 비교

목적: 매우 복잡한 문제에서 Multi-Agent가 진정한 가치를 보여줄 수 있는지 검증
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
class ComplexResult:
    """복잡한 문제 실험 결과"""
    method_name: str
    problem_id: str
    problem_category: str
    query: str
    expected_components: List[str]
    
    final_answer: str
    total_tokens: int
    total_time_ms: int
    
    # 복잡성 평가 메트릭
    answer_length: int
    component_coverage: float  # 필수 요소 포함 비율
    depth_score: float  # 답변의 깊이 점수
    coherence_score: float  # 논리적 일관성
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_name": self.method_name,
            "problem_id": self.problem_id,
            "problem_category": self.problem_category,
            "query": self.query,
            "expected_components": self.expected_components,
            "final_answer": self.final_answer,
            "total_tokens": self.total_tokens,
            "total_time_ms": self.total_time_ms,
            "answer_length": self.answer_length,
            "component_coverage": float(self.component_coverage),
            "depth_score": float(self.depth_score),
            "coherence_score": float(self.coherence_score)
        }

class ComplexProblemEvaluator:
    """복잡한 문제 평가기"""
    
    def __init__(self):
        pass
    
    def evaluate_answer(self, answer: str, expected_components: List[str]) -> Dict[str, float]:
        """복잡한 답변 평가"""
        
        # 1. 컴포넌트 커버리지 (필수 요소 포함 비율)
        component_coverage = self._calculate_component_coverage(answer, expected_components)
        
        # 2. 깊이 점수 (답변의 세부 정도)
        depth_score = self._calculate_depth_score(answer)
        
        # 3. 일관성 점수 (논리적 구조)
        coherence_score = self._calculate_coherence_score(answer)
        
        return {
            "component_coverage": component_coverage,
            "depth_score": depth_score,
            "coherence_score": coherence_score
        }
    
    def _calculate_component_coverage(self, answer: str, expected_components: List[str]) -> float:
        """필수 요소 포함 비율 계산"""
        if not expected_components:
            return 1.0
        
        answer_lower = answer.lower()
        covered = 0
        
        for component in expected_components:
            # 키워드 기반 매칭 (간단한 버전)
            component_keywords = component.lower().split()
            if any(keyword in answer_lower for keyword in component_keywords):
                covered += 1
        
        return covered / len(expected_components)
    
    def _calculate_depth_score(self, answer: str) -> float:
        """답변 깊이 점수 (0-1)"""
        # 간단한 휴리스틱: 길이, 구조적 요소들 기반
        length_score = min(len(answer) / 2000, 1.0)  # 2000자를 만점 기준
        
        # 구조적 요소들
        structure_indicators = [
            "첫째", "둘째", "셋째", "1)", "2)", "3)",
            "따라서", "그러므로", "결론적으로", "요약하면",
            "예를 들어", "구체적으로", "세부적으로",
            "반면에", "그러나", "하지만", "한편"
        ]
        
        structure_score = 0
        for indicator in structure_indicators:
            if indicator in answer:
                structure_score += 1
        structure_score = min(structure_score / 10, 1.0)  # 10개를 만점 기준
        
        return (length_score * 0.6 + structure_score * 0.4)
    
    def _calculate_coherence_score(self, answer: str) -> float:
        """논리적 일관성 점수 (0-1)"""
        # 간단한 휴리스틱: 문장 연결성, 반복 여부
        sentences = answer.split('.')
        if len(sentences) < 2:
            return 0.5
        
        # 평균 문장 길이 (너무 짧거나 길면 감점)
        avg_sentence_length = sum(len(s.strip()) for s in sentences) / len(sentences)
        length_coherence = 1.0 - abs(avg_sentence_length - 100) / 200  # 100자를 적정으로
        length_coherence = max(0.0, min(1.0, length_coherence))
        
        # 논리 연결 표현 존재
        logic_connectors = ["따라서", "그러므로", "왜냐하면", "그 이유는", "결과적으로"]
        connector_score = sum(1 for conn in logic_connectors if conn in answer) / len(logic_connectors)
        
        return (length_coherence * 0.7 + connector_score * 0.3)

class ComplexComparisonRunner:
    """복잡한 문제 비교 실험 러너"""
    
    def __init__(self):
        self.evaluator = ComplexProblemEvaluator()
        self.load_complex_problems()
    
    def load_complex_problems(self):
        """복잡한 문제 데이터셋 로드"""
        dataset_path = Path(__file__).parent.parent / "datasets" / "complex_problems.json"
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Complex problems dataset not found: {dataset_path}")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.problems = json.load(f)
        
        print(f"[+] Loaded {len(self.problems)} complex problems")
    
    def run_full_comparison(self) -> Dict[str, List[ComplexResult]]:
        """전체 복잡한 문제 비교 실험"""
        
        print("=" * 80)
        print("*** COMPLEX PROBLEM COMPARISON EXPERIMENT ***")
        print("Multi-Agent vs Single Model on Very Hard Problems")
        print("=" * 80)
        
        # 테스트할 방법들
        methods = [
            ("Multi-Agent-NONE", self._run_multi_agent_none),
            ("Single-llama3:8b", lambda p: self._run_single_model(p, "llama3:8b"))
        ]
        
        all_results = {}
        
        for method_name, method_func in methods:
            print(f"\n[*] Testing {method_name}...")
            all_results[method_name] = []
            
            for i, problem in enumerate(self.problems, 1):
                print(f"\n  [{i}/{len(self.problems)}] Problem: {problem['id']}")
                print(f"  Category: {problem['category']}")
                print(f"  Query: {problem['query'][:100]}...")
                
                try:
                    result = method_func(problem)
                    all_results[method_name].append(result)
                    
                    print(f"    → Tokens: {result.total_tokens}, Time: {result.total_time_ms}ms")
                    print(f"    → Coverage: {result.component_coverage:.3f}, Depth: {result.depth_score:.3f}, Coherence: {result.coherence_score:.3f}")
                    
                except Exception as e:
                    print(f"    → ERROR: {e}")
                    # 에러 결과도 기록
                    error_result = ComplexResult(
                        method_name=method_name,
                        problem_id=problem['id'],
                        problem_category=problem['category'],
                        query=problem['query'],
                        expected_components=problem['expected_components'],
                        final_answer=f"ERROR: {e}",
                        total_tokens=0,
                        total_time_ms=0,
                        answer_length=0,
                        component_coverage=0.0,
                        depth_score=0.0,
                        coherence_score=0.0
                    )
                    all_results[method_name].append(error_result)
                
                time.sleep(2)  # 모델 과부하 방지
        
        # 결과 저장
        self._save_results(all_results)
        
        # 최종 분석
        self._print_final_analysis(all_results)
        
        return all_results
    
    def _run_multi_agent_none(self, problem: Dict[str, Any]) -> ComplexResult:
        """Multi-Agent NONE 모드 실행"""
        pipeline = IsolationPipeline(IsolationLevel.NONE, k_samples=3)
        
        start_time = time.time()
        isolation_result = pipeline.run_experiment(
            query=problem['query'],
            expected_answer="",  # 복잡한 문제는 정답이 정해져 있지 않음
            question_id=problem['id'],
            llm_factory=create_llm_auto
        )
        
        # 복잡성 평가
        evaluation = self.evaluator.evaluate_answer(
            isolation_result.final_answer, 
            problem['expected_components']
        )
        
        return ComplexResult(
            method_name="Multi-Agent-NONE",
            problem_id=problem['id'],
            problem_category=problem['category'],
            query=problem['query'],
            expected_components=problem['expected_components'],
            final_answer=isolation_result.final_answer,
            total_tokens=isolation_result.total_tokens,
            total_time_ms=isolation_result.total_time_ms,
            answer_length=len(isolation_result.final_answer),
            component_coverage=evaluation['component_coverage'],
            depth_score=evaluation['depth_score'],
            coherence_score=evaluation['coherence_score']
        )
    
    def _run_single_model(self, problem: Dict[str, Any], model_name: str) -> ComplexResult:
        """단일 모델 실행"""
        start_time = time.time()
        
        llm = create_llm_auto(model_name)
        
        # 복잡한 문제용 프롬프트
        prompt = f"""다음은 매우 복잡한 문제입니다. 깊이 있고 체계적으로 분석하여 답변해주세요:

{problem['query']}

요구사항:
- 단계별로 논리적 분석
- 각 측면을 충분히 검토
- 구체적이고 실행 가능한 방안 제시
- 완전하고 일관된 답변 작성"""
        
        try:
            response = llm.generate(prompt, temperature=0.3, max_tokens=2048)
            
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
            
            # 복잡성 평가
            evaluation = self.evaluator.evaluate_answer(final_answer, problem['expected_components'])
            
            return ComplexResult(
                method_name=f"Single-{model_name}",
                problem_id=problem['id'],
                problem_category=problem['category'],
                query=problem['query'],
                expected_components=problem['expected_components'],
                final_answer=final_answer,
                total_tokens=tokens,
                total_time_ms=total_time,
                answer_length=len(final_answer),
                component_coverage=evaluation['component_coverage'],
                depth_score=evaluation['depth_score'],
                coherence_score=evaluation['coherence_score']
            )
            
        except Exception as e:
            raise e
    
    def _save_results(self, results: Dict[str, List[ComplexResult]]):
        """결과 저장"""
        results_file = f"experiments/results/complex_comparison_{int(time.time())}.json"
        Path(results_file).parent.mkdir(parents=True, exist_ok=True)
        
        # JSON 직렬화
        json_results = {}
        for method, result_list in results.items():
            json_results[method] = [r.to_dict() for r in result_list]
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n[+] Results saved: {results_file}")
        return results_file
    
    def _print_final_analysis(self, results: Dict[str, List[ComplexResult]]):
        """최종 분석 출력"""
        print("\n" + "=" * 80)
        print("*** COMPLEX PROBLEM ANALYSIS ***")
        print("=" * 80)
        
        # 전체 요약
        print(f"\n{'Method':<20} {'Avg Coverage':<12} {'Avg Depth':<10} {'Avg Coherence':<12} {'Avg Tokens':<10} {'Avg Time(ms)':<12}")
        print("-" * 80)
        
        method_summaries = {}
        
        for method, result_list in results.items():
            if not result_list:
                continue
                
            valid_results = [r for r in result_list if r.total_tokens > 0]
            if not valid_results:
                continue
            
            avg_coverage = sum(r.component_coverage for r in valid_results) / len(valid_results)
            avg_depth = sum(r.depth_score for r in valid_results) / len(valid_results)
            avg_coherence = sum(r.coherence_score for r in valid_results) / len(valid_results)
            avg_tokens = sum(r.total_tokens for r in valid_results) / len(valid_results)
            avg_time = sum(r.total_time_ms for r in valid_results) / len(valid_results)
            
            method_summaries[method] = {
                "coverage": avg_coverage,
                "depth": avg_depth,
                "coherence": avg_coherence,
                "tokens": avg_tokens,
                "time": avg_time
            }
            
            print(f"{method:<20} {avg_coverage:<12.3f} {avg_depth:<10.3f} {avg_coherence:<12.3f} {avg_tokens:<10.0f} {avg_time:<12.0f}")
        
        # 카테고리별 분석
        print(f"\n{'Category':<20} {'Multi Coverage':<13} {'Single Coverage':<14} {'Multi Depth':<11} {'Single Depth':<12}")
        print("-" * 80)
        
        categories = set()
        for result_list in results.values():
            categories.update(r.problem_category for r in result_list)
        
        for category in categories:
            multi_results = [r for r in results.get("Multi-Agent-NONE", []) if r.problem_category == category and r.total_tokens > 0]
            single_results = [r for r in results.get("Single-llama3:8b", []) if r.problem_category == category and r.total_tokens > 0]
            
            if multi_results and single_results:
                multi_coverage = sum(r.component_coverage for r in multi_results) / len(multi_results)
                single_coverage = sum(r.component_coverage for r in single_results) / len(single_results)
                multi_depth = sum(r.depth_score for r in multi_results) / len(multi_results)
                single_depth = sum(r.depth_score for r in single_results) / len(single_results)
                
                print(f"{category:<20} {multi_coverage:<13.3f} {single_coverage:<14.3f} {multi_depth:<11.3f} {single_depth:<12.3f}")
        
        # 최종 승부 판정
        if "Multi-Agent-NONE" in method_summaries and "Single-llama3:8b" in method_summaries:
            multi = method_summaries["Multi-Agent-NONE"]
            single = method_summaries["Single-llama3:8b"]
            
            print(f"\n{'='*20} FINAL VERDICT {'='*20}")
            
            # 품질 비교
            quality_score_multi = (multi["coverage"] + multi["depth"] + multi["coherence"]) / 3
            quality_score_single = (single["coverage"] + single["depth"] + single["coherence"]) / 3
            
            quality_advantage = ((quality_score_multi - quality_score_single) / quality_score_single) * 100
            cost_overhead = ((multi["tokens"] - single["tokens"]) / single["tokens"]) * 100
            
            print(f"Quality Advantage (Multi vs Single): {quality_advantage:+.1f}%")
            print(f"Cost Overhead (Multi vs Single): {cost_overhead:+.1f}%")
            
            if quality_advantage > 10:
                print("🏆 Multi-Agent WINS: Significant quality improvement!")
            elif quality_advantage > 0 and cost_overhead < 200:
                print("🤔 Multi-Agent VIABLE: Better quality with reasonable cost")
            elif quality_advantage > -10:
                print("🟨 MIXED RESULTS: Similar performance, consider context")
            else:
                print("😰 Single Model WINS: Better efficiency without quality loss")

def main():
    """메인 실행 함수"""
    try:
        runner = ComplexComparisonRunner()
        results = runner.run_full_comparison()
        
        print("\n" + "🎯" * 30)
        print("*** COMPLEX PROBLEM EXPERIMENT COMPLETED ***")
        print("Check detailed results in the saved JSON file.")
        
    except KeyboardInterrupt:
        print("\n[!] Experiment interrupted by user")
    except Exception as e:
        print(f"\n[!] Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()