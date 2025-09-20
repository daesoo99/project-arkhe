#!/usr/bin/env python3
"""
Project Arkhē Style Context Separation Experiment
정량적 실험: 컨텍스트 혼재 vs 분리의 실제 성능 차이 측정
"""

import sys
import os
import time
import json
from typing import Dict, List, Tuple, Any

# Project Arkhē 기존 인프라 사용
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm.simple_llm import OllamaLLM
# Remove TaskScorer import - will implement simple scoring inline
from multiroom.version_workspace_manager import (
    VersionWorkspaceManager, ContextPriority
)


class ContextSeparationBenchmark:
    """컨텍스트 분리 벤치마크 - Arkhē 스타일"""

    def __init__(self, models: List[str] = None):
        self.models = models or ["gemma:2b", "llama3:8b"]
        self.llms = {model: OllamaLLM(model) for model in self.models}
        # Simple inline scoring instead of TaskScorer
        self.workspace_manager = VersionWorkspaceManager()

        # 실험 설정
        self.setup_complex_context()

    def setup_complex_context(self):
        """복잡한 컨텍스트 설정 (실제 상황 모방)"""

        self.workspace_manager.switch_workspace("v1_research")

        # 핵심 규칙들 (PERSISTENT)
        persistent_rules = [
            "하드코딩 절대 금지, 모든 설정은 config 파일 사용",
            "실험 결과는 반드시 EXPERIMENT_LOG.md에 기록",
            "Registry 패턴으로 모든 모델 관리 필수",
            "CLAUDE.local 규칙 엄격히 준수"
        ]

        for rule in persistent_rules:
            self.workspace_manager.add_message_smart(
                rule, role="system",
                priority=ContextPriority.PERSISTENT,
                is_rule=True
            )

        # 과거 실험 히스토리 (REFERENCE)
        past_experiments = [
            "DECISION: Multi-Agent보다 Single 모델이 87.7% vs 50.2%로 우위",
            "DECISION: Shannon Entropy로 정보 복잡도 측정하기로 결정",
            "DECISION: 창의적 작업에서는 Multi-Agent가 2.2배 단어 다양성 달성",
            "DECISION: Registry 패턴이 하드코딩 문제 완전 해결",
            "DECISION: Token 최적화로 비용 40% 절감 성공"
        ]

        for exp in past_experiments:
            self.workspace_manager.add_message_smart(
                exp, role="assistant",
                priority=ContextPriority.REFERENCE,
                is_decision=True
            )

        # 최근 잡다한 대화들 (ACTIVE)
        recent_conversations = [
            "qwen2:0.5b 모델의 품질 한계 확인됨",
            "Multi-Agent에서 토큰 기하급수적 증가 문제",
            "llama3:8b Judge 모델의 효과적 성능",
            "실험 15개 질문으로 벤치마크 진행",
            "temperature 파라미터 조정 필요성",
            "k_samples 최적화 실험 완료",
            "효율성 메트릭 계산 방식 개선",
            "baseline_comparison 스크립트 개선",
            "Economic Intelligence 모듈 통합",
            "Information Asymmetry 실험 설계"
        ]

        for conv in recent_conversations:
            self.workspace_manager.add_message_smart(
                f"논의: {conv}", role="user",
                priority=ContextPriority.ACTIVE
            )

        # 시간 경과 시뮬레이션
        self.workspace_manager.auto_archive_old_messages(hours_threshold=0.1)

    def create_benchmark_questions(self) -> List[Dict[str, Any]]:
        """벤치마크 질문들 생성 (Project Arkhē 스타일)"""

        return [
            {
                "id": "Q1",
                "question": "Write a Python function to calculate the factorial of a number",
                "type": "coding",
                "expected_keywords": ["def", "factorial", "return"],
                "irrelevant_keywords": ["experiment", "multi-agent", "decision", "entropy"]
            },
            {
                "id": "Q2",
                "question": "What is the main advantage of using Registry pattern in our codebase?",
                "type": "architecture",
                "expected_keywords": ["registry", "pattern", "hardcoding"],
                "irrelevant_keywords": ["temperature", "token", "qwen2", "samples"]
            },
            {
                "id": "Q3",
                "question": "How should I handle errors in API calls?",
                "type": "general",
                "expected_keywords": ["error", "exception", "try"],
                "irrelevant_keywords": ["experiment", "baseline", "llama", "shannon"]
            },
            {
                "id": "Q4",
                "question": "What were the key findings from Multi-Agent vs Single model comparison?",
                "type": "research",
                "expected_keywords": ["multi-agent", "single", "87.7%", "50.2%"],
                "irrelevant_keywords": ["factorial", "api", "error", "exception"]
            },
            {
                "id": "Q5",
                "question": "Create a simple class for database connection",
                "type": "coding",
                "expected_keywords": ["class", "database", "connection"],
                "irrelevant_keywords": ["entropy", "decision", "experiment", "efficiency"]
            }
        ]

    def run_single_test(self, model: str, question: Dict, context: str) -> Dict[str, Any]:
        """단일 테스트 실행"""

        full_prompt = f"""
{context}

===== CURRENT QUESTION =====
{question['question']}

Please provide a clear and focused response.
"""

        start_time = time.time()

        try:
            response = self.llms[model].generate(full_prompt)
            end_time = time.time()

            response_text = response.get('response', '')

            # 성능 지표 계산
            relevance_score = self._calculate_relevance(
                response_text,
                question['expected_keywords'],
                question['irrelevant_keywords']
            )

            return {
                "success": True,
                "response": response_text,
                "response_time": end_time - start_time,
                "token_count": len(full_prompt.split()) + len(response_text.split()),
                "prompt_tokens": len(full_prompt.split()),
                "response_tokens": len(response_text.split()),
                "relevance_score": relevance_score,
                "context_length": len(context)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response_time": time.time() - start_time,
                "token_count": 0,
                "relevance_score": 0.0
            }

    def _calculate_relevance(self, response: str, expected_kw: List[str], irrelevant_kw: List[str]) -> float:
        """응답 관련성 점수 계산"""

        response_lower = response.lower()

        # 관련 키워드 점수
        relevant_count = sum(1 for kw in expected_kw if kw.lower() in response_lower)
        relevant_score = relevant_count / len(expected_kw) if expected_kw else 0

        # 무관한 키워드 페널티
        irrelevant_count = sum(1 for kw in irrelevant_kw if kw.lower() in response_lower)
        irrelevant_penalty = irrelevant_count / len(irrelevant_kw) if irrelevant_kw else 0

        # 최종 점수 (0.0 ~ 1.0)
        final_score = max(0.0, relevant_score - irrelevant_penalty * 0.5)

        return final_score

    def run_benchmark(self) -> Dict[str, Any]:
        """전체 벤치마크 실행"""

        print("=== Context Separation Benchmark (Arkhe Style) ===\n")

        questions = self.create_benchmark_questions()
        results = {
            "mixed_context": {},
            "separated_context": {},
            "summary": {}
        }

        for model in self.models:
            print(f"Testing model: {model}")

            mixed_results = []
            separated_results = []

            for question in questions:
                print(f"  Question {question['id']}: {question['question'][:50]}...")

                # A) 혼재된 컨텍스트 (기존 방식)
                all_messages = []
                if self.workspace_manager.main_thread:
                    all_messages.extend(self.workspace_manager.main_thread)

                current_workspace = self.workspace_manager.get_current_workspace()
                if current_workspace and current_workspace.messages:
                    all_messages.extend(current_workspace.messages)

                mixed_context = "\n".join([
                    f"{msg.role}: {msg.content}"
                    for msg in all_messages[-15:]  # 최근 15개
                ])

                mixed_result = self.run_single_test(model, question, mixed_context)
                mixed_results.append({**mixed_result, "question_id": question['id']})

                # B) 분리된 컨텍스트 (새로운 방식)
                separated_context = self.workspace_manager.get_smart_context()

                separated_result = self.run_single_test(model, question, separated_context)
                separated_results.append({**separated_result, "question_id": question['id']})

                # 즉시 결과 출력
                if mixed_result['success'] and separated_result['success']:
                    relevance_improvement = separated_result['relevance_score'] - mixed_result['relevance_score']
                    token_reduction = (mixed_result['token_count'] - separated_result['token_count']) / mixed_result['token_count'] * 100

                    print(f"    Relevance: {mixed_result['relevance_score']:.3f} → {separated_result['relevance_score']:.3f} (Δ{relevance_improvement:+.3f})")
                    print(f"    Tokens: {mixed_result['token_count']} → {separated_result['token_count']} ({token_reduction:+.1f}%)")

            results["mixed_context"][model] = mixed_results
            results["separated_context"][model] = separated_results

        # 전체 요약 계산
        results["summary"] = self._calculate_summary(results)

        return results

    def _calculate_summary(self, results: Dict) -> Dict[str, Any]:
        """전체 결과 요약"""

        summary = {}

        for model in self.models:
            mixed = results["mixed_context"][model]
            separated = results["separated_context"][model]

            # 성공한 테스트만 집계
            mixed_success = [r for r in mixed if r['success']]
            separated_success = [r for r in separated if r['success']]

            if mixed_success and separated_success:
                # 평균 지표 계산
                mixed_avg_relevance = sum(r['relevance_score'] for r in mixed_success) / len(mixed_success)
                separated_avg_relevance = sum(r['relevance_score'] for r in separated_success) / len(separated_success)

                mixed_avg_tokens = sum(r['token_count'] for r in mixed_success) / len(mixed_success)
                separated_avg_tokens = sum(r['token_count'] for r in separated_success) / len(separated_success)

                mixed_avg_time = sum(r['response_time'] for r in mixed_success) / len(mixed_success)
                separated_avg_time = sum(r['response_time'] for r in separated_success) / len(separated_success)

                summary[model] = {
                    "mixed_avg_relevance": mixed_avg_relevance,
                    "separated_avg_relevance": separated_avg_relevance,
                    "relevance_improvement": separated_avg_relevance - mixed_avg_relevance,
                    "relevance_improvement_pct": (separated_avg_relevance - mixed_avg_relevance) / mixed_avg_relevance * 100 if mixed_avg_relevance > 0 else 0,

                    "mixed_avg_tokens": mixed_avg_tokens,
                    "separated_avg_tokens": separated_avg_tokens,
                    "token_reduction": mixed_avg_tokens - separated_avg_tokens,
                    "token_reduction_pct": (mixed_avg_tokens - separated_avg_tokens) / mixed_avg_tokens * 100 if mixed_avg_tokens > 0 else 0,

                    "mixed_avg_time": mixed_avg_time,
                    "separated_avg_time": separated_avg_time,
                    "time_improvement": mixed_avg_time - separated_avg_time,

                    "success_rate": len(separated_success) / len(separated)
                }

        return summary

    def print_results(self, results: Dict[str, Any]):
        """결과 출력 (Arkhē 스타일)"""

        print(f"\n=== BENCHMARK RESULTS ===")

        for model, summary in results["summary"].items():
            print(f"\nModel: {model}")
            print(f"  Relevance Score:")
            print(f"    Mixed Context: {summary['mixed_avg_relevance']:.3f}")
            print(f"    Separated Context: {summary['separated_avg_relevance']:.3f}")
            print(f"    Improvement: {summary['relevance_improvement']:+.3f} ({summary['relevance_improvement_pct']:+.1f}%)")

            print(f"  Token Usage:")
            print(f"    Mixed Context: {summary['mixed_avg_tokens']:.0f} tokens")
            print(f"    Separated Context: {summary['separated_avg_tokens']:.0f} tokens")
            print(f"    Reduction: {summary['token_reduction']:.0f} tokens ({summary['token_reduction_pct']:+.1f}%)")

            print(f"  Response Time:")
            print(f"    Mixed Context: {summary['mixed_avg_time']:.2f}s")
            print(f"    Separated Context: {summary['separated_avg_time']:.2f}s")
            print(f"    Improvement: {summary['time_improvement']:+.2f}s")

            print(f"  Success Rate: {summary['success_rate']:.1%}")

    def save_results(self, results: Dict[str, Any], filename: str = None):
        """결과 저장"""

        if filename is None:
            timestamp = int(time.time())
            filename = f"context_separation_results_{timestamp}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {filename}")


def main():
    """메인 실행 함수"""

    try:
        # 간단한 연결 테스트
        test_llm = OllamaLLM("gemma:2b")
        test_response = test_llm.generate("Hello")

        if not test_response.get('response'):
            print("❌ Ollama server not accessible. Please start Ollama first:")
            print("   ollama serve")
            return

    except Exception as e:
        print(f"❌ Failed to connect to Ollama: {e}")
        print("Please ensure Ollama is running with available models")
        return

    # 벤치마크 실행
    benchmark = ContextSeparationBenchmark()

    print("Starting Context Separation Benchmark...")
    print("This will measure quantitative performance differences\n")

    results = benchmark.run_benchmark()
    benchmark.print_results(results)
    benchmark.save_results(results)

    # EXPERIMENT_LOG 업데이트 제안
    print(f"\nNext steps:")
    print(f"   1. Review detailed results in the saved JSON file")
    print(f"   2. Add findings to EXPERIMENT_LOG.md")
    print(f"   3. Consider running with more models for broader validation")


if __name__ == "__main__":
    main()