#!/usr/bin/env python3
"""
종합 모델 비교 실험 - 메모리 백업 시스템의 진짜 효과 검증
모든 사용 가능한 모델들로 비교하여 일반화 가능성 확인
"""

import sys
import os
import time
import json
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm.simple_llm import OllamaLLM


class ConsistencyMemoryManager:
    """일관성 보존 메모리 관리자 (이전 버전과 동일)"""

    def __init__(self):
        self.persistent_memory = []
        self.conversation_history = []
        self.max_context_length = 1000

    def add_persistent_info(self, info: str, category: str = "rule"):
        self.persistent_memory.append({
            "content": info,
            "category": category,
            "timestamp": time.time()
        })

    def add_conversation(self, message: str, role: str = "user"):
        self.conversation_history.append({
            "content": message,
            "role": role,
            "timestamp": time.time()
        })
        self._trim_conversation_if_needed()

    def _trim_conversation_if_needed(self):
        total_length = self._estimate_total_length()
        if total_length > self.max_context_length:
            self.conversation_history = self.conversation_history[-10:]

    def _estimate_total_length(self) -> int:
        total = 0
        for item in self.persistent_memory:
            total += len(item["content"].split())
        for item in self.conversation_history:
            total += len(item["content"].split())
        return total

    def get_context_for_ai(self, current_question: str) -> str:
        context_parts = []

        if self.persistent_memory:
            context_parts.append("=== IMPORTANT RULES (ALWAYS REMEMBER) ===")
            for item in self.persistent_memory:
                context_parts.append(f"- {item['content']}")
            context_parts.append("")

        if self.conversation_history:
            context_parts.append("=== RECENT CONVERSATION ===")
            for item in self.conversation_history[-5:]:
                context_parts.append(f"{item['role']}: {item['content']}")
            context_parts.append("")

        context_parts.append("=== CURRENT QUESTION ===")
        context_parts.append(current_question)

        return "\n".join(context_parts)


class ComprehensiveModelComparison:
    """종합 모델 비교 실험"""

    def __init__(self):
        # 사용 가능한 모든 모델들
        self.available_models = ["qwen2:0.5b", "gemma:2b", "llama3:8b", "qwen2:7b"]
        self.test_questions = self._create_comprehensive_test_questions()

    def _create_comprehensive_test_questions(self) -> List[Dict[str, Any]]:
        """포괄적인 테스트 질문들"""
        return [
            {
                "id": "Q1_coding_basic",
                "question": "Write a Python function to calculate factorial",
                "category": "coding",
                "difficulty": "easy",
                "expected_elements": ["def", "factorial", "return"],
                "rules_needed": ["no hardcoding"]
            },
            {
                "id": "Q2_coding_advanced",
                "question": "Create a database connection class with error handling",
                "category": "coding",
                "difficulty": "medium",
                "expected_elements": ["class", "database", "try", "except"],
                "rules_needed": ["no hardcoding", "config files"]
            },
            {
                "id": "Q3_architecture",
                "question": "How should I implement a model registry pattern?",
                "category": "architecture",
                "difficulty": "medium",
                "expected_elements": ["registry", "pattern", "models"],
                "rules_needed": ["registry pattern", "no hardcoding"]
            },
            {
                "id": "Q4_policy",
                "question": "What are our coding standards for configuration?",
                "category": "policy",
                "difficulty": "easy",
                "expected_elements": ["config", "standards", "hardcoding"],
                "rules_needed": ["no hardcoding", "config files"]
            },
            {
                "id": "Q5_research",
                "question": "What was the outcome of our Multi-Agent vs Single model experiment?",
                "category": "research",
                "difficulty": "hard",
                "expected_elements": ["multi-agent", "single", "experiment"],
                "rules_needed": ["experiment logging"]
            },
            {
                "id": "Q6_troubleshooting",
                "question": "How do I debug import errors in Python modules?",
                "category": "troubleshooting",
                "difficulty": "easy",
                "expected_elements": ["import", "debug", "python"],
                "rules_needed": ["systematic debugging"]
            }
        ]

    def test_single_model(self, model_name: str) -> Dict[str, Any]:
        """단일 모델 테스트"""
        print(f"Testing model: {model_name}")

        try:
            llm = OllamaLLM(model_name)
        except Exception as e:
            return {
                "model": model_name,
                "error": f"Failed to initialize: {str(e)}",
                "success": False
            }

        results = {
            "model": model_name,
            "without_backup": {},
            "with_backup": {},
            "comparison": {},
            "success": True
        }

        # A) 백업 없는 테스트
        print(f"  Testing {model_name} WITHOUT backup...")
        results["without_backup"] = self._test_without_backup(llm, model_name)

        # B) 백업 있는 테스트
        print(f"  Testing {model_name} WITH backup...")
        results["with_backup"] = self._test_with_backup(llm, model_name)

        # C) 비교 분석
        results["comparison"] = self._compare_model_results(
            results["without_backup"],
            results["with_backup"]
        )

        return results

    def _test_without_backup(self, llm: OllamaLLM, model_name: str) -> Dict[str, Any]:
        """백업 없는 기존 방식 테스트"""

        # 복잡한 대화 히스토리 생성 (50개 잡담)
        conversation = self._generate_noisy_conversation()

        # 컨텍스트 길이 제한 (최근 20개만)
        limited_context = conversation[-20:]

        question_results = []

        for question_data in self.test_questions:
            question = question_data["question"]
            context = "\n".join(limited_context) + f"\n\nQ: {question}"

            try:
                start_time = time.time()
                response = llm.generate(context)
                response_time = time.time() - start_time

                response_text = response.get('response', '')

                # 성능 평가
                performance = self._evaluate_response(question_data, response_text)

                question_results.append({
                    "question_id": question_data["id"],
                    "question": question,
                    "response": response_text,
                    "response_time": response_time,
                    "performance": performance,
                    "context_length": len(context.split())
                })

            except Exception as e:
                question_results.append({
                    "question_id": question_data["id"],
                    "question": question,
                    "response": f"Error: {str(e)}",
                    "response_time": 0,
                    "performance": {"overall_score": 0, "error": True},
                    "context_length": 0
                })

        # 전체 성과 계산
        avg_score = sum(q["performance"]["overall_score"] for q in question_results) / len(question_results)
        avg_time = sum(q["response_time"] for q in question_results) / len(question_results)

        return {
            "model": model_name,
            "method": "no_backup",
            "question_results": question_results,
            "average_score": avg_score,
            "average_response_time": avg_time,
            "total_questions": len(question_results)
        }

    def _test_with_backup(self, llm: OllamaLLM, model_name: str) -> Dict[str, Any]:
        """백업 있는 새로운 방식 테스트"""

        memory_manager = ConsistencyMemoryManager()

        # 중요 규칙들을 영구 메모리에 백업
        important_rules = [
            "Never suggest hardcoding - always use config files",
            "All experiments must be logged to EXPERIMENT_LOG.md",
            "Registry pattern must be used for all models",
            "Use systematic debugging approach for troubleshooting",
            "Multi-Agent experiments showed Single model superiority (87.7% vs 50.2%)"
        ]

        for rule in important_rules:
            memory_manager.add_persistent_info(rule, "rule")

        # 50개 잡담 대화 추가 (자동으로 trim됨)
        noisy_conversations = self._generate_noisy_conversations_list()
        for conv in noisy_conversations:
            memory_manager.add_conversation(conv["content"], conv["role"])

        question_results = []

        for question_data in self.test_questions:
            question = question_data["question"]
            context = memory_manager.get_context_for_ai(question)

            try:
                start_time = time.time()
                response = llm.generate(context)
                response_time = time.time() - start_time

                response_text = response.get('response', '')

                # 성능 평가
                performance = self._evaluate_response(question_data, response_text)

                question_results.append({
                    "question_id": question_data["id"],
                    "question": question,
                    "response": response_text,
                    "response_time": response_time,
                    "performance": performance,
                    "context_length": len(context.split())
                })

            except Exception as e:
                question_results.append({
                    "question_id": question_data["id"],
                    "question": question,
                    "response": f"Error: {str(e)}",
                    "response_time": 0,
                    "performance": {"overall_score": 0, "error": True},
                    "context_length": 0
                })

        # 전체 성과 계산
        avg_score = sum(q["performance"]["overall_score"] for q in question_results) / len(question_results)
        avg_time = sum(q["response_time"] for q in question_results) / len(question_results)

        return {
            "model": model_name,
            "method": "with_backup",
            "question_results": question_results,
            "average_score": avg_score,
            "average_response_time": avg_time,
            "total_questions": len(question_results),
            "memory_stats": memory_manager.get_stats() if hasattr(memory_manager, 'get_stats') else {}
        }

    def _generate_noisy_conversation(self) -> List[str]:
        """잡음 대화 생성 (기존 방식용)"""
        distractions = [
            "What's the weather today?", "Response about weather",
            "How to cook pasta?", "Pasta cooking instructions",
            "Tell me about quantum physics", "Quantum physics explanation",
            "What's your favorite movie?", "Movie preferences discussion",
            "Help with math homework", "Math problem solutions",
            "Translate hello to Spanish", "Hola translation",
            "Write a poem about cats", "Cat poetry composition",
            "Explain machine learning", "ML concept explanation",
            "What time is it?", "Time information",
            "Help me debug SQL", "SQL debugging advice"
        ]

        # 50개 대화 생성
        conversation = [
            "Remember these rules:",
            "1. Never suggest hardcoding - use config files",
            "2. Log all experiments to EXPERIMENT_LOG.md",
            "3. Use Registry pattern for models"
        ]

        for i in range(25):  # 25쌍 = 50개
            conversation.extend([
                distractions[i % len(distractions)],
                distractions[(i + 1) % len(distractions)]
            ])

        return conversation

    def _generate_noisy_conversations_list(self) -> List[Dict[str, str]]:
        """잡음 대화 리스트 생성 (백업 방식용)"""
        distractions = [
            {"content": "What's the weather today?", "role": "user"},
            {"content": "I can't check real-time weather", "role": "assistant"},
            {"content": "How to cook pasta?", "role": "user"},
            {"content": "Boil water, add pasta, cook 8-12 minutes", "role": "assistant"},
            {"content": "Tell me about quantum physics", "role": "user"},
            {"content": "Quantum physics studies subatomic particles", "role": "assistant"},
            {"content": "What's your favorite movie?", "role": "user"},
            {"content": "I don't watch movies as an AI", "role": "assistant"},
            {"content": "Help with math homework", "role": "user"},
            {"content": "What specific math problem?", "role": "assistant"}
        ]

        # 50개 대화 생성
        conversations = []
        for i in range(50):
            conversations.append(distractions[i % len(distractions)])

        return conversations

    def _evaluate_response(self, question_data: Dict, response: str) -> Dict[str, Any]:
        """응답 평가"""
        response_lower = response.lower()

        # 1. 기본 요소 포함 여부
        expected_elements = question_data.get("expected_elements", [])
        elements_found = sum(1 for elem in expected_elements if elem.lower() in response_lower)
        elements_score = elements_found / len(expected_elements) if expected_elements else 0

        # 2. 규칙 준수 여부
        rules_needed = question_data.get("rules_needed", [])
        rules_score = 0
        if "no hardcoding" in rules_needed:
            if "config" in response_lower or "configuration" in response_lower:
                rules_score += 0.5
            if "hardcode" not in response_lower:
                rules_score += 0.5

        # 3. 응답 품질 (길이, 구조)
        quality_score = 0
        if len(response) > 50:  # 최소 길이
            quality_score += 0.3
        if "```" in response:  # 코드 블록
            quality_score += 0.3
        if len(response.split('\n')) > 3:  # 구조화된 응답
            quality_score += 0.4

        # 4. 에러 체크
        error_penalty = 0
        if "error" in response_lower or "unable" in response_lower:
            error_penalty = 0.5

        # 전체 점수 계산
        overall_score = max(0, (elements_score * 0.4 + rules_score * 0.3 + quality_score * 0.3) - error_penalty)

        return {
            "overall_score": overall_score,
            "elements_score": elements_score,
            "rules_score": rules_score,
            "quality_score": quality_score,
            "error_penalty": error_penalty,
            "elements_found": elements_found,
            "elements_total": len(expected_elements)
        }

    def _compare_model_results(self, without_backup: Dict, with_backup: Dict) -> Dict[str, Any]:
        """모델 결과 비교"""

        improvement = with_backup["average_score"] - without_backup["average_score"]
        improvement_pct = (improvement / max(without_backup["average_score"], 0.01)) * 100

        time_diff = with_backup["average_response_time"] - without_backup["average_response_time"]

        return {
            "score_improvement": improvement,
            "score_improvement_percentage": improvement_pct,
            "time_difference": time_diff,
            "without_backup_score": without_backup["average_score"],
            "with_backup_score": with_backup["average_score"],
            "is_improvement": improvement > 0,
            "significant_improvement": improvement > 0.1  # 10% 이상 개선
        }

    def run_comprehensive_comparison(self) -> Dict[str, Any]:
        """모든 모델에 대한 종합 비교 실행"""

        print("=== Comprehensive Model Comparison ===")
        print("Testing memory backup system across all available models\n")

        all_results = {}
        failed_models = []

        # 각 모델별로 테스트 실행
        for model in self.available_models:
            print(f"\n{'='*50}")
            print(f"Testing {model}")
            print(f"{'='*50}")

            try:
                model_result = self.test_single_model(model)
                if model_result["success"]:
                    all_results[model] = model_result

                    # 즉시 결과 출력
                    comp = model_result["comparison"]
                    print(f"  Results for {model}:")
                    print(f"    Without backup: {comp['without_backup_score']:.3f}")
                    print(f"    With backup: {comp['with_backup_score']:.3f}")
                    print(f"    Improvement: {comp['score_improvement']:+.3f} ({comp['score_improvement_percentage']:+.1f}%)")
                    print(f"    Significant: {'YES' if comp['significant_improvement'] else 'NO'}")
                else:
                    failed_models.append(model)
                    print(f"  FAILED: {model_result.get('error', 'Unknown error')}")

            except Exception as e:
                failed_models.append(model)
                print(f"  FAILED: {str(e)}")

        # 전체 요약 분석
        summary = self._analyze_overall_results(all_results)

        return {
            "individual_results": all_results,
            "failed_models": failed_models,
            "summary": summary,
            "experiment_info": {
                "total_models_tested": len(all_results),
                "total_models_failed": len(failed_models),
                "test_questions": len(self.test_questions),
                "timestamp": time.time()
            }
        }

    def _analyze_overall_results(self, all_results: Dict) -> Dict[str, Any]:
        """전체 결과 분석"""

        if not all_results:
            return {"error": "No successful model tests"}

        # 모든 모델의 개선도 수집
        improvements = []
        significant_improvements = []

        for model, result in all_results.items():
            comp = result["comparison"]
            improvements.append(comp["score_improvement"])
            if comp["significant_improvement"]:
                significant_improvements.append(model)

        # 통계 계산
        avg_improvement = sum(improvements) / len(improvements)
        positive_improvements = len([i for i in improvements if i > 0])
        improvement_rate = positive_improvements / len(improvements)

        # 최고/최저 성과 모델
        best_model = max(all_results.keys(),
                        key=lambda m: all_results[m]["comparison"]["score_improvement"])
        worst_model = min(all_results.keys(),
                         key=lambda m: all_results[m]["comparison"]["score_improvement"])

        return {
            "average_improvement": avg_improvement,
            "improvement_rate": improvement_rate,
            "significant_improvement_count": len(significant_improvements),
            "significant_improvement_models": significant_improvements,
            "best_performing_model": {
                "name": best_model,
                "improvement": all_results[best_model]["comparison"]["score_improvement"]
            },
            "worst_performing_model": {
                "name": worst_model,
                "improvement": all_results[worst_model]["comparison"]["score_improvement"]
            },
            "total_models": len(all_results),
            "positive_improvement_rate": improvement_rate,
            "conclusion": self._generate_conclusion(avg_improvement, improvement_rate, len(significant_improvements))
        }

    def _generate_conclusion(self, avg_improvement: float, improvement_rate: float, significant_count: int) -> str:
        """결론 생성"""

        if avg_improvement > 0.1 and improvement_rate >= 0.75:
            return "STRONG SUCCESS: Memory backup system shows consistent significant improvements across models"
        elif avg_improvement > 0.05 and improvement_rate >= 0.5:
            return "MODERATE SUCCESS: Memory backup system shows positive trends in most models"
        elif avg_improvement > 0 and improvement_rate >= 0.5:
            return "WEAK SUCCESS: Memory backup system shows marginal improvements"
        elif improvement_rate < 0.5:
            return "MIXED RESULTS: Memory backup benefits vary significantly by model"
        else:
            return "FAILURE: Memory backup system does not provide consistent benefits"


def main():
    """메인 실행 함수"""

    experiment = ComprehensiveModelComparison()

    print("Starting comprehensive model comparison...")
    print(f"Models to test: {experiment.available_models}")
    print(f"Questions per model: {len(experiment.test_questions)}")
    print()

    # 전체 실험 실행
    results = experiment.run_comprehensive_comparison()

    # 최종 결과 출력
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")

    summary = results["summary"]

    print(f"Models tested successfully: {summary['total_models']}")
    print(f"Models failed: {len(results['failed_models'])}")
    if results['failed_models']:
        print(f"Failed models: {', '.join(results['failed_models'])}")

    print(f"\nOverall Performance:")
    print(f"  Average improvement: {summary['average_improvement']:+.3f}")
    print(f"  Models showing improvement: {summary['positive_improvement_rate']:.1%}")
    print(f"  Significant improvements: {summary['significant_improvement_count']}/{summary['total_models']}")

    print(f"\nBest performing: {summary['best_performing_model']['name']} ({summary['best_performing_model']['improvement']:+.3f})")
    print(f"Worst performing: {summary['worst_performing_model']['name']} ({summary['worst_performing_model']['improvement']:+.3f})")

    print(f"\nConclusion: {summary['conclusion']}")

    # 결과 저장
    timestamp = int(time.time())
    filename = f"comprehensive_comparison_results_{timestamp}.json"

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed results saved to: {filename}")


if __name__ == "__main__":
    main()