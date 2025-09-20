#!/usr/bin/env python3
"""
실제 AI 연동 실험: 정보 혼재 vs 컨텍스트 분리 성능 비교
Project Arkhē LLM 인프라를 사용하여 실제 AI 응답 품질 측정
"""

import sys
import os
import time
import json
from typing import Dict, List, Tuple, Any

# Project Arkhē 기존 인프라 import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm.simple_llm import OllamaLLM
from multiroom.version_workspace_manager import (
    VersionWorkspaceManager, ContextPriority, WorkspaceConcept, WorkspaceType
)
# Simple Shannon entropy calculation for this experiment
import math
from collections import Counter


def calculate_shannon_entropy(text: str) -> float:
    """Calculate Shannon entropy of text"""
    if not text:
        return 0.0

    words = text.lower().split()
    if not words:
        return 0.0

    word_counts = Counter(words)
    total_words = len(words)

    entropy = 0.0
    for count in word_counts.values():
        probability = count / total_words
        if probability > 0:
            entropy -= probability * math.log2(probability)

    return entropy


class ContextMixingExperiment:
    """정보 혼재 vs 분리 실험 클래스"""

    def __init__(self, model_name: str = "gemma:2b"):
        self.llm = OllamaLLM(model_name)
        self.workspace_manager = VersionWorkspaceManager()
        self.results = []

    def setup_complex_conversation_history(self):
        """복잡한 대화 히스토리 구성 (실제 상황 시뮬레이션)"""

        # 1달 전: 프로젝트 설정 (PERSISTENT)
        self.workspace_manager.switch_workspace("v1_research")

        persistent_rules = [
            "Code quality is top priority - no hardcoding ever",
            "All experiments must be logged in EXPERIMENT_LOG.md",
            "Follow CLAUDE.local rules strictly",
            "Registry pattern for all model management"
        ]

        for rule in persistent_rules:
            self.workspace_manager.add_message_smart(
                rule, role="system",
                priority=ContextPriority.PERSISTENT,
                is_rule=True
            )

        # 2주 전: 아키텍처 결정들 (REFERENCE)
        architecture_decisions = [
            "DECISION: Use Shannon Entropy for information measurement",
            "DECISION: Multi-Agent system shows advantages in creative tasks",
            "DECISION: Registry pattern prevents hardcoding effectively",
            "DECISION: Token optimization is crucial for performance"
        ]

        for decision in architecture_decisions:
            self.workspace_manager.add_message_smart(
                decision, role="assistant",
                priority=ContextPriority.REFERENCE,
                is_decision=True
            )

        # 1주 전: 실험 대화들 (ACTIVE -> 시간 지나면 ARCHIVED)
        experiment_conversations = [
            ("What's the current Multi-Agent vs Single model performance?", "user"),
            ("Multi-Agent shows 75% accuracy vs Single 60% in complex reasoning", "assistant"),
            ("Should we optimize token usage for better efficiency?", "user"),
            ("Yes, token optimization reduced costs by 40% without quality loss", "assistant"),
            ("How does Shannon Entropy help in our pipeline?", "user"),
            ("Shannon Entropy measures information content, helps balance complexity", "assistant"),
            ("What about creative collaboration experiments?", "user"),
            ("Creative tasks show 2.2x word diversity with Multi-Agent approach", "assistant"),
            ("Is the Registry pattern working well?", "user"),
            ("Registry pattern eliminated all hardcoding issues successfully", "assistant")
        ]

        for content, role in experiment_conversations:
            self.workspace_manager.add_message_smart(
                content, role=role,
                expires_hours=168  # 1주일 후 만료
            )

        # 어제: 버그 관련 임시 대화들 (ACTIVE -> 빨리 만료)
        bug_conversations = [
            ("Error: Import path not found in module X", "user"),
            ("Fixed: Updated sys.path to include missing directory", "assistant"),
            ("Still getting TypeError in function Y", "user"),
            ("Applied workaround: Type checking bypass for compatibility", "assistant"),
            ("Performance degradation in pipeline Z", "user"),
            ("Temporary fix: Reduced batch size from 100 to 50", "assistant"),
            ("Memory leak detected in long-running tasks", "user"),
            ("Quick patch: Added garbage collection after each iteration", "assistant")
        ]

        for content, role in bug_conversations:
            self.workspace_manager.add_message_smart(
                content, role=role,
                expires_hours=24  # 24시간 후 만료
            )

        # 시간 경과 시뮬레이션 (오래된 메시지들 아카이브)
        self.workspace_manager.auto_archive_old_messages(hours_threshold=1)

    def create_test_questions(self) -> List[Dict[str, Any]]:
        """실험용 질문들 생성"""
        return [
            {
                "question": "Write a simple Python function to calculate the average of a list of numbers",
                "type": "simple_coding",
                "expected_focus": "Just the function implementation, no complex context needed"
            },
            {
                "question": "How should I structure a new feature in this codebase?",
                "type": "architecture",
                "expected_focus": "Architecture rules and patterns, but not old bug discussions"
            },
            {
                "question": "What's our policy on code quality and hardcoding?",
                "type": "policy",
                "expected_focus": "Core rules should be maintained, recent experiments irrelevant"
            },
            {
                "question": "Explain the benefits of Shannon Entropy in our system",
                "type": "explanation",
                "expected_focus": "Past decisions relevant, but not recent bug fixes"
            },
            {
                "question": "I'm getting an import error, how to fix it?",
                "type": "debugging",
                "expected_focus": "Recent bug context might be relevant, old experiments not"
            }
        ]

    def get_mixed_context(self) -> str:
        """기존 방식: 모든 정보가 섞인 컨텍스트"""
        all_messages = []

        # 메인 스레드 모든 메시지
        all_messages.extend(self.workspace_manager.main_thread)

        # 현재 워크스페이스 모든 메시지
        current_workspace = self.workspace_manager.get_current_workspace()
        if current_workspace:
            all_messages.extend(current_workspace.messages)

        # 최근 20개 메시지 (시간순)
        all_messages.sort(key=lambda x: x.timestamp)
        recent_messages = all_messages[-20:]

        context_parts = ["=== CONVERSATION HISTORY (ALL MIXED) ==="]
        for msg in recent_messages:
            if not msg.is_expired():
                context_parts.append(f"{msg.role}: {msg.content}")

        return "\n".join(context_parts)

    def get_separated_context(self) -> str:
        """새로운 방식: 스마트 분리된 컨텍스트"""
        return self.workspace_manager.get_smart_context()

    def ask_ai_with_context(self, question: str, context: str) -> Dict[str, Any]:
        """AI에게 컨텍스트와 함께 질문"""

        full_prompt = f"""
{context}

===== CURRENT QUESTION =====
{question}

Please provide a helpful response focused on the current question.
"""

        start_time = time.time()

        try:
            response = self.llm.generate(full_prompt)
            end_time = time.time()

            response_text = response.get('response', '')

            return {
                "response": response_text,
                "response_time": end_time - start_time,
                "token_count": len(full_prompt.split()) + len(response_text.split()),
                "success": True,
                "prompt_length": len(full_prompt),
                "response_length": len(response_text)
            }

        except Exception as e:
            return {
                "response": f"Error: {str(e)}",
                "response_time": time.time() - start_time,
                "token_count": 0,
                "success": False,
                "error": str(e)
            }

    def evaluate_response_quality(self, question_info: Dict, response_data: Dict) -> Dict[str, float]:
        """응답 품질 평가"""

        response = response_data.get("response", "")
        question_type = question_info.get("type", "")

        scores = {}

        # 1. 관련성 점수 (키워드 기반)
        if question_type == "simple_coding":
            relevant_keywords = ["def", "function", "average", "return", "list"]
            irrelevant_keywords = ["experiment", "decision", "bug", "error", "architecture"]
        elif question_type == "policy":
            relevant_keywords = ["hardcoding", "quality", "rule", "policy"]
            irrelevant_keywords = ["bug", "error", "import", "fix"]
        elif question_type == "debugging":
            relevant_keywords = ["import", "error", "fix", "path"]
            irrelevant_keywords = ["experiment", "multi-agent", "entropy"]
        else:
            relevant_keywords = []
            irrelevant_keywords = []

        relevant_count = sum(1 for kw in relevant_keywords if kw.lower() in response.lower())
        irrelevant_count = sum(1 for kw in irrelevant_keywords if kw.lower() in response.lower())

        total_keywords = len(relevant_keywords) + len(irrelevant_keywords)
        if total_keywords > 0:
            scores["relevance"] = (relevant_count - irrelevant_count * 0.5) / len(relevant_keywords) if relevant_keywords else 0
            scores["focus"] = max(0, 1 - (irrelevant_count / len(irrelevant_keywords)) if irrelevant_keywords else 1)
        else:
            scores["relevance"] = 0.5
            scores["focus"] = 0.5

        # 2. 간결성 점수 (불필요한 길이 페널티)
        optimal_length = 200  # 적정 응답 길이
        actual_length = len(response)

        if actual_length <= optimal_length:
            scores["conciseness"] = 1.0
        else:
            scores["conciseness"] = max(0.1, optimal_length / actual_length)

        # 3. 정보 엔트로피 (복잡도 측정)
        try:
            scores["entropy"] = calculate_shannon_entropy(response)
        except:
            scores["entropy"] = 0.5

        # 4. 전체 품질 점수
        scores["overall_quality"] = (
            scores["relevance"] * 0.4 +
            scores["focus"] * 0.3 +
            scores["conciseness"] * 0.2 +
            (1 - min(scores["entropy"], 1.0)) * 0.1  # 낮은 엔트로피가 더 좋음
        )

        return scores

    def run_experiment(self) -> Dict[str, Any]:
        """전체 실험 실행"""

        print("=== Context Mixing vs Separation AI Experiment ===\n")

        # 1. 복잡한 대화 히스토리 구성
        print("Step 1: Setting up complex conversation history...")
        self.setup_complex_conversation_history()
        print("Created 4 persistent rules, 4 architecture decisions, 10 experiment conversations, 8 bug discussions")

        # 2. 테스트 질문들
        test_questions = self.create_test_questions()
        print(f"Prepared {len(test_questions)} test questions")

        # 3. 각 질문에 대해 두 방식으로 실험
        mixed_results = []
        separated_results = []

        print("\nStep 2: Running AI response tests...")

        for i, question_info in enumerate(test_questions):
            question = question_info["question"]
            print(f"\nQuestion {i+1}: {question[:50]}...")

            # A) 혼재된 컨텍스트로 테스트
            mixed_context = self.get_mixed_context()
            mixed_response = self.ask_ai_with_context(question, mixed_context)
            mixed_scores = self.evaluate_response_quality(question_info, mixed_response)

            mixed_result = {
                "question": question_info,
                "context_type": "mixed",
                "context_length": len(mixed_context),
                "response_data": mixed_response,
                "quality_scores": mixed_scores
            }
            mixed_results.append(mixed_result)

            # B) 분리된 컨텍스트로 테스트
            separated_context = self.get_separated_context()
            separated_response = self.ask_ai_with_context(question, separated_context)
            separated_scores = self.evaluate_response_quality(question_info, separated_response)

            separated_result = {
                "question": question_info,
                "context_type": "separated",
                "context_length": len(separated_context),
                "response_data": separated_response,
                "quality_scores": separated_scores
            }
            separated_results.append(separated_result)

            print(f"  Mixed Quality: {mixed_scores['overall_quality']:.2f}")
            print(f"  Separated Quality: {separated_scores['overall_quality']:.2f}")
            print(f"  Improvement: {separated_scores['overall_quality'] - mixed_scores['overall_quality']:.2f}")

        # 4. 결과 분석
        print("\nStep 3: Analyzing results...")

        avg_mixed_quality = sum(r["quality_scores"]["overall_quality"] for r in mixed_results) / len(mixed_results)
        avg_separated_quality = sum(r["quality_scores"]["overall_quality"] for r in separated_results) / len(separated_results)

        avg_mixed_context_length = sum(r["context_length"] for r in mixed_results) / len(mixed_results)
        avg_separated_context_length = sum(r["context_length"] for r in separated_results) / len(separated_results)

        improvement = avg_separated_quality - avg_mixed_quality
        context_reduction = (avg_mixed_context_length - avg_separated_context_length) / avg_mixed_context_length

        # 5. 상세 결과 출력
        print(f"\n=== EXPERIMENT RESULTS ===")
        print(f"Average Quality Scores:")
        print(f"  Mixed Context: {avg_mixed_quality:.3f}")
        print(f"  Separated Context: {avg_separated_quality:.3f}")
        print(f"  Improvement: +{improvement:.3f} ({improvement/avg_mixed_quality*100:+.1f}%)")

        print(f"\nContext Efficiency:")
        print(f"  Mixed Avg Length: {avg_mixed_context_length:.0f} chars")
        print(f"  Separated Avg Length: {avg_separated_context_length:.0f} chars")
        print(f"  Context Reduction: {context_reduction:.1%}")

        print(f"\nDetailed Breakdown:")
        for i, (mixed, separated) in enumerate(zip(mixed_results, separated_results)):
            q_type = mixed["question"]["type"]
            improvement = separated["quality_scores"]["overall_quality"] - mixed["quality_scores"]["overall_quality"]
            print(f"  Q{i+1} ({q_type}): {improvement:+.3f}")

        return {
            "mixed_results": mixed_results,
            "separated_results": separated_results,
            "summary": {
                "avg_mixed_quality": avg_mixed_quality,
                "avg_separated_quality": avg_separated_quality,
                "improvement": improvement,
                "improvement_percentage": improvement/avg_mixed_quality*100,
                "context_reduction": context_reduction,
                "avg_mixed_context_length": avg_mixed_context_length,
                "avg_separated_context_length": avg_separated_context_length
            }
        }


def main():
    """실험 실행 메인 함수"""

    # Ollama 서버 확인
    try:
        experiment = ContextMixingExperiment("gemma:2b")

        # 간단한 연결 테스트
        test_response = experiment.llm.generate("Hello")
        if not test_response.get('response'):
            print("❌ Ollama server not accessible. Please start Ollama first.")
            return

    except Exception as e:
        print(f"❌ Failed to connect to Ollama: {e}")
        print("Please ensure Ollama is running with 'ollama serve'")
        return

    # 실험 실행
    print("Starting Context Mixing vs Separation Experiment...")
    print("This will test if context separation actually improves AI responses\n")

    results = experiment.run_experiment()

    # 결과 저장
    timestamp = int(time.time())
    results_file = f"context_experiment_results_{timestamp}.json"

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed results saved to: {results_file}")

    # 최종 결론
    improvement = results["summary"]["improvement"]
    if improvement > 0.1:
        print(f"\nSUCCESS! Context separation shows significant improvement (+{improvement:.3f})")
        print("Your idea of preventing information mixing is validated!")
    elif improvement > 0.05:
        print(f"\nPOSITIVE! Context separation shows moderate improvement (+{improvement:.3f})")
        print("Your approach is working in the right direction!")
    else:
        print(f"\nMIXED RESULTS: Improvement is minimal (+{improvement:.3f})")
        print("May need further refinement of the separation strategy")


if __name__ == "__main__":
    main()