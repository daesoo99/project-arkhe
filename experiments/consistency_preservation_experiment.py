#!/usr/bin/env python3
"""
일관성 보존 실험 - 진짜 문제 해결
채팅이 길어져도 과거 중요 정보를 백업해서 AI가 일관성을 유지하는지 테스트
"""

import sys
import os
import time
import json
from typing import Dict, List, Any

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm.simple_llm import OllamaLLM


class ConsistencyMemoryManager:
    """일관성 보존을 위한 메모리 관리자"""

    def __init__(self):
        self.persistent_memory = []  # 영구 보존 정보
        self.conversation_history = []  # 일반 대화 히스토리
        self.max_context_length = 1000  # 컨텍스트 길이 제한

    def add_persistent_info(self, info: str, category: str = "rule"):
        """영구 보존할 중요 정보 추가"""
        self.persistent_memory.append({
            "content": info,
            "category": category,
            "timestamp": time.time()
        })

    def add_conversation(self, message: str, role: str = "user"):
        """일반 대화 추가"""
        self.conversation_history.append({
            "content": message,
            "role": role,
            "timestamp": time.time()
        })

        # 컨텍스트가 너무 길어지면 오래된 대화 제거 (중요 정보는 보존)
        self._trim_conversation_if_needed()

    def _trim_conversation_if_needed(self):
        """대화가 너무 길면 오래된 것 제거 (백업된 중요 정보는 유지)"""
        total_length = self._estimate_total_length()

        if total_length > self.max_context_length:
            # 최근 대화만 유지 (중요 정보는 persistent_memory에 있으므로 안전)
            self.conversation_history = self.conversation_history[-10:]  # 최근 10개만

    def _estimate_total_length(self) -> int:
        """전체 컨텍스트 길이 추정"""
        total = 0

        # 영구 메모리
        for item in self.persistent_memory:
            total += len(item["content"].split())

        # 대화 히스토리
        for item in self.conversation_history:
            total += len(item["content"].split())

        return total

    def get_context_for_ai(self, current_question: str) -> str:
        """AI에게 제공할 컨텍스트 구성"""
        context_parts = []

        # 1. 영구 보존 정보 (항상 포함)
        if self.persistent_memory:
            context_parts.append("=== IMPORTANT RULES (ALWAYS REMEMBER) ===")
            for item in self.persistent_memory:
                context_parts.append(f"- {item['content']}")
            context_parts.append("")

        # 2. 최근 대화 히스토리
        if self.conversation_history:
            context_parts.append("=== RECENT CONVERSATION ===")
            for item in self.conversation_history[-5:]:  # 최근 5개
                context_parts.append(f"{item['role']}: {item['content']}")
            context_parts.append("")

        # 3. 현재 질문
        context_parts.append("=== CURRENT QUESTION ===")
        context_parts.append(current_question)

        return "\n".join(context_parts)

    def get_stats(self) -> Dict[str, Any]:
        """메모리 통계"""
        return {
            "persistent_items": len(self.persistent_memory),
            "conversation_items": len(self.conversation_history),
            "total_estimated_tokens": self._estimate_total_length(),
            "persistent_memory": self.persistent_memory,
            "recent_conversations": self.conversation_history[-3:]
        }


class ConsistencyExperiment:
    """일관성 유지 실험"""

    def __init__(self, model: str = "gemma:2b"):
        self.llm = OllamaLLM(model)
        self.model_name = model

    def run_consistency_test(self) -> Dict[str, Any]:
        """일관성 유지 테스트 실행"""

        print(f"=== Consistency Preservation Test ({self.model_name}) ===\n")

        # A) 기존 방식 (메모리 백업 없음)
        print("1. Testing WITHOUT memory backup...")
        no_backup_results = self._test_without_backup()

        # B) 새로운 방식 (메모리 백업 있음)
        print("\n2. Testing WITH memory backup...")
        with_backup_results = self._test_with_backup()

        # 결과 비교
        print("\n3. Comparing results...")
        comparison = self._compare_results(no_backup_results, with_backup_results)

        return {
            "model": self.model_name,
            "no_backup": no_backup_results,
            "with_backup": with_backup_results,
            "comparison": comparison
        }

    def _test_without_backup(self) -> Dict[str, Any]:
        """백업 없는 기존 방식 테스트"""

        # 초기 규칙 설정
        conversation = []
        conversation.append("Remember: You must always follow these rules:")
        conversation.append("1. Never suggest hardcoding - always use config files")
        conversation.append("2. All experiments must be logged to EXPERIMENT_LOG.md")
        conversation.append("3. Registry pattern must be used for all models")

        # 50개의 일반 대화로 컨텍스트 오염
        distraction_topics = [
            "What's the weather like?",
            "Can you help me with math homework?",
            "Tell me about machine learning",
            "How to cook pasta?",
            "Explain quantum physics",
            "What's your favorite color?",
            "Help me debug this SQL query",
            "Translate this to Spanish",
            "Write a poem about cats",
            "Explain cryptocurrency"
        ]

        for i in range(50):
            topic = distraction_topics[i % len(distraction_topics)]
            conversation.append(f"Q{i+1}: {topic}")
            conversation.append(f"A{i+1}: [Mock response about {topic}]")

        # 컨텍스트 길이 제한 시뮬레이션 (최근 20개만 유지)
        limited_context = conversation[-20:]

        # 규칙 준수 테스트
        test_question = "Write a Python function to connect to a database"
        context = "\n".join(limited_context) + f"\n\nQ: {test_question}"

        response = self._get_ai_response(context, test_question)

        # 규칙 준수율 체크
        rule_compliance = self._check_rule_compliance(response)

        return {
            "total_conversations": len(conversation),
            "context_length": len(limited_context),
            "test_question": test_question,
            "ai_response": response,
            "rule_compliance": rule_compliance,
            "rules_remembered": len([r for r in rule_compliance.values() if r])
        }

    def _test_with_backup(self) -> Dict[str, Any]:
        """백업 있는 새로운 방식 테스트"""

        memory_manager = ConsistencyMemoryManager()

        # 중요 규칙들을 영구 메모리에 백업
        memory_manager.add_persistent_info("Never suggest hardcoding - always use config files", "rule")
        memory_manager.add_persistent_info("All experiments must be logged to EXPERIMENT_LOG.md", "rule")
        memory_manager.add_persistent_info("Registry pattern must be used for all models", "rule")

        # 50개의 일반 대화 (메모리 관리자가 자동으로 trim)
        distraction_topics = [
            "What's the weather like?",
            "Can you help me with math homework?",
            "Tell me about machine learning",
            "How to cook pasta?",
            "Explain quantum physics",
            "What's your favorite color?",
            "Help me debug this SQL query",
            "Translate this to Spanish",
            "Write a poem about cats",
            "Explain cryptocurrency"
        ]

        for i in range(50):
            topic = distraction_topics[i % len(distraction_topics)]
            memory_manager.add_conversation(f"Q{i+1}: {topic}", "user")
            memory_manager.add_conversation(f"[Mock response about {topic}]", "assistant")

        # 규칙 준수 테스트
        test_question = "Write a Python function to connect to a database"
        context = memory_manager.get_context_for_ai(test_question)

        response = self._get_ai_response(context, test_question)

        # 규칙 준수율 체크
        rule_compliance = self._check_rule_compliance(response)

        return {
            "total_conversations": 100,  # 50 Q + 50 A
            "context_length": len(context.split()),
            "test_question": test_question,
            "ai_response": response,
            "rule_compliance": rule_compliance,
            "rules_remembered": len([r for r in rule_compliance.values() if r]),
            "memory_stats": memory_manager.get_stats()
        }

    def _get_ai_response(self, context: str, question: str) -> str:
        """AI 응답 받기"""
        try:
            result = self.llm.generate(context)
            return result.get('response', 'No response')
        except Exception as e:
            return f"Error: {str(e)}"

    def _check_rule_compliance(self, response: str) -> Dict[str, bool]:
        """규칙 준수 여부 확인"""
        response_lower = response.lower()

        return {
            "no_hardcoding_mentioned": "config" in response_lower or "configuration" in response_lower,
            "no_hardcoding_avoided": "hardcode" not in response_lower or "hard-code" not in response_lower,
            "logging_mentioned": "log" in response_lower or "experiment" in response_lower,
            "registry_pattern": "registry" in response_lower or "pattern" in response_lower
        }

    def _compare_results(self, no_backup: Dict, with_backup: Dict) -> Dict[str, Any]:
        """결과 비교"""

        no_backup_score = no_backup["rules_remembered"]
        with_backup_score = with_backup["rules_remembered"]

        improvement = with_backup_score - no_backup_score
        improvement_pct = (improvement / max(no_backup_score, 1)) * 100

        return {
            "no_backup_rules_remembered": no_backup_score,
            "with_backup_rules_remembered": with_backup_score,
            "improvement": improvement,
            "improvement_percentage": improvement_pct,
            "context_efficiency": {
                "no_backup_context_length": no_backup["context_length"],
                "with_backup_context_length": with_backup["context_length"],
                "efficiency_gain": no_backup["context_length"] - with_backup["context_length"]
            }
        }


def main():
    """메인 실험 실행"""

    print("=== Consistency Preservation Experiment ===")
    print("Testing if memory backup preserves AI consistency in long conversations\n")

    # 실험 실행
    experiment = ConsistencyExperiment("gemma:2b")
    results = experiment.run_consistency_test()

    # 결과 출력
    print("\n=== FINAL RESULTS ===")
    comp = results["comparison"]

    print(f"Model: {results['model']}")
    print(f"Rules Remembered:")
    print(f"  Without Backup: {comp['no_backup_rules_remembered']}/4 rules")
    print(f"  With Backup: {comp['with_backup_rules_remembered']}/4 rules")
    print(f"  Improvement: +{comp['improvement']} rules ({comp['improvement_percentage']:+.1f}%)")

    print(f"\nContext Efficiency:")
    eff = comp["context_efficiency"]
    print(f"  Without Backup: {eff['no_backup_context_length']} words")
    print(f"  With Backup: {eff['with_backup_context_length']} words")
    print(f"  Efficiency Gain: {eff['efficiency_gain']} words saved")

    # 상세 응답 비교
    print(f"\n=== RESPONSE COMPARISON ===")
    print(f"Question: {results['no_backup']['test_question']}")

    print(f"\nWithout Backup Response:")
    print(f"'{results['no_backup']['ai_response'][:200]}...'")

    print(f"\nWith Backup Response:")
    print(f"'{results['with_backup']['ai_response'][:200]}...'")

    # 결과 저장
    timestamp = int(time.time())
    filename = f"consistency_experiment_{timestamp}.json"

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed results saved to: {filename}")

    # 결론
    if comp["improvement"] > 0:
        print(f"\nSUCCESS! Memory backup improved consistency by {comp['improvement']} rules")
        print("Your idea of preserving important information works!")
    elif comp["improvement"] == 0:
        print(f"\nNEUTRAL: Both methods performed equally well")
        print("Memory backup didn't hurt, but benefits may show in longer conversations")
    else:
        print(f"\nUNEXPECTED: Backup method performed worse")
        print("May need to adjust the memory management strategy")


if __name__ == "__main__":
    main()