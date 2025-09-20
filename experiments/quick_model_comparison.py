#!/usr/bin/env python3
"""
빠른 모델 비교 실험 - 메모리 백업 효과 검증
"""

import sys
import os
import time
import json
from typing import Dict

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm.simple_llm import OllamaLLM


class SimpleMemoryManager:
    """간단한 메모리 관리자"""

    def __init__(self):
        self.persistent_memory = []
        self.conversation_history = []

    def add_rule(self, rule: str):
        self.persistent_memory.append(rule)

    def add_conversation(self, message: str):
        self.conversation_history.append(message)
        # 최근 5개만 유지
        if len(self.conversation_history) > 5:
            self.conversation_history = self.conversation_history[-5:]

    def get_context_with_backup(self, question: str) -> str:
        context = []
        if self.persistent_memory:
            context.append("=== IMPORTANT RULES ===")
            for rule in self.persistent_memory:
                context.append(f"- {rule}")
            context.append("")

        if self.conversation_history:
            context.append("=== RECENT CHAT ===")
            context.extend(self.conversation_history[-3:])
            context.append("")

        context.append(f"Question: {question}")
        return "\n".join(context)

    def get_context_without_backup(self, question: str) -> str:
        # 최근 대화만 (규칙 손실)
        context = []
        if self.conversation_history:
            context.extend(self.conversation_history[-3:])
        context.append(f"Question: {question}")
        return "\n".join(context)


def test_single_model(model_name: str, question: str) -> Dict:
    """단일 모델 테스트"""
    print(f"Testing {model_name}...")

    try:
        llm = OllamaLLM(model_name)

        # 메모리 매니저 설정
        memory = SimpleMemoryManager()
        memory.add_rule("Never suggest hardcoding - always use config files")
        memory.add_rule("Always include error handling in code")

        # 잡담으로 컨텍스트 오염
        for i in range(10):
            memory.add_conversation(f"Random chat {i}: What about weather?")

        # 1) 백업 없이 테스트
        context_no_backup = memory.get_context_without_backup(question)
        try:
            response_no_backup = llm.generate(context_no_backup)['response']
        except:
            response_no_backup = "ERROR: Failed to generate"

        # 2) 백업 있이 테스트
        context_with_backup = memory.get_context_with_backup(question)
        try:
            response_with_backup = llm.generate(context_with_backup)['response']
        except:
            response_with_backup = "ERROR: Failed to generate"

        # 간단한 평가
        def score_response(response: str) -> int:
            score = 0
            if len(response) > 50: score += 1  # 충분한 길이
            if "config" in response.lower(): score += 1  # 규칙 반영
            if "error" in response.lower(): score += 1  # 에러 처리
            if response.startswith("ERROR:"): score = 0  # 실패
            return score

        return {
            "model": model_name,
            "question": question,
            "no_backup_response": response_no_backup[:100] + "...",
            "with_backup_response": response_with_backup[:100] + "...",
            "no_backup_score": score_response(response_no_backup),
            "with_backup_score": score_response(response_with_backup),
            "improvement": score_response(response_with_backup) - score_response(response_no_backup)
        }

    except Exception as e:
        return {
            "model": model_name,
            "error": str(e),
            "improvement": 0
        }


def main():
    """메인 실험"""
    print("=== Quick Model Comparison Experiment ===")

    models = ["qwen2:0.5b", "gemma:2b", "llama3:8b", "qwen2:7b"]
    question = "Write a Python function to connect to a database"

    results = []

    for model in models:
        result = test_single_model(model, question)
        results.append(result)

        if "error" not in result:
            print(f"{model}: {result['improvement']:+d} improvement")
        else:
            print(f"{model}: ERROR - {result['error']}")

        time.sleep(1)  # 모델 간 간격

    # 결과 저장
    timestamp = int(time.time())
    filename = f"quick_comparison_{timestamp}.json"

    with open(filename, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "question": question,
            "results": results,
            "summary": {
                "total_models": len(models),
                "improved_models": sum(1 for r in results if r.get("improvement", 0) > 0),
                "average_improvement": sum(r.get("improvement", 0) for r in results) / len(results)
            }
        }, f, indent=2)

    print(f"\n=== SUMMARY ===")
    improved_count = sum(1 for r in results if r.get("improvement", 0) > 0)
    print(f"Models improved: {improved_count}/{len(models)}")
    avg_improvement = sum(r.get("improvement", 0) for r in results) / len(results)
    print(f"Average improvement: {avg_improvement:+.2f}")
    print(f"Results saved: {filename}")

    return results


if __name__ == "__main__":
    main()