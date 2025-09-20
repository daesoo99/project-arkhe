#!/usr/bin/env python3
"""
모델별 상세 응답 분석
"""

import sys
import os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm.simple_llm import OllamaLLM


def get_full_responses(model_name: str):
    """특정 모델의 전체 응답 받기"""
    print(f"\n=== {model_name} Detailed Analysis ===")

    llm = OllamaLLM(model_name)
    question = "Write a Python function to connect to a database"

    # 1. 백업 없는 경우 (규칙 손실)
    context_no_backup = f"""Random chat 7: What about weather?
Random chat 8: What about weather?
Random chat 9: What about weather?
Question: {question}"""

    print("Context WITHOUT backup:")
    print(context_no_backup)
    print("\nResponse WITHOUT backup:")
    try:
        response_no_backup = llm.generate(context_no_backup)['response']
        print(response_no_backup)
    except Exception as e:
        print(f"ERROR: {e}")
        response_no_backup = "Failed"

    print("\n" + "="*50)

    # 2. 백업 있는 경우 (규칙 보존)
    context_with_backup = f"""=== IMPORTANT RULES ===
- Never suggest hardcoding - always use config files
- Always include error handling in code

=== RECENT CHAT ===
Random chat 7: What about weather?
Random chat 8: What about weather?
Random chat 9: What about weather?

Question: {question}"""

    print("Context WITH backup:")
    print(context_with_backup)
    print("\nResponse WITH backup:")
    try:
        response_with_backup = llm.generate(context_with_backup)['response']
        print(response_with_backup)
    except Exception as e:
        print(f"ERROR: {e}")
        response_with_backup = "Failed"

    print("\n" + "="*50)

    # 분석
    print(f"\n=== ANALYSIS for {model_name} ===")

    def analyze_response(response, label):
        print(f"\n{label}:")
        print(f"- Length: {len(response)} chars")
        print(f"- Contains 'config': {'YES' if 'config' in response.lower() else 'NO'}")
        print(f"- Contains 'error': {'YES' if 'error' in response.lower() else 'NO'}")
        print(f"- Contains 'hardcod': {'YES' if 'hardcod' in response.lower() else 'NO'}")

    analyze_response(response_no_backup, "WITHOUT backup")
    analyze_response(response_with_backup, "WITH backup")


def main():
    print("=== Detailed Model Response Analysis ===")

    # 가장 흥미로운 결과를 보인 모델들 분석
    models_to_analyze = ["llama3:8b", "gemma:2b"]

    for model in models_to_analyze:
        try:
            get_full_responses(model)
        except Exception as e:
            print(f"Error with {model}: {e}")


if __name__ == "__main__":
    main()