#!/usr/bin/env python3
"""
빠른 래퍼 테스트 - Windows 콘솔 호환
이모지 없이 ASCII만 사용
"""

import sys
import json
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.hierarchy import CostTracker, Mediator, IndependentThinker

def quick_asymmetry_test():
    """간단한 정보 비대칭 테스트"""
    print("=" * 50)
    print("정보 비대칭 빠른 테스트")
    print("=" * 50)
    
    # 간단한 주관적 질문 2개
    questions = [
        "AI 규제가 필요한가요? 이유를 제시하세요.",
        "원격근무가 더 효율적인가요?"
    ]
    
    cost_tracker = CostTracker()
    
    # 3개의 독립 에이전트
    thinkers = [
        IndependentThinker(f"Agent_{i+1}", cost_tracker, 'gemma:2b')
        for i in range(3)
    ]
    
    mediator = Mediator(thinkers, cost_tracker)
    
    for i, question in enumerate(questions, 1):
        print(f"\n질문 {i}: {question}")
        print("-" * 30)
        
        try:
            result = mediator.solve_problem(question)
            
            print("각 에이전트 답변:")
            for j, response in enumerate(result["all_responses"], 1):
                print(f"  Agent {j}: {response[:80]}...")
            
            diversity = result["shannon_entropy"]
            print(f"\n다양성 지수: {diversity:.2f}")
            
            if diversity > 1.0:
                print("결과: 높은 다양성 - 정보 비대칭 효과 확인")
            elif diversity > 0.5:
                print("결과: 중간 다양성")
            else:
                print("결과: 낮은 다양성")
                
        except Exception as e:
            print(f"오류: {e}")
    
    total_cost = cost_tracker.get_total_cost()
    print(f"\n총 비용: ${total_cost:.6f}")

def quick_integrated_test():
    """간단한 통합 시스템 테스트"""
    print("\n" + "=" * 50)
    print("통합 시스템 빠른 테스트")
    print("=" * 50)
    
    # 간단한 사실 질문 2개
    questions = [
        "프랑스의 수도는?",
        "2 + 2는?"
    ]
    
    # 1. 단일 에이전트 테스트
    print("\n단일 에이전트:")
    single_cost_tracker = CostTracker()
    single_agent = IndependentThinker("Single", single_cost_tracker, 'gemma:2b')
    
    single_results = []
    for question in questions:
        try:
            response = single_agent.solve(question)
            single_results.append(response[:50])
            print(f"  {question} -> {response[:50]}...")
        except Exception as e:
            print(f"  {question} -> 오류: {e}")
    
    single_cost = single_cost_tracker.get_total_cost()
    print(f"단일 에이전트 비용: ${single_cost:.6f}")
    
    # 2. 멀티 에이전트 테스트
    print("\n멀티 에이전트 (Arkhe 기본):")
    multi_cost_tracker = CostTracker()
    thinkers = [
        IndependentThinker(f"Multi_{i+1}", multi_cost_tracker, 'gemma:2b')
        for i in range(3)
    ]
    mediator = Mediator(thinkers, multi_cost_tracker)
    
    multi_results = []
    for question in questions:
        try:
            result = mediator.solve_problem(question)
            response = result["final_answer"]
            diversity = result["shannon_entropy"]
            multi_results.append(response[:50])
            print(f"  {question} -> {response[:50]}... (다양성: {diversity:.2f})")
        except Exception as e:
            print(f"  {question} -> 오류: {e}")
    
    multi_cost = multi_cost_tracker.get_total_cost()
    print(f"멀티 에이전트 비용: ${multi_cost:.6f}")
    
    # 비교
    cost_ratio = multi_cost / single_cost if single_cost > 0 else 0
    print(f"\n비용 비율: {cost_ratio:.1f}x (멀티/단일)")

def main():
    """메인 실행 함수"""
    print("Project Arkhe - 빠른 래퍼 테스트")
    
    try:
        quick_asymmetry_test()
        quick_integrated_test()
        
        print("\n" + "=" * 50)
        print("빠른 테스트 완료!")
        print("더 상세한 테스트는 bench_simple.py 사용")
        
    except Exception as e:
        print(f"테스트 중 오류: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())