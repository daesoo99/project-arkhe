"""
LLM 통합 테스트
실제 LLM 호출이 제대로 작동하는지 확인
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agents.hierarchy import CostTracker
from src.agents.economic_agent import EconomicAgent, FixedModelAgent

def test_llm_integration():
    """LLM 통합 테스트 실행"""
    print("=== LLM Integration Test ===\n")
    
    # 테스트 문제들
    test_problems = [
        ("단순", "프랑스의 수도는 어디인가요?"),
        ("중간", "원격근무의 장단점을 비교해주세요."),
        ("복잡", "AI 규제 정책 수립 시 고려해야 할 핵심 요소들을 분석해주세요.")
    ]
    
    for difficulty, problem in test_problems:
        print(f"\n{'='*60}")
        print(f"테스트 문제 ({difficulty}): {problem}")
        print(f"{'='*60}")
        
        # Economic Agent 테스트
        print("\n[ECONOMIC AGENT] - 동적 모델 선택")
        economic_tracker = CostTracker()
        economic_agent = EconomicAgent("Economic_Agent", economic_tracker)
        
        try:
            economic_result = economic_agent.solve(problem)
            economic_cost = economic_tracker.get_total_cost()
            
            print(f"결과: {economic_result[:150]}...")
            print(f"비용: ${economic_cost:.6f}")
            
        except Exception as e:
            print(f"오류 발생: {e}")
        
        # Fixed Model Agent 테스트  
        print("\n[FIXED MODEL AGENT] - 고정 모델")
        fixed_tracker = CostTracker()
        fixed_agent = FixedModelAgent("Fixed_Agent", fixed_tracker)
        
        try:
            fixed_result = fixed_agent.solve(problem)
            fixed_cost = fixed_tracker.get_total_cost()
            
            print(f"결과: {fixed_result[:150]}...")
            print(f"비용: ${fixed_cost:.6f}")
            
            # 비용 비교
            if economic_cost > 0 and fixed_cost > 0:
                cost_diff = ((economic_cost - fixed_cost) / fixed_cost * 100)
                print(f"비용 차이: {cost_diff:+.1f}%")
            
        except Exception as e:
            print(f"오류 발생: {e}")
    
    print(f"\n{'='*60}")
    print("테스트 완료")
    print(f"{'='*60}")

if __name__ == "__main__":
    test_llm_integration()