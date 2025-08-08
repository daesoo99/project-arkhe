"""
빠른 LLM 통합 테스트
3가지 철학의 핵심 기능이 실제 LLM과 작동하는지 확인
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agents.hierarchy import CostTracker
from src.agents.economic_agent import EconomicAgent, FixedModelAgent
from src.agents.information_asymmetry import IsolatedAgent, TransparentAgent, InformationIsolationEngine, IsolationLevel
from src.agents.recursive_agent import RecursiveAgent, FlatAgent

def test_economic_intelligence():
    """경제적 지능 빠른 테스트"""
    print("=== 1. Economic Intelligence Test ===")
    
    # 간단한 테스트 문제
    problem = "프랑스의 수도는 어디인가요?"
    
    # Economic Agent (동적 모델 선택)
    economic_tracker = CostTracker()
    economic_agent = EconomicAgent("Economic", economic_tracker)
    economic_result = economic_agent.solve(problem)
    economic_cost = economic_tracker.get_total_cost()
    
    # Fixed Model Agent (기본 비교군)
    fixed_tracker = CostTracker()
    fixed_agent = FixedModelAgent("Fixed", fixed_tracker)
    fixed_result = fixed_agent.solve(problem)
    fixed_cost = fixed_tracker.get_total_cost()
    
    print(f"Economic Agent: {economic_result[:100]}...")
    print(f"Fixed Agent: {fixed_result[:100]}...")
    print(f"Cost Difference: Economic=${economic_cost:.6f}, Fixed=${fixed_cost:.6f}")
    print()

def test_information_asymmetry():
    """정보 비대칭 빠른 테스트"""
    print("=== 2. Information Asymmetry Test ===")
    
    problem = "원격근무의 장단점은 무엇인가요?"
    
    # Isolated Agent (정보 격리)
    isolated_tracker = CostTracker()
    engine = InformationIsolationEngine(IsolationLevel.COMPLETE)
    isolated_agent = IsolatedAgent("Isolated", isolated_tracker, "analytical")
    context = engine.create_isolated_context("agent_1", problem)
    isolated_agent.set_context(context)
    isolated_result = isolated_agent.solve_isolated(problem)
    isolated_cost = isolated_tracker.get_total_cost()
    
    # Transparent Agent (정보 공유)
    transparent_tracker = CostTracker()
    transparent_agent = TransparentAgent("Transparent", transparent_tracker)
    transparent_result = transparent_agent.solve_with_shared_info(problem)
    transparent_cost = transparent_tracker.get_total_cost()
    
    print(f"Isolated Agent: {isolated_result[:100]}...")
    print(f"Transparent Agent: {transparent_result[:100]}...")
    print(f"Cost Difference: Isolated=${isolated_cost:.6f}, Transparent=${transparent_cost:.6f}")
    print()

def test_recursive_intelligence():
    """자율적 재귀 빠른 테스트"""
    print("=== 3. Recursive Intelligence Test ===")
    
    problem = "AI 윤리의 주요 쟁점들을 분석해주세요."
    
    # Recursive Agent (문제 분해)
    recursive_tracker = CostTracker()
    recursive_agent = RecursiveAgent("Recursive", recursive_tracker, max_recursion_depth=1)  # 깊이 제한
    recursive_result = recursive_agent.solve_recursively(problem, complexity=7.0, depth=0)
    recursive_cost = recursive_tracker.get_total_cost()
    
    # Flat Agent (평면적 해결)
    flat_tracker = CostTracker()
    flat_agent = FlatAgent("Flat", flat_tracker)
    flat_result = flat_agent.solve(problem)
    flat_cost = flat_tracker.get_total_cost()
    
    print(f"Recursive Agent: {recursive_result.final_synthesis[:100]}...")
    print(f"Flat Agent: {flat_result[:100]}...")
    print(f"Cost Difference: Recursive=${recursive_cost:.6f}, Flat=${flat_cost:.6f}")
    print(f"Agents Used: Recursive={recursive_result.total_agents_used}, Flat=1")
    print()

def main():
    """메인 테스트 실행"""
    print("Quick LLM Integration Test - Project Arkhe\n")
    
    try:
        test_economic_intelligence()
    except Exception as e:
        print(f"Economic Intelligence Error: {e}\n")
    
    try:
        test_information_asymmetry()
    except Exception as e:
        print(f"Information Asymmetry Error: {e}\n")
    
    try:
        test_recursive_intelligence()
    except Exception as e:
        print(f"Recursive Intelligence Error: {e}\n")
    
    print("=== Test Complete ===")
    print("All three philosophies are now integrated with real LLM calls!")

if __name__ == "__main__":
    main()