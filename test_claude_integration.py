from src.agents.hierarchy import CostTracker, Mediator
from src.agents.claude_agent import ClaudeAgent

def test_claude_integration():
    """Claude API 통합 테스트"""
    print("=== Claude API 통합 테스트 ===")
    
    # 비용 추적기 초기화
    cost_tracker = CostTracker()
    
    # Claude 에이전트 생성
    claude_agent = ClaudeAgent("Claude_Thinker", cost_tracker)
    
    # 간단한 테스트 문제들
    test_problems = [
        "2 + 2는 무엇인가요?",
        "프랑스의 수도는 어디인가요?", 
        "AI가 규제되어야 한다고 생각하시나요? 예/아니오로 답해주세요."
    ]
    
    print(f"\n총 {len(test_problems)}개 문제로 테스트 시작...")
    
    for i, problem in enumerate(test_problems, 1):
        print(f"\n--- 문제 {i}: {problem} ---")
        
        try:
            answer = claude_agent.solve(problem)
            print(f"Claude 답변: {answer[:200]}...")
            
        except Exception as e:
            print(f"오류 발생: {e}")
    
    # 총 비용 출력
    total_cost = cost_tracker.get_total_cost()
    print(f"\n--- 테스트 완료 ---")
    print(f"총 예상 비용: ${total_cost:.4f}")

if __name__ == "__main__":
    test_claude_integration()