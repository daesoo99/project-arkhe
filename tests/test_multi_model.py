"""
멀티 모델 비교 테스트
- Claude (Mock)
- GPT-OSS (Mock/Real)  
- Gemma (Ollama)
"""

from src.agents.hierarchy import CostTracker, Mediator
from src.agents.gpt_oss_agent import GPTOSSAgent, MockGPTOSSAgent

def mock_claude_solve(problem):
    """Claude Mock 응답 (실제 API 대신)"""
    mock_responses = {
        "capital": "파리입니다. 프랑스의 수도는 파리로, 센 강 유역에 위치한 역사적인 도시입니다.",
        "earth": "아니오. 지구는 구형입니다. 과학적 증거들이 이를 명확히 뒷받침합니다.",
        "math": "4입니다. 2 + 2 = 4는 기본적인 산술 연산입니다.",
        "regulation": "예. AI 기술의 급속한 발전을 고려할 때 적절한 규제 프레임워크가 필요합니다.",
        "climate": "주요 원인은 인간 활동으로 인한 온실가스 배출입니다. 특히 화석연료 사용이 가장 큰 요인입니다."
    }
    
    problem_lower = problem.lower()
    for keyword, answer in mock_responses.items():
        if keyword == "capital" and any(word in problem_lower for word in ["수도", "capital", "프랑스", "france"]):
            return f"[Claude Mock] {answer}"
        elif keyword == "earth" and any(word in problem_lower for word in ["flat", "평평", "지구", "earth"]):
            return f"[Claude Mock] {answer}"
        elif keyword == "math" and any(word in problem_lower for word in ["2+2", "2 + 2", "더하기"]):
            return f"[Claude Mock] {answer}"
        elif keyword == "regulation" and any(word in problem_lower for word in ["regulation", "규제", "ai"]):
            return f"[Claude Mock] {answer}"
        elif keyword == "climate" and any(word in problem_lower for word in ["climate", "기후", "변화"]):
            return f"[Claude Mock] {answer}"
    
    return f"[Claude Mock] 질문을 이해했지만 구체적인 답변을 제공하기 어렵습니다."

class MockClaudeAgent:
    """Claude Mock 에이전트"""
    def __init__(self, name, cost_tracker):
        self.name = name
        self.cost_tracker = cost_tracker
        
    def solve(self, problem):
        response = mock_claude_solve(problem)
        # Claude 가격 적용
        estimated_tokens = len(problem.split()) + len(response.split())
        self.cost_tracker.add_cost("claude-3-5-sonnet", estimated_tokens // 2, estimated_tokens // 2)
        return response

def test_multi_model():
    """멀티 모델 비교 테스트 실행"""
    print("=== Multi-Model Comparison Test ===\n")
    
    cost_tracker = CostTracker()
    
    # 다양한 모델 에이전트 생성
    claude_agent = MockClaudeAgent("Claude_Mock", cost_tracker)
    gpt_oss_agent = MockGPTOSSAgent("GPT_OSS_Mock", cost_tracker)
    
    # 중재자 생성
    mediator = Mediator([claude_agent, gpt_oss_agent], cost_tracker)
    
    # 테스트 문제들
    test_problems = [
        "프랑스의 수도는 어디인가요?",
        "지구는 평평한가요? 예/아니오로 답해주세요.",
        "2 + 2는 무엇인가요?",
        "AI가 규제되어야 한다고 생각하시나요? 예/아니오로 답해주세요.",
        "기후 변화의 주요 원인은 무엇인가요?"
    ]
    
    print(f"Testing with {len(test_problems)} problems...\n")
    
    results = []
    
    for i, problem in enumerate(test_problems, 1):
        print(f"{'='*60}")
        print(f"Problem {i}: {problem}")
        print(f"{'='*60}")
        
        try:
            result = mediator.solve_problem(problem)
            results.append(result)
            
            print(f"\nFinal Answer: {result['final_answer']}")
            print(f"Diversity Score (Shannon Entropy): {result['shannon_entropy']:.3f}")
            print(f"Contradiction Detection: {result['contradiction_report']}")
            
            print(f"\nIndividual Agent Responses:")
            for j, response in enumerate(result['all_responses']):
                print(f"   {j+1}. {response}")
            
        except Exception as e:
            print(f"Error occurred: {e}")
        
        print()
    
    # 최종 비용 및 성능 분석
    total_cost = cost_tracker.get_total_cost()
    print(f"\n{'='*60}")
    print(f"Test Complete - Summary")
    print(f"{'='*60}")
    print(f"Total Estimated Cost: ${total_cost:.6f}")
    print(f"Problems Processed: {len(results)}")
    
    # 다양성 점수 분석
    if results:
        avg_entropy = sum(r['shannon_entropy'] for r in results) / len(results)
        print(f"Average Response Diversity: {avg_entropy:.3f}")
        
        # 모순 발생 통계
        contradictions = sum(1 for r in results if "Found" in r['contradiction_report'])
        print(f"Contradictions Found: {contradictions}/{len(results)} problems")
    
    return results

def test_gpt_oss_real():
    """실제 GPT-OSS 모델 테스트 (Ollama 필요)"""
    print("\n=== Real GPT-OSS Model Test ===")
    
    cost_tracker = CostTracker()
    gpt_oss_real = GPTOSSAgent("GPT_OSS_Real", cost_tracker)
    
    test_problem = "2 + 2는 무엇인가요?"
    print(f"Test Problem: {test_problem}")
    
    try:
        response = gpt_oss_real.solve(test_problem)
        print(f"Response: {response}")
        print(f"Cost: ${cost_tracker.get_total_cost():.6f}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Note: Ollama가 설치되지 않았거나 GPT-OSS 모델을 사용할 수 없습니다.")

if __name__ == "__main__":
    # Mock 모델들로 기본 테스트
    test_multi_model()
    
    # 실제 GPT-OSS 테스트 (선택적)
    test_gpt_oss_real()