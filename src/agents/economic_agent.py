"""
경제적 지능 에이전트
Economic Intelligence Agent - 동적 모델 선택
"""

from .complexity_analyzer import ComplexityAnalyzer
from .hierarchy import CostTracker

class EconomicAgent:
    """복잡도 기반 동적 모델 선택을 하는 경제적 지능 에이전트"""
    
    def __init__(self, name: str, cost_tracker: CostTracker):
        self.name = name
        self.cost_tracker = cost_tracker
        self.complexity_analyzer = ComplexityAnalyzer()
        
        # Mock 모델 응답 생성기들
        self.model_responses = {
            "gemma:2b": self._gemma_response,
            "gpt-3.5-turbo": self._gpt35_response,
            "claude-3-5-sonnet": self._claude_response,
            "gpt-4": self._gpt4_response
        }
    
    def solve(self, problem: str) -> str:
        """문제 복잡도를 분석하여 최적 모델로 해결"""
        # 1. 복잡도 분석
        metrics = self.complexity_analyzer.analyze(problem)
        
        # 2. 선택된 모델로 응답 생성
        selected_model = metrics.recommended_model
        response_func = self.model_responses.get(selected_model, self._gpt35_response)
        
        response = response_func(problem, metrics)
        
        # 3. 비용 추적
        estimated_tokens = len(problem.split()) + len(response.split())
        self.cost_tracker.add_cost(selected_model, estimated_tokens // 2, estimated_tokens // 2)
        
        # 4. 선택된 모델 정보 포함하여 반환
        return f"[{self.name}|{selected_model}|{metrics.score:.1f}] {response}"
    
    def _gemma_response(self, problem: str, metrics) -> str:
        """Gemma 2B 스타일 응답 (단순, 직접적)"""
        if "수도" in problem:
            return "파리입니다."
        elif "2+2" in problem or "2 + 2" in problem:
            return "4입니다."
        elif "평평" in problem:
            return "아니오, 지구는 구형입니다."
        else:
            return "간단한 답변을 제공합니다."
    
    def _gpt35_response(self, problem: str, metrics) -> str:
        """GPT-3.5 스타일 응답 (적당한 설명)"""
        if "수도" in problem:
            return "파리입니다. 프랑스의 수도이며 센 강 유역에 위치합니다."
        elif "2+2" in problem or "2 + 2" in problem:
            return "4입니다. 기본적인 덧셈 연산의 결과입니다."
        elif "평평" in problem:
            return "아니오. 지구는 구형이며, 이는 과학적으로 증명된 사실입니다."
        elif "규제" in problem:
            return "예. AI 기술 발전을 고려할 때 적절한 규제가 필요합니다."
        elif "기후" in problem:
            return "주요 원인은 온실가스 배출입니다. 산업 활동이 주된 요인입니다."
        else:
            return "중간 수준의 상세한 답변을 제공합니다."
    
    def _claude_response(self, problem: str, metrics) -> str:
        """Claude 스타일 응답 (분석적, 구조적)"""
        if "장단점" in problem or "비교" in problem:
            return f"이 문제는 다각도 분석이 필요합니다. 주요 관점들을 비교해보면: 1) 긍정적 측면과 2) 우려사항들이 있으며, 균형잡힌 접근이 중요합니다."
        elif "철학" in problem or "의미" in problem:
            return "이는 깊이 있는 철학적 사고를 요구하는 문제입니다. 여러 관점에서 접근하여 근본적인 의미를 탐구해야 합니다."
        elif "예측" in problem:
            return "복합적 요인들을 고려한 예측이 필요합니다. 현재 트렌드와 미래 변수들을 종합적으로 분석하겠습니다."
        else:
            return "이 문제는 체계적인 분석과 다층적 사고가 필요한 복잡한 주제입니다."
    
    def _gpt4_response(self, problem: str, metrics) -> str:
        """GPT-4 스타일 응답 (매우 상세, 전문적)"""
        return f"이는 매우 복잡한 문제로 다학제적 접근이 필요합니다. {metrics.domain} 영역의 전문 지식과 {metrics.reasoning_depth}단계의 추론 과정을 통해 종합적이고 균형잡힌 관점에서 상세한 분석을 제공하겠습니다."

class FixedModelAgent:
    """고정 모델을 사용하는 기존 방식 에이전트 (Control Group)"""
    
    def __init__(self, name: str, cost_tracker: CostTracker, model: str = "gpt-3.5-turbo"):
        self.name = name
        self.cost_tracker = cost_tracker
        self.model = model
    
    def solve(self, problem: str) -> str:
        """고정된 모델로 모든 문제 해결"""
        # GPT-3.5 수준의 응답 생성
        if "수도" in problem:
            response = "파리입니다. 프랑스의 수도이며 센 강 유역에 위치합니다."
        elif "2+2" in problem or "2 + 2" in problem:
            response = "4입니다. 기본적인 덧셈 연산의 결과입니다."
        elif "평평" in problem:
            response = "아니오. 지구는 구형이며, 이는 과학적으로 증명된 사실입니다."
        elif "규제" in problem:
            response = "예. AI 기술 발전을 고려할 때 적절한 규제가 필요합니다."
        elif "기후" in problem:
            response = "주요 원인은 온실가스 배출입니다. 산업 활동이 주된 요인입니다."
        elif "장단점" in problem or "비교" in problem:
            response = "이 주제에는 여러 관점이 있습니다. 긍정적 측면과 부정적 측면을 모두 고려해야 합니다."
        elif "철학" in problem or "의미" in problem:
            response = "이는 복잡한 철학적 문제입니다. 다양한 관점에서 접근할 수 있습니다."
        else:
            response = "이 문제에 대해 표준적인 수준의 답변을 제공합니다."
        
        # 비용 추적 (모든 문제에 동일한 모델 비용)
        estimated_tokens = len(problem.split()) + len(response.split())
        self.cost_tracker.add_cost(self.model, estimated_tokens // 2, estimated_tokens // 2)
        
        return f"[{self.name}|{self.model}|Fixed] {response}"