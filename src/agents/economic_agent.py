"""
경제적 지능 에이전트
Economic Intelligence Agent - 동적 모델 선택
"""

from .complexity_analyzer import ComplexityAnalyzer
from .hierarchy import CostTracker
from ..llm.llm_interface import llm_interface, LLMConfig, ModelType, get_default_model

class EconomicAgent:
    """복잡도 기반 동적 모델 선택을 하는 경제적 지능 에이전트"""
    
    def __init__(self, name: str, cost_tracker: CostTracker):
        self.name = name
        self.cost_tracker = cost_tracker
        self.complexity_analyzer = ComplexityAnalyzer()
        
        # 모델 타입 매핑
        self.model_type_mapping = {
            "gemma:2b": ModelType.OLLAMA_LLAMA,  # 가벼운 모델로 매핑
            "gpt-3.5-turbo": ModelType.OPENAI_GPT35,
            "claude-3-5-sonnet": ModelType.CLAUDE_SONNET,
            "gpt-4": ModelType.OPENAI_GPT4
        }
    
    def solve(self, problem: str) -> str:
        """문제 복잡도를 분석하여 최적 모델로 해결"""
        # 1. 복잡도 분석
        metrics = self.complexity_analyzer.analyze(problem)
        
        # 2. 모델 타입 매핑
        selected_model_name = metrics.recommended_model
        model_type = self.model_type_mapping.get(selected_model_name, get_default_model())
        
        # 3. LLM 설정 생성
        config = LLMConfig(
            model_type=model_type,
            temperature=0.7,
            max_tokens=500,
            timeout=30
        )
        
        # 4. 실제 LLM 호출
        llm_response = llm_interface.generate_sync(problem, config)
        
        # 5. 비용 추적 (실제 토큰 수 사용)
        self.cost_tracker.add_cost(
            selected_model_name, 
            llm_response.tokens_used // 2, 
            llm_response.tokens_used // 2
        )
        
        # 6. 응답 포맷팅
        if llm_response.success:
            response_content = llm_response.content
        else:
            response_content = f"오류 발생: {llm_response.error}"
        
        return f"[{self.name}|{selected_model_name}|{metrics.score:.1f}] {response_content}"
    

class FixedModelAgent:
    """고정 모델을 사용하는 기존 방식 에이전트 (Control Group)"""
    
    def __init__(self, name: str, cost_tracker: CostTracker, model: str = "gpt-3.5-turbo"):
        self.name = name
        self.cost_tracker = cost_tracker
        self.model = model
        
        # 모델 타입 매핑
        self.model_type_mapping = {
            "gpt-3.5-turbo": ModelType.OPENAI_GPT35,
            "gpt-4": ModelType.OPENAI_GPT4,
            "claude-3-5-sonnet": ModelType.CLAUDE_SONNET,
            "gemma:2b": ModelType.OLLAMA_LLAMA
        }
    
    def solve(self, problem: str) -> str:
        """고정된 모델로 모든 문제 해결"""
        # 모델 타입 결정
        model_type = self.model_type_mapping.get(self.model, get_default_model())
        
        # LLM 설정 생성
        config = LLMConfig(
            model_type=model_type,
            temperature=0.7,
            max_tokens=500,
            timeout=30
        )
        
        # 실제 LLM 호출
        llm_response = llm_interface.generate_sync(problem, config)
        
        # 비용 추적
        self.cost_tracker.add_cost(
            self.model,
            llm_response.tokens_used // 2,
            llm_response.tokens_used // 2
        )
        
        # 응답 포맷팅
        if llm_response.success:
            response_content = llm_response.content
        else:
            response_content = f"오류 발생: {llm_response.error}"
        
        return f"[{self.name}|{self.model}|Fixed] {response_content}"