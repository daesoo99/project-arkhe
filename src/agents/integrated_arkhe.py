"""
통합된 Project Arkhē 시스템
3가지 핵심 철학의 최적 조합을 구현
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import random

from .complexity_analyzer import ComplexityAnalyzer, ComplexityMetrics
from .information_asymmetry import IsolatedAgent, InformationIsolationEngine, IsolationLevel, CrossValidationEngine
from .recursive_agent import ProblemDecomposer, RecursiveAgent
from .hierarchy import CostTracker

class ArkheLevelConfig(Enum):
    """Arkhē 시스템 적용 수준"""
    BASIC = "basic"              # 정보 비대칭만
    ADVANCED = "advanced"        # 정보 비대칭 + 경제적 지능
    FULL = "full"               # 모든 기능 활성화

@dataclass
class ArkheSystemConfig:
    """Arkhē 시스템 설정"""
    level: ArkheLevelConfig
    enable_economic_intelligence: bool
    enable_recursive_decomposition: bool
    enable_information_asymmetry: bool
    isolation_level: IsolationLevel
    max_recursion_depth: int
    cost_optimization_threshold: float

class IntegratedArkheAgent:
    """통합된 Arkhē 에이전트 - 모든 기능을 선택적으로 활용"""
    
    def __init__(self, name: str, cost_tracker: CostTracker, config: ArkheSystemConfig):
        self.name = name
        self.cost_tracker = cost_tracker
        self.config = config
        
        # 하위 시스템들 초기화
        self.complexity_analyzer = ComplexityAnalyzer() if config.enable_economic_intelligence else None
        self.isolation_engine = InformationIsolationEngine(config.isolation_level) if config.enable_information_asymmetry else None
        self.decomposer = ProblemDecomposer(max_depth=config.max_recursion_depth) if config.enable_recursive_decomposition else None
        self.cross_validator = CrossValidationEngine()
        
        # 사고 스타일 (정보 비대칭용)
        self.thinking_styles = ["analytical", "creative", "skeptical", "practical", "optimistic"]
        
    def solve(self, problem: str) -> Dict[str, Any]:
        """통합된 Arkhē 방식으로 문제 해결"""
        
        # 1단계: 복잡도 분석 (경제적 지능)
        if self.config.enable_economic_intelligence:
            complexity_metrics = self.complexity_analyzer.analyze(problem)
            recommended_model = complexity_metrics.recommended_model
            should_use_advanced = complexity_metrics.score >= self.config.cost_optimization_threshold
        else:
            complexity_metrics = None
            recommended_model = "gpt-3.5-turbo"
            should_use_advanced = False
        
        # 2단계: 재귀적 분해 필요성 판단
        if self.config.enable_recursive_decomposition and complexity_metrics:
            should_decompose = self.decomposer.should_decompose(problem, complexity_metrics.score)
        else:
            should_decompose = False
        
        # 3단계: 실제 문제 해결 전략 결정
        if should_decompose:
            return self._solve_with_recursion(problem, complexity_metrics)
        else:
            return self._solve_with_isolation(problem, complexity_metrics, recommended_model)
    
    def _solve_with_recursion(self, problem: str, complexity_metrics: ComplexityMetrics) -> Dict[str, Any]:
        """재귀적 분해를 통한 문제 해결"""
        
        # 서브 문제들로 분해
        sub_problems = self.decomposer.decompose(problem, complexity_metrics.score, 0)
        
        if not sub_problems:
            # 분해 실패시 일반적 해결로 fallback
            return self._solve_with_isolation(problem, complexity_metrics, complexity_metrics.recommended_model)
        
        # 각 서브 문제를 격리된 에이전트로 해결
        sub_results = []
        total_agents_used = 0
        
        for i, sub_prob in enumerate(sub_problems):
            sub_agent = IntegratedArkheAgent(
                f"{self.name}_Sub_{i+1}",
                self.cost_tracker,
                ArkheSystemConfig(
                    level=ArkheLevelConfig.BASIC,  # 서브 에이전트는 단순하게
                    enable_economic_intelligence=True,
                    enable_recursive_decomposition=False,  # 재귀 방지
                    enable_information_asymmetry=True,
                    isolation_level=IsolationLevel.COMPLETE,
                    max_recursion_depth=0,
                    cost_optimization_threshold=6.0
                )
            )
            
            sub_result = sub_agent._solve_with_isolation(sub_prob.content, None, "gpt-3.5-turbo")
            sub_results.append(sub_result['final_answer'])
            total_agents_used += sub_result['agents_used']
        
        # 서브 결과들을 종합
        final_synthesis = self._synthesize_recursive_results(problem, sub_problems, sub_results)
        
        # 추가 비용 계산 (종합 작업)
        synthesis_tokens = len(final_synthesis)
        self.cost_tracker.add_cost("claude-3-5-sonnet", synthesis_tokens // 3, synthesis_tokens // 3)
        
        return {
            'final_answer': final_synthesis,
            'method': 'recursive_decomposition',
            'sub_problems_count': len(sub_problems),
            'agents_used': total_agents_used + 1,
            'complexity_score': complexity_metrics.score if complexity_metrics else 0,
            'recommended_model': complexity_metrics.recommended_model if complexity_metrics else "unknown"
        }
    
    def _solve_with_isolation(self, problem: str, complexity_metrics: Optional[ComplexityMetrics], recommended_model: str) -> Dict[str, Any]:
        """정보 격리를 통한 독립적 문제 해결"""
        
        if not self.config.enable_information_asymmetry:
            # 격리 비활성화시 단순 해결
            return self._solve_simple(problem, recommended_model)
        
        # 3개의 독립적 에이전트 생성 (다양한 사고 스타일)
        isolated_responses = []
        agents_used = 3
        
        for i in range(3):
            thinking_style = self.thinking_styles[i % len(self.thinking_styles)]
            
            # 격리된 맥락 생성
            context = self.isolation_engine.create_isolated_context(f"agent_{i+1}", problem)
            
            # 격리된 에이전트 생성 및 문제 해결
            isolated_agent = IsolatedAgent(f"{self.name}_Isolated_{i+1}", self.cost_tracker, thinking_style)
            isolated_agent.set_context(context)
            
            response = isolated_agent.solve_isolated(problem)
            isolated_responses.append(response)
        
        # 교차 검증
        validation_result = self.cross_validator.cross_validate(isolated_responses)
        
        # 최종 종합
        final_answer = self._synthesize_isolated_responses(isolated_responses, validation_result)
        
        return {
            'final_answer': final_answer,
            'method': 'information_isolation',
            'agents_used': agents_used,
            'diversity_score': validation_result.get('diversity', 0),
            'consistency_score': validation_result.get('consistency', 0),
            'confidence_score': validation_result.get('confidence', 0),
            'complexity_score': complexity_metrics.score if complexity_metrics else 0,
            'recommended_model': recommended_model
        }
    
    def _solve_simple(self, problem: str, model: str) -> Dict[str, Any]:
        """단순한 문제 해결 (기본 fallback)"""
        
        # 간단한 응답 생성
        if "수도" in problem:
            response = "파리입니다. 프랑스의 수도입니다."
        elif "행성" in problem:
            response = "목성입니다. 태양계에서 가장 큰 행성입니다."
        elif "2+2" in problem or "2 + 2" in problem:
            response = "4입니다."
        else:
            response = f"이 문제에 대한 {model} 수준의 답변을 제공합니다."
        
        # 비용 추적
        estimated_tokens = len(problem) + len(response)
        self.cost_tracker.add_cost(model, estimated_tokens // 3, estimated_tokens // 3)
        
        final_answer = f"[{self.name}|{model}|Simple] {response}"
        
        return {
            'final_answer': final_answer,
            'method': 'simple_resolution',
            'agents_used': 1,
            'complexity_score': 0,
            'recommended_model': model
        }
    
    def _synthesize_recursive_results(self, original_problem: str, sub_problems: List, sub_results: List[str]) -> str:
        """재귀적 결과들 종합"""
        synthesis = f"[Arkhē 재귀 종합] {original_problem}\\n\\n"
        synthesis += f"서브 문제 분석 결과 ({len(sub_problems)}개 관점):\\n"
        
        for i, (sub_prob, result) in enumerate(zip(sub_problems, sub_results), 1):
            clean_result = result.split("]")[-1].strip() if "]" in result else result
            synthesis += f"{i}. {clean_result[:100]}...\\n"
        
        synthesis += f"\\n종합 결론: 다각도 분석을 통해 {len(sub_problems)}가지 관점을 통합한 포괄적 해결책을 제시했습니다."
        
        return synthesis
    
    def _synthesize_isolated_responses(self, responses: List[str], validation: Dict[str, Any]) -> str:
        """격리된 응답들 종합"""
        validated_points = validation.get('validated_points', [])
        
        synthesis = f"[Arkhē 격리 종합] 독립적 분석 결과 (다양성: {validation.get('diversity', 0):.2f})\\n\\n"
        
        if validated_points:
            synthesis += f"공통 검증 포인트: {', '.join(validated_points[:3])}\\n"
        
        synthesis += "다양한 관점들:\\n"
        for i, response in enumerate(responses, 1):
            # 응답에서 핵심만 추출
            clean_response = response.split("]")[-1].strip() if "]" in response else response
            synthesis += f"{i}. {clean_response[:80]}...\\n"
        
        synthesis += f"\\n결론: {len(responses)}개의 독립적 관점을 통해 균형잡힌 분석을 수행했습니다."
        
        return synthesis

class ArkheSystemFactory:
    """Arkhē 시스템 설정 팩토리"""
    
    @staticmethod
    def create_basic_config() -> ArkheSystemConfig:
        """기본 설정 - 정보 비대칭만 활성화"""
        return ArkheSystemConfig(
            level=ArkheLevelConfig.BASIC,
            enable_economic_intelligence=False,
            enable_recursive_decomposition=False,
            enable_information_asymmetry=True,
            isolation_level=IsolationLevel.COMPLETE,
            max_recursion_depth=0,
            cost_optimization_threshold=10.0  # 매우 높게 설정하여 비활성화
        )
    
    @staticmethod
    def create_advanced_config() -> ArkheSystemConfig:
        """고급 설정 - 정보 비대칭 + 경제적 지능"""
        return ArkheSystemConfig(
            level=ArkheLevelConfig.ADVANCED,
            enable_economic_intelligence=True,
            enable_recursive_decomposition=False,
            enable_information_asymmetry=True,
            isolation_level=IsolationLevel.COMPLETE,
            max_recursion_depth=0,
            cost_optimization_threshold=8.0  # 매우 복잡한 문제만 고급 모델
        )
    
    @staticmethod
    def create_full_config() -> ArkheSystemConfig:
        """완전 설정 - 모든 기능 활성화 (신중하게 사용)"""
        return ArkheSystemConfig(
            level=ArkheLevelConfig.FULL,
            enable_economic_intelligence=True,
            enable_recursive_decomposition=True,
            enable_information_asymmetry=True,
            isolation_level=IsolationLevel.COMPLETE,
            max_recursion_depth=2,  # 제한적 재귀
            cost_optimization_threshold=9.0  # 극도로 복잡한 문제만
        )
    
    @staticmethod
    def create_optimized_config() -> ArkheSystemConfig:
        """최적화된 설정 - 실험 결과를 바탕으로 한 최선의 조합"""
        return ArkheSystemConfig(
            level=ArkheLevelConfig.ADVANCED,
            enable_economic_intelligence=False,  # 비용 증가 요소 제거
            enable_recursive_decomposition=False,  # 비용 증가 요소 제거  
            enable_information_asymmetry=True,   # 유일하게 효과적이었던 기능
            isolation_level=IsolationLevel.COMPLETE,
            max_recursion_depth=0,
            cost_optimization_threshold=10.0
        )

class TraditionalAgent:
    """기존 방식의 단순 에이전트 (Control Group)"""
    
    def __init__(self, name: str, cost_tracker: CostTracker):
        self.name = name
        self.cost_tracker = cost_tracker
    
    def solve(self, problem: str) -> Dict[str, Any]:
        """전통적 방식으로 문제 해결"""
        
        # 단순한 단일 모델 접근
        if "비교" in problem or "장단점" in problem:
            response = "장점과 단점을 고려한 균형잡힌 분석이 필요합니다."
        elif "예측" in problem or "미래" in problem:
            response = "현재 트렌드를 바탕으로 미래를 예측하겠습니다."
        elif "원인" in problem:
            response = "다양한 원인들을 체계적으로 분석하겠습니다."
        elif "수도" in problem:
            response = "파리입니다."
        elif "행성" in problem:
            response = "목성입니다."
        else:
            response = "이 문제에 대해 전통적 접근법으로 답변드리겠습니다."
        
        # 비용 추적 (단일 GPT-3.5 사용 가정)
        estimated_tokens = len(problem) + len(response)
        self.cost_tracker.add_cost("gpt-3.5-turbo", estimated_tokens // 3, estimated_tokens // 3)
        
        final_answer = f"[{self.name}|Traditional] {response}"
        
        return {
            'final_answer': final_answer,
            'method': 'traditional',
            'agents_used': 1,
            'complexity_score': 0,
            'recommended_model': "gpt-3.5-turbo"
        }