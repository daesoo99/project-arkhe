"""
자율적 재귀 에이전트 (Autonomous Recursion)
복잡한 문제를 자동으로 서브 문제들로 분해하고 서브팀을 생성
"""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
from ..llm.llm_interface import llm_interface, LLMConfig, ModelType, get_default_model

@dataclass
class SubProblem:
    """서브 문제 정의"""
    id: str
    content: str
    complexity: float
    parent_id: str = None

@dataclass
class RecursionResult:
    """재귀 실행 결과"""
    original_problem: str
    sub_problems: List[SubProblem]
    sub_results: List[str]
    final_synthesis: str
    recursion_depth: int
    total_agents_used: int

class ProblemDecomposer:
    """문제 분해기 - 복잡한 문제를 서브 문제들로 분해"""
    
    def __init__(self, max_depth: int = 3, max_sub_problems: int = 5):
        self.max_depth = max_depth
        self.max_sub_problems = max_sub_problems
        
        # 분해 가능한 문제 패턴들
        self.decomposable_patterns = {
            'comparison': {
                'patterns': [r'비교.*분석', r'장단점', r'차이점', r'pros.*cons'],
                'decomposer': self._decompose_comparison
            },
            'multi_factor': {
                'patterns': [r'원인.*나열', r'방법.*제시', r'요인.*분석', r'단계.*설명'],
                'decomposer': self._decompose_multi_factor
            },
            'prediction': {
                'patterns': [r'예측.*분석', r'미래.*영향', r'전망.*제시'],
                'decomposer': self._decompose_prediction
            },
            'philosophical': {
                'patterns': [r'철학.*의미', r'윤리.*문제', r'가치.*판단'],
                'decomposer': self._decompose_philosophical
            }
        }
    
    def should_decompose(self, problem: str, complexity: float) -> bool:
        """문제가 분해될 필요가 있는지 판단"""
        # 복잡도가 6.0 이상이고 분해 가능한 패턴이 있으면 분해
        if complexity < 6.0:
            return False
            
        problem_lower = problem.lower()
        for category, info in self.decomposable_patterns.items():
            for pattern in info['patterns']:
                if re.search(pattern, problem_lower):
                    return True
        return False
    
    def decompose(self, problem: str, complexity: float, depth: int = 0) -> List[SubProblem]:
        """문제를 서브 문제들로 분해"""
        if depth >= self.max_depth or not self.should_decompose(problem, complexity):
            return []
        
        problem_lower = problem.lower()
        
        # 적절한 분해 방식 선택
        for category, info in self.decomposable_patterns.items():
            for pattern in info['patterns']:
                if re.search(pattern, problem_lower):
                    return info['decomposer'](problem, complexity, depth)
        
        return []
    
    def _decompose_comparison(self, problem: str, complexity: float, depth: int) -> List[SubProblem]:
        """비교/분석 문제 분해"""
        base_id = f"comp_{depth}"
        sub_problems = [
            SubProblem(f"{base_id}_1", f"{problem}의 긍정적 측면을 분석해주세요.", complexity * 0.6),
            SubProblem(f"{base_id}_2", f"{problem}의 부정적 측면을 분석해주세요.", complexity * 0.6),
            SubProblem(f"{base_id}_3", f"위 분석을 바탕으로 균형잡힌 결론을 도출해주세요.", complexity * 0.7)
        ]
        return sub_problems[:self.max_sub_problems]
    
    def _decompose_multi_factor(self, problem: str, complexity: float, depth: int) -> List[SubProblem]:
        """다요인 분석 문제 분해"""
        base_id = f"multi_{depth}"
        sub_problems = [
            SubProblem(f"{base_id}_1", f"{problem}의 직접적 요인들을 분석해주세요.", complexity * 0.7),
            SubProblem(f"{base_id}_2", f"{problem}의 간접적 요인들을 분석해주세요.", complexity * 0.7),
            SubProblem(f"{base_id}_3", f"각 요인들 간의 상호작용을 분석해주세요.", complexity * 0.8)
        ]
        return sub_problems[:self.max_sub_problems]
    
    def _decompose_prediction(self, problem: str, complexity: float, depth: int) -> List[SubProblem]:
        """예측 문제 분해"""
        base_id = f"pred_{depth}"
        sub_problems = [
            SubProblem(f"{base_id}_1", f"{problem}와 관련된 현재 상황을 분석해주세요.", complexity * 0.6),
            SubProblem(f"{base_id}_2", f"관련 기술/사회적 트렌드를 분석해주세요.", complexity * 0.7),
            SubProblem(f"{base_id}_3", f"잠재적 변수들과 시나리오를 제시해주세요.", complexity * 0.8),
            SubProblem(f"{base_id}_4", f"종합적 예측과 대응방안을 제시해주세요.", complexity * 0.9)
        ]
        return sub_problems[:self.max_sub_problems]
    
    def _decompose_philosophical(self, problem: str, complexity: float, depth: int) -> List[SubProblem]:
        """철학적 문제 분해"""
        base_id = f"phil_{depth}"
        sub_problems = [
            SubProblem(f"{base_id}_1", f"{problem}에 대한 다양한 철학적 관점들을 제시해주세요.", complexity * 0.7),
            SubProblem(f"{base_id}_2", f"각 관점의 논리적 근거를 분석해주세요.", complexity * 0.8),
            SubProblem(f"{base_id}_3", f"현실적 적용 가능성을 평가해주세요.", complexity * 0.7),
            SubProblem(f"{base_id}_4", f"통합적 관점에서 결론을 도출해주세요.", complexity * 0.9)
        ]
        return sub_problems[:self.max_sub_problems]

class RecursiveAgent:
    """재귀적 문제 해결 에이전트"""
    
    def __init__(self, name: str, cost_tracker, max_recursion_depth: int = 3):
        self.name = name
        self.cost_tracker = cost_tracker
        self.decomposer = ProblemDecomposer(max_depth=max_recursion_depth)
        self.total_agents_created = 0
    
    def solve_recursively(self, problem: str, complexity: float = 5.0, depth: int = 0) -> RecursionResult:
        """재귀적으로 문제 해결"""
        
        # 재귀 종료 조건
        if depth >= self.decomposer.max_depth or not self.decomposer.should_decompose(problem, complexity):
            # 단순 해결
            simple_result = self._solve_simple(problem, complexity)
            return RecursionResult(
                original_problem=problem,
                sub_problems=[],
                sub_results=[simple_result],
                final_synthesis=simple_result,
                recursion_depth=depth,
                total_agents_used=1
            )
        
        # 문제 분해
        sub_problems = self.decomposer.decompose(problem, complexity, depth)
        
        if not sub_problems:
            # 분해 불가능하면 단순 해결
            simple_result = self._solve_simple(problem, complexity)
            return RecursionResult(
                original_problem=problem,
                sub_problems=[],
                sub_results=[simple_result],
                final_synthesis=simple_result,
                recursion_depth=depth,
                total_agents_used=1
            )
        
        print(f"{'  ' * depth}DECOMPOSE: {len(sub_problems)} sub-problems created (depth {depth})")
        
        # 각 서브 문제를 재귀적으로 해결
        sub_results = []
        agents_used = len(sub_problems)  # 현재 레벨에서 사용된 에이전트
        
        for i, sub_problem in enumerate(sub_problems):
            print(f"{'  ' * depth}|- Sub-problem {i+1}: {sub_problem.content[:50]}...")
            
            # 재귀 호출 (더 깊은 분해 가능)
            sub_result = self.solve_recursively(sub_problem.content, sub_problem.complexity, depth + 1)
            sub_results.append(sub_result.final_synthesis)
            agents_used += sub_result.total_agents_used
        
        # 서브 결과들을 종합
        final_synthesis = self._synthesize_results(problem, sub_problems, sub_results)
        
        return RecursionResult(
            original_problem=problem,
            sub_problems=sub_problems,
            sub_results=sub_results,
            final_synthesis=final_synthesis,
            recursion_depth=depth,
            total_agents_used=agents_used
        )
    
    def _solve_simple(self, problem: str, complexity: float) -> str:
        """단순 문제 해결 (재귀 없음)"""
        
        # LLM 설정 생성
        config = LLMConfig(
            model_type=get_default_model(),
            temperature=0.7,
            max_tokens=300,
            timeout=30
        )
        
        # 복잡도에 따른 프롬프트 조정
        if complexity <= 3.0:
            prompt = f"간단명료하게 답변해주세요: {problem}"
        elif complexity <= 6.0:
            prompt = f"적당한 수준의 설명을 포함하여 답변해주세요: {problem}"
        else:
            prompt = f"상세하고 전문적으로 분석해서 답변해주세요: {problem}"
        
        # 실제 LLM 호출
        llm_response = llm_interface.generate_sync(prompt, config)
        
        # 비용 추적
        self.cost_tracker.add_cost("gpt-3.5-turbo", llm_response.tokens_used // 2, llm_response.tokens_used // 2)
        
        # 응답 처리
        if llm_response.success:
            return llm_response.content
        else:
            # fallback 응답
            return f"복잡도 {complexity:.1f}의 문제에 대한 분석적 답변입니다."
    
    def _synthesize_results(self, original_problem: str, sub_problems: List[SubProblem], sub_results: List[str]) -> str:
        """서브 결과들을 종합하여 최종 답변 생성"""
        
        synthesis = f"[종합 분석] {original_problem}에 대한 다각도 분석 결과:\n\n"
        
        for i, (sub_prob, result) in enumerate(zip(sub_problems, sub_results), 1):
            synthesis += f"{i}. {result}\n"
        
        synthesis += f"\n결론: 위 {len(sub_results)}가지 관점을 종합하면, 균형잡힌 접근이 필요하며 지속적인 모니터링과 조정이 중요합니다."
        
        # 종합 작업 비용 추가
        estimated_tokens = len(synthesis)
        self.cost_tracker.add_cost("claude-3-5-sonnet", estimated_tokens // 3, estimated_tokens // 3)
        
        return synthesis

class FlatAgent:
    """기존 방식 - 평면적 에이전트 (Control Group)"""
    
    def __init__(self, name: str, cost_tracker):
        self.name = name
        self.cost_tracker = cost_tracker
    
    def solve(self, problem: str) -> str:
        """평면적 해결 (재귀 없음)"""
        
        # LLM 설정 생성
        config = LLMConfig(
            model_type=get_default_model(),
            temperature=0.7,
            max_tokens=400,
            timeout=30
        )
        
        # 단순한 프롬프트 (재귀 분해 없이 직접 해결)
        prompt = f"다음 문제를 종합적으로 분석하여 답변해주세요: {problem}"
        
        # 실제 LLM 호출
        llm_response = llm_interface.generate_sync(prompt, config)
        
        # 비용 추적
        self.cost_tracker.add_cost("gpt-3.5-turbo", llm_response.tokens_used // 2, llm_response.tokens_used // 2)
        
        # 응답 처리
        if llm_response.success:
            response_content = llm_response.content
        else:
            # fallback 응답
            response_content = "이 문제에 대해 종합적인 분석을 제공합니다."
        
        return f"[{self.name}|Flat] {response_content}"