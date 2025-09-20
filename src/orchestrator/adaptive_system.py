#!/usr/bin/env python3
"""
Project Arkhē - Adaptive Multi-Agent System
실험 결과를 바탕으로 한 최적화된 문제 해결 시스템
"""

import re
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

class ProblemType(Enum):
    SIMPLE_FACT = "simple_fact"
    MODERATE_REASONING = "moderate_reasoning" 
    COMPLEX_PROOF = "complex_proof"
    CREATIVE_TASK = "creative_task"
    SYSTEM_DESIGN = "system_design"

class ModelTier(Enum):
    FAST = "qwen2:0.5b"        # 빠른 처리
    BALANCED = "qwen2:7b"      # 균형잡힌 성능
    POWERFUL = "llama3:8b"     # 최고 성능

@dataclass
class ProblemAnalysis:
    """문제 분석 결과"""
    complexity_score: float
    problem_type: ProblemType
    keyword_count: int
    requires_multi_step: bool
    has_creative_element: bool
    estimated_tokens: int

@dataclass 
class SystemConfiguration:
    """시스템 구성 정보"""
    use_multi_agent: bool
    draft_model: ModelTier
    review_model: Optional[ModelTier]
    judge_model: ModelTier
    expected_performance: float
    expected_efficiency: float
    rationale: str

class AdaptiveOrchestrator:
    """문제별 최적 구성을 선택하는 지능형 오케스트레이터"""
    
    def __init__(self):
        self.performance_history = {}
        self.load_experimental_knowledge()
    
    def load_experimental_knowledge(self):
        """기존 실험에서 학습한 패턴들"""
        self.knowledge_base = {
            # Phase 1 결과: 0.5B 모델에서는 Single 압도적
            "small_model_single_advantage": {
                "condition": "model_size < 1B",
                "recommendation": "single_model",
                "confidence": 0.95
            },
            
            # Phase 2 결과: 7B 모델에서만 Multi-Agent 우위
            "sweet_spot_multi_agent": {
                "condition": "3.0 <= complexity <= 8.0 and model_size >= 7B",
                "recommendation": "multi_agent", 
                "confidence": 0.80
            },
            
            # Phase 3 결과: 복잡한 문제에서 Single 복귀
            "complex_single_return": {
                "condition": "complexity > 9.0",
                "recommendation": "single_powerful",
                "confidence": 0.75
            },
            
            # 정보 비대칭 실험: 완전 공유 or 완전 독립이 최적
            "information_flow": {
                "avoid_partial_sharing": True,
                "prefer_complete_modes": True
            }
        }
    
    def analyze_problem(self, prompt: str) -> ProblemAnalysis:
        """문제를 분석하여 특성 파악"""
        
        # 복잡도 계산 (키워드 기반 + 길이 기반)
        complexity_indicators = {
            # 수학/논리 키워드
            "prove|theorem|proof|mathematical|logic": 3.0,
            "algorithm|complexity|optimization": 2.5,
            
            # 창작/설계 키워드  
            "design|create|innovative|imagine": 2.0,
            "story|creative|novel|fiction": 2.5,
            
            # 시스템/기술 키워드
            "system|architecture|distributed|scalable": 3.0,
            "implement|code|programming": 2.0,
            
            # 단순 사실 키워드
            "what is|who is|where is|when": 1.0,
            "capital|name|date|number": 1.5,
        }
        
        complexity_score = 1.0
        problem_type = ProblemType.SIMPLE_FACT
        
        for pattern, score in complexity_indicators.items():
            if re.search(pattern, prompt.lower()):
                complexity_score = max(complexity_score, score)
                
                if "prove|theorem|mathematical" in pattern:
                    problem_type = ProblemType.COMPLEX_PROOF
                elif "design|create|story" in pattern:
                    problem_type = ProblemType.CREATIVE_TASK
                elif "system|architecture|implement" in pattern:
                    problem_type = ProblemType.SYSTEM_DESIGN
                elif score > 2.0:
                    problem_type = ProblemType.MODERATE_REASONING
        
        # 길이 기반 복잡도 보정
        if len(prompt) > 200:
            complexity_score *= 1.3
        elif len(prompt) > 100:
            complexity_score *= 1.1
            
        # 다단계 추론 필요성 판단
        multi_step_indicators = ["step by step", "first", "then", "finally", "because", "therefore"]
        requires_multi_step = any(indicator in prompt.lower() for indicator in multi_step_indicators)
        
        # 창작 요소 판단
        creative_indicators = ["creative", "innovative", "imagine", "story", "design"]
        has_creative_element = any(indicator in prompt.lower() for indicator in creative_indicators)
        
        return ProblemAnalysis(
            complexity_score=complexity_score,
            problem_type=problem_type,
            keyword_count=len(prompt.split()),
            requires_multi_step=requires_multi_step,
            has_creative_element=has_creative_element,
            estimated_tokens=len(prompt.split()) * 1.5  # 추정 응답 토큰
        )
    
    def select_optimal_configuration(self, analysis: ProblemAnalysis) -> SystemConfiguration:
        """문제 분석 결과를 바탕으로 최적 구성 선택"""
        
        complexity = analysis.complexity_score
        prob_type = analysis.problem_type
        
        # Rule 1: 매우 단순한 문제 - 빠른 단일 모델
        if complexity <= 2.0 and not analysis.requires_multi_step:
            return SystemConfiguration(
                use_multi_agent=False,
                draft_model=None,
                review_model=None, 
                judge_model=ModelTier.FAST,
                expected_performance=0.85,
                expected_efficiency=0.60,
                rationale="Simple fact - fast single model optimal"
            )
        
        # Rule 2: Multi-Agent 스위트스팟 (복잡도 3-8, 다단계 추론)
        elif 3.0 <= complexity <= 8.0 and analysis.requires_multi_step:
            return SystemConfiguration(
                use_multi_agent=True,
                draft_model=ModelTier.BALANCED,    # qwen2:7b
                review_model=ModelTier.BALANCED,   # qwen2:7b
                judge_model=ModelTier.POWERFUL,    # llama3:8b
                expected_performance=0.80,
                expected_efficiency=0.35,
                rationale="Sweet spot for Multi-Agent collaboration"
            )
        
        # Rule 3: 복잡한 증명/설계 - 강력한 단일 모델
        elif complexity > 8.0 or prob_type == ProblemType.COMPLEX_PROOF:
            return SystemConfiguration(
                use_multi_agent=False,
                draft_model=None,
                review_model=None,
                judge_model=ModelTier.POWERFUL,
                expected_performance=0.75,
                expected_efficiency=0.45,
                rationale="Complex reasoning requires deep single model"
            )
        
        # Rule 4: 창작 작업 - Multi-Agent 다양성 활용
        elif analysis.has_creative_element and prob_type == ProblemType.CREATIVE_TASK:
            return SystemConfiguration(
                use_multi_agent=True,
                draft_model=ModelTier.BALANCED,
                review_model=ModelTier.BALANCED,
                judge_model=ModelTier.POWERFUL,
                expected_performance=0.70,
                expected_efficiency=0.30,
                rationale="Creative tasks benefit from diverse perspectives"
            )
        
        # Rule 5: 기본값 - 균형잡힌 단일 모델
        else:
            return SystemConfiguration(
                use_multi_agent=False,
                draft_model=None,
                review_model=None,
                judge_model=ModelTier.BALANCED,
                expected_performance=0.70,
                expected_efficiency=0.50,
                rationale="Balanced single model for general cases"
            )
    
    def execute_optimal_solution(self, prompt: str, llm_factory) -> Dict[str, Any]:
        """최적 구성으로 문제 해결 실행"""
        
        start_time = time.time()
        
        # 1. 문제 분석
        analysis = self.analyze_problem(prompt)
        
        # 2. 최적 구성 선택
        config = self.select_optimal_configuration(analysis)
        
        print(f"Problem Analysis:")
        print(f"  Complexity: {analysis.complexity_score:.1f}")
        print(f"  Type: {analysis.problem_type.value}")
        print(f"  Multi-step: {analysis.requires_multi_step}")
        print(f"  Creative: {analysis.has_creative_element}")
        
        print(f"\nSelected Configuration:")
        print(f"  Mode: {'Multi-Agent' if config.use_multi_agent else 'Single Model'}")
        print(f"  Model: {config.judge_model.value}")
        print(f"  Expected Performance: {config.expected_performance:.2f}")
        print(f"  Expected Efficiency: {config.expected_efficiency:.2f}")
        print(f"  Rationale: {config.rationale}")
        
        # 3. 실행
        if config.use_multi_agent:
            result = self._execute_multi_agent(prompt, config, llm_factory)
        else:
            result = self._execute_single_model(prompt, config, llm_factory)
        
        end_time = time.time()
        
        # 4. 결과 정리
        result.update({
            "analysis": analysis,
            "configuration": config,
            "execution_time": end_time - start_time,
            "adaptive_decision": True
        })
        
        return result
    
    def _execute_single_model(self, prompt: str, config: SystemConfiguration, llm_factory) -> Dict[str, Any]:
        """단일 모델 실행"""
        llm = llm_factory(config.judge_model.value)
        response_dict = llm.generate(prompt)
        response = response_dict.get('response', str(response_dict)) if isinstance(response_dict, dict) else str(response_dict)
        
        return {
            "mode": "single_model",
            "model": config.judge_model.value,
            "response": response,
            "tokens": len(prompt.split()) + len(response.split()),
            "stages": 1
        }
    
    def _execute_multi_agent(self, prompt: str, config: SystemConfiguration, llm_factory) -> Dict[str, Any]:
        """Multi-Agent 실행 (B-approach 방식)"""
        from .pipeline import run_3stage_with_context
        
        result = run_3stage_with_context(llm_factory, prompt)
        
        # 토큰 계산
        total_tokens = len(prompt.split()) * 3  # 각 단계별 프롬프트
        for stage in ["draft_responses", "review_responses", "final"]:
            if stage in result:
                if isinstance(result[stage], list):
                    for r in result[stage]:
                        total_tokens += len(str(r).split())
                else:
                    total_tokens += len(str(result[stage]).split())
        
        return {
            "mode": "multi_agent",
            "models": f"{config.draft_model.value}+{config.judge_model.value}",
            "response": result.get("final", ""),
            "tokens": total_tokens,
            "stages": 3,
            "details": result
        }

def create_adaptive_system():
    """적응형 시스템 인스턴스 생성"""
    return AdaptiveOrchestrator()

# 사용 예시
if __name__ == "__main__":
    # 테스트용 문제들
    test_problems = [
        "서울의 인구는 몇 명인가요?",  # Simple fact
        "기후변화 문제를 해결하기 위한 3단계 전략을 수립하시오.", # Moderate reasoning
        "베르트랑 공준을 수학적으로 증명하시오.", # Complex proof
        "시간여행 패러독스를 해결하는 SF 소설 줄거리를 창작하시오." # Creative task
    ]
    
    orchestrator = AdaptiveOrchestrator()
    
    for problem in test_problems:
        print(f"\nTesting: {problem[:50]}...")
        analysis = orchestrator.analyze_problem(problem)
        config = orchestrator.select_optimal_configuration(analysis)
        print(f"   -> {config.rationale}")