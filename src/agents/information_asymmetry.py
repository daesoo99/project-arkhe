"""
정보 비대칭 에이전트 시스템 (Intentional Information Asymmetry)
- 의도적 정보 격리를 통한 독립적 사고
- 군중사고 방지 및 다양성 증대
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import random
import hashlib
from enum import Enum
from ..llm.llm_interface import llm_interface, LLMConfig, ModelType, get_default_model

class IsolationLevel(Enum):
    """격리 수준 정의"""
    COMPLETE = "complete"  # 완전 격리 - 아무 정보도 공유하지 않음
    PARTIAL = "partial"    # 부분 격리 - 제한적 힌트만 제공
    MINIMAL = "minimal"    # 최소 격리 - 기본적인 맥락만 차단

@dataclass 
class ThinkingContext:
    """사고 맥락 정보"""
    agent_id: str
    problem: str
    available_info: List[str]  # 사용 가능한 정보들
    hidden_info: Set[str]      # 의도적으로 숨겨진 정보들
    isolation_level: IsolationLevel
    thinking_seed: str         # 사고 다양성을 위한 시드

class InformationIsolationEngine:
    """정보 격리 엔진 - 에이전트별로 다른 정보 접근 제어"""
    
    def __init__(self, isolation_level: IsolationLevel = IsolationLevel.COMPLETE):
        self.isolation_level = isolation_level
        self.global_context = {}  # 전역 정보 저장소
        self.agent_contexts = {}  # 에이전트별 격리된 맥락
        
    def create_isolated_context(self, agent_id: str, problem: str, base_info: List[str] = None) -> ThinkingContext:
        """에이전트별 격리된 사고 맥락 생성"""
        if base_info is None:
            base_info = []
            
        # 사고 다양성을 위한 고유 시드 생성
        thinking_seed = self._generate_thinking_seed(agent_id, problem)
        
        # 격리 수준에 따른 정보 필터링
        available_info, hidden_info = self._filter_information(agent_id, base_info)
        
        context = ThinkingContext(
            agent_id=agent_id,
            problem=problem,
            available_info=available_info,
            hidden_info=hidden_info,
            isolation_level=self.isolation_level,
            thinking_seed=thinking_seed
        )
        
        self.agent_contexts[agent_id] = context
        return context
    
    def _generate_thinking_seed(self, agent_id: str, problem: str) -> str:
        """사고 다양성을 위한 고유 시드 생성"""
        # 에이전트ID + 문제 해시로 고유한 사고 방향 생성
        combined = f"{agent_id}_{problem}_{random.randint(1000, 9999)}"
        return hashlib.md5(combined.encode()).hexdigest()[:8]
    
    def _filter_information(self, agent_id: str, base_info: List[str]) -> tuple:
        """격리 수준에 따른 정보 필터링"""
        available_info = base_info.copy()
        hidden_info = set()
        
        if self.isolation_level == IsolationLevel.COMPLETE:
            # 완전 격리: 다른 에이전트의 응답 정보 완전 차단
            filtered_info = [info for info in available_info if not self._is_other_agent_info(info, agent_id)]
            hidden_info = set(available_info) - set(filtered_info)
            available_info = filtered_info
            
        elif self.isolation_level == IsolationLevel.PARTIAL:
            # 부분 격리: 일부 힌트는 제공하되 구체적 답변은 차단
            available_info = [self._sanitize_info(info, agent_id) for info in available_info]
            
        # MINIMAL인 경우 대부분의 정보 유지
        
        return available_info, hidden_info
    
    def _is_other_agent_info(self, info: str, current_agent_id: str) -> bool:
        """다른 에이전트의 응답 정보인지 판단"""
        # 간단한 휴리스틱: 다른 에이전트 ID가 포함되어 있으면 차단
        other_agents = ["agent", "thinker", "경제", "재귀", "평면"]
        return any(agent.lower() in info.lower() for agent in other_agents if agent != current_agent_id.lower())
    
    def _sanitize_info(self, info: str, agent_id: str) -> str:
        """부분 격리를 위한 정보 정제"""
        # 구체적 답변은 제거하고 방향성만 제공
        if len(info) > 50:
            return info[:30] + "... [세부사항 격리됨]"
        return info

class IsolatedAgent:
    """정보 격리 상태에서 독립적으로 사고하는 에이전트"""
    
    def __init__(self, name: str, cost_tracker, thinking_style: str = "analytical"):
        self.name = name
        self.cost_tracker = cost_tracker
        self.thinking_style = thinking_style
        self.context: Optional[ThinkingContext] = None
        
        # 사고 스타일별 특성
        self.thinking_styles = {
            "analytical": "논리적이고 체계적인 분석을 선호",
            "creative": "창의적이고 혁신적인 접근을 선호", 
            "practical": "현실적이고 실용적인 해결책을 선호",
            "skeptical": "비판적이고 의문을 제기하는 접근",
            "optimistic": "긍정적이고 가능성에 초점을 맞춤"
        }
    
    def set_context(self, context: ThinkingContext):
        """격리된 사고 맥락 설정"""
        self.context = context
    
    def solve(self, problem: str) -> str:
        """Mediator 호환성을 위한 solve 메서드"""
        return self.solve_isolated(problem)
    
    def solve_isolated(self, problem: str) -> str:
        """격리된 상태에서 독립적 문제 해결"""
        if not self.context:
            return "격리 맥락이 설정되지 않았습니다."
        
        # 사고 스타일에 따른 프롬프트 조정
        style_prompt = self._create_style_specific_prompt(problem)
        
        # LLM 설정 생성 (기본 모델 사용)
        config = LLMConfig(
            model_type=get_default_model(),
            temperature=0.8,  # 다양성을 위해 높게 설정
            max_tokens=400,
            timeout=30
        )
        
        # 실제 LLM 호출
        llm_response = llm_interface.generate_sync(style_prompt, config)
        
        # 비용 추적
        self.cost_tracker.add_cost("gpt-3.5-turbo", llm_response.tokens_used // 2, llm_response.tokens_used // 2)
        
        # 응답 포맷팅
        if llm_response.success:
            response_content = llm_response.content
        else:
            # fallback 응답
            response_content = self._create_fallback_response(problem)
        
        return f"[{self.name}|{self.thinking_style}|{self.context.thinking_seed}] {response_content}"
    
    def _get_approach_modifier(self) -> str:
        """사고 시드와 스타일을 바탕으로 접근 방식 결정"""
        seed_hash = int(self.context.thinking_seed, 16) % 5
        style_desc = self.thinking_styles.get(self.thinking_style, "균형적")
        
        modifiers = [
            f"({style_desc}, 관점 A)",
            f"({style_desc}, 관점 B)", 
            f"({style_desc}, 관점 C)",
            f"({style_desc}, 관점 D)",
            f"({style_desc}, 관점 E)"
        ]
        
        return modifiers[seed_hash]
    
    def _create_style_specific_prompt(self, problem: str) -> str:
        """사고 스타일에 따른 전용 프롬프트 생성"""
        base_prompt = f"다음 문제에 대해 답변해주세요: {problem}\n\n"
        
        style_instruction = {
            "analytical": "논리적이고 체계적인 분석을 통해 단계별로 접근해주세요. 데이터와 증거를 중시하며 객관적 관점에서 답변해주세요.",
            "creative": "창의적이고 혁신적인 관점에서 접근해주세요. 기존 틀에서 벗어난 새로운 아이디어나 해결책을 제시해주세요.",
            "practical": "현실적이고 실용적인 관점에서 답변해주세요. 실제 적용 가능성과 구현 방법에 중점을 두어주세요.",
            "skeptical": "비판적이고 의문을 제기하는 관점에서 접근해주세요. 잠재적 문제점이나 한계를 지적하며 신중하게 검토해주세요.",
            "optimistic": "긍정적이고 희망적인 관점에서 답변해주세요. 가능성과 기회에 초점을 맞춰주세요."
        }
        
        instruction = style_instruction.get(self.thinking_style, "균형잡힌 관점에서 답변해주세요.")
        
        return base_prompt + instruction
    
    def _create_fallback_response(self, problem: str) -> str:
        """LLM 호출 실패 시 fallback 응답 생성"""
        style_responses = {
            "analytical": "이 문제는 체계적 분석이 필요합니다. 데이터 수집과 단계별 검토를 통해 접근해야 합니다.",
            "creative": "이 문제에는 혁신적 접근이 필요합니다. 기존 관념을 벗어난 새로운 관점을 시도해볼 수 있습니다.",
            "practical": "이 문제는 실용적 해결책이 중요합니다. 실현 가능한 방법론을 중심으로 접근해야 합니다.",
            "skeptical": "이 문제에는 신중한 검토가 필요합니다. 잠재적 위험과 한계를 면밀히 살펴봐야 합니다.",
            "optimistic": "이 문제는 새로운 기회의 관점에서 볼 수 있습니다. 긍정적 가능성을 탐색해볼 가치가 있습니다."
        }
        
        return style_responses.get(self.thinking_style, "이 문제에 대한 균형잡힌 분석이 필요합니다.")
    

class TransparentAgent:
    """완전한 정보 공유 상태에서 동작하는 기존 방식 에이전트"""
    
    def __init__(self, name: str, cost_tracker):
        self.name = name
        self.cost_tracker = cost_tracker
        self.shared_context = []  # 다른 에이전트들의 정보 접근 가능
        
    def add_shared_context(self, info: str):
        """다른 에이전트의 정보 추가"""
        self.shared_context.append(info)
    
    def solve(self, problem: str) -> str:
        """Mediator 호환성을 위한 solve 메서드"""
        return self.solve_with_shared_info(problem)
    
    def solve_with_shared_info(self, problem: str) -> str:
        """공유된 정보를 활용한 문제 해결"""
        
        # 공유된 맥락 정보 활용
        context_info = ""
        if self.shared_context:
            context_info = "; ".join(self.shared_context[-3:])  # 최근 3개만
            
        # 공유 정보를 포함한 프롬프트 생성
        if context_info:
            enhanced_prompt = f"다음 문제에 답변해주세요: {problem}\n\n다른 전문가들의 의견 참고: {context_info}\n\n위 의견들을 참고하여 종합적이고 균형잡힌 답변을 제공해주세요."
        else:
            enhanced_prompt = f"다음 문제에 답변해주세요: {problem}"
        
        # LLM 설정 생성
        config = LLMConfig(
            model_type=get_default_model(),
            temperature=0.5,  # 일관성을 위해 낮게 설정
            max_tokens=400,
            timeout=30
        )
        
        # 실제 LLM 호출
        llm_response = llm_interface.generate_sync(enhanced_prompt, config)
        
        # 비용 추적
        self.cost_tracker.add_cost("gpt-3.5-turbo", llm_response.tokens_used // 2, llm_response.tokens_used // 2)
        
        # 응답 포맷팅
        if llm_response.success:
            response_content = llm_response.content
        else:
            # fallback 응답
            response_content = "종합적 분석을 통한 체계적 접근이 중요합니다."
        
        return f"[{self.name}|Shared] {response_content}"

class CrossValidationEngine:
    """교차 검증 엔진 - 독립적 결과들의 일치/불일치 분석"""
    
    def __init__(self):
        self.validation_metrics = {}
    
    def cross_validate(self, responses: List[str]) -> Dict[str, Any]:
        """독립적 응답들의 교차 검증"""
        if len(responses) < 2:
            return {"confidence": 0.0, "consistency": 0.0, "diversity": 0.0}
        
        # 일치도 분석 (키워드 겹침 정도)
        consistency = self._calculate_consistency(responses)
        
        # 다양성 분석 (응답 간 차이점)
        diversity = self._calculate_diversity(responses)
        
        # 신뢰도 (일치하는 핵심 포인트들)
        confidence = self._calculate_confidence(responses, consistency)
        
        return {
            "confidence": confidence,
            "consistency": consistency, 
            "diversity": diversity,
            "validated_points": self._extract_validated_points(responses),
            "divergent_views": self._extract_divergent_views(responses)
        }
    
    def _calculate_consistency(self, responses: List[str]) -> float:
        """응답 간 일치도 계산"""
        if len(responses) < 2:
            return 0.0
        
        # 간단한 키워드 겹침 분석
        all_words = []
        response_words = []
        
        for response in responses:
            words = set(response.lower().split())
            response_words.append(words)
            all_words.extend(words)
        
        if not all_words:
            return 0.0
        
        common_words = set(all_words)
        for words in response_words:
            common_words &= words
        
        return len(common_words) / len(set(all_words))
    
    def _calculate_diversity(self, responses: List[str]) -> float:
        """응답 간 다양성 계산"""
        if len(responses) < 2:
            return 0.0
        
        # 응답 길이와 유니크한 관점 수 기반
        unique_responses = set(responses)
        return len(unique_responses) / len(responses)
    
    def _calculate_confidence(self, responses: List[str], consistency: float) -> float:
        """종합적 신뢰도 계산"""
        # 일치도가 너무 높으면 군중사고 의심, 너무 낮으면 신뢰도 하락
        optimal_consistency = 0.6
        confidence = 1.0 - abs(consistency - optimal_consistency)
        return max(0.0, confidence)
    
    def _extract_validated_points(self, responses: List[str]) -> List[str]:
        """검증된 핵심 포인트들 추출"""
        # 간단한 구현: 여러 응답에서 반복되는 키워드들
        word_counts = {}
        for response in responses:
            words = response.lower().split()
            for word in words:
                if len(word) > 3:  # 의미있는 단어만
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        # 2회 이상 등장하는 단어들을 검증된 포인트로 간주
        validated = [word for word, count in word_counts.items() if count >= 2]
        return validated[:5]  # 상위 5개
    
    def _extract_divergent_views(self, responses: List[str]) -> List[str]:
        """분기된 관점들 추출"""
        # 각 응답의 고유한 특징들 추출
        divergent = []
        for i, response in enumerate(responses):
            unique_aspects = [part.strip() for part in response.split('.') if len(part.strip()) > 10]
            if unique_aspects:
                divergent.append(f"관점{i+1}: {unique_aspects[0]}")
        
        return divergent