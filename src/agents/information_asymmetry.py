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
        
        # 사고 시드와 스타일을 바탕으로 다양한 접근
        approach_modifier = self._get_approach_modifier()
        
        # 문제별 다양한 관점 생성
        if "비교" in problem or "장단점" in problem:
            response = self._solve_comparison_isolated(problem, approach_modifier)
        elif "예측" in problem or "미래" in problem:
            response = self._solve_prediction_isolated(problem, approach_modifier)
        elif "원인" in problem or "요인" in problem:
            response = self._solve_causal_isolated(problem, approach_modifier)
        elif "철학" in problem or "의미" in problem:
            response = self._solve_philosophical_isolated(problem, approach_modifier)
        else:
            response = self._solve_general_isolated(problem, approach_modifier)
        
        # 비용 추적
        estimated_tokens = len(problem) + len(response)
        self.cost_tracker.add_cost("gpt-3.5-turbo", estimated_tokens // 3, estimated_tokens // 3)
        
        return f"[{self.name}|{self.thinking_style}|{self.context.thinking_seed}] {response}"
    
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
    
    def _solve_comparison_isolated(self, problem: str, modifier: str) -> str:
        """격리된 상태에서 비교 문제 해결"""
        if "analytical" in self.thinking_style:
            return f"{modifier} 체계적 분석 결과: 장점은 효율성과 혁신이며, 단점은 위험성과 비용입니다. 데이터 기반 접근이 필요합니다."
        elif "creative" in self.thinking_style:
            return f"{modifier} 창의적 관점: 기존 패러다임을 벗어난 새로운 가능성이 있지만, 예측 불가능한 부작용도 고려해야 합니다."
        elif "skeptical" in self.thinking_style:
            return f"{modifier} 비판적 검토: 표면적 장점들이 과장되었을 가능성이 있으며, 숨겨진 리스크들을 더 면밀히 조사해야 합니다."
        else:
            return f"{modifier} 균형적 평가: 긍정적 측면과 우려사항을 동시에 고려한 신중한 접근이 필요합니다."
    
    def _solve_prediction_isolated(self, problem: str, modifier: str) -> str:
        """격리된 상태에서 예측 문제 해결"""
        if "optimistic" in self.thinking_style:
            return f"{modifier} 낙관적 전망: 기술 발전과 사회적 적응을 통해 긍정적 변화가 예상되며, 새로운 기회들이 창출될 것입니다."
        elif "skeptical" in self.thinking_style:
            return f"{modifier} 신중한 예측: 불확실성이 높으며, 여러 리스크 시나리오를 고려한 대비책이 필요합니다."
        else:
            return f"{modifier} 균형적 예측: 기술적 진보와 사회적 도전이 공존하는 복합적 미래가 예상됩니다."
    
    def _solve_causal_isolated(self, problem: str, modifier: str) -> str:
        """격리된 상태에서 원인 분석"""
        return f"{modifier} 다층적 원인 분석: 직접적 요인들과 근본적 구조적 문제들을 구분하여 접근해야 합니다."
    
    def _solve_philosophical_isolated(self, problem: str, modifier: str) -> str:
        """격리된 상태에서 철학적 문제 해결"""
        return f"{modifier} 철학적 탐구: 존재론적, 인식론적 관점에서 근본적 질문들을 제기하며, 다원적 가치 체계를 고려해야 합니다."
    
    def _solve_general_isolated(self, problem: str, modifier: str) -> str:
        """격리된 상태에서 일반적 문제 해결"""
        return f"{modifier} 종합적 접근: 다각도 분석을 통한 포괄적 이해와 실행 가능한 해결 방안이 필요합니다."

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
        if self.shared_context:
            context_summary = "공유 정보 참고: " + "; ".join(self.shared_context[-3:])  # 최근 3개만
        else:
            context_summary = ""
        
        # 문제 해결 (공유 정보의 영향으로 유사한 답변 경향)
        if "비교" in problem:
            base_response = "장점은 효율성과 혁신, 단점은 위험과 비용입니다. 균형잡힌 접근이 필요합니다."
        elif "예측" in problem:
            base_response = "기술 발전과 사회 변화가 지속되며, 기회와 도전이 공존할 것으로 예상됩니다."
        elif "철학" in problem:
            base_response = "복잡한 철학적 문제로, 다양한 관점의 균형적 고려가 필요합니다."
        else:
            base_response = "종합적 분석을 통한 체계적 접근이 중요합니다."
        
        # 공유 정보의 영향으로 답변이 수렴하는 경향 시뮬레이션
        if self.shared_context and any("효율성" in ctx for ctx in self.shared_context):
            base_response = base_response.replace("혁신", "효율성 중심의 혁신")
        
        response = base_response
        if context_summary:
            response += f" [{context_summary[:50]}... 참고]"
        
        # 비용 추적
        estimated_tokens = len(problem) + len(response) + len(context_summary)
        self.cost_tracker.add_cost("gpt-3.5-turbo", estimated_tokens // 3, estimated_tokens // 3)
        
        return f"[{self.name}|Shared] {response}"

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