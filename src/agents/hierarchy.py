# -*- coding: utf-8 -*-
"""
Project Arkhē - Hierarchical Agent System
변경 요약: ollama 직접 의존 제거, simple_llm로 일원화
- 모든 LLM 호출이 create_llm_auto()를 통해 이루어져 환경 독립성 확보
- HTTP/Mock 자동 폴백으로 어떤 환경에서든 동작
"""

import math
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# LLM 호출 일원화 - ollama 직접 의존 제거
from src.llm.simple_llm import create_llm_auto

@dataclass
class CostItem:
    """개별 비용 항목"""
    model: str
    input_tokens: int
    output_tokens: int
    unit_cost_in: float
    unit_cost_out: float
    
    @property
    def total_cost(self) -> float:
        return (self.input_tokens / 1_000_000 * self.unit_cost_in) + \
               (self.output_tokens / 1_000_000 * self.unit_cost_out)

class CostTracker:
    """API 호출 비용을 토큰 사용량 기준으로 추적"""
    
    def __init__(self):
        self.items: List[CostItem] = []
        # 1M 토큰당 가격 (USD)
        self._model_prices = {
            "qwen2:0.5b": {"input": 0.05, "output": 0.05},
            "gemma:2b": {"input": 0.15, "output": 0.15},
            "llama3:8b": {"input": 0.5, "output": 0.5},
            "claude-3-5-sonnet": {"input": 3.0, "output": 15.0},
            "gpt-4": {"input": 30.0, "output": 60.0},
            "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
            "gpt-oss-20b": {"input": 0.0, "output": 0.0},  # 오픈소스 무료
            "gpt-oss-120b": {"input": 0.0, "output": 0.0}  # 오픈소스 무료
        }

    def add_cost(self, model: str, input_tokens: int, output_tokens: int, 
                 unit_in: float = None, unit_out: float = None):
        """단일 API 호출 비용 추가 (음수 방지 안정성 보강)"""
        # 음수/None 방지
        input_tokens = max(0, int(input_tokens or 0))
        output_tokens = max(0, int(output_tokens or 0))
        
        # 단가 설정
        if unit_in is None or unit_out is None:
            prices = self._model_prices.get(model, {"input": 0.1, "output": 0.1})
            unit_in = unit_in or prices["input"]
            unit_out = unit_out or prices["output"]
        
        item = CostItem(model, input_tokens, output_tokens, unit_in, unit_out)
        self.items.append(item)

    def get_total_cost(self) -> float:
        """누적 총 비용 반환"""
        return sum(item.total_cost for item in self.items)
    
    def get_model_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """모델별 비용 분석"""
        breakdown = {}
        for item in self.items:
            if item.model not in breakdown:
                breakdown[item.model] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_cost": 0.0
                }
            
            breakdown[item.model]["calls"] += 1
            breakdown[item.model]["input_tokens"] += item.input_tokens
            breakdown[item.model]["output_tokens"] += item.output_tokens
            breakdown[item.model]["total_cost"] += item.total_cost
            
        return breakdown

def shannon_entropy(texts: List[str]) -> float:
    """Shannon entropy 계산 (표준 Python)"""
    if not texts:
        return 0.0
    
    # 응답을 정규화하여 비교 (최대 160자)
    signatures = [" ".join((text or "").split())[:160] for text in texts]
    
    # 빈도 계산
    counter = Counter(signatures)
    total = sum(counter.values()) or 1
    
    # Shannon entropy 계산
    entropy = 0.0
    for count in counter.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    
    return entropy

def detect_contradiction(texts: List[str]) -> str:
    """모순 검출 (표준 Python)"""
    if not texts:
        return "None"
    
    # 소문자 변환
    lows = [(text or "").lower() for text in texts]
    
    # 간단한 yes/no 모순 검출
    has_yes = any((" yes " in " " + text + " " or "예" in text or "맞다" in text) for text in lows)
    has_no = any((" no " in " " + text + " " or "아니오" in text or "아니" in text or "틀렸다" in text) for text in lows)
    
    contradictions = []
    
    if has_yes and has_no:
        contradictions.append("Yes/No contradiction detected")
    
    # 추가 모순 패턴들
    has_true = any("true" in text or "참" in text for text in lows)
    has_false = any("false" in text or "거짓" in text for text in lows)
    
    if has_true and has_false:
        contradictions.append("True/False contradiction detected")
    
    return "; ".join(contradictions) if contradictions else "None"

class IndependentThinker:
    """독립적으로 문제를 해결하는 에이전트"""
    
    def __init__(self, name: str, cost_tracker: CostTracker, model: str = "gemma:2b"):
        self.name = name
        self.model = model
        self.cost_tracker = cost_tracker
        # LLM 인스턴스 생성 (create_llm_auto 사용)
        self.llm = create_llm_auto(model)

    def solve(self, problem: str) -> str:
        """주어진 문제를 지정된 LLM을 사용하여 해결"""
        try:
            # 한국어 프롬프트로 통일
            formatted_prompt = f"한국어로 간결하게 답하세요.\n\n질문: {problem}\n\n답변:"
            
            # simple_llm을 통한 통일된 호출
            response = self.llm.generate(
                formatted_prompt,
                temperature=0.2,
                max_tokens=220,
                timeout=60
            )
            
            # 응답 텍스트 추출
            text = (response.get("response") or "").strip()
            
            # 토큰 수 추출 (Ollama 메타데이터 or 추정)
            eval_count = response.get("eval_count")
            if isinstance(eval_count, int) and eval_count > 0:
                tokens = eval_count
            else:
                # 토큰 수 추정 (단어 수 기반)
                tokens = max(8, len(text.split()) + len(formatted_prompt.split()) // 2)
            
            # 입력/출력 토큰을 대략 절반씩 분배 (간이 추정)
            input_tokens = tokens // 2
            output_tokens = tokens - input_tokens
            
            # 비용 추적
            self.cost_tracker.add_cost(self.model, input_tokens, output_tokens)
            
            return text if text else f"[{self.name}] No response generated"
            
        except Exception as e:
            error_msg = f"[{self.name}] Error: {str(e)}"
            # 에러 발생시에도 최소 비용 기록
            self.cost_tracker.add_cost(self.model, 10, 5)  # 최소 추정
            return error_msg

class BiasDetector:
    """응답 편향성을 감지하는 간단한 모듈 (하위 호환성)"""
    
    def __init__(self):
        self.contradiction_pairs = [
            ("yes", "no"),
            ("true", "false"), 
            ("agree", "disagree"),
            ("correct", "incorrect"),
            ("예", "아니오"),
            ("맞다", "틀렸다")
        ]

    def calculate_shannon_entropy(self, responses: List[str]) -> float:
        """Shannon entropy를 사용한 응답 다양성 계산 (하위 호환성)"""
        return shannon_entropy(responses)

    def detect_simple_contradictions(self, responses: List[str]) -> str:
        """응답 목록에서 간단하고 직접적인 모순 감지 (하위 호환성)"""
        return detect_contradiction(responses)

class Mediator:
    """여러 독립 사고자의 결과를 집계하고 최종 답변을 제공"""
    
    def __init__(self, thinkers: List[IndependentThinker], cost_tracker: CostTracker):
        self.thinkers = thinkers
        self.cost_tracker = cost_tracker
        self.bias_detector = BiasDetector()  # 하위 호환성
        self.aggregation_strategy = "rule_based"  # 기본 전략

    def solve_problem(self, problem: str) -> Dict[str, Any]:
        """모든 독립 사고자를 통해 문제 해결 오케스트레이션"""
        print(f"\n--- Solving Problem: '{problem}' ---")
        
        # 모든 thinker로부터 응답 수집
        responses = []
        for thinker in self.thinkers:
            if hasattr(thinker, "solve"):
                response = thinker.solve(problem)
                responses.append(response)
                print(f"  {thinker.name}: {response[:60]}...")
            else:
                # 문자열이나 다른 객체인 경우
                responses.append(str(thinker))
        
        # 간단한 규칙 기반 집계: 첫 번째 유효한 응답 사용
        final_answer = next((r for r in responses if r and not r.startswith("Error")), 
                          "No valid answer could be determined.")
        
        # 정보 이론적 분석
        entropy = shannon_entropy(responses)
        contradiction_report = detect_contradiction(responses)
        
        print(f"  Shannon Entropy: {entropy:.3f}")
        print(f"  Contradictions: {contradiction_report}")
        print(f"  Final Answer: {final_answer[:80]}...")
        
        return {
            "problem": problem,
            "final_answer": final_answer,
            "all_responses": responses,
            "shannon_entropy": entropy,
            "contradiction_report": contradiction_report,
            # 추가 메타데이터
            "thinker_count": len(self.thinkers),
            "response_count": len(responses),
            "total_cost": self.cost_tracker.get_total_cost(),
            "aggregation_strategy": self.aggregation_strategy
        }

# 편의를 위한 팩토리 함수들
def create_independent_thinker(name: str, model: str = "gemma:2b", 
                             cost_tracker: CostTracker = None) -> IndependentThinker:
    """IndependentThinker 생성 헬퍼"""
    if cost_tracker is None:
        cost_tracker = CostTracker()
    return IndependentThinker(name, cost_tracker, model)

def create_multi_agent_system(agents_config: List[Dict[str, str]], 
                            shared_cost_tracker: CostTracker = None) -> Mediator:
    """멀티 에이전트 시스템 생성 헬퍼"""
    if shared_cost_tracker is None:
        shared_cost_tracker = CostTracker()
    
    thinkers = []
    for config in agents_config:
        name = config.get("name", "Agent")
        model = config.get("model", "gemma:2b")
        thinker = IndependentThinker(name, shared_cost_tracker, model)
        thinkers.append(thinker)
    
    return Mediator(thinkers, shared_cost_tracker)