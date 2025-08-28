# -*- coding: utf-8 -*-
"""
Project Arkhē - ThoughtAggregator
사고과정 압축 및 정보 최적화 컴포넌트

핵심 기능:
- 공통 핵심 아이디어 추출
- 개별 독창적 접근법 분석
- "공통 핵심 + 개별 특징" 형태로 압축된 컨텍스트 생성
"""

import sys
import os
from typing import List, Dict, Any
from dataclasses import dataclass

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from llm.simple_llm import create_llm_auto

@dataclass
class ThoughtAnalysis:
    """사고 분석 결과"""
    common_core: str           # 공통 핵심 아이디어
    unique_approaches: List[str]  # 각각의 독창적 접근법
    compressed_context: str    # 압축된 전체 컨텍스트
    original_tokens: int       # 원본 토큰 수
    compressed_tokens: int     # 압축 후 토큰 수
    compression_ratio: float   # 압축률 (1.0 = 압축 없음)

class ThoughtAggregator:
    """사고과정 분석 및 압축 컴포넌트"""
    
    def __init__(self, model_name: str = "qwen2:0.5b"):
        """
        Args:
            model_name: 분석용 LLM 모델 (경제적인 모델 권장)
        """
        self.model_name = model_name
        self.llm = create_llm_auto(model_name)
        
    def analyze_thoughts(self, responses: List[str], context: str = "") -> ThoughtAnalysis:
        """
        여러 응답을 분석하여 공통 요소와 개별 특징을 추출
        
        Args:
            responses: 분석할 응답들 (Draft 또는 Review 결과들)
            context: 추가 컨텍스트 (원본 질문 등)
            
        Returns:
            ThoughtAnalysis: 분석 결과
        """
        if not responses:
            return ThoughtAnalysis("", [], "", 0, 0, 1.0)
        
        # 토큰 수 계산 (원본)
        original_text = " | ".join(responses)
        original_tokens = self._count_tokens(original_text + context)
        
        # 공통 요소 추출
        common_core = self._extract_common_elements(responses, context)
        
        # 개별 특징 분석
        unique_approaches = self._identify_unique_approaches(responses, common_core)
        
        # 압축된 컨텍스트 생성
        compressed_context = self._create_compressed_context(common_core, unique_approaches, context)
        
        # 압축 후 토큰 수
        compressed_tokens = self._count_tokens(compressed_context)
        compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
        
        # 압축 실패 감지: 원본보다 길어지면 원본 사용
        if compression_ratio > 1.0:
            print(f"  압축 실패 감지: {compression_ratio:.2f} > 1.0, 원본 사용")
            compressed_context = original_text  # 원본으로 폴백
            compressed_tokens = original_tokens
            compression_ratio = 1.0
        
        return ThoughtAnalysis(
            common_core=common_core,
            unique_approaches=unique_approaches,
            compressed_context=compressed_context,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compression_ratio
        )
    
    def _extract_common_elements(self, responses: List[str], context: str) -> str:
        """공통 핵심 아이디어 추출"""
        if len(responses) <= 1:
            return responses[0] if responses else ""
        
        prompt = f"""다음 여러 답변들을 분석하여 공통된 결론과 사고과정을 추출해주세요.

원본 질문: {context}

답변들:
{chr(10).join(f"{i+1}. {resp}" for i, resp in enumerate(responses))}

작업:
1. 공통된 최종 결론을 찾으세요
2. 각 답변이 그 결론에 도달한 사고과정/근거/접근방식을 파악하세요
3. 결론 + 사고과정을 간결하게 정리하세요

출력 형식:
[공통 결론] 모든 답변이 동의하는 핵심 내용
[사고과정] A는 X관점으로, B는 Y관점으로, C는 Z관점으로 접근

공통 핵심과 사고과정:"""

        try:
            response = self.llm.generate(prompt, temperature=0.3, max_tokens=200)
            if isinstance(response, dict):
                return response.get("response", "").strip()
            return str(response).strip()
        except Exception as e:
            print(f"공통 요소 추출 오류: {e}")
            # 폴백: 첫 번째 응답 반환
            return responses[0] if responses else ""
    
    def _identify_unique_approaches(self, responses: List[str], common_core: str) -> List[str]:
        """개별 독창적 접근법 분석"""
        if len(responses) <= 1:
            return []
        
        prompt = f"""다음 답변들에서 각각의 독특한 사고과정과 접근방식을 찾아주세요.

공통 결론과 사고과정: {common_core}

답변들:
{chr(10).join(f"{i+1}. {resp}" for i, resp in enumerate(responses))}

작업:
1. 각 답변의 독특한 추론 방식, 근거, 관점을 찾으세요
2. 다른 답변과 구별되는 사고의 특징을 파악하세요
3. 각 사고과정의 차별점을 간결하게 표현하세요

출력 형식:
답변1: [고유한 사고과정/관점]
답변2: [고유한 사고과정/관점]
답변3: [고유한 사고과정/관점]

독특한 사고과정들:"""

        try:
            response = self.llm.generate(prompt, temperature=0.4, max_tokens=300)
            if isinstance(response, dict):
                result_text = response.get("response", "").strip()
            else:
                result_text = str(response).strip()
            
            # 결과를 줄 단위로 분리하여 리스트로 변환
            unique_approaches = [
                line.strip() 
                for line in result_text.split('\n') 
                if line.strip() and not line.strip().startswith('-')
            ]
            
            return unique_approaches[:len(responses)]  # 최대 응답 수만큼만
        except Exception as e:
            print(f"개별 특징 분석 오류: {e}")
            # 폴백: 각 응답의 앞부분을 간단히 반환
            return [resp[:50] + "..." for resp in responses[:3]]
    
    def _create_compressed_context(self, common_core: str, unique_approaches: List[str], context: str) -> str:
        """압축된 컨텍스트 생성"""
        if not common_core and not unique_approaches:
            return context
        
        prompt = f"""다음 정보를 바탕으로 사고과정이 보존된 압축 컨텍스트를 생성해주세요.

원본 질문: {context}

공통 결론과 사고과정: {common_core}

독특한 사고과정들:
{chr(10).join(f"- {approach}" for approach in unique_approaches)}

작업:
1. 공통 결론을 1-2줄로 간결하게 정리
2. 사고과정의 핵심만 단어/구문 수준으로 압축
3. 원본보다 반드시 짧게 만들 것
4. 불필요한 설명 절대 추가 금지

출력 형식 (반드시 간결하게):
[결론] 공통 결론 (1줄)
[사고] 접근방식 차이 (핵심 단어들만)

압축된 컨텍스트 (원본보다 짧게):"""

        try:
            response = self.llm.generate(prompt, temperature=0.2, max_tokens=150)
            if isinstance(response, dict):
                compressed = response.get("response", "").strip()
            else:
                compressed = str(response).strip()
            
            # 폴백 처리: LLM 호출 실패 시
            if not compressed:
                # 간단한 템플릿 기반 압축
                base = f"[핵심] {common_core[:100]}" if common_core else ""
                unique_summary = " | ".join(approach[:50] for approach in unique_approaches[:3])
                compressed = f"{base}\n[특징] {unique_summary}" if unique_summary else base
                
            return compressed if compressed else context
            
        except Exception as e:
            print(f"컨텍스트 압축 오류: {e}")
            # 폴백: 간단한 문자열 조합
            parts = []
            if common_core:
                parts.append(f"핵심: {common_core[:100]}")
            if unique_approaches:
                parts.append(f"특징: {' | '.join(unique_approaches[:2])}")
            return "\n".join(parts) if parts else context
    
    def _count_tokens(self, text: str) -> int:
        """토큰 수 계산"""
        try:
            import tiktoken
            encoder = tiktoken.encoding_for_model("gpt-4")
            return len(encoder.encode(text))
        except ImportError:
            return len(text.split())  # 폴백