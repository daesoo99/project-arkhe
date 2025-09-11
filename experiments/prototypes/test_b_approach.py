# -*- coding: utf-8 -*-
"""
B안 구현 및 테스트: 프롬프트 개선 방식
Review 프롬프트에서 사고과정 분석하도록 개선
"""

import sys
import time
from typing import List, Dict, Any
sys.path.append('.')

from src.llm.simple_llm import create_llm_auto

class BApproachReviewer:
    """B안: 프롬프트 개선 방식 Reviewer"""
    
    def __init__(self, model_name: str = "qwen2:0.5b"):
        self.model_name = model_name
        self.llm = create_llm_auto(model_name)
    
    def review_with_thought_analysis(self, draft_responses: List[str], question: str) -> Dict[str, Any]:
        """사고과정 분석을 통한 리뷰"""
        
        # 개선된 프롬프트: 사고과정 분석 모드
        prompt = f"""다음 질문에 대한 3개의 Draft 답변을 분석하여 종합적인 리뷰를 작성하세요.

질문: {question}

Draft 답변들:
{chr(10).join(f"Draft {i+1}: {resp}" for i, resp in enumerate(draft_responses))}

작업:
1. 공통 핵심 아이디어 추출: 모든 Draft가 동의하는 핵심 내용
2. 독특한 접근 분석: 각 Draft의 독창적인 사고과정과 근거 파악
3. 통합 답변 생성: 공통 아이디어 + 가장 합리적인 독창적 접근법들을 결합

출력 형식:
## 공통 핵심
[모든 Draft가 동의하는 내용]

## 독창적 접근법들
- Draft 1 특징: [고유한 사고과정/근거]  
- Draft 2 특징: [고유한 사고과정/근거]
- Draft 3 특징: [고유한 사고과정/근거]

## 통합 판단
[공통 아이디어를 기반으로, 가장 설득력 있는 독창적 접근법들을 통합한 최종 답변]

최종 답변:"""

        try:
            start_time = time.time()
            response = self.llm.generate(prompt, temperature=0.4, max_tokens=400)
            latency_ms = int((time.time() - start_time) * 1000)
            
            if isinstance(response, dict):
                result_text = response.get("response", "").strip()
            else:
                result_text = str(response).strip()
            
            return {
                "review_response": result_text,
                "latency_ms": latency_ms,
                "token_count": len(prompt.split()) + len(result_text.split()),
                "success": True
            }
            
        except Exception as e:
            return {
                "review_response": "",
                "latency_ms": 0,
                "token_count": 0,
                "success": False,
                "error": str(e)
            }

def test_b_approach():
    """B안 사고과정 분석 테스트"""
    
    print("=" * 60)
    print("B안 테스트: 프롬프트 개선 방식")
    print("=" * 60)
    
    # 테스트 케이스
    test_cases = [
        {
            "question": "What is the capital of South Korea?",
            "drafts": [
                "Seoul is the capital of South Korea. I know this because it has 10 million people, making it the largest city, so it must be the capital.",
                "The capital city of South Korea is Seoul. I figured this out by thinking about K-pop and Samsung headquarters - they're both in Seoul, indicating it's the economic center and therefore the capital.",
                "Seoul is South Korea's capital. My reasoning: Seoul has Gyeongbokgung Palace and other historical sites, showing it has been the political center for centuries."
            ]
        },
        {
            "question": "What is 2+2?",
            "drafts": [
                "2+2=4. I added 2 and 2.",
                "2+2 equals 4. I used basic addition.",
                "The answer is 4. I calculated 2 plus 2."
            ]
        }
    ]
    
    reviewer = BApproachReviewer()
    
    for i, test_case in enumerate(test_cases):
        print(f"\n{i+1}. 테스트 케이스: {test_case['question']}")
        print("-" * 40)
        
        # Draft 답변들 출력
        for j, draft in enumerate(test_case['drafts']):
            print(f"Draft {j+1}: {draft}")
        
        # B안 리뷰 실행
        result = reviewer.review_with_thought_analysis(test_case['drafts'], test_case['question'])
        
        if result['success']:
            print(f"\nB안 Review 결과:")
            print(result['review_response'])
            print(f"\n메트릭:")
            print(f"- 응답 시간: {result['latency_ms']}ms")
            print(f"- 토큰 수: {result['token_count']}")
        else:
            print(f"오류: {result.get('error', 'Unknown error')}")
        
        print("\n" + "="*60)

def compare_token_efficiency():
    """토큰 효율성 비교"""
    
    print("토큰 효율성 분석")
    print("=" * 40)
    
    # 기존 방식 (단순 누적)
    drafts = [
        "Seoul is the capital of South Korea. I know this because it has 10 million people, making it the largest city, so it must be the capital.",
        "The capital city of South Korea is Seoul. I figured this out by thinking about K-pop and Samsung headquarters - they're both in Seoul, indicating it's the economic center and therefore the capital.",
        "Seoul is South Korea's capital. My reasoning: Seoul has Gyeongbokgung Palace and other historical sites, showing it has been the political center for centuries."
    ]
    
    # 1. 원본 누적 방식 토큰 수
    original_cumulative = "질문: What is the capital of South Korea?\n\n" + "\n".join(f"Draft {i+1}: {draft}" for i, draft in enumerate(drafts))
    original_tokens = len(original_cumulative.split())
    
    # 2. B안 프롬프트 토큰 수 (구조화된 분석)
    b_approach_prompt_base = "다음 질문에 대한 3개의 Draft 답변을 분석하여 종합적인 리뷰를 작성하세요."
    b_approach_full = len(BApproachReviewer().review_with_thought_analysis(drafts, "What is the capital of South Korea?"))
    
    print(f"원본 누적 방식: {original_tokens} 토큰")
    print(f"B안 구조화 분석: 프롬프트 구조화로 효율성 개선 예상")
    print(f"핵심: 단순 나열 → 구조화된 사고과정 분석")

if __name__ == "__main__":
    test_b_approach()
    print("\n")
    compare_token_efficiency()