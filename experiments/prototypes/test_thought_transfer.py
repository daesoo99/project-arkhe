# -*- coding: utf-8 -*-
"""
사고 전달 실험: Draft→Review 사고과정 전달 테스트
A안(ThoughtAggregator) vs B안(프롬프트개선) 사고과정 전달 비교
"""

import sys
import time
from typing import List, Dict, Any
sys.path.append('.')

from src.llm.simple_llm import create_llm_auto
from src.orchestrator.thought_aggregator import ThoughtAggregator
from test_b_approach import BApproachReviewer

class ThoughtTransferTester:
    """사고과정 전달 테스트"""
    
    def __init__(self):
        self.llm = create_llm_auto("qwen2:0.5b")
        self.aggregator = ThoughtAggregator(model_name="qwen2:0.5b")
        self.b_reviewer = BApproachReviewer(model_name="qwen2:0.5b")
    
    def create_draft_responses(self, question: str, num_drafts: int = 3) -> List[Dict[str, Any]]:
        """Draft 응답 생성"""
        
        draft_prompts = [
            f"질문: {question}\n\n답변하고 당신의 사고과정도 설명하세요:",
            f"다음 질문에 답하세요. 추론 과정을 포함하세요: {question}",
            f"{question}\n\n논리적 근거와 함께 답변하세요:"
        ]
        
        drafts = []
        for i, prompt in enumerate(draft_prompts[:num_drafts]):
            try:
                response = self.llm.generate(prompt, temperature=0.3 + i*0.1, max_tokens=200)
                if isinstance(response, dict):
                    draft_text = response.get("response", "").strip()
                else:
                    draft_text = str(response).strip()
                
                drafts.append({
                    "id": f"draft_{i+1}",
                    "response": draft_text,
                    "token_count": len(prompt.split()) + len(draft_text.split())
                })
            except Exception as e:
                drafts.append({
                    "id": f"draft_{i+1}",
                    "response": f"Error: {e}",
                    "token_count": 0
                })
        
        return drafts
    
    def test_a_approach(self, drafts: List[Dict], question: str) -> Dict[str, Any]:
        """A안 테스트: ThoughtAggregator 사용"""
        
        draft_responses = [d["response"] for d in drafts]
        
        try:
            # 1단계: 사고과정 압축
            start_time = time.time()
            analysis = self.aggregator.analyze_thoughts(draft_responses, question)
            compression_time = int((time.time() - start_time) * 1000)
            
            # 2단계: 압축된 결과를 Review에게 전달
            review_prompt = f"""다음은 3개 Draft의 압축된 사고과정입니다:

{analysis.compressed_context}

이를 바탕으로 최종 답변을 작성하세요:"""
            
            start_time = time.time()
            review_response = self.llm.generate(review_prompt, temperature=0.4, max_tokens=300)
            review_time = int((time.time() - start_time) * 1000)
            
            if isinstance(review_response, dict):
                review_text = review_response.get("response", "").strip()
            else:
                review_text = str(review_response).strip()
            
            return {
                "approach": "A안 (ThoughtAggregator)",
                "compressed_context": analysis.compressed_context,
                "compression_ratio": analysis.compression_ratio,
                "review_response": review_text,
                "total_time_ms": compression_time + review_time,
                "compression_time_ms": compression_time,
                "review_time_ms": review_time,
                "success": True
            }
            
        except Exception as e:
            return {
                "approach": "A안 (ThoughtAggregator)",
                "success": False,
                "error": str(e)
            }
    
    def test_b_approach(self, drafts: List[Dict], question: str) -> Dict[str, Any]:
        """B안 테스트: 프롬프트 개선"""
        
        draft_responses = [d["response"] for d in drafts]
        
        try:
            result = self.b_reviewer.review_with_thought_analysis(draft_responses, question)
            result["approach"] = "B안 (프롬프트개선)"
            return result
            
        except Exception as e:
            return {
                "approach": "B안 (프롬프트개선)",  
                "success": False,
                "error": str(e)
            }

def run_thought_transfer_experiment():
    """사고 전달 실험 실행"""
    
    print("=" * 70)
    print("사고과정 전달 실험: A안 vs B안")
    print("=" * 70)
    
    tester = ThoughtTransferTester()
    
    # 테스트 질문들
    test_questions = [
        "What is the capital of South Korea?",
        "What is 2+2?", 
        "Why do seasons change?",
        "What are the benefits of renewable energy?"
    ]
    
    results = []
    
    for i, question in enumerate(test_questions):
        print(f"\n{i+1}. 질문: {question}")
        print("=" * 50)
        
        # Draft 응답 생성
        print("Draft 응답 생성 중...")
        drafts = tester.create_draft_responses(question)
        
        for j, draft in enumerate(drafts):
            print(f"Draft {j+1}: {draft['response'][:100]}{'...' if len(draft['response']) > 100 else ''}")
        
        # A안 테스트
        print(f"\nA안 (ThoughtAggregator) 테스트...")
        a_result = tester.test_a_approach(drafts, question)
        
        # B안 테스트  
        print(f"B안 (프롬프트개선) 테스트...")
        b_result = tester.test_b_approach(drafts, question)
        
        # 결과 출력
        print(f"\n--- 결과 비교 ---")
        
        if a_result['success']:
            print(f"A안:")
            print(f"  압축된 컨텍스트: {a_result.get('compressed_context', '')[:150]}...")
            print(f"  압축률: {a_result.get('compression_ratio', 0):.3f}")
            print(f"  Review 응답: {a_result.get('review_response', '')[:150]}...")
            print(f"  총 시간: {a_result.get('total_time_ms', 0)}ms")
        else:
            print(f"A안 실패: {a_result.get('error', '')}")
        
        if b_result['success']:
            print(f"B안:")
            print(f"  Review 응답: {b_result.get('review_response', '')[:150]}...")
            print(f"  응답 시간: {b_result.get('latency_ms', 0)}ms")
            print(f"  토큰 수: {b_result.get('token_count', 0)}")
        else:
            print(f"B안 실패: {b_result.get('error', '')}")
        
        # 사고과정 전달 품질 평가
        print(f"\n--- 사고과정 전달 평가 ---")
        evaluate_thought_transfer_quality(drafts, a_result, b_result)
        
        results.append({
            "question": question,
            "drafts": drafts,
            "a_result": a_result,
            "b_result": b_result
        })
    
    # 전체 결과 요약
    print_experiment_summary(results)
    
    return results

def evaluate_thought_transfer_quality(drafts: List[Dict], a_result: Dict, b_result: Dict):
    """사고과정 전달 품질 평가"""
    
    # Draft들에서 핵심 사고과정 키워드 추출
    draft_keywords = set()
    for draft in drafts:
        words = draft['response'].lower().split()
        # 사고과정 관련 키워드들 (reasoning, because, think, etc.)
        thinking_words = [w for w in words if any(kw in w for kw in 
                         ['because', 'think', 'reason', 'figur', 'know', 'logic', 'explain'])]
        draft_keywords.update(thinking_words)
    
    print(f"Draft 사고과정 키워드: {list(draft_keywords)[:5]}...")
    
    # A안에서 키워드 보존도
    if a_result['success'] and draft_keywords:
        a_keywords_preserved = sum(1 for kw in draft_keywords 
                                 if kw in a_result.get('review_response', '').lower())
        a_preservation_rate = a_keywords_preserved / len(draft_keywords)
        print(f"A안 사고과정 보존률: {a_preservation_rate:.2f}")
    
    # B안에서 키워드 보존도  
    if b_result['success'] and draft_keywords:
        b_keywords_preserved = sum(1 for kw in draft_keywords 
                                 if kw in b_result.get('review_response', '').lower())
        b_preservation_rate = b_keywords_preserved / len(draft_keywords)
        print(f"B안 사고과정 보존률: {b_preservation_rate:.2f}")

def print_experiment_summary(results: List[Dict]):
    """실험 결과 요약"""
    
    print("\n" + "=" * 70)
    print("실험 결과 요약")
    print("=" * 70)
    
    a_success_count = sum(1 for r in results if r['a_result']['success'])
    b_success_count = sum(1 for r in results if r['b_result']['success'])
    
    print(f"성공률:")
    print(f"  A안: {a_success_count}/{len(results)} ({a_success_count/len(results)*100:.1f}%)")
    print(f"  B안: {b_success_count}/{len(results)} ({b_success_count/len(results)*100:.1f}%)")
    
    # 평균 응답 시간
    a_times = [r['a_result'].get('total_time_ms', 0) for r in results if r['a_result']['success']]
    b_times = [r['b_result'].get('latency_ms', 0) for r in results if r['b_result']['success']]
    
    if a_times:
        print(f"평균 응답 시간:")
        print(f"  A안: {sum(a_times)/len(a_times):.1f}ms")
    if b_times:
        print(f"  B안: {sum(b_times)/len(b_times):.1f}ms")

if __name__ == "__main__":
    results = run_thought_transfer_experiment()