# -*- coding: utf-8 -*-
"""
전체 파이프라인 통합 테스트
Draft→Review→Judge 완전한 플로우에서 A안 vs B안 최종 성능 비교
"""

import sys
import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
sys.path.append('.')

from src.llm.simple_llm import create_llm_auto
from src.orchestrator.thought_aggregator import ThoughtAggregator
from test_b_approach import BApproachReviewer
from src.utils.token_counter import UnifiedTokenCounter

@dataclass
class PipelineResult:
    """파이프라인 실행 결과"""
    approach: str
    question: str
    draft_responses: List[str]
    review_responses: List[str]
    judge_response: str
    total_tokens: int
    total_time_ms: int
    accuracy_score: float
    success: bool
    error: Optional[str] = None

class FullPipelineTester:
    """전체 파이프라인 테스트"""
    
    def __init__(self):
        self.llm = create_llm_auto("qwen2:0.5b")
        self.aggregator = ThoughtAggregator(model_name="qwen2:0.5b")
        self.b_reviewer = BApproachReviewer(model_name="qwen2:0.5b")
        self.token_counter = UnifiedTokenCounter()
    
    def run_a_approach_pipeline(self, question: str, expected_answer: str) -> PipelineResult:
        """A안 전체 파이프라인 실행"""
        
        start_time = time.time()
        total_tokens = 0
        
        try:
            # 1. Draft 단계
            draft_responses = []
            for i in range(3):
                prompt = f"질문: {question}\n\n답변하고 사고과정을 설명하세요:"
                response = self.llm.generate(prompt, temperature=0.3 + i*0.1, max_tokens=200)
                
                if isinstance(response, dict):
                    draft_text = response.get("response", "").strip()
                else:
                    draft_text = str(response).strip()
                
                draft_responses.append(draft_text)
                total_tokens += len(prompt.split()) + len(draft_text.split())
            
            # 2. Review 단계 (ThoughtAggregator 사용)
            review_responses = []
            for reviewer_id in range(2):
                # 사고과정 압축
                analysis = self.aggregator.analyze_thoughts(draft_responses, question)
                
                # 압축된 컨텍스트로 리뷰
                review_prompt = f"""압축된 사고과정을 분석하여 리뷰하세요:

{analysis.compressed_context}

질문: {question}
검토자 관점 {reviewer_id + 1}에서 비판적으로 분석하고 개선된 답변을 제시하세요:"""

                response = self.llm.generate(review_prompt, temperature=0.4, max_tokens=300)
                
                if isinstance(response, dict):
                    review_text = response.get("response", "").strip()
                else:
                    review_text = str(response).strip()
                
                review_responses.append(review_text)
                total_tokens += len(review_prompt.split()) + len(review_text.split())
            
            # 3. Judge 단계
            judge_prompt = f"""질문: {question}

Review 결과들:
Review 1: {review_responses[0]}
Review 2: {review_responses[1]}

두 리뷰를 종합하여 최종 정답을 결정하세요:"""

            judge_response = self.llm.generate(judge_prompt, temperature=0.2, max_tokens=200)
            
            if isinstance(judge_response, dict):
                judge_text = judge_response.get("response", "").strip()
            else:
                judge_text = str(judge_response).strip()
                
            total_tokens += len(judge_prompt.split()) + len(judge_text.split())
            
            # 정확도 평가
            accuracy = self._evaluate_accuracy(judge_text, expected_answer)
            
            return PipelineResult(
                approach="A안 (ThoughtAggregator)",
                question=question,
                draft_responses=draft_responses,
                review_responses=review_responses,
                judge_response=judge_text,
                total_tokens=total_tokens,
                total_time_ms=int((time.time() - start_time) * 1000),
                accuracy_score=accuracy,
                success=True
            )
            
        except Exception as e:
            return PipelineResult(
                approach="A안 (ThoughtAggregator)",
                question=question,
                draft_responses=[],
                review_responses=[],
                judge_response="",
                total_tokens=0,
                total_time_ms=0,
                accuracy_score=0.0,
                success=False,
                error=str(e)
            )
    
    def run_b_approach_pipeline(self, question: str, expected_answer: str) -> PipelineResult:
        """B안 전체 파이프라인 실행"""
        
        start_time = time.time()
        total_tokens = 0
        
        try:
            # 1. Draft 단계 (동일)
            draft_responses = []
            for i in range(3):
                prompt = f"질문: {question}\n\n답변하고 사고과정을 설명하세요:"
                response = self.llm.generate(prompt, temperature=0.3 + i*0.1, max_tokens=200)
                
                if isinstance(response, dict):
                    draft_text = response.get("response", "").strip()
                else:
                    draft_text = str(response).strip()
                
                draft_responses.append(draft_text)
                total_tokens += len(prompt.split()) + len(draft_text.split())
            
            # 2. Review 단계 (개선된 프롬프트 사용)
            review_responses = []
            for reviewer_id in range(2):
                review_prompt = f"""다음 질문에 대한 3개의 Draft 답변을 분석하여 리뷰하세요.

질문: {question}

Draft 답변들:
{chr(10).join(f"Draft {i+1}: {resp}" for i, resp in enumerate(draft_responses))}

검토자 {reviewer_id + 1} 관점에서:
1. 공통 핵심 아이디어 추출
2. 독특한 접근법들의 장단점 분석  
3. 가장 합리적인 통합 답변 제시

답변:"""

                response = self.llm.generate(review_prompt, temperature=0.4, max_tokens=300)
                
                if isinstance(response, dict):
                    review_text = response.get("response", "").strip()
                else:
                    review_text = str(response).strip()
                
                review_responses.append(review_text)
                total_tokens += len(review_prompt.split()) + len(review_text.split())
            
            # 3. Judge 단계 (동일)
            judge_prompt = f"""질문: {question}

Review 결과들:
Review 1: {review_responses[0]}
Review 2: {review_responses[1]}

두 리뷰를 종합하여 최종 정답을 결정하세요:"""

            judge_response = self.llm.generate(judge_prompt, temperature=0.2, max_tokens=200)
            
            if isinstance(judge_response, dict):
                judge_text = judge_response.get("response", "").strip()
            else:
                judge_text = str(judge_response).strip()
                
            total_tokens += len(judge_prompt.split()) + len(judge_text.split())
            
            # 정확도 평가
            accuracy = self._evaluate_accuracy(judge_text, expected_answer)
            
            return PipelineResult(
                approach="B안 (프롬프트개선)",
                question=question,
                draft_responses=draft_responses,
                review_responses=review_responses,
                judge_response=judge_text,
                total_tokens=total_tokens,
                total_time_ms=int((time.time() - start_time) * 1000),
                accuracy_score=accuracy,
                success=True
            )
            
        except Exception as e:
            return PipelineResult(
                approach="B안 (프롬프트개선)",
                question=question,
                draft_responses=[],
                review_responses=[],
                judge_response="",
                total_tokens=0,
                total_time_ms=0,
                accuracy_score=0.0,
                success=False,
                error=str(e)
            )
    
    def run_single_model_baseline(self, question: str, expected_answer: str) -> PipelineResult:
        """Single 모델 베이스라인"""
        
        start_time = time.time()
        
        try:
            prompt = f"질문: {question}\n\n답변하세요:"
            response = self.llm.generate(prompt, temperature=0.3, max_tokens=200)
            
            if isinstance(response, dict):
                answer_text = response.get("response", "").strip()
            else:
                answer_text = str(response).strip()
            
            total_tokens = len(prompt.split()) + len(answer_text.split())
            accuracy = self._evaluate_accuracy(answer_text, expected_answer)
            
            return PipelineResult(
                approach="Single Model",
                question=question,
                draft_responses=[],
                review_responses=[],
                judge_response=answer_text,
                total_tokens=total_tokens,
                total_time_ms=int((time.time() - start_time) * 1000),
                accuracy_score=accuracy,
                success=True
            )
            
        except Exception as e:
            return PipelineResult(
                approach="Single Model",
                question=question,
                draft_responses=[],
                review_responses=[],
                judge_response="",
                total_tokens=0,
                total_time_ms=0,
                accuracy_score=0.0,
                success=False,
                error=str(e)
            )
    
    def _evaluate_accuracy(self, response: str, expected_answer: str) -> float:
        """정확도 평가 (간단한 키워드 매칭)"""
        response_lower = response.lower()
        expected_lower = expected_answer.lower()
        
        # 핵심 키워드가 포함되어 있는지 확인
        if expected_lower in response_lower:
            return 1.0
        
        # 부분 매칭 (키워드 기반)
        expected_words = set(expected_lower.split())
        response_words = set(response_lower.split())
        
        if not expected_words:
            return 0.0
            
        matching_words = expected_words.intersection(response_words)
        return len(matching_words) / len(expected_words)

def run_full_pipeline_comparison():
    """전체 파이프라인 A안 vs B안 vs Single 비교"""
    
    print("=" * 80)
    print("전체 파이프라인 비교: A안 vs B안 vs Single Model")
    print("=" * 80)
    
    tester = FullPipelineTester()
    
    # 테스트 케이스
    test_cases = [
        {
            "question": "What is the capital of South Korea?",
            "expected_answer": "Seoul"
        },
        {
            "question": "What is 2+2?",
            "expected_answer": "4"
        },
        {
            "question": "What is the largest planet in our solar system?",
            "expected_answer": "Jupiter"
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "expected_answer": "Shakespeare"
        }
    ]
    
    all_results = []
    
    for i, test_case in enumerate(test_cases):
        question = test_case["question"]
        expected = test_case["expected_answer"]
        
        print(f"\n{i+1}. 테스트: {question}")
        print("=" * 60)
        
        # A안 실행
        print("A안 (ThoughtAggregator) 실행 중...")
        a_result = tester.run_a_approach_pipeline(question, expected)
        
        # B안 실행
        print("B안 (프롬프트개선) 실행 중...")
        b_result = tester.run_b_approach_pipeline(question, expected)
        
        # Single 모델 실행
        print("Single Model 실행 중...")
        single_result = tester.run_single_model_baseline(question, expected)
        
        # 결과 출력
        print(f"\n--- 결과 비교 ---")
        for result in [a_result, b_result, single_result]:
            if result.success:
                print(f"{result.approach}:")
                print(f"  최종 답변: {result.judge_response[:100]}...")
                print(f"  정확도: {result.accuracy_score:.2f}")
                print(f"  토큰 수: {result.total_tokens}")
                print(f"  실행 시간: {result.total_time_ms}ms")
                print(f"  효율성: {result.accuracy_score/result.total_tokens:.4f}")
            else:
                print(f"{result.approach}: 실패 - {result.error}")
            print()
        
        all_results.extend([a_result, b_result, single_result])
    
    # 전체 결과 요약
    print_final_summary(all_results)
    
    # 결과를 파일로 저장
    save_results_to_file(all_results)
    
    return all_results

def print_final_summary(results: List[PipelineResult]):
    """최종 결과 요약"""
    
    print("\n" + "=" * 80)
    print("최종 실험 결과 요약")
    print("=" * 80)
    
    # 접근법별로 그룹화
    approaches = {}
    for result in results:
        if result.success:
            if result.approach not in approaches:
                approaches[result.approach] = []
            approaches[result.approach].append(result)
    
    for approach, approach_results in approaches.items():
        if not approach_results:
            continue
            
        avg_accuracy = sum(r.accuracy_score for r in approach_results) / len(approach_results)
        avg_tokens = sum(r.total_tokens for r in approach_results) / len(approach_results)
        avg_time = sum(r.total_time_ms for r in approach_results) / len(approach_results)
        avg_efficiency = avg_accuracy / avg_tokens if avg_tokens > 0 else 0
        
        print(f"\n{approach}:")
        print(f"  성공률: {len(approach_results)}/{len([r for r in results if r.approach == approach])} 건")
        print(f"  평균 정확도: {avg_accuracy:.3f}")
        print(f"  평균 토큰 수: {avg_tokens:.1f}")
        print(f"  평균 실행 시간: {avg_time:.1f}ms")
        print(f"  효율성 (정확도/토큰): {avg_efficiency:.6f}")

def save_results_to_file(results: List[PipelineResult]):
    """결과를 파일로 저장"""
    
    # 직렬화 가능한 형태로 변환
    serializable_results = []
    for result in results:
        serializable_results.append({
            "approach": result.approach,
            "question": result.question,
            "judge_response": result.judge_response,
            "total_tokens": result.total_tokens,
            "total_time_ms": result.total_time_ms,
            "accuracy_score": result.accuracy_score,
            "success": result.success,
            "error": result.error
        })
    
    # 타임스탬프 포함한 파일명
    timestamp = int(time.time())
    filename = f"results/full_pipeline_comparison_{timestamp}.json"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        print(f"\n결과 저장됨: {filename}")
    except Exception as e:
        print(f"파일 저장 실패: {e}")

if __name__ == "__main__":
    results = run_full_pipeline_comparison()