# -*- coding: utf-8 -*-
"""
개선된 Judge 프롬프트 테스트
Judge가 원본 Draft들도 보고 Review들의 오류를 무시할 수 있도록 개선
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

@dataclass
class ImprovedPipelineResult:
    """개선된 파이프라인 실행 결과"""
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

class ImprovedJudgeTester:
    """개선된 Judge 프롬프트 테스트"""
    
    def __init__(self):
        self.llm = create_llm_auto("qwen2:0.5b")
        self.aggregator = ThoughtAggregator(model_name="qwen2:0.5b")
        self.b_reviewer = BApproachReviewer(model_name="qwen2:0.5b")
    
    def run_improved_a_approach(self, question: str, expected_answer: str) -> ImprovedPipelineResult:
        """A안 + 개선된 Judge"""
        
        start_time = time.time()
        total_tokens = 0
        
        try:
            # 1. Draft 단계
            draft_responses = []
            for i in range(3):
                prompt = f"질문: {question}\n\n답변하고 사고과정을 설명하세요:"
                response = self.llm.generate(prompt, temperature=0.3 + i*0.1, max_tokens=150)
                
                if isinstance(response, dict):
                    draft_text = response.get("response", "").strip()
                else:
                    draft_text = str(response).strip()
                
                draft_responses.append(draft_text)
                total_tokens += len(prompt.split()) + len(draft_text.split())
            
            # 2. Review 단계 (A안: ThoughtAggregator 사용)
            review_responses = []
            for reviewer_id in range(2):
                # 사고과정 압축
                analysis = self.aggregator.analyze_thoughts(draft_responses, question)
                
                # 압축된 컨텍스트로 리뷰
                review_prompt = f"""압축된 사고과정을 분석하여 리뷰하세요:

{analysis.compressed_context}

질문: {question}
검토자 관점 {reviewer_id + 1}에서 비판적으로 분석하고 개선된 답변을 제시하세요:"""

                response = self.llm.generate(review_prompt, temperature=0.4, max_tokens=200)
                
                if isinstance(response, dict):
                    review_text = response.get("response", "").strip()
                else:
                    review_text = str(response).strip()
                
                review_responses.append(review_text)
                total_tokens += len(review_prompt.split()) + len(review_text.split())
            
            # 3. 개선된 Judge 단계
            judge_prompt = f"""질문: {question}

Draft 원본들:
{chr(10).join(f"Draft {i+1}: {draft}" for i, draft in enumerate(draft_responses))}

Review 분석들:
Review 1: {review_responses[0]}
Review 2: {review_responses[1]}

작업 지시:
1. 원래 질문에 정확히 답하는 것이 최우선입니다
2. Draft들에서 유용한 정보나 사고과정을 찾으세요
3. Review들에서 도움이 되는 분석이 있다면 참고하세요
4. 하지만 Draft나 Review가 명백히 틀렸다면 무시하고 질문에 직접 답하세요
5. 간결하고 정확한 최종 답변을 제시하세요

최종 답변:"""

            judge_response = self.llm.generate(judge_prompt, temperature=0.2, max_tokens=150)
            
            if isinstance(judge_response, dict):
                judge_text = judge_response.get("response", "").strip()
            else:
                judge_text = str(judge_response).strip()
                
            total_tokens += len(judge_prompt.split()) + len(judge_text.split())
            
            # 정확도 평가
            accuracy = self._evaluate_accuracy(judge_text, expected_answer)
            
            return ImprovedPipelineResult(
                approach="A안 + 개선된 Judge",
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
            return ImprovedPipelineResult(
                approach="A안 + 개선된 Judge",
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
    
    def run_improved_b_approach(self, question: str, expected_answer: str) -> ImprovedPipelineResult:
        """B안 + 개선된 Judge"""
        
        start_time = time.time()
        total_tokens = 0
        
        try:
            # 1. Draft 단계 (동일)
            draft_responses = []
            for i in range(3):
                prompt = f"질문: {question}\n\n답변하고 사고과정을 설명하세요:"
                response = self.llm.generate(prompt, temperature=0.3 + i*0.1, max_tokens=150)
                
                if isinstance(response, dict):
                    draft_text = response.get("response", "").strip()
                else:
                    draft_text = str(response).strip()
                
                draft_responses.append(draft_text)
                total_tokens += len(prompt.split()) + len(draft_text.split())
            
            # 2. Review 단계 (B안: 개선된 프롬프트 사용)
            review_responses = []
            for reviewer_id in range(2):
                review_prompt = f"""질문: {question}

Draft 답변들을 분석하여 리뷰하세요:
{chr(10).join(f"Draft {i+1}: {resp}" for i, resp in enumerate(draft_responses))}

검토자 {reviewer_id + 1} 관점에서:
1. Draft들에서 올바른 정보 식별
2. 잘못되거나 불완전한 부분 지적
3. 개선된 답변 제시

리뷰 결과:"""

                response = self.llm.generate(review_prompt, temperature=0.4, max_tokens=200)
                
                if isinstance(response, dict):
                    review_text = response.get("response", "").strip()
                else:
                    review_text = str(response).strip()
                
                review_responses.append(review_text)
                total_tokens += len(review_prompt.split()) + len(review_text.split())
            
            # 3. 개선된 Judge 단계 (동일한 프롬프트)
            judge_prompt = f"""질문: {question}

Draft 원본들:
{chr(10).join(f"Draft {i+1}: {draft}" for i, draft in enumerate(draft_responses))}

Review 분석들:
Review 1: {review_responses[0]}
Review 2: {review_responses[1]}

작업 지시:
1. 원래 질문에 정확히 답하는 것이 최우선입니다
2. Draft들에서 유용한 정보나 사고과정을 찾으세요
3. Review들에서 도움이 되는 분석이 있다면 참고하세요
4. 하지만 Draft나 Review가 명백히 틀렸다면 무시하고 질문에 직접 답하세요
5. 간결하고 정확한 최종 답변을 제시하세요

최종 답변:"""

            judge_response = self.llm.generate(judge_prompt, temperature=0.2, max_tokens=150)
            
            if isinstance(judge_response, dict):
                judge_text = judge_response.get("response", "").strip()
            else:
                judge_text = str(judge_response).strip()
                
            total_tokens += len(judge_prompt.split()) + len(judge_text.split())
            
            # 정확도 평가
            accuracy = self._evaluate_accuracy(judge_text, expected_answer)
            
            return ImprovedPipelineResult(
                approach="B안 + 개선된 Judge",
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
            return ImprovedPipelineResult(
                approach="B안 + 개선된 Judge",
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
    
    def run_single_model_baseline(self, question: str, expected_answer: str) -> ImprovedPipelineResult:
        """Single 모델 베이스라인"""
        
        start_time = time.time()
        
        try:
            prompt = f"질문: {question}\n\n답변하세요:"
            response = self.llm.generate(prompt, temperature=0.3, max_tokens=100)
            
            if isinstance(response, dict):
                answer_text = response.get("response", "").strip()
            else:
                answer_text = str(response).strip()
            
            total_tokens = len(prompt.split()) + len(answer_text.split())
            accuracy = self._evaluate_accuracy(answer_text, expected_answer)
            
            return ImprovedPipelineResult(
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
            return ImprovedPipelineResult(
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
        """정확도 평가"""
        response_lower = response.lower()
        expected_lower = expected_answer.lower()
        
        # 핵심 키워드가 포함되어 있는지 확인
        if expected_lower in response_lower:
            return 1.0
        
        # 부분 매칭
        expected_words = set(expected_lower.split())
        response_words = set(response_lower.split())
        
        if not expected_words:
            return 0.0
            
        matching_words = expected_words.intersection(response_words)
        return len(matching_words) / len(expected_words)

def run_improved_judge_comparison():
    """개선된 Judge 프롬프트 비교 실험"""
    
    print("=" * 80)
    print("개선된 Judge 프롬프트 테스트")
    print("=" * 80)
    
    tester = ImprovedJudgeTester()
    
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
        
        # 개선된 A안 실행
        print("개선된 A안 실행 중...")
        a_result = tester.run_improved_a_approach(question, expected)
        
        # 개선된 B안 실행  
        print("개선된 B안 실행 중...")
        b_result = tester.run_improved_b_approach(question, expected)
        
        # Single 모델 실행
        print("Single Model 실행 중...")
        single_result = tester.run_single_model_baseline(question, expected)
        
        # 결과 출력
        print(f"\n--- 결과 비교 ---")
        for result in [a_result, b_result, single_result]:
            if result.success:
                print(f"{result.approach}:")
                print(f"  최종 답변: {result.judge_response[:80]}...")
                print(f"  정확도: {result.accuracy_score:.2f}")
                print(f"  토큰 수: {result.total_tokens}")
                print(f"  실행 시간: {result.total_time_ms}ms")
                print(f"  효율성: {result.accuracy_score/result.total_tokens:.6f}")
            else:
                print(f"{result.approach}: 실패 - {result.error}")
            print()
        
        all_results.extend([a_result, b_result, single_result])
    
    # 전체 결과 요약
    print_improved_summary(all_results)
    
    return all_results

def print_improved_summary(results: List[ImprovedPipelineResult]):
    """개선 결과 요약"""
    
    print("\n" + "=" * 80)
    print("개선된 Judge 프롬프트 결과 요약")
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
        print(f"  평균 정확도: {avg_accuracy:.3f}")
        print(f"  평균 토큰 수: {avg_tokens:.1f}")
        print(f"  평균 실행 시간: {avg_time:.1f}ms")
        print(f"  효율성 (정확도/토큰): {avg_efficiency:.6f}")

if __name__ == "__main__":
    results = run_improved_judge_comparison()