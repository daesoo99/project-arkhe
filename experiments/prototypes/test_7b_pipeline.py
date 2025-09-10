# -*- coding: utf-8 -*-
"""
7B 모델 파이프라인 테스트
모든 Agent를 qwen2:7b로 업그레이드하여 성능 재측정
"""

import sys
import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
sys.path.append('.')

from src.llm.simple_llm import create_llm_auto
from src.orchestrator.thought_aggregator import ThoughtAggregator

@dataclass
class Model7BResult:
    """7B 모델 실행 결과"""
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

class Pipeline7BTester:
    """7B 모델 파이프라인 테스트"""
    
    def __init__(self):
        # 모든 Agent를 7B 모델로 설정
        self.draft_llm = create_llm_auto("qwen2:7b")
        self.review_llm = create_llm_auto("qwen2:7b") 
        self.judge_llm = create_llm_auto("qwen2:7b")
        self.aggregator = ThoughtAggregator(model_name="qwen2:7b")
    
    def run_7b_a_approach(self, question: str, expected_answer: str) -> Model7BResult:
        """A안 7B: ThoughtAggregator 사용"""
        
        start_time = time.time()
        total_tokens = 0
        
        try:
            print(f"  Draft 단계 시작...")
            # 1. Draft 단계 (7B 모델)
            draft_responses = []
            for i in range(3):
                prompt = f"질문: {question}\n\n답변하고 사고과정을 설명하세요:"
                response = self.draft_llm.generate(prompt, temperature=0.3 + i*0.1, max_tokens=200)
                
                if isinstance(response, dict):
                    draft_text = response.get("response", "").strip()
                else:
                    draft_text = str(response).strip()
                
                draft_responses.append(draft_text)
                total_tokens += len(prompt.split()) + len(draft_text.split())
                print(f"    Draft {i+1} 완료")
            
            print(f"  Review 단계 시작...")
            # 2. Review 단계 (7B + ThoughtAggregator)
            review_responses = []
            for reviewer_id in range(2):
                # 사고과정 압축
                analysis = self.aggregator.analyze_thoughts(draft_responses, question)
                
                # 압축된 컨텍스트로 리뷰
                review_prompt = f"""압축된 사고과정을 분석하여 리뷰하세요:

{analysis.compressed_context}

질문: {question}
검토자 관점 {reviewer_id + 1}에서:
1. 제시된 정보의 정확성 검증
2. 누락된 중요한 관점이나 정보 확인
3. 더 나은 답변을 위한 개선 사항 제시

리뷰 결과:"""

                response = self.review_llm.generate(review_prompt, temperature=0.4, max_tokens=250)
                
                if isinstance(response, dict):
                    review_text = response.get("response", "").strip()
                else:
                    review_text = str(response).strip()
                
                review_responses.append(review_text)
                total_tokens += len(review_prompt.split()) + len(review_text.split())
                print(f"    Review {reviewer_id+1} 완료")
            
            print(f"  Judge 단계 시작...")
            # 3. Judge 단계 (7B)
            judge_prompt = f"""질문: {question}

Draft 원본들:
{chr(10).join(f"Draft {i+1}: {draft}" for i, draft in enumerate(draft_responses))}

Review 분석들:
Review 1: {review_responses[0]}
Review 2: {review_responses[1]}

작업 지시:
1. 원래 질문에 정확하게 답하는 것이 최우선입니다
2. Draft들과 Review들을 종합적으로 분석하세요
3. 정확한 정보는 채택하고 잘못된 정보는 배제하세요
4. 간결하고 정확한 최종 답변을 제시하세요

최종 답변:"""

            judge_response = self.judge_llm.generate(judge_prompt, temperature=0.2, max_tokens=200)
            
            if isinstance(judge_response, dict):
                judge_text = judge_response.get("response", "").strip()
            else:
                judge_text = str(judge_response).strip()
                
            total_tokens += len(judge_prompt.split()) + len(judge_text.split())
            print(f"    Judge 완료")
            
            # 정확도 평가
            accuracy = self._evaluate_accuracy(judge_text, expected_answer)
            
            return Model7BResult(
                approach="A안 7B (ThoughtAggregator)",
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
            return Model7BResult(
                approach="A안 7B (ThoughtAggregator)",
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
    
    def run_7b_b_approach(self, question: str, expected_answer: str) -> Model7BResult:
        """B안 7B: 프롬프트 개선"""
        
        start_time = time.time()
        total_tokens = 0
        
        try:
            print(f"  Draft 단계 시작...")
            # 1. Draft 단계 (동일)
            draft_responses = []
            for i in range(3):
                prompt = f"질문: {question}\n\n답변하고 사고과정을 설명하세요:"
                response = self.draft_llm.generate(prompt, temperature=0.3 + i*0.1, max_tokens=200)
                
                if isinstance(response, dict):
                    draft_text = response.get("response", "").strip()
                else:
                    draft_text = str(response).strip()
                
                draft_responses.append(draft_text)
                total_tokens += len(prompt.split()) + len(draft_text.split())
                print(f"    Draft {i+1} 완료")
            
            print(f"  Review 단계 시작...")
            # 2. Review 단계 (7B + 개선된 프롬프트)
            review_responses = []
            for reviewer_id in range(2):
                review_prompt = f"""질문: {question}

Draft 답변들을 분석하여 리뷰하세요:
{chr(10).join(f"Draft {i+1}: {resp}" for i, resp in enumerate(draft_responses))}

검토자 {reviewer_id + 1} 관점에서:
1. 공통 핵심 내용 파악: 모든 Draft가 동의하는 부분
2. 차별점 분석: 각 Draft만의 독특한 접근이나 정보
3. 정확성 검증: 사실적 오류나 논리적 문제점 식별
4. 통합 개선안: 가장 정확하고 완전한 답변 제시

분석 결과:"""

                response = self.review_llm.generate(review_prompt, temperature=0.4, max_tokens=250)
                
                if isinstance(response, dict):
                    review_text = response.get("response", "").strip()
                else:
                    review_text = str(response).strip()
                
                review_responses.append(review_text)
                total_tokens += len(review_prompt.split()) + len(review_text.split())
                print(f"    Review {reviewer_id+1} 완료")
            
            print(f"  Judge 단계 시작...")
            # 3. Judge 단계 (동일한 프롬프트)
            judge_prompt = f"""질문: {question}

Draft 원본들:
{chr(10).join(f"Draft {i+1}: {draft}" for i, draft in enumerate(draft_responses))}

Review 분석들:
Review 1: {review_responses[0]}
Review 2: {review_responses[1]}

작업 지시:
1. 원래 질문에 정확하게 답하는 것이 최우선입니다
2. Draft들과 Review들을 종합적으로 분석하세요
3. 정확한 정보는 채택하고 잘못된 정보는 배제하세요
4. 간결하고 정확한 최종 답변을 제시하세요

최종 답변:"""

            judge_response = self.judge_llm.generate(judge_prompt, temperature=0.2, max_tokens=200)
            
            if isinstance(judge_response, dict):
                judge_text = judge_response.get("response", "").strip()
            else:
                judge_text = str(judge_response).strip()
                
            total_tokens += len(judge_prompt.split()) + len(judge_text.split())
            print(f"    Judge 완료")
            
            # 정확도 평가
            accuracy = self._evaluate_accuracy(judge_text, expected_answer)
            
            return Model7BResult(
                approach="B안 7B (프롬프트개선)",
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
            return Model7BResult(
                approach="B안 7B (프롬프트개선)",
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
    
    def run_7b_single_baseline(self, question: str, expected_answer: str) -> Model7BResult:
        """7B Single 모델 베이스라인"""
        
        start_time = time.time()
        
        try:
            print(f"  Single 모델 실행...")
            prompt = f"질문: {question}\n\n답변하세요:"
            response = self.judge_llm.generate(prompt, temperature=0.3, max_tokens=150)
            
            if isinstance(response, dict):
                answer_text = response.get("response", "").strip()
            else:
                answer_text = str(response).strip()
            
            total_tokens = len(prompt.split()) + len(answer_text.split())
            accuracy = self._evaluate_accuracy(answer_text, expected_answer)
            
            return Model7BResult(
                approach="Single 7B Model",
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
            return Model7BResult(
                approach="Single 7B Model",
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

def run_7b_model_comparison():
    """7B 모델 비교 실험"""
    
    print("=" * 80)
    print("7B 모델 파이프라인 비교 실험")
    print("=" * 80)
    
    tester = Pipeline7BTester()
    
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
        },
        {
            "question": "What is the speed of light?",
            "expected_answer": "300000000"  # 3x10^8 m/s
        }
    ]
    
    all_results = []
    
    for i, test_case in enumerate(test_cases):
        question = test_case["question"]
        expected = test_case["expected_answer"]
        
        print(f"\n{i+1}. 테스트: {question}")
        print("=" * 60)
        
        # A안 7B 실행
        print("A안 7B (ThoughtAggregator) 실행 중...")
        a_result = tester.run_7b_a_approach(question, expected)
        
        # B안 7B 실행  
        print("B안 7B (프롬프트개선) 실행 중...")
        b_result = tester.run_7b_b_approach(question, expected)
        
        # Single 7B 실행
        print("Single 7B Model 실행 중...")
        single_result = tester.run_7b_single_baseline(question, expected)
        
        # 결과 출력
        print(f"\n--- 결과 비교 ---")
        for result in [a_result, b_result, single_result]:
            if result.success:
                print(f"{result.approach}:")
                print(f"  최종 답변: {result.judge_response[:100]}...")
                print(f"  정확도: {result.accuracy_score:.2f}")
                print(f"  토큰 수: {result.total_tokens}")
                print(f"  실행 시간: {result.total_time_ms}ms")
                print(f"  효율성: {result.accuracy_score/result.total_tokens:.6f}")
                
                # Draft들 샘플 출력
                if result.draft_responses:
                    print(f"  Draft 샘플: {result.draft_responses[0][:80]}...")
            else:
                print(f"{result.approach}: 실패 - {result.error}")
            print()
        
        all_results.extend([a_result, b_result, single_result])
    
    # 전체 결과 요약
    print_7b_summary(all_results)
    
    # 결과를 파일로 저장
    save_7b_results(all_results)
    
    return all_results

def print_7b_summary(results: List[Model7BResult]):
    """7B 모델 결과 요약"""
    
    print("\n" + "=" * 80)
    print("7B 모델 실험 결과 요약")
    print("=" * 80)
    
    # 접근법별로 그룹화
    approaches = {}
    for result in results:
        if result.success:
            if result.approach not in approaches:
                approaches[result.approach] = []
            approaches[result.approach].append(result)
    
    print(f"\n📊 성능 비교:")
    for approach, approach_results in approaches.items():
        if not approach_results:
            continue
            
        avg_accuracy = sum(r.accuracy_score for r in approach_results) / len(approach_results)
        avg_tokens = sum(r.total_tokens for r in approach_results) / len(approach_results)
        avg_time = sum(r.total_time_ms for r in approach_results) / len(approach_results)
        avg_efficiency = avg_accuracy / avg_tokens if avg_tokens > 0 else 0
        
        print(f"\n{approach}:")
        print(f"  평균 정확도: {avg_accuracy:.3f} ({avg_accuracy*100:.1f}%)")
        print(f"  평균 토큰 수: {avg_tokens:.1f}")
        print(f"  평균 실행 시간: {avg_time:.1f}ms")
        print(f"  효율성 (정확도/토큰): {avg_efficiency:.6f}")
    
    # 0.5B vs 7B 비교 (참고용)
    print(f"\n📈 0.5B 대비 개선 예상:")
    print(f"  지식 정확도: 크게 개선될 것으로 예상")
    print(f"  추론 능력: 향상될 것으로 예상") 
    print(f"  실행 시간: 5-10배 증가")
    print(f"  토큰 효율성: Multi-Agent의 진정한 가치 확인 가능")

def save_7b_results(results: List[Model7BResult]):
    """7B 모델 결과 저장"""
    
    # 직렬화 가능한 형태로 변환
    serializable_results = []
    for result in results:
        serializable_results.append({
            "approach": result.approach,
            "question": result.question,
            "draft_responses": result.draft_responses,
            "review_responses": result.review_responses,
            "judge_response": result.judge_response,
            "total_tokens": result.total_tokens,
            "total_time_ms": result.total_time_ms,
            "accuracy_score": result.accuracy_score,
            "success": result.success,
            "error": result.error
        })
    
    # 타임스탬프 포함한 파일명
    timestamp = int(time.time())
    filename = f"results/7b_pipeline_comparison_{timestamp}.json"
    
    try:
        import os
        os.makedirs("results", exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 결과 저장됨: {filename}")
    except Exception as e:
        print(f"파일 저장 실패: {e}")

if __name__ == "__main__":
    print("🚀 7B 모델 파이프라인 테스트 시작...")
    print("⚠️  주의: 7B 모델은 0.5B 대비 상당히 느릴 수 있습니다.")
    
    # qwen2:7b 모델 확인
    print("\n🔍 qwen2:7b 모델 확인 중...")
    import subprocess
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if "qwen2:7b" not in result.stdout:
            print("❌ qwen2:7b 모델이 설치되지 않았습니다.")
            print("📥 다음 명령어로 설치하세요: ollama pull qwen2:7b")
            exit(1)
        else:
            print("✅ qwen2:7b 모델 확인됨")
    except Exception as e:
        print(f"⚠️  ollama 상태 확인 실패: {e}")
        print("수동으로 'ollama list'를 실행하여 qwen2:7b가 있는지 확인하세요.")
    
    input("\nEnter를 눌러서 실험을 시작하세요...")
    results = run_7b_model_comparison()