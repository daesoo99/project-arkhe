# -*- coding: utf-8 -*-
"""
계층적 Multi-Agent 비교 실험
Option 1: Draft(7B) → Review(7B) → Judge(14B)
Option 2: Draft(7B) → Judge(14B) (Review 제거)
"""

import sys
import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
sys.path.append('.')

from src.llm.simple_llm import create_llm_auto

@dataclass
class HierarchicalResult:
    """계층적 실험 결과"""
    approach: str
    question: str
    draft_responses: List[str]
    review_responses: List[str]  # Option 2에서는 빈 리스트
    judge_response: str
    total_tokens: int
    total_time_ms: int
    accuracy_score: float
    success: bool
    error: Optional[str] = None

class HierarchicalTester:
    """계층적 Multi-Agent 테스트"""
    
    def __init__(self):
        self.draft_llm = create_llm_auto("qwen2:0.5b")   # 저렴한 모델
        self.review_llm = create_llm_auto("qwen2:7b")    # 중간 모델
        self.judge_llm = create_llm_auto("llama3:8b")  # 고급 모델
    
    def run_option1_pipeline(self, question: str, expected_answer: str) -> HierarchicalResult:
        """Option 1: Draft → Review → Judge (3단계)"""
        
        start_time = time.time()
        total_tokens = 0
        
        try:
            print(f"    Option 1 실행 중...")
            
            # 1. Draft 단계 (7B)
            print(f"      Draft 단계...")
            draft_responses = []
            for i in range(3):
                prompt = f"질문: {question}\n\n답변하고 사고과정을 자세히 설명하세요:"
                response = self.draft_llm.generate(prompt, temperature=0.3 + i*0.1, max_tokens=200)
                
                if isinstance(response, dict):
                    draft_text = response.get("response", "").strip()
                else:
                    draft_text = str(response).strip()
                
                draft_responses.append(draft_text)
                total_tokens += len(prompt.split()) + len(draft_text.split())
            
            # 2. Review 단계 (7B)
            print(f"      Review 단계...")
            review_responses = []
            for reviewer_id in range(2):
                review_prompt = f"""질문: {question}

Draft 답변들을 분석하여 리뷰하세요:
{chr(10).join(f"Draft {i+1}: {resp}" for i, resp in enumerate(draft_responses))}

검토자 {reviewer_id + 1} 관점에서:
1. 공통 핵심 내용 파악
2. 각 Draft의 사고과정 논리적 검증
3. 사실적 오류나 추론 문제 식별
4. 개선된 통합 답변 제시

분석 결과:"""

                response = self.review_llm.generate(review_prompt, temperature=0.4, max_tokens=250)
                
                if isinstance(response, dict):
                    review_text = response.get("response", "").strip()
                else:
                    review_text = str(response).strip()
                
                review_responses.append(review_text)
                total_tokens += len(review_prompt.split()) + len(review_text.split())
            
            # 3. Judge 단계 (14B)
            print(f"      Judge 단계...")
            judge_prompt = f"""질문: {question}

Draft 원본들 (사고과정 포함):
{chr(10).join(f"Draft {i+1}: {draft}" for i, draft in enumerate(draft_responses))}

Review 분석들:
Review 1: {review_responses[0]}
Review 2: {review_responses[1]}

고급 판정자로서의 작업:
1. Draft들의 사고과정을 논리적으로 검증하세요
2. Review들의 분석이 타당한지 평가하세요  
3. 더 넓은 지식과 추론으로 오류를 수정하세요
4. 가장 정확하고 완전한 최종 답변을 제시하세요

특히 Draft나 Review에서 논리적 오류, 사실 오류가 있다면 
더 큰 모델로서의 지식과 추론력으로 수정해주세요.

최종 판정:"""

            judge_response = self.judge_llm.generate(judge_prompt, temperature=0.2, max_tokens=200)
            
            if isinstance(judge_response, dict):
                judge_text = judge_response.get("response", "").strip()
            else:
                judge_text = str(judge_response).strip()
                
            total_tokens += len(judge_prompt.split()) + len(judge_text.split())
            
            # 정확도 평가
            accuracy = self._evaluate_accuracy(judge_text, expected_answer)
            
            return HierarchicalResult(
                approach="Option 1 (Draft→Review→Judge)",
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
            return HierarchicalResult(
                approach="Option 1 (Draft→Review→Judge)",
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
    
    def run_option2_pipeline(self, question: str, expected_answer: str) -> HierarchicalResult:
        """Option 2: Draft → Judge (2단계, Review 제거)"""
        
        start_time = time.time()
        total_tokens = 0
        
        try:
            print(f"    Option 2 실행 중...")
            
            # 1. Draft 단계 (7B) - 동일
            print(f"      Draft 단계...")
            draft_responses = []
            for i in range(3):
                prompt = f"질문: {question}\n\n답변하고 사고과정을 자세히 설명하세요:"
                response = self.draft_llm.generate(prompt, temperature=0.3 + i*0.1, max_tokens=200)
                
                if isinstance(response, dict):
                    draft_text = response.get("response", "").strip()
                else:
                    draft_text = str(response).strip()
                
                draft_responses.append(draft_text)
                total_tokens += len(prompt.split()) + len(draft_text.split())
            
            # 2. Judge 단계 (14B) - Review 없이 직접
            print(f"      Judge 단계...")
            judge_prompt = f"""질문: {question}

3개의 Draft 답변들 (각각의 사고과정 포함):
{chr(10).join(f"Draft {i+1}: {draft}" for i, draft in enumerate(draft_responses))}

고급 판정자로서의 작업:
1. 각 Draft의 답변과 사고과정을 철저히 분석하세요
2. 논리적 오류, 사실적 오류, 추론 문제를 식별하세요
3. Draft들 간의 일치점과 차이점을 파악하세요
4. 더 큰 모델로서의 광범위한 지식으로 검증하세요
5. 모든 정보를 종합하여 가장 정확한 답변을 도출하세요

특별 지시: Draft들이 모두 틀렸다면, 당신의 지식으로 올바른 답변을 제시하세요.
Draft들의 사고과정에서 좋은 부분은 활용하되, 오류는 수정하세요.

최종 판정:"""

            judge_response = self.judge_llm.generate(judge_prompt, temperature=0.2, max_tokens=200)
            
            if isinstance(judge_response, dict):
                judge_text = judge_response.get("response", "").strip()
            else:
                judge_text = str(judge_response).strip()
                
            total_tokens += len(judge_prompt.split()) + len(judge_text.split())
            
            # 정확도 평가
            accuracy = self._evaluate_accuracy(judge_text, expected_answer)
            
            return HierarchicalResult(
                approach="Option 2 (Draft→Judge)",
                question=question,
                draft_responses=draft_responses,
                review_responses=[],  # Review 없음
                judge_response=judge_text,
                total_tokens=total_tokens,
                total_time_ms=int((time.time() - start_time) * 1000),
                accuracy_score=accuracy,
                success=True
            )
            
        except Exception as e:
            return HierarchicalResult(
                approach="Option 2 (Draft→Judge)",
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
    
    def run_single_8b_baseline(self, question: str, expected_answer: str) -> HierarchicalResult:
        """Single 8B 모델 베이스라인"""
        
        start_time = time.time()
        
        try:
            print(f"    Single 8B 실행 중...")
            prompt = f"질문: {question}\n\n답변하세요:"
            response = self.judge_llm.generate(prompt, temperature=0.3, max_tokens=150)
            
            if isinstance(response, dict):
                answer_text = response.get("response", "").strip()
            else:
                answer_text = str(response).strip()
            
            total_tokens = len(prompt.split()) + len(answer_text.split())
            accuracy = self._evaluate_accuracy(answer_text, expected_answer)
            
            return HierarchicalResult(
                approach="Single 8B Model",
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
            return HierarchicalResult(
                approach="Single 8B Model",
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

def run_hierarchical_comparison():
    """계층적 구조 비교 실험"""
    
    print("=" * 80)
    print("계층적 Multi-Agent 비교 실험")
    print("Option 1: Draft(0.5B) → Review(7B) → Judge(8B)")
    print("Option 2: Draft(0.5B) → Judge(8B)")
    print("=" * 80)
    
    tester = HierarchicalTester()
    
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
            "expected_answer": "300000000"
        }
    ]
    
    all_results = []
    
    for i, test_case in enumerate(test_cases):
        question = test_case["question"]
        expected = test_case["expected_answer"]
        
        print(f"\n{i+1}. 테스트: {question}")
        print("=" * 60)
        
        # Option 1: Draft → Review → Judge
        print("  Option 1 (3단계) 실행...")
        option1_result = tester.run_option1_pipeline(question, expected)
        
        # Option 2: Draft → Judge 
        print("  Option 2 (2단계) 실행...")
        option2_result = tester.run_option2_pipeline(question, expected)
        
        # Single 8B 베이스라인
        print("  Single 8B 실행...")
        single_result = tester.run_single_8b_baseline(question, expected)
        
        # 결과 출력
        print(f"\n--- 결과 비교 ---")
        for result in [option1_result, option2_result, single_result]:
            if result.success:
                print(f"{result.approach}:")
                print(f"  최종 답변: {result.judge_response[:80]}...")
                print(f"  정확도: {result.accuracy_score:.2f}")
                print(f"  토큰 수: {result.total_tokens}")
                print(f"  실행 시간: {result.total_time_ms}ms")
                print(f"  효율성: {result.accuracy_score/result.total_tokens:.6f}")
                
                # Review 존재 여부
                if result.review_responses:
                    print(f"  Review 샘플: {result.review_responses[0][:60]}...")
            else:
                print(f"{result.approach}: 실패 - {result.error}")
            print()
        
        all_results.extend([option1_result, option2_result, single_result])
    
    # 전체 결과 요약
    print_hierarchical_summary(all_results)
    
    # 결과 저장
    save_hierarchical_results(all_results)
    
    return all_results

def print_hierarchical_summary(results: List[HierarchicalResult]):
    """계층적 실험 결과 요약"""
    
    print("\n" + "=" * 80)
    print("계층적 Multi-Agent 실험 결과 요약")
    print("=" * 80)
    
    # 접근법별 그룹화
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
    
    # Review 단계의 가치 분석
    option1_results = [r for r in results if r.approach.startswith("Option 1") and r.success]
    option2_results = [r for r in results if r.approach.startswith("Option 2") and r.success]
    
    if option1_results and option2_results:
        print(f"\n🔍 Review 단계 가치 분석:")
        
        opt1_avg_acc = sum(r.accuracy_score for r in option1_results) / len(option1_results)
        opt2_avg_acc = sum(r.accuracy_score for r in option2_results) / len(option2_results)
        
        opt1_avg_tokens = sum(r.total_tokens for r in option1_results) / len(option1_results)
        opt2_avg_tokens = sum(r.total_tokens for r in option2_results) / len(option2_results)
        
        opt1_avg_time = sum(r.total_time_ms for r in option1_results) / len(option1_results)
        opt2_avg_time = sum(r.total_time_ms for r in option2_results) / len(option2_results)
        
        accuracy_diff = opt1_avg_acc - opt2_avg_acc
        token_diff = opt1_avg_tokens - opt2_avg_tokens
        time_diff = opt1_avg_time - opt2_avg_time
        
        print(f"  정확도 차이: {accuracy_diff:+.3f} ({'Option 1 우위' if accuracy_diff > 0 else 'Option 2 우위'})")
        print(f"  토큰 차이: {token_diff:+.1f} ({'Review 비용' if token_diff > 0 else 'Review 절약'})")
        print(f"  시간 차이: {time_diff:+.1f}ms ({'Review 지연' if time_diff > 0 else 'Review 가속'})")
        
        # Review의 가치 판정
        if accuracy_diff > 0.1:  # 10% 이상 정확도 개선
            print(f"  💡 결론: Review 단계가 정확도 개선에 효과적")
        elif accuracy_diff < -0.1:  # 10% 이상 정확도 하락
            print(f"  ⚠️  결론: Review 단계가 오히려 성능 저하 초래")
        else:
            print(f"  📊 결론: Review 단계의 효과 미미, 효율성 고려 필요")

def save_hierarchical_results(results: List[HierarchicalResult]):
    """계층적 실험 결과 저장"""
    
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
    
    timestamp = int(time.time())
    filename = f"results/hierarchical_comparison_{timestamp}.json"
    
    try:
        import os
        os.makedirs("results", exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 결과 저장됨: {filename}")
    except Exception as e:
        print(f"파일 저장 실패: {e}")

if __name__ == "__main__":
    print("계층적 Multi-Agent 비교 실험 시작...")
    print("주의: llama3:8b 모델 사용으로 시간이 오래 걸릴 수 있습니다.")
    
    # llama3:8b 모델 확인
    print("\nllama3:8b 모델 확인 중...")
    import subprocess
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if "llama3:8b" not in result.stdout:
            print("llama3:8b 모델이 설치되지 않았습니다.")
            print("다음 명령어로 설치하세요: ollama pull llama3:8b")
            print("용량: 약 4.7GB")
            exit(1)
        else:
            print("llama3:8b 모델 확인됨")
    except Exception as e:
        print(f"ollama 상태 확인 실패: {e}")
        print("수동으로 'ollama list'를 실행하여 llama3:8b가 있는지 확인하세요.")
    
    results = run_hierarchical_comparison()