# -*- coding: utf-8 -*-
"""
Multi-Agent vs Single Model Benchmark Comparison
GSM8K 수학 문제로 통계적으로 유의미한 비교 실험
"""

import sys
import time
import json
import random
from typing import List, Dict, Any
from dataclasses import dataclass
import statistics
sys.path.append('.')

# Registry 시스템 사용으로 하드코딩 제거
from src.registry.model_registry import get_model_registry

@dataclass
class BenchmarkResult:
    """벤치마크 실험 결과"""
    method: str
    question: str
    expected: str
    predicted: str
    correct: bool
    tokens: int
    time_ms: int
    reasoning: str = ""

class BenchmarkTester:
    """벤치마크 테스터 - Registry 기반 (하드코딩 제거)"""
    
    def __init__(self, environment: str = "development"):
        print(f">>> 벤치마크 테스터 초기화 - 환경: {environment}")
        
        # Registry를 통한 설정 기반 모델 초기화 (하드코딩 제거!)
        self.registry = get_model_registry(environment)
        
        # 역할별 모델 할당 (config/models.yaml 기반)
        self.draft_llm = self.registry.get_model("undergraduate")  # 학부연구생 역할
        self.review_llm = self.registry.get_model("graduate")      # 석박사 역할
        self.judge_llm = self.registry.get_model("professor")      # 교수님 역할
        
        # 설정 정보 출력
        print(f"  Draft 모델: {self.registry.get_model_name('undergraduate')}")
        print(f"  Review 모델: {self.registry.get_model_name('graduate')}")
        print(f"  Judge 모델: {self.registry.get_model_name('professor')}")
    
    def run_single_model(self, question: str, expected: str) -> BenchmarkResult:
        """Single 8B 모델"""
        start_time = time.time()
        
        prompt = f"Question: {question}\n\nSolve step by step and provide the final numerical answer:"
        response = self.judge_llm.generate(prompt, temperature=0.1, max_tokens=300)
        
        if isinstance(response, dict):
            answer = response.get("response", "").strip()
        else:
            answer = str(response).strip()
        
        tokens = len(prompt.split()) + len(answer.split())
        time_ms = int((time.time() - start_time) * 1000)
        
        # Extract numerical answer
        predicted = self.extract_number(answer)
        correct = self.is_correct(predicted, expected)
        
        return BenchmarkResult(
            method=f"Single {self.registry.get_model_name('professor')}",
            question=question,
            expected=expected,
            predicted=predicted,
            correct=correct,
            tokens=tokens,
            time_ms=time_ms,
            reasoning=answer
        )
    
    def run_multi_agent(self, question: str, expected: str) -> BenchmarkResult:
        """Multi-Agent (Draft -> Review -> Judge)"""
        start_time = time.time()
        total_tokens = 0
        
        # Draft stage (3 drafts)
        draft_responses = []
        for i in range(3):
            prompt = f"Question: {question}\n\nSolve step by step and show your reasoning:"
            response = self.draft_llm.generate(prompt, temperature=0.2 + i*0.1, max_tokens=200)
            
            if isinstance(response, dict):
                draft = response.get("response", "").strip()
            else:
                draft = str(response).strip()
            
            draft_responses.append(draft)
            total_tokens += len(prompt.split()) + len(draft.split())
        
        # Review stage (2 reviews)
        review_responses = []
        for reviewer_id in range(2):
            review_prompt = f"""Question: {question}

Draft solutions to review:
{chr(10).join(f"Draft {i+1}: {resp}" for i, resp in enumerate(draft_responses))}

As reviewer {reviewer_id + 1}:
1. Check mathematical accuracy of each draft
2. Identify common approaches and differences  
3. Point out calculation errors or logical flaws
4. Suggest the most reliable solution path

Review:"""

            response = self.review_llm.generate(review_prompt, temperature=0.3, max_tokens=250)
            if isinstance(response, dict):
                review = response.get("response", "").strip()
            else:
                review = str(response).strip()
            
            review_responses.append(review)
            total_tokens += len(review_prompt.split()) + len(review.split())
        
        # Judge stage
        judge_prompt = f"""Question: {question}

Draft solutions:
{chr(10).join(f"Draft {i+1}: {draft}" for i, draft in enumerate(draft_responses))}

Review analyses:
Review 1: {review_responses[0]}
Review 2: {review_responses[1]}

As final judge:
1. Evaluate the mathematical correctness of each draft
2. Consider the reviewers' feedback carefully
3. Identify the most accurate solution approach
4. Provide the correct final numerical answer
5. If all drafts are wrong, solve it yourself with your knowledge

Final answer with reasoning:"""

        response = self.judge_llm.generate(judge_prompt, temperature=0.1, max_tokens=200)
        if isinstance(response, dict):
            final_answer = response.get("response", "").strip()
        else:
            final_answer = str(response).strip()
        
        total_tokens += len(judge_prompt.split()) + len(final_answer.split())
        time_ms = int((time.time() - start_time) * 1000)
        
        predicted = self.extract_number(final_answer)
        correct = self.is_correct(predicted, expected)
        
        return BenchmarkResult(
            method="Multi-Agent",
            question=question,
            expected=expected,
            predicted=predicted,
            correct=correct,
            tokens=total_tokens,
            time_ms=time_ms,
            reasoning=final_answer
        )
    
    def extract_number(self, text: str) -> str:
        """텍스트에서 숫자 답안 추출"""
        import re
        
        # Look for patterns like "answer is 42", "= 42", "42.", etc.
        patterns = [
            r'(?:answer is|answer:|final answer is|final answer:)\s*([+-]?\d+(?:\.\d+)?)',
            r'=\s*([+-]?\d+(?:\.\d+)?)',
            r'([+-]?\d+(?:\.\d+)?)\s*$',
            r'([+-]?\d+(?:\.\d+)?)\s*\.',
        ]
        
        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(1)
        
        # Fallback: find any number in the text
        numbers = re.findall(r'([+-]?\d+(?:\.\d+)?)', text)
        if numbers:
            return numbers[-1]  # Return the last number found
        
        return ""
    
    def is_correct(self, predicted: str, expected: str) -> bool:
        """정답 여부 확인"""
        if not predicted:
            return False
        
        try:
            pred_num = float(predicted)
            exp_num = float(expected)
            return abs(pred_num - exp_num) < 0.001  # Allow small floating point errors
        except:
            return predicted.strip().lower() == expected.strip().lower()

def load_gsm8k_sample(n_samples: int = 100) -> List[Dict[str, str]]:
    """GSM8K 스타일 수학 문제 생성 (실제 데이터셋 없을 경우 샘플 생성)"""
    
    # 간단한 수학 문제들 (다양한 난이도)
    problems = [
        # Level 1: Basic arithmetic
        {"question": "Sarah has 15 apples. She gives 7 to her friend. How many apples does she have left?", "answer": "8"},
        {"question": "A store sells pencils for $2 each. If Tom buys 6 pencils, how much does he pay?", "answer": "12"},
        {"question": "There are 24 students in a class. If they form groups of 4, how many groups are there?", "answer": "6"},
        {"question": "Lisa saves $5 every week. How much money will she have after 8 weeks?", "answer": "40"},
        {"question": "A box contains 36 chocolates. If 4 friends share them equally, how many chocolates does each friend get?", "answer": "9"},
        
        # Level 2: Multi-step problems
        {"question": "John has 20 marbles. He gives 5 to his sister and 3 to his brother. Then he buys 8 more marbles. How many marbles does he have now?", "answer": "20"},
        {"question": "A pizza is cut into 8 slices. Mike eats 3 slices and Jenny eats 2 slices. What fraction of the pizza is left?", "answer": "0.375"},
        {"question": "A car travels 60 miles in 1 hour. At this speed, how many miles will it travel in 2.5 hours?", "answer": "150"},
        {"question": "A recipe calls for 3 cups of flour to make 12 cookies. How many cups of flour are needed to make 20 cookies?", "answer": "5"},
        {"question": "James has $50. He spends $12 on lunch and $8 on a book. Then he earns $15 from his job. How much money does he have now?", "answer": "45"},
        
        # Level 3: Complex word problems  
        {"question": "A school has 180 students. 2/3 of them play sports. Of those who play sports, 1/4 play basketball. How many students play basketball?", "answer": "30"},
        {"question": "A store offers a 20% discount on all items. If a shirt originally costs $25, what is the final price after discount?", "answer": "20"},
        {"question": "Two trains leave stations 240 miles apart at the same time, traveling toward each other. One train travels at 60 mph and the other at 80 mph. After how many hours will they meet?", "answer": "1.714"},
        {"question": "A rectangular garden is 12 meters long and 8 meters wide. If fencing costs $3 per meter, how much will it cost to fence the entire perimeter?", "answer": "120"},
        {"question": "Amy's age is 3 times Bob's age. The sum of their ages is 32. How old is Amy?", "answer": "24"},
    ]
    
    # 문제 확장 (변형 생성)
    extended_problems = []
    for _ in range(n_samples):
        problem = random.choice(problems)
        extended_problems.append(problem)
    
    return extended_problems

def run_benchmark_experiment(n_samples: int = 100, environment: str = "development"):
    """벤치마크 실험 실행 - Registry 기반"""
    
    print(f"=== Multi-Agent vs Single Model Benchmark (Registry 기반) ===")
    print(f"Testing {n_samples} math problems")
    print(f"환경: {environment}")
    
    tester = BenchmarkTester(environment)
    
    # 역할별 모델 정보 출력
    draft_model = tester.registry.get_model_name('undergraduate')
    review_model = tester.registry.get_model_name('graduate')
    judge_model = tester.registry.get_model_name('professor')
    
    print(f"Models: Single({judge_model}) vs Multi-Agent({draft_model}->{review_model}->{judge_model})")
    print("=" * 80)
    problems = load_gsm8k_sample(n_samples)
    
    single_results = []
    multi_results = []
    
    for i, problem in enumerate(problems):
        question = problem["question"]
        expected = problem["answer"]
        
        print(f"\nProblem {i+1}/{len(problems)}: {question[:50]}...")
        
        # Run Single Model
        print("  Running Single Model...")
        single_result = tester.run_single_model(question, expected)
        single_results.append(single_result)
        
        # Run Multi-Agent  
        print("  Running Multi-Agent...")
        multi_result = tester.run_multi_agent(question, expected)
        multi_results.append(multi_result)
        
        # Progress update
        single_acc = sum(r.correct for r in single_results) / len(single_results)
        multi_acc = sum(r.correct for r in multi_results) / len(multi_results)
        
        print(f"  Single: {single_result.correct} ({single_acc:.2%} so far)")
        print(f"  Multi:  {multi_result.correct} ({multi_acc:.2%} so far)")
    
    # Statistical analysis
    analyze_results(single_results, multi_results)
    
    # Save results
    save_benchmark_results(single_results, multi_results)
    
    return single_results, multi_results

def analyze_results(single_results: List[BenchmarkResult], multi_results: List[BenchmarkResult]):
    """통계적 결과 분석"""
    
    print(f"\n" + "=" * 60)
    print("BENCHMARK RESULTS ANALYSIS")
    print("=" * 60)
    
    # Accuracy comparison
    single_accuracy = [r.correct for r in single_results]
    multi_accuracy = [r.correct for r in multi_results]
    
    single_acc_mean = statistics.mean(single_accuracy)
    multi_acc_mean = statistics.mean(multi_accuracy)
    
    print(f"\nACCURACY COMPARISON:")
    print(f"Single Model:  {single_acc_mean:.1%} ({sum(single_accuracy)}/{len(single_accuracy)})")
    print(f"Multi-Agent:   {multi_acc_mean:.1%} ({sum(multi_accuracy)}/{len(multi_accuracy)})")
    print(f"Improvement:   {multi_acc_mean - single_acc_mean:+.1%}")
    
    # Efficiency comparison
    single_tokens = [r.tokens for r in single_results]
    multi_tokens = [r.tokens for r in multi_results]
    
    single_time = [r.time_ms for r in single_results]
    multi_time = [r.time_ms for r in multi_results]
    
    print(f"\nEFFICIENCY COMPARISON:")
    print(f"Average Tokens - Single: {statistics.mean(single_tokens):.0f}, Multi: {statistics.mean(multi_tokens):.0f}")
    print(f"Token Ratio: {statistics.mean(multi_tokens) / statistics.mean(single_tokens):.1f}x")
    print(f"Average Time - Single: {statistics.mean(single_time):.0f}ms, Multi: {statistics.mean(multi_time):.0f}ms") 
    print(f"Time Ratio: {statistics.mean(multi_time) / statistics.mean(single_time):.1f}x")
    
    # Statistical significance (simple test)
    correct_diff = sum(multi_accuracy) - sum(single_accuracy)
    print(f"\nSTATISTICAL SUMMARY:")
    print(f"Correct answers difference: {correct_diff:+d}")
    print(f"Sample size: {len(single_results)}")
    
    if abs(correct_diff) >= 5:  # Simple threshold
        winner = "Multi-Agent" if correct_diff > 0 else "Single Model"
        print(f"Result: {winner} shows meaningful advantage")
    else:
        print(f"Result: No meaningful difference in accuracy")
    
    # Efficiency verdict
    token_ratio = statistics.mean(multi_tokens) / statistics.mean(single_tokens)
    acc_improvement = multi_acc_mean - single_acc_mean
    
    print(f"\nFINAL VERDICT:")
    if acc_improvement > 0.05:  # 5% improvement
        print(f"✓ Multi-Agent provides significant accuracy improvement ({acc_improvement:.1%})")
        if token_ratio < 10:
            print(f"✓ Token cost is reasonable ({token_ratio:.1f}x)")
        else:
            print(f"⚠ Token cost is high ({token_ratio:.1f}x) - consider simpler problems")
    else:
        print(f"✗ Multi-Agent does not provide significant accuracy benefit")
        print(f"✗ Token overhead is {token_ratio:.1f}x - Single Model is more efficient")

def save_benchmark_results(single_results: List[BenchmarkResult], multi_results: List[BenchmarkResult]):
    """결과 저장"""
    
    results_data = {
        "experiment_info": {
            "timestamp": int(time.time()),
            "n_samples": len(single_results),
            "environment": getattr(single_results[0] if single_results else None, 'environment', 'unknown'),
            "models": {
                "single": single_results[0].method.split()[-1] if single_results else "unknown",
                "multi_draft": "undergraduate_role", 
                "multi_review": "graduate_role",
                "multi_judge": "professor_role"
            }
        },
        "single_results": [],
        "multi_results": []
    }
    
    for result in single_results:
        results_data["single_results"].append({
            "question": result.question,
            "expected": result.expected,
            "predicted": result.predicted,
            "correct": result.correct,
            "tokens": result.tokens,
            "time_ms": result.time_ms
        })
    
    for result in multi_results:
        results_data["multi_results"].append({
            "question": result.question,
            "expected": result.expected,
            "predicted": result.predicted,
            "correct": result.correct,
            "tokens": result.tokens,
            "time_ms": result.time_ms
        })
    
    timestamp = int(time.time())
    filename = f"results/benchmark_comparison_{timestamp}.json"
    
    try:
        import os
        os.makedirs("results", exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
        print(f"📊 Load this file to analyze results in detail")
        
    except Exception as e:
        print(f"❌ Failed to save results: {e}")

if __name__ == "__main__":
    print(">>> Multi-Agent vs Single Model Benchmark 시작 (Registry 기반)")
    print(">>> 큰 샘플 크기에서는 상당한 시간이 소요됩니다")
    
    # 환경 선택
    print("\n>>> 테스트 환경 선택:")
    print("  1. development (빠른 모델)")
    print("  2. test (중간 성능)")
    print("  3. production (고성능, 시간 오래 걸림)")
    
    choice = input("\n환경 선택 (1-3, 기본값=1): ").strip() or "1"
    environments = {"1": "development", "2": "test", "3": "production"}
    environment = environments.get(choice, "development")
    
    # 샘플 크기 선택
    try:
        n = int(input("\n테스트할 문제 수 입력 (권장: 50-100): "))
        n = max(10, min(200, n))  # 10-200 사이로 제한
    except:
        n = 50
    
    print(f"\n>>> {environment} 환경에서 {n}개 수학 문제 테스트...")
    print(">>> 각 문제마다 Single Model과 Multi-Agent 둘 다 해결")
    
    input("\nEnter를 눌러 실험을 시작하세요...")
    
    single_results, multi_results = run_benchmark_experiment(n, environment)
    
    print(f"\n>>> 벤치마크 실험 완료! (Registry 기반)")
    print(f">>> 위의 결과 요약과 저장된 JSON 파일을 확인하세요")