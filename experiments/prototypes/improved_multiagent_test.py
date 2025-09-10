# -*- coding: utf-8 -*-
"""
개선된 Multi-Agent 테스트 - "교수님" 권위 구조
Draft(학부연구생) -> Review(석박사) -> Judge(교수님)
"""

import sys
import time
import json
from typing import List, Dict, Any
from dataclasses import dataclass
sys.path.append('.')

from src.llm.simple_llm import create_llm_auto

@dataclass 
class ImprovedResult:
    """개선된 실험 결과"""
    method: str
    question: str
    expected: str
    predicted: str
    correct: bool
    tokens: int
    time_ms: int
    draft_responses: List[str] = None
    review_responses: List[str] = None
    judge_reasoning: str = ""

class ImprovedMultiAgentTester:
    """개선된 Multi-Agent 테스터"""
    
    def __init__(self):
        print("🎓 학계 모델 구조 로딩...")
        self.undergraduate = create_llm_auto("qwen2:0.5b")  # 학부연구생
        self.graduate = create_llm_auto("qwen2:7b")         # 석박사  
        self.professor = create_llm_auto("llama3:8b")       # 교수님
        print("✅ 학계 구조 준비 완료")
    
    def run_original_multiagent(self, question: str, expected: str) -> ImprovedResult:
        """기존 Multi-Agent (휘둘리는 Judge)"""
        start_time = time.time()
        total_tokens = 0
        
        # Draft stage (학부연구생들)
        draft_responses = []
        for i in range(3):
            prompt = f"Question: {question}\n\nSolve step by step:"
            response = self.undergraduate.generate(prompt, temperature=0.2 + i*0.1, max_tokens=150)
            
            if isinstance(response, dict):
                draft = response.get("response", "").strip()
            else:
                draft = str(response).strip()
            
            draft_responses.append(draft)
            total_tokens += len(prompt.split()) + len(draft.split())
        
        # Review stage (석박사들)
        review_responses = []
        for reviewer_id in range(2):
            review_prompt = f"""Question: {question}

Undergraduate students' attempts:
{chr(10).join(f"Student {i+1}: {resp}" for i, resp in enumerate(draft_responses))}

As graduate student reviewer {reviewer_id + 1}:
1. Check mathematical accuracy
2. Identify good approaches and errors
3. Provide analysis

Review:"""

            response = self.graduate.generate(review_prompt, temperature=0.3, max_tokens=200)
            if isinstance(response, dict):
                review = response.get("response", "").strip()
            else:
                review = str(response).strip()
            
            review_responses.append(review)
            total_tokens += len(review_prompt.split()) + len(review.split())
        
        # Judge stage (기존 스타일 - 휘둘리는 교수님)
        judge_prompt = f"""Question: {question}

Student attempts:
{chr(10).join(f"Student {i+1}: {draft}" for i, draft in enumerate(draft_responses))}

Graduate reviews:
Review 1: {review_responses[0]}
Review 2: {review_responses[1]}

As judge:
1. Analyze all the provided information
2. Consider the student attempts and graduate reviews
3. Synthesize a final answer based on the collective input

Final answer:"""

        response = self.professor.generate(judge_prompt, temperature=0.1, max_tokens=150)
        if isinstance(response, dict):
            judge_answer = response.get("response", "").strip()
        else:
            judge_answer = str(response).strip()
        
        total_tokens += len(judge_prompt.split()) + len(judge_answer.split())
        time_ms = int((time.time() - start_time) * 1000)
        
        predicted = self.extract_answer(judge_answer)
        correct = self.is_correct(predicted, expected)
        
        return ImprovedResult(
            method="Original Multi-Agent (Collaborative Judge)",
            question=question,
            expected=expected,
            predicted=predicted,
            correct=correct,
            tokens=total_tokens,
            time_ms=time_ms,
            draft_responses=draft_responses,
            review_responses=review_responses,
            judge_reasoning=judge_answer
        )
    
    def run_improved_multiagent(self, question: str, expected: str) -> ImprovedResult:
        """개선된 Multi-Agent (교수님 권위)"""
        start_time = time.time()
        total_tokens = 0
        
        # Draft stage (동일)
        draft_responses = []
        for i in range(3):
            prompt = f"Question: {question}\n\nAs undergraduate researcher, solve this step by step:"
            response = self.undergraduate.generate(prompt, temperature=0.2 + i*0.1, max_tokens=150)
            
            if isinstance(response, dict):
                draft = response.get("response", "").strip()
            else:
                draft = str(response).strip()
            
            draft_responses.append(draft)
            total_tokens += len(prompt.split()) + len(draft.split())
        
        # Review stage (동일)
        review_responses = []
        for reviewer_id in range(2):
            review_prompt = f"""Question: {question}

Undergraduate attempts:
{chr(10).join(f"Student {i+1}: {resp}" for i, resp in enumerate(draft_responses))}

As graduate student, provide critical analysis:
1. Which approaches show promise?
2. What errors do you spot?
3. What improvements would you suggest?

Critical review:"""

            response = self.graduate.generate(review_prompt, temperature=0.3, max_tokens=200)
            if isinstance(response, dict):
                review = response.get("response", "").strip()
            else:
                review = str(response).strip()
            
            review_responses.append(review)
            total_tokens += len(review_prompt.split()) + len(review.split())
        
        # Judge stage (새로운 스타일 - 권위적 교수님)
        judge_prompt = f"""I am a professor and expert in this field.

PROBLEM: {question}

Undergraduate students' attempts:
{chr(10).join(f"Student {i+1}: {draft}" for i, draft in enumerate(draft_responses))}

Graduate students' reviews:
Review 1: {review_responses[0]}
Review 2: {review_responses[1]}

PROFESSOR'S INDEPENDENT EXPERT JUDGMENT:

First, let me solve this problem using my expertise:
1. I will approach this problem with my professional knowledge
2. I will examine if any undergraduate ideas are worth incorporating
3. I will verify if graduate reviews are sound
4. I will make the final decision based on my expertise

Key principles:
- I reference others' work but make independent judgments
- If students or graduates are wrong, I correct them without hesitation
- I adopt good ideas and discard bad ones
- My expertise takes precedence in the final decision

Professional analysis and final answer:"""

        response = self.professor.generate(judge_prompt, temperature=0.1, max_tokens=200)
        if isinstance(response, dict):
            judge_answer = response.get("response", "").strip()
        else:
            judge_answer = str(response).strip()
        
        total_tokens += len(judge_prompt.split()) + len(judge_answer.split())
        time_ms = int((time.time() - start_time) * 1000)
        
        predicted = self.extract_answer(judge_answer)
        correct = self.is_correct(predicted, expected)
        
        return ImprovedResult(
            method="Improved Multi-Agent (Authoritative Professor)",
            question=question,
            expected=expected,
            predicted=predicted,
            correct=correct,
            tokens=total_tokens,
            time_ms=time_ms,
            draft_responses=draft_responses,
            review_responses=review_responses,
            judge_reasoning=judge_answer
        )
    
    def run_single_professor(self, question: str, expected: str) -> ImprovedResult:
        """Single 교수님 모델"""
        start_time = time.time()
        
        prompt = f"As an expert professor, solve this problem: {question}\n\nProvide the answer:"
        response = self.professor.generate(prompt, temperature=0.1, max_tokens=150)
        
        if isinstance(response, dict):
            answer = response.get("response", "").strip()
        else:
            answer = str(response).strip()
        
        tokens = len(prompt.split()) + len(answer.split())
        time_ms = int((time.time() - start_time) * 1000)
        
        predicted = self.extract_answer(answer)
        correct = self.is_correct(predicted, expected)
        
        return ImprovedResult(
            method="Single Professor Model",
            question=question,
            expected=expected,
            predicted=predicted,
            correct=correct,
            tokens=tokens,
            time_ms=time_ms,
            judge_reasoning=answer
        )
    
    def extract_answer(self, text: str) -> str:
        """답변에서 숫자 추출"""
        import re
        
        patterns = [
            r'(?:answer is|answer:|final answer is|final answer:)\s*([+-]?\d+(?:\.\d+)?)',
            r'(?:result is|result:)\s*([+-]?\d+(?:\.\d+)?)',
            r'=\s*([+-]?\d+(?:\.\d+)?)',
            r'([+-]?\d+(?:\.\d+)?)\s*$',
            r'([+-]?\d+(?:\.\d+)?)\s*\.',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(1)
        
        numbers = re.findall(r'([+-]?\d+(?:\.\d+)?)', text)
        if numbers:
            return numbers[-1]
        
        return ""
    
    def is_correct(self, predicted: str, expected: str) -> bool:
        """정답 여부 확인"""
        if not predicted:
            return False
        
        try:
            pred_num = float(predicted)
            exp_num = float(expected)
            return abs(pred_num - exp_num) < 0.01
        except:
            return predicted.strip().lower() == expected.strip().lower()

def run_authority_comparison_test():
    """권위 구조 비교 테스트"""
    
    print("=" * 80)
    print("🎓 교수님 권위 구조 비교 테스트")
    print("Original vs Improved Multi-Agent vs Single Professor")
    print("=" * 80)
    
    tester = ImprovedMultiAgentTester()
    
    # 기본 테스트 케이스 - 실제 벤치마크에서 실패했던 문제들
    test_cases = [
        {"question": "Sarah has 15 apples. She gives away 7. How many are left?", "expected": "8"},
        {"question": "A shirt costs $25. With 20% discount, what is the final price?", "expected": "20"},
        {"question": "If 4 friends share 36 chocolates equally, how many does each get?", "expected": "9"},
        {"question": "What is 240 divided by 140?", "expected": "1.714"},
        {"question": "Rectangle is 12m by 8m. What is the perimeter?", "expected": "40"},
        {"question": "Two trains 240 miles apart, speeds 60 and 80 mph. Meeting time?", "expected": "1.714"},
    ]
    
    all_results = []
    
    for i, test_case in enumerate(test_cases):
        question = test_case["question"]
        expected = test_case["expected"]
        
        print(f"\n📝 Problem {i+1}: {question}")
        print("-" * 60)
        
        # Original Multi-Agent
        print("🤝 Original Multi-Agent (Collaborative)...")
        original_result = tester.run_original_multiagent(question, expected)
        
        # Improved Multi-Agent  
        print("👨‍🏫 Improved Multi-Agent (Authoritative)...")
        improved_result = tester.run_improved_multiagent(question, expected)
        
        # Single Professor
        print("🎯 Single Professor...")
        single_result = tester.run_single_professor(question, expected)
        
        # 결과 출력
        results = [original_result, improved_result, single_result]
        print(f"\n📊 Results:")
        
        for result in results:
            status = "✅" if result.correct else "❌"
            print(f"  {status} {result.method}")
            print(f"      Predicted: {result.predicted} (Expected: {result.expected})")
            print(f"      Tokens: {result.tokens}, Time: {result.time_ms}ms")
            
            # Judge의 reasoning 샘플
            if "Multi-Agent" in result.method and result.judge_reasoning:
                reasoning_sample = result.judge_reasoning[:100].replace('\n', ' ')
                print(f"      Judge reasoning: {reasoning_sample}...")
        
        all_results.extend(results)
    
    # 종합 분석
    analyze_authority_results(all_results)
    save_authority_results(all_results)
    
    return all_results

def analyze_authority_results(results: List[ImprovedResult]):
    """권위 구조 결과 분석"""
    
    print(f"\n" + "=" * 80)
    print("📊 권위 구조 비교 분석")
    print("=" * 80)
    
    # 방법별 그룹화
    methods = {}
    for result in results:
        if result.method not in methods:
            methods[result.method] = []
        methods[result.method].append(result)
    
    print(f"\n🎯 전체 성능 비교:")
    for method, method_results in methods.items():
        correct_count = sum(1 for r in method_results if r.correct)
        total_count = len(method_results)
        accuracy = correct_count / total_count if total_count > 0 else 0
        
        avg_tokens = sum(r.tokens for r in method_results) / total_count
        avg_time = sum(r.time_ms for r in method_results) / total_count
        
        print(f"\n{method}:")
        print(f"  정확도: {accuracy:.1%} ({correct_count}/{total_count})")
        print(f"  평균 토큰: {avg_tokens:.0f}")
        print(f"  평균 시간: {avg_time:.0f}ms")
    
    # 개선 효과 분석
    original_results = methods.get("Original Multi-Agent (Collaborative Judge)", [])
    improved_results = methods.get("Improved Multi-Agent (Authoritative Professor)", [])
    single_results = methods.get("Single Professor Model", [])
    
    if original_results and improved_results:
        orig_acc = sum(1 for r in original_results if r.correct) / len(original_results)
        imp_acc = sum(1 for r in improved_results if r.correct) / len(improved_results)
        single_acc = sum(1 for r in single_results if r.correct) / len(single_results)
        
        print(f"\n🔍 권위 구조 개선 효과:")
        print(f"  Original Multi-Agent: {orig_acc:.1%}")
        print(f"  Improved Multi-Agent: {imp_acc:.1%}")
        print(f"  Single Professor: {single_acc:.1%}")
        
        improvement = imp_acc - orig_acc
        vs_single = imp_acc - single_acc
        
        print(f"  개선 효과: {improvement:+.1%}")
        if improvement > 0.05:
            print(f"  ✅ 권위적 Judge가 협력적 Judge보다 효과적!")
        
        print(f"  vs Single: {vs_single:+.1%}")
        if vs_single > 0:
            print(f"  ✅ Multi-Agent가 Single Model을 능가!")
        elif vs_single > -0.1:
            print(f"  📊 Single Model과 비슷한 성능")
        else:
            print(f"  ❌ 여전히 Single Model이 우세")

def save_authority_results(results: List[ImprovedResult]):
    """권위 구조 실험 결과 저장"""
    
    timestamp = int(time.time())
    filename = f"results/authority_comparison_{timestamp}.json"
    
    serializable_results = []
    for result in results:
        serializable_results.append({
            "method": result.method,
            "question": result.question,
            "expected": result.expected,
            "predicted": result.predicted,
            "correct": result.correct,
            "tokens": result.tokens,
            "time_ms": result.time_ms,
            "draft_responses": result.draft_responses,
            "review_responses": result.review_responses,
            "judge_reasoning": result.judge_reasoning
        })
    
    try:
        import os
        os.makedirs("results", exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 권위 구조 실험 결과 저장: {filename}")
        
    except Exception as e:
        print(f"❌ 저장 실패: {e}")

if __name__ == "__main__":
    print("🎓 교수님 권위 구조 테스트를 시작합니다...")
    print("📝 기존 협력적 Judge vs 권위적 교수님 Judge 비교")
    print("⏱️  예상 소요 시간: 10-15분")
    
    input("\nPress Enter to start authority structure test...")
    
    results = run_authority_comparison_test()
    
    print(f"\n✅ 권위 구조 비교 테스트 완료!")
    print(f"📈 교수님의 권위가 Multi-Agent 성능을 향상시켰는지 확인하세요!")