# -*- coding: utf-8 -*-
"""
개선된 Multi-Agent 테스트 - ModelRegistry 적용 버전
Draft(학부연구생) -> Review(석박사) -> Judge(교수님)

BEFORE: 하드코딩된 모델명
AFTER: config/models.yaml 기반 역할 할당
"""

import sys
import time
import json
from typing import List, Dict, Any
from dataclasses import dataclass
sys.path.append('.')

# 기존 하드코딩 대신 ModelRegistry 사용
from src.registry.model_registry import get_model_registry, ModelRegistry

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

class ImprovedMultiAgentTesterV2:
    """개선된 Multi-Agent 테스터 - 모듈화 버전"""
    
    def __init__(self, environment: str = "development"):
        print(">>> 학계 모델 구조 로딩... (Registry 기반)")
        
        # ModelRegistry를 통한 설정 기반 모델 로딩
        self.registry = get_model_registry(environment)
        
        # 역할별 모델 할당 (하드코딩 제거!)
        self.undergraduate = self.registry.get_model("undergraduate")  # config에서 "small"
        self.graduate = self.registry.get_model("graduate")           # config에서 "medium"
        self.professor = self.registry.get_model("professor")         # config에서 "large"
        
        # 설정 정보 출력
        print(f"  학부연구생: {self.registry.get_model_name('undergraduate')}")
        print(f"  석박사: {self.registry.get_model_name('graduate')}")
        print(f"  교수님: {self.registry.get_model_name('professor')}")
        print(">>> 학계 구조 준비 완료 (Registry 기반)")
    
    def run_original_multiagent(self, question: str, expected: str) -> ImprovedResult:
        """기존 Multi-Agent (휘둘리는 Judge) - 모듈화 버전"""
        start_time = time.time()
        total_tokens = 0
        
        # Draft stage (학부연구생들)
        draft_responses = []
        for i in range(3):
            prompt = f"Question: {question}\n\nSolve step by step:"
            response = self.undergraduate.generate(prompt, temperature=0.2 + i*0.1, max_tokens=150)
            
            if response.get("success", False):
                draft_responses.append(response["response"])
                total_tokens += self._count_tokens(prompt, response["response"])
                time.sleep(0.1)  # 폴라이트 지연
        
        # Review stage (석박사)
        review_responses = []
        draft_summary = f"학부연구생 답변들:\n" + "\n".join([f"{i+1}) {resp}" for i, resp in enumerate(draft_responses)])
        
        for i in range(2):
            review_prompt = f"""Question: {question}

{draft_summary}

위 학부연구생들의 답변을 검토하고 개선해주세요. 옳은 부분은 유지하고 틀린 부분은 수정해주세요:"""
            
            response = self.graduate.generate(review_prompt, temperature=0.3, max_tokens=200)
            
            if response.get("success", False):
                review_responses.append(response["response"])
                total_tokens += self._count_tokens(review_prompt, response["response"])
        
        # Judge stage (교수님) 
        all_responses = draft_responses + review_responses
        judge_prompt = f"""Question: {question}

학생들의 다양한 답변:
{chr(10).join([f"- {resp}" for resp in all_responses])}

교수님으로서 최종 정답을 내려주시고, 학생들의 답변 중 가장 좋은 부분들을 종합해주세요:"""
        
        judge_response = self.professor.generate(judge_prompt, temperature=0.2, max_tokens=200)
        
        final_answer = ""
        judge_reasoning = ""
        if judge_response.get("success", False):
            judge_reasoning = judge_response["response"]
            final_answer = judge_response["response"].split('\n')[0]  # 첫 줄을 최종 답변으로
            total_tokens += self._count_tokens(judge_prompt, judge_response["response"])
        
        time_ms = int((time.time() - start_time) * 1000)
        correct = expected.lower().strip() in final_answer.lower()
        
        return ImprovedResult(
            method="Original MultiAgent (Registry)",
            question=question,
            expected=expected,
            predicted=final_answer,
            correct=correct,
            tokens=total_tokens,
            time_ms=time_ms,
            draft_responses=draft_responses,
            review_responses=review_responses,
            judge_reasoning=judge_reasoning
        )
    
    def run_improved_multiagent(self, question: str, expected: str) -> ImprovedResult:
        """개선된 Multi-Agent (권위적 Judge) - 모듈화 버전"""
        start_time = time.time()
        total_tokens = 0
        
        # Draft stage (학부연구생들) - 더 구체적 지시
        draft_responses = []
        for i in range(3):
            prompt = f"""Question: {question}

학부연구생으로서 이 문제를 해결해보세요. 단계별로 설명하고, 확실하지 않은 부분이 있으면 "잘 모르겠습니다"라고 솔직히 말해주세요:"""
            
            response = self.undergraduate.generate(prompt, temperature=0.3 + i*0.1, max_tokens=150)
            
            if response.get("success", False):
                draft_responses.append(response["response"])
                total_tokens += self._count_tokens(prompt, response["response"])
                time.sleep(0.1)
        
        # Review stage (석박사) - 비판적 검토
        review_responses = []
        draft_summary = f"학부연구생들의 시도:\n" + "\n".join([f"답변{i+1}: {resp}" for i, resp in enumerate(draft_responses)])
        
        for i in range(2):
            review_prompt = f"""Question: {question}

{draft_summary}

석박사로서 위 답변들을 비판적으로 분석해주세요:
1) 각 답변의 장단점
2) 올바른 접근 방향
3) 예상되는 정답 (but 100% 확신은 없음)"""
            
            response = self.graduate.generate(review_prompt, temperature=0.2, max_tokens=200)
            
            if response.get("success", False):
                review_responses.append(response["response"])
                total_tokens += self._count_tokens(review_prompt, response["response"])
        
        # Judge stage (교수님) - 권위적 최종 결정
        all_discussions = f"""학부연구생들의 시도:
{chr(10).join([f"• {resp}" for resp in draft_responses])}

석박사들의 분석:
{chr(10).join([f"• {resp}" for resp in review_responses])}"""
        
        judge_prompt = f"""Question: {question}

{all_discussions}

교수님으로서 위 모든 논의를 바탕으로 최종 정답을 확정해주세요. 학생들의 좋은 아이디어는 인정하되, 틀린 부분은 명확히 정정해주세요. 

최종 답변을 첫 줄에 명확히 제시해주세요:"""
        
        judge_response = self.professor.generate(judge_prompt, temperature=0.1, max_tokens=250)
        
        final_answer = ""
        judge_reasoning = ""
        if judge_response.get("success", False):
            judge_reasoning = judge_response["response"]
            final_answer = judge_response["response"].split('\n')[0].strip()
            total_tokens += self._count_tokens(judge_prompt, judge_response["response"])
        
        time_ms = int((time.time() - start_time) * 1000)
        correct = expected.lower().strip() in final_answer.lower()
        
        return ImprovedResult(
            method="Improved MultiAgent (Registry)",
            question=question,
            expected=expected,
            predicted=final_answer,
            correct=correct,
            tokens=total_tokens,
            time_ms=time_ms,
            draft_responses=draft_responses,
            review_responses=review_responses,
            judge_reasoning=judge_reasoning
        )
    
    def run_single_baseline(self, question: str, expected: str) -> ImprovedResult:
        """Single Model 기준선 - 모듈화 버전"""
        start_time = time.time()
        
        # config에서 baseline_single 역할 사용 
        baseline_model = self.registry.get_model("baseline_single")
        
        prompt = f"Question: {question}\n\nSolve this step by step:"
        response = baseline_model.generate(prompt, temperature=0.2, max_tokens=200)
        
        final_answer = ""
        if response.get("success", False):
            final_answer = response["response"].split('\n')[0].strip()
        
        time_ms = int((time.time() - start_time) * 1000)
        total_tokens = self._count_tokens(prompt, response.get("response", ""))
        correct = expected.lower().strip() in final_answer.lower()
        
        return ImprovedResult(
            method=f"Single {self.registry.get_model_name('baseline_single')} (Registry)",
            question=question,
            expected=expected,
            predicted=final_answer,
            correct=correct,
            tokens=total_tokens,
            time_ms=time_ms
        )
    
    def _count_tokens(self, prompt: str, response: str) -> int:
        """간단한 토큰 카운터"""
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("gpt2")
            return len(encoding.encode(prompt + response))
        except:
            # tiktoken 없으면 대략적 추정
            return len((prompt + response).split()) * 1.3

def main():
    """메인 실행 함수 - 설정 기반 테스트"""
    print(">>> 모듈화된 Multi-Agent 테스트 시작")
    
    # 환경별 테스트
    environments = ["development", "test"]  # production은 시간이 오래 걸려서 제외
    
    test_questions = [
        {"question": "What is the capital of South Korea?", "expected": "Seoul"},
        {"question": "What is 2 + 2?", "expected": "4"}, 
        {"question": "Name a primary color.", "expected": "red"}
    ]
    
    for env in environments:
        print(f"\n>>> 환경: {env}")
        tester = ImprovedMultiAgentTesterV2(environment=env)
        
        # 첫 번째 질문만 테스트 (시간 절약)
        q = test_questions[0]
        
        print(f"\n>>> 질문: {q['question']}")
        print(f">>> 정답: {q['expected']}")
        
        # 세 가지 방법 비교
        results = []
        
        print("\n>>> 1. 기존 Multi-Agent 테스트...")
        result1 = tester.run_original_multiagent(q["question"], q["expected"])
        results.append(result1)
        
        print("\n>>> 2. 개선된 Multi-Agent 테스트...")  
        result2 = tester.run_improved_multiagent(q["question"], q["expected"])
        results.append(result2)
        
        print("\n>>> 3. Single Model 기준선...")
        result3 = tester.run_single_baseline(q["question"], q["expected"])
        results.append(result3)
        
        # 결과 요약
        print(f"\n>>> {env} 환경 결과:")
        for r in results:
            status = "OK" if r.correct else "FAIL"
            efficiency = r.correct / (r.tokens / 100) if r.tokens > 0 else 0
            print(f"  [{status}] {r.method}")
            print(f"         정확도: {r.correct}, 토큰: {r.tokens}, 시간: {r.time_ms}ms, 효율성: {efficiency:.3f}")

if __name__ == "__main__":
    main()