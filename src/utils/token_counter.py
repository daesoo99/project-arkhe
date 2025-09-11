#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project Arkhē - Unified Token Counter
GPT tiktoken 기준 통일된 토큰 계산기
"""

import tiktoken
from typing import Dict, List, Union
from dataclasses import dataclass

@dataclass
class TokenUsage:
    """토큰 사용량 정보"""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    
    def to_dict(self) -> Dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens
        }

class UnifiedTokenCounter:
    """GPT tiktoken 기준 통일 토큰 계산기"""
    
    def __init__(self, model_name: str = "gpt-4"):
        """
        Args:
            model_name: 기준이 될 GPT 모델명 (토크나이저 결정)
        """
        try:
            self.encoder = tiktoken.encoding_for_model(model_name)
            self.model_name = model_name
        except KeyError:
            # 기본값으로 cl100k_base 사용 (gpt-4, gpt-3.5-turbo 등에서 사용)
            print(f"[!] Unknown model {model_name}, using cl100k_base encoder")
            self.encoder = tiktoken.get_encoding("cl100k_base")
            self.model_name = "gpt-4"
    
    def count_tokens(self, text: Union[str, List[str]]) -> int:
        """텍스트의 토큰 수 계산"""
        if isinstance(text, list):
            # 리스트인 경우 모든 텍스트 합계
            total = 0
            for t in text:
                if t:  # None이나 빈 문자열 체크
                    total += len(self.encoder.encode(str(t)))
            return total
        else:
            if not text:
                return 0
            return len(self.encoder.encode(str(text)))
    
    def calculate_usage(self, input_text: str, output_text: str) -> TokenUsage:
        """입력/출력 텍스트의 토큰 사용량 계산"""
        input_tokens = self.count_tokens(input_text)
        output_tokens = self.count_tokens(output_text)
        
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens
        )
    
    def calculate_multi_stage_usage(self, 
                                  stage_inputs: List[str], 
                                  stage_outputs: List[List[str]]) -> TokenUsage:
        """Multi-Agent의 다단계 토큰 사용량 계산"""
        total_input = 0
        total_output = 0
        
        # 각 단계별 계산
        for stage_input, stage_output_list in zip(stage_inputs, stage_outputs):
            # 입력 토큰
            total_input += self.count_tokens(stage_input)
            
            # 출력 토큰 (해당 단계의 모든 샘플)
            for output in stage_output_list:
                total_output += self.count_tokens(output)
        
        return TokenUsage(
            input_tokens=total_input,
            output_tokens=total_output,
            total_tokens=total_input + total_output
        )
    
    def get_model_info(self) -> Dict[str, str]:
        """사용 중인 토크나이저 정보"""
        return {
            "encoder_name": self.encoder.name,
            "base_model": self.model_name,
            "description": f"Unified token counting based on {self.model_name} tokenizer"
        }

class TokenTracker:
    """실험 전체의 토큰 사용량 추적"""
    
    def __init__(self):
        self.counter = UnifiedTokenCounter()
        self.usage_log = []
    
    def log_usage(self, method: str, category: str, question_id: str, 
                  input_text: str, output_text: str) -> TokenUsage:
        """토큰 사용량 기록"""
        usage = self.counter.calculate_usage(input_text, output_text)
        
        log_entry = {
            "method": method,
            "category": category,
            "question_id": question_id,
            "usage": usage.to_dict(),
            "timestamp": __import__('time').time()
        }
        
        self.usage_log.append(log_entry)
        return usage
    
    def log_multi_agent_usage(self, method: str, category: str, question_id: str,
                            stage_inputs: List[str], stage_outputs: List[List[str]]) -> TokenUsage:
        """Multi-Agent 토큰 사용량 기록"""
        usage = self.counter.calculate_multi_stage_usage(stage_inputs, stage_outputs)
        
        log_entry = {
            "method": method,
            "category": category,
            "question_id": question_id,
            "usage": usage.to_dict(),
            "stage_count": len(stage_inputs),
            "timestamp": __import__('time').time()
        }
        
        self.usage_log.append(log_entry)
        return usage
    
    def get_summary(self) -> Dict[str, any]:
        """전체 토큰 사용량 요약"""
        if not self.usage_log:
            return {"total_usage": 0, "method_breakdown": {}}
        
        method_totals = {}
        category_totals = {}
        
        for entry in self.usage_log:
            method = entry["method"]
            category = entry["category"]
            total = entry["usage"]["total_tokens"]
            
            # 메서드별 집계
            if method not in method_totals:
                method_totals[method] = {"count": 0, "total_tokens": 0}
            method_totals[method]["count"] += 1
            method_totals[method]["total_tokens"] += total
            
            # 카테고리별 집계
            if category not in category_totals:
                category_totals[category] = {"count": 0, "total_tokens": 0}
            category_totals[category]["count"] += 1
            category_totals[category]["total_tokens"] += total
        
        return {
            "total_entries": len(self.usage_log),
            "total_tokens": sum(entry["usage"]["total_tokens"] for entry in self.usage_log),
            "method_breakdown": method_totals,
            "category_breakdown": category_totals,
            "tokenizer_info": self.counter.get_model_info()
        }
    
    def save_log(self, filepath: str):
        """토큰 사용 로그 저장"""
        import json
        from pathlib import Path
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        export_data = {
            "tokenizer_info": self.counter.get_model_info(),
            "summary": self.get_summary(),
            "detailed_log": self.usage_log
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print(f"[+] Token usage log saved: {filepath}")

# 전역 인스턴스
global_token_tracker = TokenTracker()

def test_token_counter():
    """토큰 카운터 테스트"""
    print("=" * 50)
    print("*** TOKEN COUNTER TEST ***")
    print("=" * 50)
    
    counter = UnifiedTokenCounter()
    
    # 테스트 텍스트들
    test_cases = [
        "Hello world!",
        "What is 2 + 2?",
        "Write a Python function that returns the sum of two numbers.",
        "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes 4 into muffins for her friends every day. She sells the remainder at the farmers' market daily for $2 per egg. How much money does she make every day?"
    ]
    
    print(f"Using tokenizer: {counter.get_model_info()['encoder_name']}")
    print()
    
    for i, text in enumerate(test_cases, 1):
        tokens = counter.count_tokens(text)
        print(f"{i}. Text: {text[:50]}{'...' if len(text) > 50 else ''}")
        print(f"   Tokens: {tokens}")
        print()
    
    # Multi-stage 테스트
    stage_inputs = [
        "Answer this question: What is 2+2?",
        "Review this answer: 4",
        "Final judgment on: 4"
    ]
    
    stage_outputs = [
        ["4", "Four", "2+2=4"],
        ["The answer 4 is correct", "4 is right"],
        ["Final answer: 4"]
    ]
    
    multi_usage = counter.calculate_multi_stage_usage(stage_inputs, stage_outputs)
    print(f"Multi-stage example:")
    print(f"  Input tokens: {multi_usage.input_tokens}")
    print(f"  Output tokens: {multi_usage.output_tokens}")
    print(f"  Total tokens: {multi_usage.total_tokens}")

if __name__ == "__main__":
    test_token_counter()