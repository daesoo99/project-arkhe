#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shannon Entropy 승급 정책 빠른 데모
핵심 기능 검증용 간소화 실험
"""

import sys
import os
import json
import time
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from orchestrator.experimental_pipeline import create_entropy_experiment_pipeline
from llm.simple_llm import create_llm_auto

def run_demo():
    """빠른 데모 실행"""
    print("*** SHANNON ENTROPY 승급 정책 데모 ***")
    print("=" * 50)
    
    # 테스트 질문들
    test_questions = [
        {"id": "simple", "query": "대한민국의 수도는?", "expected": "서울"},
        {"id": "complex", "query": "Python에서 리스트를 정렬하는 메서드는?", "expected": "sort()"}
    ]
    
    # 두 가지 임계치 조건
    conditions = [
        {"tau1": 0.8, "tau2": 1.0, "name": "보통 임계치"},
        {"tau1": 1.2, "tau2": 1.4, "name": "높은 임계치"}
    ]
    
    results = []
    
    for condition in conditions:
        print(f"\n[*] {condition['name']}: τ1={condition['tau1']}, τ2={condition['tau2']}")
        
        # 파이프라인 생성
        pipeline = create_entropy_experiment_pipeline(
            tau1=condition["tau1"], 
            tau2=condition["tau2"], 
            k_samples=3
        )
        
        for question in test_questions:
            print(f"  Testing: {question['id']} - {question['query']}")
            
            start_time = time.time()
            try:
                result = pipeline.run(
                    query=question["query"],
                    llm_factory=create_llm_auto,
                    experiment_id=f"demo_{condition['name']}_{question['id']}"
                )
                
                # 결과 요약
                metrics = result["metrics"]
                steps_executed = metrics["executed_steps"]
                total_steps = metrics["total_steps"]
                total_time = time.time() - start_time
                
                cost_savings = (total_steps - steps_executed) / total_steps * 100
                
                print(f"    → {steps_executed}/{total_steps} steps ({cost_savings:.0f}% cost reduction)")
                print(f"    → Time: {total_time:.1f}s")
                print(f"    → Final: {result['final'][:50]}...")
                
                results.append({
                    "condition": condition["name"],
                    "question": question["id"],
                    "steps_executed": steps_executed,
                    "total_steps": total_steps,
                    "cost_savings_pct": cost_savings,
                    "time_sec": total_time,
                    "final_answer": result["final"]
                })
                
            except Exception as e:
                print(f"    → ERROR: {e}")
    
    # 결과 요약
    print("\n" + "=" * 50)
    print("[+] 데모 완료 - 결과 요약:")
    
    for result in results:
        print(f"  {result['condition']} / {result['question']}: "
              f"{result['cost_savings_pct']:.0f}% 비용 절약, "
              f"{result['time_sec']:.1f}초")
    
    # 평균 비용 절약 계산
    avg_savings = sum(r["cost_savings_pct"] for r in results) / len(results)
    print(f"\n[*] 평균 비용 절약: {avg_savings:.1f}%")
    
    return results

if __name__ == "__main__":
    run_demo()