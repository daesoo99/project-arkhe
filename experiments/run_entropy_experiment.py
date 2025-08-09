#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project Arkhē - Shannon Entropy 승급 정책 실험 실행기
가설: H(초안 샘플들)이 임계치 τ를 넘을 때만 상위 단계로 승급하면 비용↓, 품질 유지 가능

실험 설계:
- τ 스윕: τ ∈ {0.6, 0.8, 1.0, 1.2}
- k 샘플: k ∈ {2, 3, 5}  
- 측정: 비용, 지연, 품질, 승급율
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from itertools import product

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from orchestrator.experimental_pipeline import create_entropy_experiment_pipeline
from llm.simple_llm import create_llm_auto

def load_experimental_questions(dataset_path: str = None) -> Dict[str, List[Dict]]:
    """실험 질문 데이터셋 로드"""
    if dataset_path is None:
        dataset_path = Path(__file__).parent.parent / "datasets" / "experimental_questions.json"
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data

def run_single_experiment(question: Dict[str, Any], tau1: float, tau2: float, 
                        k_samples: int, run_id: int) -> Dict[str, Any]:
    """단일 실험 실행"""
    
    # 실험 파이프라인 생성
    pipeline = create_entropy_experiment_pipeline(tau1=tau1, tau2=tau2, k_samples=k_samples)
    
    # 실험 ID 생성
    experiment_id = f"tau1_{tau1}_tau2_{tau2}_k_{k_samples}_run_{run_id}_{question['id']}"
    
    # 파이프라인 실행
    start_time = time.time()
    result = pipeline.run(
        query=question["query"], 
        llm_factory=create_llm_auto,
        experiment_id=experiment_id
    )
    end_time = time.time()
    
    # 결과에 실험 메타데이터 추가
    result["experiment_metadata"] = {
        "question_id": question["id"],
        "question_category": question.get("category", "unknown"),
        "question_difficulty": question.get("difficulty", "unknown"),
        "tau1": tau1,
        "tau2": tau2,
        "k_samples": k_samples,
        "run_id": run_id,
        "experiment_id": experiment_id,
        "total_experiment_time": end_time - start_time,
        "expected_answer": question.get("expected_answer", None)
    }
    
    return result

def run_parameter_sweep():
    """파라미터 스윕 실험 실행"""
    
    print("=" * 70)
    print("*** SHANNON ENTROPY 승급 정책 실험 ***")
    print("=" * 70)
    
    # 실험 파라미터 (간소화)
    tau1_values = [0.8, 1.2]  # 2개 조건
    tau2_values = [1.0, 1.4]  # 2개 조건
    k_values = [3]            # 1개 조건
    num_runs = 2              # 각 조건별 2회 반복
    
    # 데이터셋 로드
    questions_data = load_experimental_questions()
    
    # 폐쇄형 질문 5개만 사용 (빠른 데모)
    test_questions = questions_data["closed_form_questions"][:5]
    
    print(f"실험 설정:")
    print(f"  τ1 values: {tau1_values}")
    print(f"  τ2 values: {tau2_values}")
    print(f"  k values: {k_values}")
    print(f"  Questions: {len(test_questions)}")
    print(f"  Runs per condition: {num_runs}")
    
    total_experiments = len(tau1_values) * len(tau2_values) * len(k_values) * len(test_questions) * num_runs
    print(f"  Total experiments: {total_experiments}")
    print()
    
    # 결과 저장용
    all_results = []
    experiment_count = 0
    
    # 파라미터 조합별 실험
    for tau1, tau2, k in product(tau1_values, tau2_values, k_values):
        
        print(f"[*] Condition: τ1={tau1}, τ2={tau2}, k={k}")
        condition_results = []
        
        for question in test_questions:
            for run_id in range(num_runs):
                experiment_count += 1
                
                print(f"  [{experiment_count}/{total_experiments}] {question['id']} (run {run_id + 1})")
                
                try:
                    result = run_single_experiment(question, tau1, tau2, k, run_id)
                    condition_results.append(result)
                    all_results.append(result)
                    
                    # 간단한 진행 상황 출력
                    metrics = result["metrics"]
                    print(f"    Executed: {metrics['executed_steps']}/{metrics['total_steps']} steps, "
                          f"Time: {metrics['total_time_ms']}ms")
                    
                except Exception as e:
                    print(f"    ERROR: {e}")
                    # 에러도 기록
                    error_result = {
                        "error": str(e),
                        "experiment_metadata": {
                            "question_id": question["id"],
                            "tau1": tau1, "tau2": tau2, "k_samples": k,
                            "run_id": run_id,
                            "failed": True
                        }
                    }
                    all_results.append(error_result)
        
        # 조건별 요약
        successful_runs = [r for r in condition_results if "error" not in r]
        if successful_runs:
            avg_executed = sum(r["metrics"]["executed_steps"] for r in successful_runs) / len(successful_runs)
            avg_time = sum(r["metrics"]["total_time_ms"] for r in successful_runs) / len(successful_runs)
            print(f"  [+] Condition Summary: Avg steps={avg_executed:.1f}, Avg time={avg_time:.0f}ms")
        print()
    
    # 전체 결과 저장
    results_file = f"experiments/results/entropy_experiment_{int(time.time())}.json"
    Path(results_file).parent.mkdir(parents=True, exist_ok=True)
    
    final_results = {
        "experiment_type": "shannon_entropy_promotion_policy",
        "timestamp": time.time(),
        "parameters": {
            "tau1_values": tau1_values,
            "tau2_values": tau2_values, 
            "k_values": k_values,
            "num_runs": num_runs,
            "total_questions": len(test_questions)
        },
        "results": all_results,
        "summary": {
            "total_experiments": len(all_results),
            "successful_experiments": len([r for r in all_results if "error" not in r]),
            "failed_experiments": len([r for r in all_results if "error" in r])
        }
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    print("=" * 70)
    print("[+] 실험 완료!")
    print(f"[-] 결과 저장: {results_file}")
    print(f"[*] 총 실험: {final_results['summary']['total_experiments']}")
    print(f"[+] 성공: {final_results['summary']['successful_experiments']}")
    print(f"[-] 실패: {final_results['summary']['failed_experiments']}")
    print("=" * 70)
    
    return results_file

def run_pilot_experiment():
    """파일럿 실험 - 작은 규모로 빠른 검증"""
    
    print("*** PILOT EXPERIMENT - Shannon Entropy 승급 정책 ***")
    print("=" * 50)
    
    # 데이터셋 로드
    questions_data = load_experimental_questions()
    
    # 간단한 질문 3개만 사용
    pilot_questions = questions_data["closed_form_questions"][:3]
    
    # 2개 조건만 테스트
    conditions = [
        {"tau1": 0.8, "tau2": 1.0, "k": 3},  # 보통 임계치
        {"tau1": 1.2, "tau2": 1.4, "k": 3}   # 높은 임계치 (더 많은 스킵 예상)
    ]
    
    results = []
    
    for i, condition in enumerate(conditions):
        print(f"\n[*] Condition {i+1}: {condition}")
        
        for question in pilot_questions:
            print(f"  Testing: {question['id']}")
            
            try:
                result = run_single_experiment(
                    question, 
                    condition["tau1"], 
                    condition["tau2"],
                    condition["k"], 
                    run_id=0
                )
                
                results.append(result)
                
                # 결과 요약
                metrics = result["metrics"]
                print(f"    → {metrics['executed_steps']}/{metrics['total_steps']} steps, "
                      f"{metrics['total_time_ms']}ms")
                
            except Exception as e:
                print(f"    → ERROR: {e}")
    
    print(f"\n[+] Pilot experiment completed: {len(results)} successful runs")
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Shannon Entropy 승급 정책 실험")
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot",
                       help="실험 모드: pilot (빠른 검증) 또는 full (전체 실험)")
    
    args = parser.parse_args()
    
    if args.mode == "pilot":
        run_pilot_experiment()
    else:
        run_parameter_sweep()