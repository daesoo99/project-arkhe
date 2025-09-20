#!/usr/bin/env python3
"""
Complex Reasoning Test: Multi-Agent vs Single Model
테스트 복잡한 추론 문제에서의 성능 비교
"""

import json
import time
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm.simple_llm import create_llm_auto
from orchestrator.pipeline import run_3stage_with_context
from utils.scorers import score_task

def load_complex_tasks():
    """복잡한 추론 문제 로드"""
    tasks_file = Path(__file__).parent.parent / "prompts" / "complex_tasks.jsonl"
    tasks = []
    
    with open(tasks_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    
    return tasks

def test_single_model(task, llm_func):
    """Single model 테스트"""
    start_time = time.time()
    
    # Single llama3:8b로 테스트
    llm = llm_func("llama3:8b")
    response_dict = llm.generate(task["prompt"])
    response = response_dict.get('response', str(response_dict)) if isinstance(response_dict, dict) else str(response_dict)
    
    end_time = time.time()
    
    # 토큰 계산 (추정)
    tokens = len(task["prompt"].split()) + len(response.split())
    
    return {
        "response": response,
        "time": end_time - start_time,
        "tokens": tokens,
        "model": "llama3:8b"
    }

def test_multi_agent(task, llm_func):
    """Multi-Agent 테스트 (qwen2:7b → qwen2:7b → llama3:8b)"""
    start_time = time.time()
    
    # B안 방식으로 실행 (사고과정 직접 전달)
    result = run_3stage_with_context(llm_func, task["prompt"])
    
    end_time = time.time()
    
    # 토큰 계산 (누적)
    total_tokens = 0
    for stage in ["draft_responses", "review_responses", "final"]:
        if stage in result:
            if isinstance(result[stage], list):
                for r in result[stage]:
                    total_tokens += len(str(r).split())
            else:
                total_tokens += len(str(result[stage]).split())
    
    total_tokens += len(task["prompt"].split()) * 6  # 각 단계별 프롬프트
    
    return {
        "response": result.get("final", ""),
        "time": end_time - start_time,
        "tokens": total_tokens,
        "model": "qwen2:7b+qwen2:7b+llama3:8b",
        "details": result
    }

def calculate_metrics(result, task):
    """성능 메트릭 계산"""
    # 스코어링 (간단한 유사도 체크)
    score = score_task(task["type"], task["answer"], result["response"])
    
    # 효율성 = 정확도 / (토큰수/100)
    efficiency = score["score"] / (result["tokens"] / 100) if result["tokens"] > 0 else 0
    
    return {
        "accuracy": score["score"],
        "tokens": result["tokens"],
        "time": result["time"],
        "efficiency": efficiency,
        "reasoning": score.get("reasoning", "")
    }

def main():
    """메인 실험 실행"""
    print("Complex Reasoning Test: Multi-Agent vs Single Model")
    print("=" * 60)
    
    # LLM 생성 함수
    llm_func = create_llm_auto
    
    # 복잡한 문제 로드
    tasks = load_complex_tasks()
    print(f"Load complex tasks: {len(tasks)}")
    
    results = []
    
    # 각 문제에 대해 테스트
    for i, task in enumerate(tasks[:3]):  # 처음 3개만 테스트
        print(f"\nTest {i+1}: {task['id']} (complexity: {task['complexity']})")
        print(f"Problem: {task['prompt'][:100]}...")
        
        try:
            # Single model 테스트
            print("  Testing Single Model...")
            single_result = test_single_model(task, llm_func)
            single_metrics = calculate_metrics(single_result, task)
            
            # Multi-Agent 테스트
            print("  Testing Multi-Agent...")
            multi_result = test_multi_agent(task, llm_func)
            multi_metrics = calculate_metrics(multi_result, task)
            
            # 결과 저장
            result = {
                "task": task,
                "single": {**single_result, **single_metrics},
                "multi": {**multi_result, **multi_metrics}
            }
            results.append(result)
            
            # 중간 결과 출력
            print(f"    Single: accuracy {single_metrics['accuracy']:.2f}, "
                  f"토큰 {single_metrics['tokens']}, 시간 {single_metrics['time']:.1f}s")
            print(f"    Multi:  accuracy {multi_metrics['accuracy']:.2f}, "
                  f"토큰 {multi_metrics['tokens']}, 시간 {multi_metrics['time']:.1f}s")
            print(f"    Efficiency: Single {single_metrics['efficiency']:.4f} vs "
                  f"Multi {multi_metrics['efficiency']:.4f}")
                  
        except Exception as e:
            print(f"    Error: {e}")
            continue
    
    # 전체 결과 분석
    print("\n" + "=" * 60)
    print("Overall Results Analysis")
    print("=" * 60)
    
    if results:
        # 평균 계산
        single_avg = {
            "accuracy": sum(r["single"]["accuracy"] for r in results) / len(results),
            "tokens": sum(r["single"]["tokens"] for r in results) / len(results),
            "time": sum(r["single"]["time"] for r in results) / len(results),
            "efficiency": sum(r["single"]["efficiency"] for r in results) / len(results)
        }
        
        multi_avg = {
            "accuracy": sum(r["multi"]["accuracy"] for r in results) / len(results),
            "tokens": sum(r["multi"]["tokens"] for r in results) / len(results),
            "time": sum(r["multi"]["time"] for r in results) / len(results),
            "efficiency": sum(r["multi"]["efficiency"] for r in results) / len(results)
        }
        
        print(f"Single Model Average:")
        print(f"  Accuracy: {single_avg['accuracy']:.3f}")
        print(f"  Tokens: {single_avg['tokens']:.0f}")
        print(f"  Time: {single_avg['time']:.1f}s")
        print(f"  Efficiency: {single_avg['efficiency']:.4f}")
        
        print(f"\nMulti-Agent Average:")
        print(f"  정확도: {multi_avg['accuracy']:.3f}")
        print(f"  토큰수: {multi_avg['tokens']:.0f}")
        print(f"  시간: {multi_avg['time']:.1f}s")
        print(f"  효율성: {multi_avg['efficiency']:.4f}")
        
        print(f"\nPerformance Comparison:")
        acc_ratio = multi_avg['accuracy'] / single_avg['accuracy'] if single_avg['accuracy'] > 0 else 0
        token_ratio = multi_avg['tokens'] / single_avg['tokens'] if single_avg['tokens'] > 0 else 0
        eff_ratio = multi_avg['efficiency'] / single_avg['efficiency'] if single_avg['efficiency'] > 0 else 0
        
        print(f"  Accuracy ratio (Multi/Single): {acc_ratio:.2f}")
        print(f"  Token ratio (Multi/Single): {token_ratio:.2f}")
        print(f"  Efficiency ratio (Multi/Single): {eff_ratio:.2f}")
        
        # 결과 저장
        timestamp = int(time.time())
        output_file = Path(__file__).parent.parent / "results" / f"complex_reasoning_results_{timestamp}.json"
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "experiment": "complex_reasoning_test",
                "timestamp": timestamp,
                "results": results,
                "summary": {
                    "single_avg": single_avg,
                    "multi_avg": multi_avg,
                    "ratios": {
                        "accuracy": acc_ratio,
                        "tokens": token_ratio,
                        "efficiency": eff_ratio
                    }
                }
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved: {output_file}")
        
        # 결론 출력
        print(f"\nExperiment Conclusion:")
        if acc_ratio > 1.1:
            print(f"  Multi-Agent wins in accuracy by {((acc_ratio-1)*100):.1f}%!")
        elif acc_ratio < 0.9:
            print(f"  Single Model wins in accuracy by {((1/acc_ratio-1)*100):.1f}%")
        else:
            print(f"  Similar accuracy levels")
            
        if eff_ratio > 1.1:
            print(f"  Multi-Agent wins in efficiency by {((eff_ratio-1)*100):.1f}%!")
        elif eff_ratio < 0.9:
            print(f"  Single Model wins in efficiency by {((1/eff_ratio-1)*100):.1f}%")
        else:
            print(f"  Similar efficiency levels")

if __name__ == "__main__":
    main()