#!/usr/bin/env python3
"""
Project Arkhē - Adaptive System Test
새로운 문제들에 대한 지능형 모델 배치 테스트
"""

import json
import time
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm.simple_llm import create_llm_auto
from orchestrator.adaptive_system import AdaptiveOrchestrator
from utils.scorers import score_task

def load_adaptive_test_problems():
    """새로운 테스트 문제들 로드"""
    tasks_file = Path(__file__).parent.parent / "prompts" / "adaptive_test_problems.jsonl"
    tasks = []
    
    with open(tasks_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    
    return tasks

def test_baseline_approach(task, llm_func):
    """기존 방식: 항상 llama3:8b Single 사용"""
    start_time = time.time()
    
    llm = llm_func("llama3:8b")
    response_dict = llm.generate(task["prompt"])
    response = response_dict.get('response', str(response_dict)) if isinstance(response_dict, dict) else str(response_dict)
    
    end_time = time.time()
    
    # 토큰 계산 (추정)
    tokens = len(task["prompt"].split()) + len(response.split())
    
    return {
        "approach": "baseline",
        "model_used": "llama3:8b",
        "response": response,
        "time": end_time - start_time,
        "tokens": tokens,
        "decision_rationale": "Always use powerful single model"
    }

def test_adaptive_approach(task, llm_func, orchestrator):
    """새로운 방식: 적응형 모델 선택"""
    start_time = time.time()
    
    result = orchestrator.execute_optimal_solution(task["prompt"], llm_func)
    
    end_time = time.time()
    result["time"] = end_time - start_time
    result["approach"] = "adaptive"
    
    return result

def calculate_metrics(result, task):
    """성능 메트릭 계산"""
    # 정확도 평가
    score = score_task(task.get("type", "reason"), task["answer"], result["response"])
    accuracy = score["score"]
    
    # 효율성 계산
    tokens = result.get("tokens", 0)
    efficiency = accuracy / (tokens / 100) if tokens > 0 else 0
    
    return {
        "accuracy": accuracy,
        "tokens": tokens,
        "time": result.get("time", 0),
        "efficiency": efficiency,
        "reasoning": score.get("reasoning", "")
    }

def main():
    """메인 실험 실행"""
    print("Adaptive System Test: Smart Model Selection")
    print("=" * 60)
    
    # 시스템 초기화
    orchestrator = AdaptiveOrchestrator()
    llm_func = create_llm_auto
    
    # 테스트 문제 로드
    tasks = load_adaptive_test_problems()
    print(f"Loaded test problems: {len(tasks)}")
    
    results = []
    
    # 각 문제에 대해 테스트
    for i, task in enumerate(tasks):
        print(f"\n--- Test {i+1}: {task['id']} ---")
        print(f"Expected: {task['expected_config']}, Complexity: {task['expected_complexity']}")
        print(f"Problem: {task['prompt'][:80]}...")
        
        try:
            # Baseline 테스트 (항상 llama3:8b)
            print("\\n  [BASELINE] Testing with llama3:8b...")
            baseline_result = test_baseline_approach(task, llm_func)
            baseline_metrics = calculate_metrics(baseline_result, task)
            
            print("\\n  [ADAPTIVE] Testing with smart selection...")
            # Adaptive 테스트 (지능형 선택)
            adaptive_result = test_adaptive_approach(task, llm_func, orchestrator)
            adaptive_metrics = calculate_metrics(adaptive_result, task)
            
            # 결과 비교
            result = {
                "task": task,
                "baseline": {**baseline_result, **baseline_metrics},
                "adaptive": {**adaptive_result, **adaptive_metrics}
            }
            results.append(result)
            
            # 중간 결과 출력
            print(f"\\n  Results:")
            print(f"    Baseline:  accuracy {baseline_metrics['accuracy']:.2f}, tokens {baseline_metrics['tokens']}, time {baseline_metrics['time']:.1f}s")
            print(f"    Adaptive:  accuracy {adaptive_metrics['accuracy']:.2f}, tokens {adaptive_metrics['tokens']}, time {adaptive_metrics['time']:.1f}s")
            
            # 효율성 비교
            efficiency_gain = (adaptive_metrics['efficiency'] / baseline_metrics['efficiency'] - 1) * 100 if baseline_metrics['efficiency'] > 0 else 0
            accuracy_change = (adaptive_metrics['accuracy'] - baseline_metrics['accuracy']) * 100
            
            print(f"    Adaptive vs Baseline:")
            print(f"       Accuracy: {accuracy_change:+.1f}%")  
            print(f"       Efficiency: {efficiency_gain:+.1f}%")
            print(f"       Decision: {adaptive_result.get('configuration', {}).get('rationale', 'N/A')}")
            
        except Exception as e:
            print(f"    Error: {e}")
            continue
    
    # 전체 결과 분석
    print("\\n" + "=" * 60)
    print("Overall Performance Analysis")
    print("=" * 60)
    
    if results:
        # 카테고리별 분석
        categories = {}
        for result in results:
            task_type = result["task"].get("type", "unknown")
            if task_type not in categories:
                categories[task_type] = {"baseline": [], "adaptive": []}
            
            categories[task_type]["baseline"].append(result["baseline"])
            categories[task_type]["adaptive"].append(result["adaptive"])
        
        print("\\n📊 Performance by Category:")
        
        overall_baseline = {"accuracy": 0, "tokens": 0, "time": 0, "efficiency": 0}
        overall_adaptive = {"accuracy": 0, "tokens": 0, "time": 0, "efficiency": 0}
        
        for category, data in categories.items():
            print(f"\\n🏷️  {category.upper()}:")
            
            # 평균 계산
            baseline_avg = {
                "accuracy": sum(r["accuracy"] for r in data["baseline"]) / len(data["baseline"]),
                "tokens": sum(r["tokens"] for r in data["baseline"]) / len(data["baseline"]), 
                "time": sum(r["time"] for r in data["baseline"]) / len(data["baseline"]),
                "efficiency": sum(r["efficiency"] for r in data["baseline"]) / len(data["baseline"])
            }
            
            adaptive_avg = {
                "accuracy": sum(r["accuracy"] for r in data["adaptive"]) / len(data["adaptive"]),
                "tokens": sum(r["tokens"] for r in data["adaptive"]) / len(data["adaptive"]),
                "time": sum(r["time"] for r in data["adaptive"]) / len(data["adaptive"]), 
                "efficiency": sum(r["efficiency"] for r in data["adaptive"]) / len(data["adaptive"])
            }
            
            print(f"    Baseline:  acc {baseline_avg['accuracy']:.2f}, tok {baseline_avg['tokens']:.0f}, eff {baseline_avg['efficiency']:.3f}")
            print(f"    Adaptive:  acc {adaptive_avg['accuracy']:.2f}, tok {adaptive_avg['tokens']:.0f}, eff {adaptive_avg['efficiency']:.3f}")
            
            # 개선율 계산
            acc_improvement = (adaptive_avg['accuracy'] - baseline_avg['accuracy']) * 100
            token_reduction = (1 - adaptive_avg['tokens'] / baseline_avg['tokens']) * 100 if baseline_avg['tokens'] > 0 else 0
            eff_improvement = (adaptive_avg['efficiency'] / baseline_avg['efficiency'] - 1) * 100 if baseline_avg['efficiency'] > 0 else 0
            
            print(f"    📈 Improvement: acc {acc_improvement:+.1f}%, tok {token_reduction:+.1f}%, eff {eff_improvement:+.1f}%")
            
            # 전체 평균에 추가
            for key in overall_baseline:
                overall_baseline[key] += baseline_avg[key] * len(data["baseline"])
                overall_adaptive[key] += adaptive_avg[key] * len(data["adaptive"])
        
        # 전체 평균 계산
        total_count = len(results)
        for key in overall_baseline:
            overall_baseline[key] /= total_count
            overall_adaptive[key] /= total_count
        
        print(f"\\n🎯 OVERALL PERFORMANCE:")
        print(f"  Baseline:  acc {overall_baseline['accuracy']:.2f}, tok {overall_baseline['tokens']:.0f}, eff {overall_baseline['efficiency']:.3f}")
        print(f"  Adaptive:  acc {overall_adaptive['accuracy']:.2f}, tok {overall_adaptive['tokens']:.0f}, eff {overall_adaptive['efficiency']:.3f}")
        
        # 전체 개선율
        overall_acc_improvement = (overall_adaptive['accuracy'] - overall_baseline['accuracy']) * 100
        overall_token_reduction = (1 - overall_adaptive['tokens'] / overall_baseline['tokens']) * 100
        overall_eff_improvement = (overall_adaptive['efficiency'] / overall_baseline['efficiency'] - 1) * 100
        
        print(f"  📈 Overall Improvement:")
        print(f"     Accuracy: {overall_acc_improvement:+.1f}%")
        print(f"     Token Efficiency: {overall_token_reduction:+.1f}%") 
        print(f"     Overall Efficiency: {overall_eff_improvement:+.1f}%")
        
        # 결과 저장
        timestamp = int(time.time())
        output_file = Path(__file__).parent.parent / "results" / f"adaptive_system_results_{timestamp}.json"
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "experiment": "adaptive_system_test",
                "timestamp": timestamp,
                "results": results,
                "summary": {
                    "overall_baseline": overall_baseline,
                    "overall_adaptive": overall_adaptive,
                    "improvement_rates": {
                        "accuracy": overall_acc_improvement,
                        "token_efficiency": overall_token_reduction,
                        "overall_efficiency": overall_eff_improvement
                    }
                }
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\\nResults saved: {output_file}")
        
        # 최종 결론
        print(f"\\n🎯 Conclusion:")
        if overall_eff_improvement > 10:
            print(f"  ✅ Adaptive system shows {overall_eff_improvement:.1f}% efficiency improvement!")
            print(f"  🎯 Smart model selection is significantly better than one-size-fits-all")
        elif overall_eff_improvement > 0:
            print(f"  ✅ Adaptive system shows modest {overall_eff_improvement:.1f}% improvement")
            print(f"  💡 Targeted optimization provides measurable benefits")
        else:
            print(f"  🤔 Adaptive system shows {overall_eff_improvement:.1f}% change")
            print(f"  📚 More analysis needed to understand the patterns")

if __name__ == "__main__":
    main()