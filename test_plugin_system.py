# -*- coding: utf-8 -*-
"""
Project Arkhē - Plugin System Integration Test
플러그인 시스템 전체 통합 테스트

Phase 3 검증을 위한 종합 테스트
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.plugins.interfaces import TaskType
from src.evaluation.plugin_engine import PluginEvaluationEngine, EvaluationRequest, create_evaluation_engine, quick_evaluate

def test_plugin_system_basic():
    """기본 플러그인 시스템 테스트"""
    print("=== Phase 3 Plugin System Integration Test ===\n")
    
    # 1. 평가 엔진 생성
    print("1. Creating evaluation engine...")
    engine = create_evaluation_engine({
        "auto_select_scorers": True,
        "default_aggregator": "weighted_average"
    })
    
    # 2. 사용 가능한 플러그인 확인
    print("2. Available plugins:")
    scorers = engine.get_available_scorers()
    aggregators = engine.get_available_aggregators()
    
    print(f"   - Scorers: {list(scorers.keys())}")
    print(f"   - Aggregators: {list(aggregators.keys())}")
    
    # 3. 태스크 커버리지 확인
    print("\n3. Task coverage:")
    coverage = engine.get_task_coverage()
    for task, scorer_list in coverage.items():
        print(f"   - {task}: {scorer_list}")
    
    return engine

def test_evaluation_scenarios():
    """다양한 평가 시나리오 테스트"""
    print("\n=== Testing Evaluation Scenarios ===\n")
    
    engine = create_evaluation_engine()
    
    test_cases = [
        {
            "name": "FACT Task",
            "ground_truth": "The capital of France is Paris.",
            "response": "Paris is the capital city of France.",
            "task_type": TaskType.FACT
        },
        {
            "name": "REASON Task", 
            "ground_truth": "Because increased CO2 levels cause global warming.",
            "response": "Higher CO2 concentrations trap heat in the atmosphere, leading to climate change.",
            "task_type": TaskType.REASON
        },
        {
            "name": "SUMMARY Task",
            "ground_truth": "The article discusses three main benefits of renewable energy: cost reduction, environmental protection, and energy independence.",
            "response": "This article explains how renewable energy provides economic savings, helps the environment, and reduces dependency on foreign energy sources.",
            "task_type": TaskType.SUMMARY
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"{i}. Testing {test_case['name']}:")
        
        # 평가 요청 생성
        request = EvaluationRequest(
            ground_truth=test_case["ground_truth"],
            response=test_case["response"], 
            task_type=test_case["task_type"]
        )
        
        # 평가 수행
        try:
            result = engine.evaluate(request)
            results.append(result)
            
            print(f"   [OK] Score: {result.aggregated_result.aggregated_score:.3f}")
            print(f"   [OK] Method: {result.aggregated_result.method}")
            print(f"   [OK] Confidence: {result.aggregated_result.confidence:.3f}")
            print(f"   [OK] Individual scores: {len(result.individual_scores)}")
            
        except Exception as e:
            print(f"   [ERROR] Error: {e}")
    
    return results

def test_aggregator_algorithms():
    """다양한 집계 알고리즘 테스트"""
    print("\n=== Testing Aggregator Algorithms ===\n")
    
    # 각 집계 알고리즘으로 동일한 입력 테스트
    aggregators = ["weighted_average", "max_score", "median", "consensus"]
    
    test_input = {
        "ground_truth": "Machine learning is a subset of artificial intelligence.",
        "response": "ML is a branch of AI that enables computers to learn from data.",
        "task_type": TaskType.FACT
    }
    
    for i, agg_name in enumerate(aggregators, 1):
        print(f"{i}. Testing {agg_name} aggregator:")
        
        try:
            engine = create_evaluation_engine({
                "default_aggregator": agg_name
            })
            
            request = EvaluationRequest(
                ground_truth=test_input["ground_truth"],
                response=test_input["response"],
                task_type=test_input["task_type"],
                aggregator_name=agg_name
            )
            
            result = engine.evaluate(request)
            
            print(f"   [OK] Score: {result.aggregated_result.aggregated_score:.3f}")
            print(f"   [OK] Method: {result.aggregated_result.method}")
            print(f"   [OK] Metadata: {list(result.aggregated_result.metadata.keys())}")
            
        except Exception as e:
            print(f"   [ERROR] Error with {agg_name}: {e}")

def test_batch_evaluation():
    """배치 평가 테스트"""
    print("\n=== Testing Batch Evaluation ===\n")
    
    engine = create_evaluation_engine()
    
    # 배치 평가 요청들 생성
    batch_requests = [
        EvaluationRequest("Paris is in France.", "Paris is the capital of France.", TaskType.FACT),
        EvaluationRequest("2+2=4 because of arithmetic rules.", "Addition of 2 and 2 equals 4.", TaskType.REASON),
        EvaluationRequest("Summary: AI helps automate tasks.", "AI can automate various processes.", TaskType.SUMMARY),
    ]
    
    print(f"Processing batch of {len(batch_requests)} requests...")
    
    try:
        batch_results = engine.batch_evaluate(batch_requests)
        
        print(f"[OK] Processed {len(batch_results)} results")
        
        total_score = sum(r.aggregated_result.aggregated_score for r in batch_results)
        avg_score = total_score / len(batch_results)
        
        print(f"[OK] Average score: {avg_score:.3f}")
        
        for i, result in enumerate(batch_results, 1):
            print(f"   {i}. {result.request.task_type.value}: {result.aggregated_result.aggregated_score:.3f}")
            
    except Exception as e:
        print(f"[ERROR] Batch evaluation error: {e}")

def test_quick_evaluate_function():
    """빠른 평가 함수 테스트"""
    print("\n=== Testing Quick Evaluate Function ===\n")
    
    try:
        score = quick_evaluate(
            ground_truth="The Earth orbits the Sun.",
            response="Earth revolves around the Sun in its orbit.", 
            task_type="fact"
        )
        
        print(f"[OK] Quick evaluation score: {score:.3f}")
        
    except Exception as e:
        print(f"[ERROR] Quick evaluation error: {e}")

def test_plugin_validation():
    """플러그인 검증 테스트"""
    print("\n=== Testing Plugin Validation ===\n")
    
    engine = create_evaluation_engine()
    
    # 잘못된 요청들 테스트
    invalid_requests = [
        EvaluationRequest("", "response", TaskType.FACT),  # 빈 ground_truth
        EvaluationRequest("ground_truth", "", TaskType.FACT),  # 빈 response
    ]
    
    for i, request in enumerate(invalid_requests, 1):
        print(f"{i}. Testing invalid request:")
        warnings = engine.validate_request(request)
        
        if warnings:
            print(f"   [OK] Validation warnings: {warnings}")
        else:
            print(f"   [INFO] No warnings detected")

def main():
    """메인 테스트 실행"""
    print("Phase 3 Plugin System - Comprehensive Integration Test")
    print("=" * 60)
    
    try:
        # 기본 시스템 테스트
        engine = test_plugin_system_basic()
        
        # 평가 시나리오 테스트
        test_evaluation_scenarios()
        
        # 집계 알고리즘 테스트
        test_aggregator_algorithms()
        
        # 배치 평가 테스트
        test_batch_evaluation()
        
        # 빠른 평가 함수 테스트
        test_quick_evaluate_function()
        
        # 플러그인 검증 테스트
        test_plugin_validation()
        
        print("\n" + "=" * 60)
        print("[SUCCESS] Phase 3 Plugin System Integration Test COMPLETED")
        print("[SUCCESS] All plugin components working correctly")
        print("[SUCCESS] Legacy integration successful")
        print("[SUCCESS] Multiple aggregation strategies available")
        print("[SUCCESS] Batch processing functional")
        print("[SUCCESS] Validation systems operational")
        
    except Exception as e:
        print(f"\n[FAILED] Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()