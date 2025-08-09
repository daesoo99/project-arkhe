#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project Arkhe - REAL Economic Intelligence Test
진짜 경제적 지능: 정보 이론 + 승격 정책 + 실측 비용 + Shannon Entropy
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from orchestrator.economic_intelligence import EconomicIntelligencePipeline, InformationTheory
from llm.simple_llm import create_llm_auto
import time
import json

def run_economic_intelligence_test():
    """진짜 경제적 지능 테스트"""
    
    print("=" * 70)
    print("🧠 PROJECT ARKHE: REAL ECONOMIC INTELLIGENCE TEST")
    print("Features: Shannon Entropy | Promotion Policies | Multi-sampling")
    print("Cost Model: Token-based Real Cost (not constants)")
    print("=" * 70)
    
    # 경제적 지능 파이프라인 생성
    ei_pipeline = EconomicIntelligencePipeline(
        cost_sensitivity=0.3,  # 비용 민감도
        utility_weight=0.7     # 유용성 가중치
    )
    
    # LLM Factory
    def llm_factory(model_name: str):
        return create_llm_auto(model_name)
    
    # 테스트 쿼리
    test_queries = [
        "What is artificial intelligence?",  # 간단 - 승급 안될 수도
        "Explain quantum computing and its applications in cryptography.",  # 복잡 - 승급될 것
        "How do neural networks learn?",  # 중간 - 2단계까지?
    ]
    
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*20} TEST {i}/3 {'='*20}")
        print(f"Query: {query}")
        
        try:
            # 경제적 지능 실행
            result = ei_pipeline.execute(query, llm_factory)
            results.append(result)
            
            # 결과 출력
            print(f"\n📊 ECONOMIC INTELLIGENCE RESULT:")
            print(f"  Executed stages: {result.executed_stages}/{result.total_stages}")
            print(f"  Promotions: {result.promotion_decisions}")
            print(f"  Entropy progression: {[f'{e:.3f}' for e in result.entropy_progression]}")
            print(f"  Total cost: ${result.total_cost:.4f}")
            print(f"  Total time: {result.total_time:.0f}ms")
            print(f"  Economic efficiency: {result.economic_efficiency:.4f}")
            print(f"  Cost saved: {result.cost_saved_ratio:.1%}")
            print(f"  Final answer: {result.final_answer[:100]}...")
            
            # 단계별 상세
            print(f"\n📈 STAGE DETAILS:")
            for j, metrics in enumerate(result.stage_metrics):
                print(f"  Stage {j+1} ({metrics.model}): "
                      f"{metrics.tokens_used} tokens, "
                      f"${metrics.real_cost:.4f}, "
                      f"{metrics.total_latency:.0f}ms, "
                      f"{metrics.tokens_per_second:.1f} tok/s")
        
        except Exception as e:
            print(f"❌ Test {i} failed: {e}")
            results.append(None)
    
    # 전체 결과 분석
    print(f"\n{'='*70}")
    print("🎯 ECONOMIC INTELLIGENCE ANALYSIS")
    print(f"{'='*70}")
    
    successful_results = [r for r in results if r is not None]
    
    if successful_results:
        # 승급률 분석
        total_possible_promotions = len(successful_results) * 2  # 각 쿼리당 2번 승급 기회
        actual_promotions = sum(sum(r.promotion_decisions) for r in successful_results)
        promotion_rate = actual_promotions / total_possible_promotions if total_possible_promotions > 0 else 0
        
        # 비용 분석
        avg_cost = sum(r.total_cost for r in successful_results) / len(successful_results)
        avg_stages = sum(r.executed_stages for r in successful_results) / len(successful_results)
        avg_efficiency = sum(r.economic_efficiency for r in successful_results) / len(successful_results)
        avg_savings = sum(r.cost_saved_ratio for r in successful_results) / len(successful_results)
        
        print(f"Successful tests: {len(successful_results)}/{len(test_queries)}")
        print(f"Promotion rate: {promotion_rate:.1%}")
        print(f"Average stages executed: {avg_stages:.1f}/3")
        print(f"Average cost: ${avg_cost:.4f}")
        print(f"Average efficiency: {avg_efficiency:.4f}")
        print(f"Average cost savings: {avg_savings:.1%}")
        
        # 정보 이론 분석
        all_entropies = []
        for r in successful_results:
            all_entropies.extend(r.entropy_progression)
        
        if all_entropies:
            print(f"\n🔬 INFORMATION THEORY ANALYSIS:")
            print(f"  Entropy range: {min(all_entropies):.3f} - {max(all_entropies):.3f}")
            print(f"  Average entropy: {sum(all_entropies)/len(all_entropies):.3f}")
        
        # 경제적 지능 검증
        print(f"\n✅ ECONOMIC INTELLIGENCE VALIDATION:")
        
        if avg_savings > 0:
            print(f"  💰 COST EFFICIENCY: {avg_savings:.1%} saved vs naive execution")
        else:
            print(f"  ⚠️  COST OVERHEAD: {-avg_savings:.1%} extra cost")
            
        if promotion_rate < 0.8:
            print(f"  🎯 SMART ROUTING: {(1-promotion_rate):.1%} of stages skipped intelligently")
        else:
            print(f"  🔄 HIGH COMPLEXITY: {promotion_rate:.1%} promotion rate indicates complex queries")
            
        if avg_efficiency > 0:
            print(f"  📈 EFFICIENCY GAIN: {avg_efficiency:.4f} utility per time unit")
        else:
            print(f"  📉 EFFICIENCY LOSS: {avg_efficiency:.4f} - needs tuning")
    
    else:
        print("❌ All tests failed!")
    
    print(f"\n{'='*70}")
    print("🏆 REAL ECONOMIC INTELLIGENCE TEST COMPLETE")
    print(f"{'='*70}")
    
    return results

if __name__ == "__main__":
    results = run_economic_intelligence_test()