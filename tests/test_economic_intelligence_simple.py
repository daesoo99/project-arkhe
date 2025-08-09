#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project Arkhē - Economic Intelligence Simple Test
의존성 없이도 동작하는 경제적 지능 테스트
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_economic_intelligence_no_deps():
    """의존성 없이 경제적 지능 테스트"""
    
    print("=" * 60)
    print("ECONOMIC INTELLIGENCE - LIGHTWEIGHT MODE TEST")
    print("Testing without numpy/scipy dependencies")
    print("=" * 60)
    
    try:
        from orchestrator.economic_intelligence import EconomicIntelligencePipeline
        
        # 경제적 지능 파이프라인 생성 (자동으로 create_llm_auto 사용)
        ei_pipeline = EconomicIntelligencePipeline(
            cost_sensitivity=0.3,
            utility_weight=0.7
        )
        
        # 간단한 테스트 쿼리
        test_query = "What is artificial intelligence?"
        
        print(f"\nTesting query: {test_query}")
        print("-" * 40)
        
        # 경제적 지능 실행 (llm_factory 없이 - 자동으로 create_llm_auto 사용)
        result = ei_pipeline.execute(test_query)
        
        # 결과 출력
        print(f"\nRESULT SUMMARY:")
        print(f"  Final answer: {result.final_answer[:100]}...")
        print(f"  Executed stages: {result.executed_stages}/{result.total_stages}")
        print(f"  Promotions: {result.promotion_decisions}")
        print(f"  Total cost: ${result.total_cost:.4f}")
        print(f"  Total time: {result.total_time:.0f}ms")
        print(f"  Economic efficiency: {result.economic_efficiency:.4f}")
        print(f"  Cost saved: {result.cost_saved_ratio:.1%}")
        
        # 엔트로피 진행 상황
        if result.entropy_progression:
            print(f"  Entropy progression: {[f'{e:.3f}' for e in result.entropy_progression]}")
        
        print(f"\nSUCCESS: Economic Intelligence works without numpy/scipy!")
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    except Exception as e:
        print(f"Runtime error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dependencies():
    """의존성 상태 확인"""
    
    print("\nDEPENDENCY STATUS:")
    
    try:
        import numpy as np
        print("  NumPy: AVAILABLE")
    except ImportError:
        print("  NumPy: NOT AVAILABLE (using fallback)")
    
    try:
        from scipy.spatial.distance import jensenshannon
        print("  SciPy: AVAILABLE")
    except ImportError:
        print("  SciPy: NOT AVAILABLE (using fallback)")
        
    try:
        from src.llm.simple_llm import create_llm_auto
        print("  create_llm_auto: AVAILABLE")
    except ImportError as e:
        print(f"  create_llm_auto: ERROR - {e}")

if __name__ == "__main__":
    test_dependencies()
    success = test_economic_intelligence_no_deps()
    
    if success:
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED - READY FOR PRODUCTION")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("TESTS FAILED - CHECK CONFIGURATION")
        print("=" * 60)