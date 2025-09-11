#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project Arkhē - Acceptance Criteria Verification
수용 기준 검증 테스트
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_contextual_pipeline_ac():
    """ContextualPipeline AC 검증"""
    print("=" * 60)
    print("ACCEPTANCE CRITERIA 1: CONTEXTUAL PIPELINE")
    print("=" * 60)
    
    try:
        from src.llm.simple_llm import create_llm_auto
        from src.orchestrator.pipeline import run_3stage_with_context
        
        print("Testing: run_3stage_with_context(create_llm_auto, '대한민국의 수도는?')")
        res = run_3stage_with_context(create_llm_auto, "대한민국의 수도는?")
        
        print(f"+ FINAL: {res['final'][:120]}...")
        print(f"+ CTX-keys: {list(res['context'].keys())}")
        print(f"+ METRICS: executed_steps={res['metrics']['executed_steps']}, skipped_steps={res['metrics']['skipped_steps']}")
        
        return True
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ei_lite_mode():
    """EI Lite 모드 검증"""
    print("\n" + "=" * 60)
    print("ACCEPTANCE CRITERIA 2: EI LITE MODE")
    print("=" * 60)
    
    # 환경변수 설정
    os.environ["ARKHE_EI_MODE"] = "lite"
    
    try:
        from src.agents.economic_intelligence import EconomicIntelligenceAgent
        
        agent = EconomicIntelligenceAgent()
        print("Testing: EI Agent in LITE mode (no numpy/scipy required)")
        
        result = agent.execute("What is AI?")
        
        print(f"+ SUCCESS: EI Lite mode works")
        print(f"  Final answer: {result.final_answer[:100]}...")
        print(f"  Executed stages: {result.executed_stages}/{result.total_stages}")
        print(f"  Total cost: ${result.total_cost:.4f}")
        
        return True
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 환경변수 정리
        if "ARKHE_EI_MODE" in os.environ:
            del os.environ["ARKHE_EI_MODE"]

def test_ei_strict_mode():
    """EI Strict 모드 검증"""
    print("\n" + "=" * 60)
    print("ACCEPTANCE CRITERIA 3: EI STRICT MODE")  
    print("=" * 60)
    
    # 환경변수 설정
    os.environ["ARKHE_EI_MODE"] = "strict"
    
    try:
        from src.agents.economic_intelligence import EconomicIntelligenceAgent
        
        agent = EconomicIntelligenceAgent()
        print("Testing: EI Agent in STRICT mode (requires numpy/scipy)")
        
        result = agent.execute("What is machine learning?")
        
        print(f"+ SUCCESS: EI Strict mode works")
        print(f"  Final answer: {result.final_answer[:100]}...")
        print(f"  Executed stages: {result.executed_stages}/{result.total_stages}")
        print(f"  Entropy progression: {[f'{e:.3f}' for e in result.entropy_progression]}")
        
        return True
        
    except RuntimeError as e:
        if "requires numpy + scipy" in str(e):
            print(f"+ EXPECTED: {e}")
            return True
        else:
            print(f"- UNEXPECTED RuntimeError: {e}")
            return False
            
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 환경변수 정리  
        if "ARKHE_EI_MODE" in os.environ:
            del os.environ["ARKHE_EI_MODE"]

def test_ei_auto_mode():
    """EI Auto 모드 검증 (기본)"""
    print("\n" + "=" * 60)
    print("ACCEPTANCE CRITERIA 4: EI AUTO MODE (DEFAULT)")
    print("=" * 60)
    
    try:
        from src.agents.economic_intelligence import EconomicIntelligenceAgent
        
        agent = EconomicIntelligenceAgent()
        print("Testing: EI Agent in AUTO mode (default behavior)")
        
        result = agent.execute("Explain neural networks")
        
        print(f"+ SUCCESS: EI Auto mode works")
        print(f"  Final answer: {result.final_answer[:100]}...")
        print(f"  Executed stages: {result.executed_stages}/{result.total_stages}")
        print(f"  Promotions: {result.promotion_decisions}")
        print(f"  Economic efficiency: {result.economic_efficiency:.4f}")
        print(f"  Cost saved: {result.cost_saved_ratio:.1%}")
        
        return True
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backward_compatibility():
    """기존 테스트 스크립트 호환성 검증"""
    print("\n" + "=" * 60)
    print("ACCEPTANCE CRITERIA 5: BACKWARD COMPATIBILITY")
    print("=" * 60)
    
    try:
        # 기존 스키마 체크
        from src.agents.economic_intelligence import EconomicIntelligenceAgent, EconomicIntelligencePipeline
        
        # 별칭 확인
        agent1 = EconomicIntelligenceAgent()
        agent2 = EconomicIntelligencePipeline()  # 하위 호환성 별칭
        
        print("+ SUCCESS: EconomicIntelligencePipeline alias works")
        
        # 기존 시그니처 호환성
        result = agent1.execute("Test query", mode="auto")
        
        required_fields = [
            'executed_stages', 'total_stages', 'promotion_decisions', 
            'entropy_progression', 'total_cost', 'total_time', 
            'economic_efficiency', 'cost_saved_ratio', 'final_answer'
        ]
        
        for field in required_fields:
            if not hasattr(result, field):
                print(f"- MISSING FIELD: {field}")
                return False
                
        print(f"+ SUCCESS: All required fields present in result schema")
        
        # stage_metrics 세부 필드 검증
        if result.stage_metrics:
            stage = result.stage_metrics[0]
            stage_required = ['tokens_used', 'real_cost', 'total_latency', 'tokens_per_second']
            for field in stage_required:
                if not hasattr(stage, field):
                    print(f"- MISSING STAGE FIELD: {field}")
                    return False
                    
        print(f"+ SUCCESS: Stage metrics schema compatible")
        return True
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("PROJECT ARKHE - ACCEPTANCE CRITERIA VERIFICATION")
    print("=" * 70)
    
    tests = [
        test_contextual_pipeline_ac,
        test_ei_lite_mode,
        test_ei_strict_mode, 
        test_ei_auto_mode,
        test_backward_compatibility
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"- TEST EXCEPTION: {e}")
            results.append(False)
    
    print("\n" + "=" * 70)
    print("# ACCEPTANCE CRITERIA RESULTS")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results), 1):
        status = "+ PASS" if result else "- FAIL"
        print(f"  AC{i}: {test.__name__} - {status}")
    
    print(f"\n> SUMMARY: {passed}/{total} criteria passed")
    
    if passed == total:
        print("* ALL ACCEPTANCE CRITERIA SATISFIED!")
        print("+ READY FOR PRODUCTION")
    else:
        print("!  Some criteria need attention")
    
    print("\n" + "=" * 70)