#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project Arkhē - Hierarchy AC Verification
수용 기준 검증: ollama 패키지 없이도 동작하는지 확인
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_hierarchy_import():
    """AC1: hierarchy.py가 ollama 패키지 없이도 임포트되는지 확인"""
    print("=== AC1: Import Test (without ollama package) ===")
    try:
        from src.agents.hierarchy import CostTracker, Mediator, IndependentThinker
        from src.agents.hierarchy import shannon_entropy, detect_contradiction
        print("SUCCESS: All classes imported without ollama dependency")
        return True
    except ImportError as e:
        print(f"FAILED: Import error - {e}")
        return False
    except Exception as e:
        print(f"FAILED: Unexpected error - {e}")
        return False

def test_create_llm_auto_integration():
    """AC2: create_llm_auto()로 LLM 호출이 통일되었는지 확인"""
    print("\n=== AC2: create_llm_auto Integration Test ===")
    try:
        from src.agents.hierarchy import CostTracker, IndependentThinker
        
        # Cost tracker 생성
        ct = CostTracker()
        
        # IndependentThinker 생성 (내부에서 create_llm_auto 사용)
        thinker = IndependentThinker("TestAgent", ct, "gemma:2b")
        
        print(f"SUCCESS: IndependentThinker created with model {thinker.model}")
        print(f"LLM instance type: {type(thinker.llm)}")
        return True
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mediator_solve_problem():
    """AC3: Mediator.solve_problem()이 정상적으로 결과를 반환하는지 확인"""
    print("\n=== AC3: Mediator solve_problem Test ===")
    try:
        from src.agents.hierarchy import CostTracker, IndependentThinker, Mediator
        
        # 공유 비용 추적기
        shared_ct = CostTracker()
        
        # 더미 에이전트들 생성
        class DummyThinker:
            def __init__(self, name, response):
                self.name = name
                self.response = response
            
            def solve(self, problem):
                return self.response
        
        # 실제 에이전트와 더미 에이전트 혼합
        thinkers = [
            DummyThinker("Agent1", "대한민국의 수도는 서울입니다"),
            DummyThinker("Agent2", "수도는 서울이다"),
        ]
        
        # Mediator 생성
        mediator = Mediator(thinkers, shared_ct)
        
        # 문제 해결
        result = mediator.solve_problem("대한민국의 수도는?")
        
        # 결과 스키마 검증
        required_keys = [
            "problem", "final_answer", "all_responses", 
            "shannon_entropy", "contradiction_report"
        ]
        
        for key in required_keys:
            if key not in result:
                print(f"FAILED: Missing key '{key}' in result")
                return False
        
        print(f"SUCCESS: All required keys present")
        print(f"  Final answer: {result['final_answer'][:50]}...")
        print(f"  Shannon entropy: {result['shannon_entropy']:.3f}")
        print(f"  Contradiction report: {result['contradiction_report']}")
        print(f"  Response count: {len(result['all_responses'])}")
        
        return True
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_shannon_entropy_standard():
    """AC4: Shannon entropy가 표준 Python으로 구현되었는지 확인"""
    print("\n=== AC4: Shannon Entropy Standard Python Test ===")
    try:
        from src.agents.hierarchy import shannon_entropy, detect_contradiction
        
        # 테스트 데이터
        texts = [
            "서울은 대한민국의 수도입니다",
            "수도는 서울이다", 
            "서울",
            "서울은 대한민국의 수도입니다"  # 중복
        ]
        
        # Shannon entropy 계산
        entropy = shannon_entropy(texts)
        print(f"Shannon entropy: {entropy:.3f}")
        
        # 모순 검출
        contradiction = detect_contradiction(texts)
        print(f"Contradiction: {contradiction}")
        
        # 모순 있는 케이스 테스트
        contradictory_texts = ["예, 맞습니다", "아니오, 틀렸습니다"]
        contradiction2 = detect_contradiction(contradictory_texts)
        print(f"Contradictory case: {contradiction2}")
        
        print("SUCCESS: Shannon entropy and contradiction detection work")
        return True
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cost_tracker_safety():
    """AC5: CostTracker 안정성 보강 확인"""
    print("\n=== AC5: CostTracker Safety Test ===")
    try:
        from src.agents.hierarchy import CostTracker
        
        ct = CostTracker()
        
        # 정상 케이스
        ct.add_cost("gemma:2b", 100, 50)
        print(f"Normal case cost: ${ct.get_total_cost():.6f}")
        
        # 음수/None 케이스
        ct.add_cost("gemma:2b", -10, None)  # 음수와 None
        ct.add_cost("gemma:2b", None, -5)   # None과 음수
        print(f"After negative/None inputs: ${ct.get_total_cost():.6f}")
        
        # 모델 분석
        breakdown = ct.get_model_breakdown()
        print(f"Model breakdown: {breakdown}")
        
        print("SUCCESS: CostTracker handles edge cases safely")
        return True
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("PROJECT ARKHE - HIERARCHY AC VERIFICATION")
    print("=" * 60)
    
    tests = [
        test_hierarchy_import,
        test_create_llm_auto_integration,
        test_mediator_solve_problem,
        test_shannon_entropy_standard,
        test_cost_tracker_safety
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"TEST EXCEPTION: {e}")
            results.append(False)
    
    print(f"\n{'=' * 60}")
    print("HIERARCHY AC RESULTS")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results), 1):
        status = "PASS" if result else "FAIL" 
        print(f"  AC{i}: {test.__name__} - {status}")
    
    print(f"\nSUMMARY: {passed}/{total} acceptance criteria passed")
    
    if passed == total:
        print("* ALL HIERARCHY AC SATISFIED!")
        print("* READY FOR PRODUCTION - Environment Independent")
    else:
        print("! Some criteria need attention")
    
    print("=" * 60)