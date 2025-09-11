#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick AC verification"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Test 1: ContextualPipeline
print("=== AC1: ContextualPipeline ===")
try:
    from src.llm.simple_llm import create_llm_auto
    from src.orchestrator.pipeline import run_3stage_with_context
    res = run_3stage_with_context(create_llm_auto, "What is 2+2?")
    print(f"SUCCESS: final={res['final'][:50]}...")
    ac1_pass = True
except Exception as e:
    print(f"FAILED: {e}")
    ac1_pass = False

# Test 2: EI Lite Mode
print("\n=== AC2: EI Lite Mode ===")
os.environ["ARKHE_EI_MODE"] = "lite"
try:
    from src.agents.economic_intelligence import EconomicIntelligenceAgent
    agent = EconomicIntelligenceAgent()
    result = agent.execute("What is AI?")
    print(f"SUCCESS: executed_stages={result.executed_stages}")
    ac2_pass = True
except Exception as e:
    print(f"FAILED: {e}")
    ac2_pass = False
finally:
    if "ARKHE_EI_MODE" in os.environ:
        del os.environ["ARKHE_EI_MODE"]

# Test 3: Backward compatibility
print("\n=== AC3: Backward Compatibility ===")
try:
    from src.agents.economic_intelligence import EconomicIntelligencePipeline
    agent = EconomicIntelligencePipeline()
    print("SUCCESS: EconomicIntelligencePipeline alias works")
    ac3_pass = True
except Exception as e:
    print(f"FAILED: {e}")
    ac3_pass = False

# Summary
passed = sum([ac1_pass, ac2_pass, ac3_pass])
print(f"\n=== SUMMARY: {passed}/3 AC PASSED ===")
if passed == 3:
    print("ALL ACCEPTANCE CRITERIA SATISFIED!")
else:
    print("Some criteria need attention")