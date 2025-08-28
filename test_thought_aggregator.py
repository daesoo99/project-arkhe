# -*- coding: utf-8 -*-
"""
ThoughtAggregator 단순 테스트
"""

import sys
import os
sys.path.append('.')

def test_thought_aggregator():
    """ThoughtAggregator 기본 테스트"""
    
    # 사고과정이 다른 테스트 응답들
    test_responses = [
        "Seoul is the capital of South Korea. I know this because it has 10 million people, making it the largest city, so it must be the capital.",
        "The capital city of South Korea is Seoul. I figured this out by thinking about K-pop and Samsung headquarters - they're both in Seoul, indicating it's the economic center and therefore the capital.", 
        "Seoul is South Korea's capital. My reasoning: Seoul has Gyeongbokgung Palace and other historical sites, showing it has been the political center for centuries."
    ]
    
    test_context = "What is the capital of South Korea?"
    
    try:
        from src.orchestrator.thought_aggregator import ThoughtAggregator
        
        print("Creating ThoughtAggregator...")
        aggregator = ThoughtAggregator(model_name="qwen2:0.5b")
        
        print("Analyzing thoughts...")
        analysis = aggregator.analyze_thoughts(test_responses, test_context)
        
        print("\n=== Analysis Results ===")
        print(f"Common core: {analysis.common_core}")
        print(f"Unique approaches: {analysis.unique_approaches}")
        print(f"Compressed context: {analysis.compressed_context}")
        print(f"Original tokens: {analysis.original_tokens}")
        print(f"Compressed tokens: {analysis.compressed_tokens}")
        print(f"Compression ratio: {analysis.compression_ratio:.2f}")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_thought_aggregator()
    if success:
        print("\nTest completed successfully!")
    else:
        print("\nTest failed!")