# -*- coding: utf-8 -*-
"""
간단한 사고과정 압축 테스트
"""

import sys
sys.path.append('.')

def test_simple_reasoning():
    """간단한 사고과정이 있는 경우 테스트"""
    
    # 더 간단한 사고과정
    test_responses = [
        "2+2=4. I added 2 and 2.",
        "2+2 equals 4. I used basic addition.",
        "The answer is 4. I calculated 2 plus 2."
    ]
    
    test_context = "What is 2+2?"
    
    try:
        from src.orchestrator.thought_aggregator import ThoughtAggregator
        
        print("Testing simple reasoning compression...")
        aggregator = ThoughtAggregator(model_name="qwen2:0.5b")
        analysis = aggregator.analyze_thoughts(test_responses, test_context)
        
        print(f"\nOriginal tokens: {analysis.original_tokens}")
        print(f"Compressed tokens: {analysis.compressed_tokens}")
        print(f"Compression ratio: {analysis.compression_ratio:.3f}")
        print(f"Token reduction: {(1 - analysis.compression_ratio) * 100:.1f}%")
        
        print(f"\nCommon core: {analysis.common_core}")
        print(f"Unique approaches: {analysis.unique_approaches}")
        print(f"\nCompressed context: {analysis.compressed_context}")
        
        return analysis.compression_ratio < 1.0  # 압축 성공 여부
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = test_simple_reasoning()
    print(f"\nSimple reasoning compression successful: {success}")