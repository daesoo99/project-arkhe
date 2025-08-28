# -*- coding: utf-8 -*-
"""
ThoughtAggregator A방안 실험 실행 스크립트
사용자가 직접 실행할 수 있는 간단한 실험 코드
"""

import sys
import os
import time
sys.path.append('.')

def run_basic_test():
    """기본 ThoughtAggregator 테스트"""
    print("=" * 60)
    print("1. Basic ThoughtAggregator Test")
    print("=" * 60)
    
    try:
        from src.orchestrator.thought_aggregator import ThoughtAggregator
        
        # 테스트 데이터
        test_responses = [
            "Seoul is the capital of South Korea. It's a modern city with advanced technology.",
            "The capital city of South Korea is Seoul, which has a population of about 10 million people.", 
            "Seoul, South Korea's capital, is famous for K-pop, technology companies, and traditional palaces."
        ]
        
        test_context = "What is the capital of South Korea?"
        
        print(f"Question: {test_context}")
        print(f"Number of responses to analyze: {len(test_responses)}")
        print()
        
        # ThoughtAggregator 실행
        aggregator = ThoughtAggregator(model_name="qwen2:0.5b")
        analysis = aggregator.analyze_thoughts(test_responses, test_context)
        
        # 결과 출력
        print("RESULTS:")
        print(f"- Original tokens: {analysis.original_tokens}")
        print(f"- Compressed tokens: {analysis.compressed_tokens}")  
        print(f"- Compression ratio: {analysis.compression_ratio:.3f}")
        print(f"- Token reduction: {(1 - analysis.compression_ratio) * 100:.1f}%")
        print()
        print(f"- Common core: {analysis.common_core}")
        print(f"- Unique approaches: {len(analysis.unique_approaches)}")
        for i, approach in enumerate(analysis.unique_approaches):
            print(f"  {i+1}. {approach}")
        print()
        print(f"- Compressed context: {analysis.compressed_context}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_pipeline_comparison():
    """파이프라인 비교 실험"""
    print("\n" + "=" * 60)
    print("2. Pipeline Comparison Experiment") 
    print("=" * 60)
    
    try:
        from src.orchestrator.thought_compression_pipeline import run_comparison_experiment
        
        # 테스트 질문
        test_query = "Explain machine learning in simple terms"
        print(f"Test query: {test_query}")
        print()
        
        # 비교 실험 실행
        results = run_comparison_experiment(test_query)
        
        # 결과 정리
        print("\n" + "=" * 60)
        print("COMPARISON RESULTS:")
        print("=" * 60)
        
        compression = results["compression"]
        baseline = results["baseline"] 
        comparison = results["comparison"]
        
        print(f"A방안 (ThoughtAggregator):")
        print(f"  - Tokens: {compression['total_tokens']}")
        print(f"  - Time: {compression['total_time']:.2f}s")
        print(f"  - Answer: {compression['final_answer'][:100]}...")
        print()
        
        print(f"기존 방식 (Cumulative):")
        print(f"  - Tokens: {baseline['total_tokens']}")  
        print(f"  - Time: {baseline['total_time']:.2f}s")
        print(f"  - Answer: {baseline['final_answer'][:100]}...")
        print()
        
        print(f"Comparison:")
        print(f"  - Token reduction: {comparison['token_reduction'] * 100:.1f}%")
        print(f"  - Time change: {comparison['time_change'] * 100:+.1f}%")
        print(f"  - Compression ratio: {comparison['compression_ratio']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_multiple_queries():
    """여러 질문으로 테스트"""
    print("\n" + "=" * 60)
    print("3. Multiple Query Test")
    print("=" * 60)
    
    queries = [
        "What is 2+2?",
        "How do you make coffee?", 
        "Explain quantum physics",
        "What are the benefits of exercise?"
    ]
    
    try:
        from src.orchestrator.thought_aggregator import ThoughtAggregator
        
        aggregator = ThoughtAggregator(model_name="qwen2:0.5b")
        results = []
        
        for i, query in enumerate(queries):
            print(f"\nQuery {i+1}: {query}")
            
            # 3개의 다양한 응답 시뮬레이션 (실제로는 LLM이 생성)
            test_responses = [
                f"Response 1 to: {query}",
                f"Alternative answer for: {query}",  
                f"Different perspective on: {query}"
            ]
            
            analysis = aggregator.analyze_thoughts(test_responses, query)
            
            result = {
                "query": query,
                "original_tokens": analysis.original_tokens,
                "compressed_tokens": analysis.compressed_tokens,
                "compression_ratio": analysis.compression_ratio,
                "token_reduction": (1 - analysis.compression_ratio) * 100
            }
            
            results.append(result)
            
            print(f"  - Token reduction: {result['token_reduction']:.1f}%")
            print(f"  - Compression ratio: {result['compression_ratio']:.3f}")
        
        print(f"\nSUMMARY:")
        avg_reduction = sum(r['token_reduction'] for r in results) / len(results)
        avg_ratio = sum(r['compression_ratio'] for r in results) / len(results)
        
        print(f"- Average token reduction: {avg_reduction:.1f}%")
        print(f"- Average compression ratio: {avg_ratio:.3f}")
        print(f"- Tested queries: {len(results)}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback  
        traceback.print_exc()
        return False

def main():
    """메인 실험 실행"""
    print("ThoughtAggregator Experiment Runner")
    print("사용자 직접 실행용 - 결과를 Claude에게 복사해서 분석 요청하세요!")
    print()
    
    # 1. 기본 테스트
    success1 = run_basic_test()
    
    # 2. 파이프라인 비교 (선택적)
    try_pipeline = input("\n파이프라인 비교 실험도 실행하시겠습니까? (y/n): ").lower().strip()
    success2 = True
    if try_pipeline in ['y', 'yes', '예']:
        success2 = run_pipeline_comparison()
    
    # 3. 다중 쿼리 테스트 (선택적)
    try_multiple = input("\n다중 쿼리 테스트도 실행하시겠습니까? (y/n): ").lower().strip()  
    success3 = True
    if try_multiple in ['y', 'yes', '예']:
        success3 = run_multiple_queries()
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETED!")
    print("=" * 60)
    print("위 결과를 Claude에게 복사해서 분석을 요청하세요.")
    print(f"Success: Basic={success1}, Pipeline={success2}, Multiple={success3}")

if __name__ == "__main__":
    main()