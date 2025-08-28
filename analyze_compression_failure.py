# -*- coding: utf-8 -*-
"""
압축 실패 원인 분석 스크립트
복잡한 사고과정에서 압축이 실패하는 이유를 단계별로 분석
"""

import sys
sys.path.append('.')

def analyze_compression_failure():
    """압축 실패 원인 단계별 분석"""
    
    print("=" * 60)
    print("복잡한 사고과정 압축 실패 원인 분석")
    print("=" * 60)
    
    # 복잡한 사고과정 (실패 케이스)
    complex_responses = [
        "Seoul is the capital of South Korea. I know this because it has 10 million people, making it the largest city, so it must be the capital.",
        "The capital city of South Korea is Seoul. I figured this out by thinking about K-pop and Samsung headquarters - they're both in Seoul, indicating it's the economic center and therefore the capital.", 
        "Seoul is South Korea's capital. My reasoning: Seoul has Gyeongbokgung Palace and other historical sites, showing it has been the political center for centuries."
    ]
    
    # 간단한 사고과정 (성공 케이스)
    simple_responses = [
        "2+2=4. I added 2 and 2.",
        "2+2 equals 4. I used basic addition.",
        "The answer is 4. I calculated 2 plus 2."
    ]
    
    try:
        from src.orchestrator.thought_aggregator import ThoughtAggregator
        
        aggregator = ThoughtAggregator(model_name="qwen2:0.5b")
        
        print("\n1. 성공 케이스 분석 (간단한 사고과정)")
        print("-" * 40)
        simple_analysis = aggregator.analyze_thoughts(simple_responses, "What is 2+2?")
        
        print(f"원본: {len(' | '.join(simple_responses))} 글자")
        print(f"공통 핵심: {len(simple_analysis.common_core)} 글자")
        print(f"개별 특징 수: {len(simple_analysis.unique_approaches)}")
        print(f"압축 결과: {len(simple_analysis.compressed_context)} 글자")
        print(f"압축률: {simple_analysis.compression_ratio:.3f}")
        
        print(f"\n공통 핵심 내용:")
        print(f"'{simple_analysis.common_core}'")
        print(f"\n개별 특징들:")
        for i, approach in enumerate(simple_analysis.unique_approaches):
            print(f"  {i+1}. '{approach}'")
        
        print("\n" + "="*60)
        print("2. 실패 케이스 분석 (복잡한 사고과정)")
        print("-" * 40)
        
        complex_analysis = aggregator.analyze_thoughts(complex_responses, "What is the capital of South Korea?")
        
        print(f"원본: {len(' | '.join(complex_responses))} 글자")
        print(f"공통 핵심: {len(complex_analysis.common_core)} 글자")  
        print(f"개별 특징 수: {len(complex_analysis.unique_approaches)}")
        print(f"최종 압축 결과: {len(complex_analysis.compressed_context)} 글자")
        print(f"압축률: {complex_analysis.compression_ratio:.3f}")
        
        print(f"\n공통 핵심 내용:")
        print(f"'{complex_analysis.common_core[:200]}{'...' if len(complex_analysis.common_core) > 200 else ''}'")
        print(f"\n개별 특징들:")
        for i, approach in enumerate(complex_analysis.unique_approaches):
            print(f"  {i+1}. '{approach[:100]}{'...' if len(approach) > 100 else ''}'")
        
        # 실패 원인 분석
        print("\n" + "="*60)
        print("3. 실패 원인 분석")
        print("-" * 40)
        
        # 중복 정도 측정
        simple_text = ' | '.join(simple_responses)
        complex_text = ' | '.join(complex_responses)
        
        # 간단한 단어 중복도 계산
        simple_words = set(simple_text.lower().split())
        complex_words = set(complex_text.lower().split())
        
        print(f"간단한 케이스 고유 단어 수: {len(simple_words)}")
        print(f"복잡한 케이스 고유 단어 수: {len(complex_words)}")
        
        # 각 응답의 길이 분석
        print(f"\n응답 길이 분석:")
        print("간단한 케이스:")
        for i, resp in enumerate(simple_responses):
            print(f"  응답{i+1}: {len(resp.split())} 단어")
            
        print("복잡한 케이스:")  
        for i, resp in enumerate(complex_responses):
            print(f"  응답{i+1}: {len(resp.split())} 단어")
            
        # 공통점 vs 차이점 비율
        print(f"\n압축 전후 길이 비교:")
        print(f"간단한 케이스: {len(simple_text)} → {len(simple_analysis.compressed_context)} (비율: {len(simple_analysis.compressed_context)/len(simple_text):.3f})")
        print(f"복잡한 케이스: {len(complex_text)} → {len(complex_analysis.compressed_context)} (비율: {len(complex_analysis.compressed_context)/len(complex_text):.3f})")
        
        return {
            "simple_success": simple_analysis.compression_ratio < 1.0,
            "complex_success": complex_analysis.compression_ratio < 1.0,
            "simple_ratio": simple_analysis.compression_ratio,
            "complex_ratio": complex_analysis.compression_ratio
        }
        
    except Exception as e:
        print(f"분석 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_step_by_step():
    """단계별 압축 과정 분석"""
    print("\n" + "="*60)
    print("4. 단계별 압축 과정 분석")
    print("="*60)
    
    # 복잡한 응답으로 각 단계별 결과 확인
    responses = [
        "Seoul is the capital of South Korea. I know this because it has 10 million people, making it the largest city, so it must be the capital.",
        "The capital city of South Korea is Seoul. I figured this out by thinking about K-pop and Samsung headquarters - they're both in Seoul, indicating it's the economic center and therefore the capital.", 
        "Seoul is South Korea's capital. My reasoning: Seoul has Gyeongbokgung Palace and other historical sites, showing it has been the political center for centuries."
    ]
    
    try:
        from src.orchestrator.thought_aggregator import ThoughtAggregator
        
        aggregator = ThoughtAggregator(model_name="qwen2:0.5b")
        context = "What is the capital of South Korea?"
        
        # 1단계: 공통 요소 추출
        print("1단계: 공통 요소 추출")
        print("-" * 30)
        common_core = aggregator._extract_common_elements(responses, context)
        print(f"결과: '{common_core}'")
        print(f"길이: {len(common_core)} 글자, {len(common_core.split())} 단어")
        
        # 2단계: 개별 특징 분석  
        print(f"\n2단계: 개별 특징 분석")
        print("-" * 30)
        unique_approaches = aggregator._identify_unique_approaches(responses, common_core)
        print(f"개수: {len(unique_approaches)}개")
        total_unique_length = sum(len(approach) for approach in unique_approaches)
        print(f"총 길이: {total_unique_length} 글자")
        for i, approach in enumerate(unique_approaches):
            print(f"  특징{i+1}: '{approach[:80]}{'...' if len(approach) > 80 else ''}'")
        
        # 3단계: 최종 압축
        print(f"\n3단계: 최종 압축")
        print("-" * 30)
        compressed = aggregator._create_compressed_context(common_core, unique_approaches, context)
        print(f"결과: '{compressed[:150]}{'...' if len(compressed) > 150 else ''}'")
        print(f"길이: {len(compressed)} 글자, {len(compressed.split())} 단어")
        
        # 원본과 비교
        original_text = ' | '.join(responses)
        print(f"\n비교 결과:")
        print(f"원본: {len(original_text)} 글자")
        print(f"1단계 후: {len(common_core)} 글자 (비율: {len(common_core)/len(original_text):.3f})")
        print(f"2단계 후: {len(common_core) + total_unique_length} 글자 (비율: {(len(common_core) + total_unique_length)/len(original_text):.3f})")
        print(f"3단계 후: {len(compressed)} 글자 (비율: {len(compressed)/len(original_text):.3f})")
        
    except Exception as e:
        print(f"단계별 분석 오류: {e}")

if __name__ == "__main__":
    results = analyze_compression_failure()
    analyze_step_by_step()
    
    if results:
        print(f"\n최종 결과:")
        print(f"간단한 케이스 압축 성공: {results['simple_success']} (비율: {results['simple_ratio']:.3f})")
        print(f"복잡한 케이스 압축 성공: {results['complex_success']} (비율: {results['complex_ratio']:.3f})")