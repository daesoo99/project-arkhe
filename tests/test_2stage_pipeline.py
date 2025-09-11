#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project Arkhē - 2단계 경제적 지능 파이프라인 테스트
qwen2:0.5b (초안) → gemma:2b (검토/최종) 
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from orchestrator.pipeline import Pipeline, PipelineStep, AggregationStrategy
from llm.simple_llm import create_llm_auto
import time

def create_2stage_economic_pipeline():
    """2단계 경제적 지능 파이프라인 생성"""
    
    # 1단계: qwen2:0.5b로 초안 생성 (비용 계수: 0.8)
    draft_step = PipelineStep(
        model="qwen2:0.5b",
        prompt="다음 질문에 간단하고 명확하게 답변하세요: {query}",
        temperature=0.3,
        max_tokens=200,
        timeout=30
    )
    
    # 2단계: gemma:2b로 검토 및 최종 답변 (비용 계수: 1.0)  
    review_step = PipelineStep(
        model="gemma:2b",
        prompt="다음 초안 답변을 검토하고 개선된 최종 답변을 제공하세요.\n\n초안: {draft}\n\n원래 질문: {query}\n\n최종 답변:",
        temperature=0.2,
        max_tokens=400,
        timeout=60
    )
    
    return Pipeline(
        name="Economic-2Stage",
        steps=[draft_step, review_step],
        aggregation=AggregationStrategy.FIRST_VALID
    )

def test_economic_intelligence():
    """2-stage economic intelligence pipeline test"""
    
    print("=== Project Arkhe: 2-Stage Economic Intelligence Pipeline Test ===")
    print("Pipeline: qwen2:0.5b (draft) -> gemma:2b (review)")
    print("Cost model: 0.8*n1 + 1.0*n2")
    print("=" * 60)
    
    # 파이프라인 생성
    pipeline = create_2stage_economic_pipeline()
    
    # Test queries
    test_queries = [
        "What is the capital of South Korea?",
        "List 3 advantages of AI.",
        "What is the difference between Python lists and tuples?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[Test {i}/3] {query}")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            # Stage 1: Draft generation (qwen2:0.5b)
            print("Stage 1 (Draft): qwen2:0.5b running...")
            llm_draft = create_llm_auto("qwen2:0.5b")
            draft_start = time.time()
            draft_response = llm_draft.generate(pipeline.steps[0].prompt.format(query=query))
            draft_time = int((time.time() - draft_start) * 1000)
            print(f"  Draft result: {str(draft_response)[:100]}...")
            print(f"  Draft time: {draft_time}ms")
            
            # Stage 2: Review and finalize (gemma:2b)
            print("Stage 2 (Review): gemma:2b running...")
            llm_review = create_llm_auto("gemma:2b") 
            review_start = time.time()
            final_response = llm_review.generate(
                pipeline.steps[1].prompt.format(draft=str(draft_response), query=query)
            )
            review_time = int((time.time() - review_start) * 1000)
            print(f"  Final result: {str(final_response)[:100]}...")
            print(f"  Review time: {review_time}ms")
            
            # Cost calculation
            total_time = int((time.time() - start_time) * 1000)
            cost_score = 0.8 * 1 + 1.0 * 1  # Each model called once
            
            print(f"\n  Total time: {total_time}ms")
            print(f"  Economic cost: {cost_score:.1f} (0.8 + 1.0)")
            print(f"  Efficiency: {1000/total_time:.3f} queries/sec")
            
        except Exception as e:
            print(f"  Error: {str(e)}")
        
    print("\n=== Test Completed ===")
    print("Next: 3-stage pipeline test after llama3:8b installation")

if __name__ == "__main__":
    test_economic_intelligence()