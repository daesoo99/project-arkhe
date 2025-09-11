#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project Arkhe - 3-Stage Economic Intelligence Pipeline Test
qwen2:0.5b (draft) -> gemma:2b (review) -> llama3:8b (judge)
Complete Economic Intelligence Implementation: 0.8 + 1.0 + 4.0 = 5.8
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from orchestrator.pipeline import Pipeline, PipelineStep, AggregationStrategy
from llm.simple_llm import create_llm_auto
import time

def create_3stage_economic_pipeline():
    """3-stage economic intelligence pipeline: Draft -> Review -> Judge"""
    
    # Stage 1: qwen2:0.5b - Quick draft (cost: 0.8)
    draft_step = PipelineStep(
        model="qwen2:0.5b",
        prompt="Answer this question quickly and concisely: {query}",
        temperature=0.3,
        max_tokens=150,
        timeout=30
    )
    
    # Stage 2: gemma:2b - Review and improve (cost: 1.0)  
    review_step = PipelineStep(
        model="gemma:2b",
        prompt="Review and improve the following draft answer.\n\nOriginal question: {query}\nDraft answer: {draft}\n\nProvide an improved answer:",
        temperature=0.2,
        max_tokens=300,
        timeout=60
    )
    
    # Stage 3: llama3:8b - Final judgment (cost: 4.0)
    judge_step = PipelineStep(
        model="llama3:8b",
        prompt="Provide the final, highest quality answer by judging and refining the previous responses.\n\nQuestion: {query}\nDraft: {draft}\nReview: {review}\n\nFinal answer:",
        temperature=0.1,
        max_tokens=400,
        timeout=120
    )
    
    return Pipeline(
        name="Economic-3Stage",
        steps=[draft_step, review_step, judge_step],
        aggregation=AggregationStrategy.FIRST_VALID
    )

def test_3stage_economic_intelligence():
    """Test the complete 3-stage economic intelligence pipeline"""
    
    print("=== Project Arkhe: 3-Stage Economic Intelligence Pipeline ===")
    print("Pipeline: qwen2:0.5b -> gemma:2b -> llama3:8b")
    print("Cost model: 0.8 + 1.0 + 4.0 = 5.8")
    print("Economic Intelligence: Draft -> Review -> Judge")
    print("=" * 65)
    
    # Create pipeline
    pipeline = create_3stage_economic_pipeline()
    
    # Test queries
    test_queries = [
        "What are the main advantages of artificial intelligence?",
        "Explain the concept of machine learning in simple terms.",
        "What is the difference between AI and human intelligence?"
    ]
    
    total_results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[Test {i}/3] {query}")
        print("-" * 60)
        
        start_time = time.time()
        results = {"query": query, "stages": []}
        
        try:
            # Stage 1: Draft (qwen2:0.5b)
            print("Stage 1 (Draft): qwen2:0.5b executing...")
            llm_draft = create_llm_auto("qwen2:0.5b")
            draft_start = time.time()
            draft_response = llm_draft.generate(pipeline.steps[0].prompt.format(query=query))
            draft_time = int((time.time() - draft_start) * 1000)
            draft_text = str(draft_response.get('response', str(draft_response)))
            
            print(f"  Draft result: {draft_text[:80]}...")
            print(f"  Draft time: {draft_time}ms (cost: 0.8)")
            results["stages"].append({"stage": "draft", "time": draft_time, "cost": 0.8})
            
            # Stage 2: Review (gemma:2b)
            print("Stage 2 (Review): gemma:2b executing...")
            llm_review = create_llm_auto("gemma:2b") 
            review_start = time.time()
            review_response = llm_review.generate(
                pipeline.steps[1].prompt.format(query=query, draft=draft_text)
            )
            review_time = int((time.time() - review_start) * 1000)
            review_text = str(review_response.get('response', str(review_response)))
            
            print(f"  Review result: {review_text[:80]}...")
            print(f"  Review time: {review_time}ms (cost: 1.0)")
            results["stages"].append({"stage": "review", "time": review_time, "cost": 1.0})
            
            # Stage 3: Judge (llama3:8b)
            print("Stage 3 (Judge): llama3:8b executing...")
            llm_judge = create_llm_auto("llama3:8b")
            judge_start = time.time()
            final_response = llm_judge.generate(
                pipeline.steps[2].prompt.format(query=query, draft=draft_text, review=review_text)
            )
            judge_time = int((time.time() - judge_start) * 1000)
            final_text = str(final_response.get('response', str(final_response)))
            
            print(f"  Final result: {final_text[:80]}...")
            print(f"  Judge time: {judge_time}ms (cost: 4.0)")
            results["stages"].append({"stage": "judge", "time": judge_time, "cost": 4.0})
            
            # Calculate totals
            total_time = int((time.time() - start_time) * 1000)
            total_cost = 0.8 + 1.0 + 4.0  # Economic intelligence cost model
            efficiency = 1000 / total_time  # queries per second
            
            print(f"\n  === Stage Summary ===")
            print(f"  Total time: {total_time}ms")
            print(f"  Economic cost: {total_cost:.1f} (0.8 + 1.0 + 4.0)")
            print(f"  Efficiency: {efficiency:.3f} queries/sec")
            print(f"  Cost per second: {total_cost * efficiency:.3f}")
            
            results.update({
                "total_time": total_time,
                "total_cost": total_cost,
                "efficiency": efficiency,
                "final_answer": final_text
            })
            total_results.append(results)
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            results["error"] = str(e)
            total_results.append(results)
    
    # Final summary
    print("\n" + "=" * 65)
    print("=== 3-STAGE ECONOMIC INTELLIGENCE SUMMARY ===")
    
    if total_results:
        successful_tests = [r for r in total_results if "error" not in r]
        if successful_tests:
            avg_time = sum(r["total_time"] for r in successful_tests) / len(successful_tests)
            avg_cost = sum(r["total_cost"] for r in successful_tests) / len(successful_tests)
            avg_efficiency = sum(r["efficiency"] for r in successful_tests) / len(successful_tests)
            
            print(f"Successful tests: {len(successful_tests)}/{len(total_results)}")
            print(f"Average time: {avg_time:.0f}ms")
            print(f"Average cost: {avg_cost:.1f}")
            print(f"Average efficiency: {avg_efficiency:.3f} queries/sec")
            
            # Compare with previous results
            print(f"\n=== COMPARISON WITH PREVIOUS TESTS ===")
            print(f"Single gemma:2b:     cost ~1.0,  time ~3000ms")
            print(f"2-stage pipeline:    cost 1.8,   time ~2800ms") 
            print(f"3-stage pipeline:    cost 5.8,   time ~{avg_time:.0f}ms")
            print(f"Cost increase: {5.8/1.8:.1f}x vs 2-stage, {5.8/1.0:.1f}x vs single")
        else:
            print("All tests failed!")
    
    print("\n=== ECONOMIC INTELLIGENCE VALIDATION COMPLETE ===")
    print("Next: Compare quality improvements vs cost increases")
    return total_results

if __name__ == "__main__":
    results = test_3stage_economic_intelligence()