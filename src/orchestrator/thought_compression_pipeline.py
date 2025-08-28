# -*- coding: utf-8 -*-
"""
Project Arkhē - ThoughtAggregator 통합 파이프라인
사고과정 압축을 통한 토큰 효율성 개선 파이프라인

A방안: ThoughtAggregator 컴포넌트 통합
- Draft → ThoughtAggregator → Review → ThoughtAggregator → Judge
- 각 단계 사이에 압축 프로세스 추가
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .pipeline import ContextualPipelineStep, ContextualStepResult, ContextualPipeline
from .thought_aggregator import ThoughtAggregator

@dataclass
class ThoughtCompressionStep:
    """사고 압축 단계 정의"""
    aggregator_model: str = "qwen2:0.5b"  # 압축용 모델 (경제적)
    compress_threshold: int = 100  # 압축 임계값 (토큰 수)
    
class ThoughtCompressionPipeline(ContextualPipeline):
    """사고과정 압축 파이프라인"""
    
    def __init__(self, name: str, steps: List[ContextualPipelineStep], 
                 compression_steps: List[ThoughtCompressionStep] = None):
        super().__init__(name, steps)
        self.compression_steps = compression_steps or []
        self.thought_aggregators = {}  # 단계별 ThoughtAggregator 인스턴스
        
    def _execute_contextual_step(self, step_id: int, step: ContextualPipelineStep, 
                                llm_factory) -> ContextualStepResult:
        """압축 기능이 포함된 단계 실행"""
        
        # 기본 단계 실행
        result = super()._execute_contextual_step(step_id, step, llm_factory)
        
        # 압축 단계가 있고, 성공한 경우에만 압축 적용
        if (result.success and not result.skipped and 
            step_id < len(self.compression_steps) and 
            self.compression_steps[step_id]):
            
            compression_step = self.compression_steps[step_id]
            
            # 압축 임계값 체크
            if result.eval_count > compression_step.compress_threshold:
                # 이전 응답들과 함께 압축 수행
                compressed_context = self._apply_thought_compression(
                    step_id, result, compression_step, llm_factory
                )
                
                if compressed_context:
                    # 압축된 컨텍스트로 현재 컨텍스트 업데이트
                    self.context[f"{step.output_key}_compressed"] = compressed_context
                    print(f"  Step {step_id + 1} compressed: {result.eval_count} → {len(compressed_context.split())} tokens")
        
        return result
    
    def _apply_thought_compression(self, step_id: int, current_result: ContextualStepResult,
                                 compression_step: ThoughtCompressionStep, llm_factory) -> str:
        """사고과정 압축 적용"""
        try:
            # ThoughtAggregator 인스턴스 생성 (단계별로 재사용)
            if step_id not in self.thought_aggregators:
                self.thought_aggregators[step_id] = ThoughtAggregator(
                    model_name=compression_step.aggregator_model
                )
            
            aggregator = self.thought_aggregators[step_id]
            
            # 이전 단계들의 응답 수집
            previous_responses = []
            for i in range(step_id + 1):  # 현재 단계까지 포함
                for result in self.results:
                    if result.step_id == i and result.success and not result.skipped:
                        previous_responses.append(result.response)
            
            if not previous_responses:
                return ""
            
            # 현재 질문을 컨텍스트로 사용
            context = self.context.get("query", "")
            
            # 사고과정 분석 및 압축
            analysis = aggregator.analyze_thoughts(previous_responses, context)
            
            print(f"    Compression ratio: {analysis.compression_ratio:.2f}")
            print(f"    Common core length: {len(analysis.common_core)}")
            print(f"    Unique approaches: {len(analysis.unique_approaches)}")
            
            return analysis.compressed_context
            
        except Exception as e:
            print(f"  Compression error at step {step_id}: {e}")
            return ""

def create_thought_compression_pipeline() -> ThoughtCompressionPipeline:
    """사고과정 압축 파이프라인 생성 (A방안)"""
    
    steps = [
        # Draft 단계: 3개 샘플 생성
        ContextualPipelineStep(
            model="qwen2:0.5b",
            prompt_template="Answer this question with your best reasoning: {query}",
            output_key="draft1",
            temperature=0.8
        ),
        ContextualPipelineStep(
            model="qwen2:0.5b", 
            prompt_template="Provide a different approach to this question: {query}",
            output_key="draft2",
            temperature=0.8
        ),
        ContextualPipelineStep(
            model="qwen2:0.5b",
            prompt_template="Think creatively about this question: {query}",
            output_key="draft3", 
            temperature=0.8
        ),
        
        # Review 단계: 압축된 정보를 바탕으로 검토
        ContextualPipelineStep(
            model="qwen2:0.5b",
            prompt_template="""Review and improve the following analysis:

Question: {query}

Previous Analysis:
{draft1_compressed}
{draft2_compressed}
{draft3_compressed}

Provide an improved answer:""",
            output_key="review1",
            temperature=0.6,
            required_keys=["draft1_compressed", "draft2_compressed", "draft3_compressed"]
        ),
        ContextualPipelineStep(
            model="qwen2:0.5b",
            prompt_template="""Alternative review of the analysis:

Question: {query}

Previous Analysis:
{draft1_compressed}
{draft2_compressed}  
{draft3_compressed}

Provide your alternative perspective:""",
            output_key="review2",
            temperature=0.6,
            required_keys=["draft1_compressed", "draft2_compressed", "draft3_compressed"]
        ),
        
        # Judge 단계: 최종 판단
        ContextualPipelineStep(
            model="llama3:8b",
            prompt_template="""Make the final decision based on all analysis:

Question: {query}

Compressed Reviews:
{review1_compressed}
{review2_compressed}

Provide the final, authoritative answer:""",
            output_key="final",
            temperature=0.3,
            required_keys=["review1_compressed", "review2_compressed"]
        )
    ]
    
    # 압축 설정: Draft 단계들 이후와 Review 단계들 이후에 압축 적용
    compression_steps = [
        ThoughtCompressionStep(aggregator_model="qwen2:0.5b", compress_threshold=50),  # draft1 후
        ThoughtCompressionStep(aggregator_model="qwen2:0.5b", compress_threshold=50),  # draft2 후  
        ThoughtCompressionStep(aggregator_model="qwen2:0.5b", compress_threshold=50),  # draft3 후
        ThoughtCompressionStep(aggregator_model="qwen2:0.5b", compress_threshold=50),  # review1 후
        ThoughtCompressionStep(aggregator_model="qwen2:0.5b", compress_threshold=50),  # review2 후
        None  # final 단계는 압축하지 않음
    ]
    
    return ThoughtCompressionPipeline(
        name="ThoughtCompression-A방안",
        steps=steps,
        compression_steps=compression_steps
    )

def create_baseline_pipeline() -> ContextualPipeline:
    """비교용 기존 방식 파이프라인 (누적 프롬프트)"""
    
    steps = [
        # Draft 단계
        ContextualPipelineStep(
            model="qwen2:0.5b",
            prompt_template="Answer this question: {query}",
            output_key="draft1",
            temperature=0.8
        ),
        ContextualPipelineStep(
            model="qwen2:0.5b",
            prompt_template="Answer this question: {query}",
            output_key="draft2", 
            temperature=0.8
        ),
        ContextualPipelineStep(
            model="qwen2:0.5b",
            prompt_template="Answer this question: {query}",
            output_key="draft3",
            temperature=0.8
        ),
        
        # Review 단계: 모든 이전 결과를 누적
        ContextualPipelineStep(
            model="qwen2:0.5b",
            prompt_template="""Review these draft answers:

Question: {query}

Draft 1: {draft1}
Draft 2: {draft2}  
Draft 3: {draft3}

Provide an improved answer:""",
            output_key="review1",
            temperature=0.6,
            required_keys=["draft1", "draft2", "draft3"]
        ),
        ContextualPipelineStep(
            model="qwen2:0.5b",
            prompt_template="""Review these draft answers from different perspective:

Question: {query}

Draft 1: {draft1}
Draft 2: {draft2}
Draft 3: {draft3}

Provide alternative review:""",
            output_key="review2",
            temperature=0.6,
            required_keys=["draft1", "draft2", "draft3"]
        ),
        
        # Judge 단계: 모든 이전 결과를 누적
        ContextualPipelineStep(
            model="llama3:8b",
            prompt_template="""Make final decision based on all analysis:

Question: {query}

Draft 1: {draft1}
Draft 2: {draft2}
Draft 3: {draft3}

Review 1: {review1}
Review 2: {review2}

Final answer:""",
            output_key="final",
            temperature=0.3,
            required_keys=["draft1", "draft2", "draft3", "review1", "review2"]
        )
    ]
    
    return ContextualPipeline("Baseline-Cumulative", steps)

def run_comparison_experiment(query: str, llm_factory=None) -> Dict[str, Any]:
    """A방안 vs 기존 방식 비교 실험"""
    
    if llm_factory is None:
        from src.llm.simple_llm import create_llm_auto
        llm_factory = create_llm_auto
    
    print(f"Comparing thought compression approaches for: {query}\n")
    
    results = {}
    
    # A방안: ThoughtAggregator 파이프라인
    print("=" * 50)
    print("A방안: ThoughtAggregator 압축 파이프라인")
    print("=" * 50)
    
    compression_pipeline = create_thought_compression_pipeline()
    start_time = time.time()
    compression_result = compression_pipeline.run(query, llm_factory)
    compression_time = time.time() - start_time
    
    results["compression"] = {
        "final_answer": compression_result["final"],
        "total_tokens": compression_result["metrics"]["total_tokens"],
        "total_time": compression_time,
        "context_keys": list(compression_result["context"].keys()),
        "step_results": len(compression_result["step_results"])
    }
    
    print(f"Final: {compression_result['final'][:100]}...")
    print(f"Tokens: {compression_result['metrics']['total_tokens']}")
    print(f"Time: {compression_time:.2f}s\n")
    
    # 기존 방식: 누적 프롬프트
    print("=" * 50) 
    print("기존 방식: 누적 프롬프트 파이프라인")
    print("=" * 50)
    
    baseline_pipeline = create_baseline_pipeline()
    start_time = time.time()
    baseline_result = baseline_pipeline.run(query, llm_factory)
    baseline_time = time.time() - start_time
    
    results["baseline"] = {
        "final_answer": baseline_result["final"],
        "total_tokens": baseline_result["metrics"]["total_tokens"],
        "total_time": baseline_time,
        "context_keys": list(baseline_result["context"].keys()),
        "step_results": len(baseline_result["step_results"])
    }
    
    print(f"Final: {baseline_result['final'][:100]}...")
    print(f"Tokens: {baseline_result['metrics']['total_tokens']}")
    print(f"Time: {baseline_time:.2f}s\n")
    
    # 비교 결과
    print("=" * 50)
    print("비교 결과")
    print("=" * 50)
    
    token_reduction = 1 - (results["compression"]["total_tokens"] / results["baseline"]["total_tokens"])
    time_change = (results["compression"]["total_time"] / results["baseline"]["total_time"]) - 1
    
    print(f"토큰 절약: {token_reduction:.1%}")
    print(f"시간 변화: {time_change:+.1%}")
    
    results["comparison"] = {
        "token_reduction": token_reduction,
        "time_change": time_change,
        "compression_ratio": results["compression"]["total_tokens"] / results["baseline"]["total_tokens"]
    }
    
    return results

# 테스트 함수
def test_thought_compression():
    """사고과정 압축 파이프라인 테스트"""
    
    test_queries = [
        "What is the capital of South Korea?",
        "Explain the concept of machine learning in simple terms",
        "How do you solve the equation 2x + 5 = 15?",
        "What are the main causes of climate change?"
    ]
    
    for query in test_queries[:1]:  # 첫 번째만 테스트
        print(f"\n{'='*60}")
        print(f"Testing query: {query}")  
        print('='*60)
        
        try:
            result = run_comparison_experiment(query)
            print(f"Test completed for: {query}")
        except Exception as e:
            print(f"Test failed for: {query} - {e}")
            
if __name__ == "__main__":
    test_thought_compression()