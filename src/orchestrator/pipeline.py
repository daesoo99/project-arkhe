# -*- coding: utf-8 -*-
"""
Project Arkhē - 파이프라인 오케스트레이터
멀티에이전트 파이프라인 빌더 및 실행기

V2 Features:
- ContextualPipeline: 컨텍스트 전달형 파이프라인
- 단계별 출력을 다음 단계 입력으로 자동 연결
- 조건부 승격 (promote_if) 지원
"""

import time
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

class AggregationStrategy(Enum):
    """결과 집계 전략"""
    MAJORITY_VOTE = "majority"      # 다수결
    FIRST_VALID = "first_valid"     # 첫 번째 유효한 답변
    JUDGE_MODEL = "judge"           # 판정 모델 사용
    WEIGHTED_AVERAGE = "weighted"   # 가중 평균
    CONSENSUS = "consensus"         # 합의 도출

@dataclass
class PipelineStep:
    """파이프라인 단계 정의"""
    model: str                      # 모델 ID (예: "gemma:2b", "gpt-4o-mini")
    prompt: str                     # 프롬프트 템플릿
    temperature: float = 0.2        # 온도 설정
    max_tokens: int = 512           # 최대 토큰 수
    timeout: int = 120             # 타임아웃 (초)
    retry_count: int = 1           # 재시도 횟수

@dataclass 
class StepResult:
    """단계 실행 결과"""
    step_id: int
    model: str
    response: str
    latency_ms: int
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class Pipeline:
    """멀티에이전트 파이프라인"""
    
    def __init__(self, name: str, steps: List[PipelineStep], 
                 aggregation: AggregationStrategy = AggregationStrategy.MAJORITY_VOTE):
        self.name = name
        self.steps = steps
        self.aggregation = aggregation
        self.results: List[StepResult] = []
        
    def execute(self, base_prompt: str, llm_factory=None) -> Dict[str, Any]:
        """
        파이프라인 실행
        
        Args:
            base_prompt: 기본 프롬프트
            llm_factory: LLM 인스턴스 팩토리 함수
            
        Returns:
            실행 결과 딕셔너리
        """
        if not llm_factory:
            # 기본 팩토리 (Ollama 사용)
            llm_factory = self._default_llm_factory
            
        start_time = time.time()
        self.results = []
        
        # 각 단계 순차 실행
        for i, step in enumerate(self.steps):
            step_result = self._execute_step(i, step, base_prompt, llm_factory)
            self.results.append(step_result)
            
        # 결과 집계
        final_result = self._aggregate_results()
        
        total_time = time.time() - start_time
        
        # StepResult 객체를 딕셔너리로 변환
        step_results_dict = []
        for r in self.results:
            step_dict = {
                "step_id": r.step_id,
                "model": r.model,
                "response": r.response,
                "latency_ms": r.latency_ms,
                "success": r.success,
                "error": r.error,
                "metadata": r.metadata
            }
            step_results_dict.append(step_dict)
        
        return {
            "pipeline_name": self.name,
            "final_answer": final_result["answer"],
            "confidence": final_result["confidence"],
            "step_results": step_results_dict,
            "total_latency_ms": int(total_time * 1000),
            "successful_steps": sum(1 for r in self.results if r.success),
            "total_steps": len(self.steps),
            "aggregation_method": self.aggregation.value,
            "aggregation_details": final_result.get("details", "")
        }
    
    def _execute_step(self, step_id: int, step: PipelineStep, 
                      base_prompt: str, llm_factory) -> StepResult:
        """단일 단계 실행"""
        try:
            # 프롬프트 템플릿 적용
            formatted_prompt = step.prompt.format(base_prompt=base_prompt)
            
            # LLM 인스턴스 생성 및 실행
            llm = llm_factory(step.model)
            start_time = time.time()
            
            response = llm.generate(
                formatted_prompt,
                temperature=step.temperature,
                max_tokens=step.max_tokens,
                timeout=step.timeout
            )
            
            latency = int((time.time() - start_time) * 1000)
            
            return StepResult(
                step_id=step_id,
                model=step.model,
                response=response.get("response", ""),
                latency_ms=latency,
                success=True,
                metadata=response
            )
            
        except Exception as e:
            return StepResult(
                step_id=step_id,
                model=step.model,
                response="",
                latency_ms=0,
                success=False,
                error=str(e)
            )
    
    def _aggregate_results(self) -> Dict[str, Any]:
        """결과 집계"""
        successful_results = [r for r in self.results if r.success]
        
        if not successful_results:
            return {
                "answer": "[ERROR] 모든 단계 실패",
                "confidence": 0.0,
                "details": "No successful steps"
            }
        
        if self.aggregation == AggregationStrategy.MAJORITY_VOTE:
            return self._majority_vote(successful_results)
        elif self.aggregation == AggregationStrategy.FIRST_VALID:
            return self._first_valid(successful_results)
        elif self.aggregation == AggregationStrategy.CONSENSUS:
            return self._consensus(successful_results)
        else:
            # 기본값: 첫 번째 성공 결과
            return self._first_valid(successful_results)
    
    def _majority_vote(self, results: List[StepResult]) -> Dict[str, Any]:
        """다수결 집계"""
        responses = [r.response.strip().lower() for r in results if r.response.strip()]
        
        if not responses:
            return {"answer": "", "confidence": 0.0, "details": "No valid responses"}
        
        # 응답 빈도 계산
        response_counts = {}
        for response in responses:
            # 간단한 정규화 (첫 50글자로 비교)
            key = response[:50]
            response_counts[key] = response_counts.get(key, 0) + 1
        
        # 최다 득표 응답
        most_common = max(response_counts.items(), key=lambda x: x[1])
        confidence = most_common[1] / len(responses)
        
        # 원본 응답 찾기
        for r in results:
            if r.response.strip().lower()[:50] == most_common[0]:
                return {
                    "answer": r.response,
                    "confidence": confidence,
                    "details": f"Votes: {most_common[1]}/{len(responses)}"
                }
        
        return {"answer": most_common[0], "confidence": confidence, "details": "Majority vote"}
    
    def _first_valid(self, results: List[StepResult]) -> Dict[str, Any]:
        """첫 번째 유효 결과"""
        if results:
            return {
                "answer": results[0].response,
                "confidence": 1.0,
                "details": f"First valid from {results[0].model}"
            }
        return {"answer": "", "confidence": 0.0, "details": "No results"}
    
    def _consensus(self, results: List[StepResult]) -> Dict[str, Any]:
        """합의 기반 집계 (단순 버전)"""
        if len(results) == 1:
            return self._first_valid(results)
        
        # 모든 응답 결합
        combined = " | ".join([r.response for r in results if r.response])
        confidence = len(results) / len(self.steps)  # 성공률 기반
        
        return {
            "answer": combined,
            "confidence": confidence,
            "details": f"Consensus from {len(results)} agents"
        }
    
    def _default_llm_factory(self, model_id: str):
        """기본 LLM 팩토리 (자동 감지)"""
        from src.llm.simple_llm import create_llm_auto
        return create_llm_auto(model_id)

class PipelineBuilder:
    """파이프라인 빌더"""
    
    @staticmethod
    def create_single_agent(model: str, name: str = None) -> Pipeline:
        """단일 에이전트 파이프라인"""
        if not name:
            name = f"Single-{model}"
        
        steps = [PipelineStep(
            model=model,
            prompt="{base_prompt}"
        )]
        
        return Pipeline(name, steps, AggregationStrategy.FIRST_VALID)
    
    @staticmethod  
    def create_multi_independent(models: List[str], name: str = None) -> Pipeline:
        """독립 멀티에이전트 파이프라인"""
        if not name:
            name = f"Multi-{len(models)}"
        
        steps = [
            PipelineStep(model=model, prompt="{base_prompt}")
            for model in models
        ]
        
        return Pipeline(name, steps, AggregationStrategy.MAJORITY_VOTE)
    
    @staticmethod
    def create_sequential_refinement(models: List[str], name: str = None) -> Pipeline:
        """순차적 개선 파이프라인"""
        if not name:
            name = f"Sequential-{len(models)}"
        
        steps = []
        
        # 첫 번째 단계: 초안 작성
        if models:
            steps.append(PipelineStep(
                model=models[0],
                prompt="{base_prompt}"
            ))
        
        # 나머지 단계: 개선 및 재평가
        for model in models[1:]:
            steps.append(PipelineStep(
                model=model,
                prompt="다음 답변을 검토하고 개선하세요:\n\n질문: {base_prompt}\n\n이전 답변: [이전 단계 결과]\n\n개선된 답변:"
            ))
        
        return Pipeline(name, steps, AggregationStrategy.FIRST_VALID)

# 미리 정의된 파이프라인들
PRESET_PIPELINES = {
    "single_8b": PipelineBuilder.create_single_agent("llama3:8b"),
    "triple_2b": PipelineBuilder.create_multi_independent(["gemma:2b"] * 3),
    "draft_review_final": PipelineBuilder.create_sequential_refinement(["gemma:2b", "mistral:7b", "llama3:8b"])
}

# =============================================================================
# V2: Contextual Pipeline - 컨텍스트 전달형 파이프라인
# =============================================================================

@dataclass
class ContextualPipelineStep:
    """컨텍스트 전달형 파이프라인 단계"""
    model: str                              # 모델 ID
    prompt_template: str                    # 프롬프트 템플릿 ("{query}", "{draft}", etc.)
    output_key: str                         # 이 단계 결과를 저장할 컨텍스트 키
    temperature: float = 0.2                # 온도 설정
    max_tokens: int = 512                   # 최대 토큰 수
    timeout: int = 120                      # 타임아웃 (초)
    promote_if: Optional[Callable[[Dict], bool]] = None  # 승급 조건 함수
    required_keys: List[str] = field(default_factory=list)  # 필요한 컨텍스트 키들

@dataclass
class ContextualStepResult:
    """컨텍스트 단계 실행 결과"""
    step_id: int
    model: str
    output_key: str
    response: str
    latency_ms: int
    eval_count: int = 0
    eval_duration_ns: int = 0
    success: bool = True
    skipped: bool = False
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ContextualPipeline:
    """컨텍스트 전달형 파이프라인"""
    
    def __init__(self, name: str, steps: List[ContextualPipelineStep]):
        self.name = name
        self.steps = steps
        self.context = {}
        self.results = []
        
    def run(self, query: str, llm_factory=None) -> Dict[str, Any]:
        """
        컨텍스트 전달형 파이프라인 실행
        
        Args:
            query: 초기 쿼리
            llm_factory: LLM 팩토리 함수 (기본값: create_llm_auto)
            
        Returns:
            {
                "final": 최종 답변,
                "context": 전체 컨텍스트 딕셔너리, 
                "metrics": 실행 메트릭,
                "step_results": 단계별 결과
            }
        """
        if llm_factory is None:
            from src.llm.simple_llm import create_llm_auto
            llm_factory = create_llm_auto
            
        # 초기 컨텍스트 설정
        self.context = {"query": query}
        self.results = []
        
        start_time = time.time()
        executed_steps = 0
        skipped_steps = 0
        
        # 각 단계 순차 실행
        for step_id, step in enumerate(self.steps):
            step_result = self._execute_contextual_step(step_id, step, llm_factory)
            self.results.append(step_result)
            
            if step_result.skipped:
                skipped_steps += 1
                print(f"  Step {step_id + 1} ({step.model}) skipped by promote_if")
                continue
                
            if step_result.success:
                executed_steps += 1
                # 성공한 경우 컨텍스트에 결과 저장
                self.context[step.output_key] = step_result.response
                print(f"  Step {step_id + 1} ({step.model}) -> context['{step.output_key}']")
            else:
                print(f"  Step {step_id + 1} ({step.model}) failed: {step_result.error}")
        
        total_time = time.time() - start_time
        
        # 최종 결과 결정
        final_answer = self._determine_final_answer()
        
        # 메트릭 계산
        metrics = {
            "total_time_ms": int(total_time * 1000),
            "executed_steps": executed_steps,
            "skipped_steps": skipped_steps,
            "total_steps": len(self.steps),
            "success_rate": executed_steps / max(len(self.steps) - skipped_steps, 1),
            "total_cost": sum(r.eval_count * 0.001 for r in self.results if r.success),  # 추정 비용
            "total_tokens": sum(r.eval_count for r in self.results if r.success),
            "avg_latency": sum(r.latency_ms for r in self.results if r.success) / max(executed_steps, 1)
        }
        
        return {
            "final": final_answer,
            "context": dict(self.context),  # 복사본 반환
            "metrics": metrics,
            "step_results": [self._step_result_to_dict(r) for r in self.results]
        }
    
    def _execute_contextual_step(self, step_id: int, step: ContextualPipelineStep, 
                                llm_factory) -> ContextualStepResult:
        """단일 컨텍스트 단계 실행"""
        
        # 승급 조건 체크
        if step.promote_if and not step.promote_if(self.context):
            return ContextualStepResult(
                step_id=step_id,
                model=step.model,
                output_key=step.output_key,
                response="",
                latency_ms=0,
                success=True,
                skipped=True
            )
        
        # 필요한 키 체크
        missing_keys = [k for k in step.required_keys if k not in self.context]
        if missing_keys:
            return ContextualStepResult(
                step_id=step_id,
                model=step.model,
                output_key=step.output_key,
                response="",
                latency_ms=0,
                success=False,
                error=f"Missing required keys: {missing_keys}"
            )
        
        try:
            # 프롬프트 템플릿 적용
            formatted_prompt = step.prompt_template.format(**self.context)
            
            # LLM 실행
            llm = llm_factory(step.model)
            start_time = time.time()
            
            response = llm.generate(
                formatted_prompt,
                temperature=step.temperature,
                max_tokens=step.max_tokens
            )
            
            latency = int((time.time() - start_time) * 1000)
            
            # 응답 처리
            if isinstance(response, dict):
                response_text = response.get("response", str(response))
                eval_count = response.get("eval_count", len(response_text.split()))
                eval_duration = response.get("eval_duration", 0)
            else:
                response_text = str(response)
                eval_count = len(response_text.split())
                eval_duration = 0
            
            return ContextualStepResult(
                step_id=step_id,
                model=step.model,
                output_key=step.output_key,
                response=response_text,
                latency_ms=latency,
                eval_count=eval_count,
                eval_duration_ns=eval_duration,
                success=True,
                metadata=response if isinstance(response, dict) else None
            )
            
        except Exception as e:
            return ContextualStepResult(
                step_id=step_id,
                model=step.model,
                output_key=step.output_key,
                response="",
                latency_ms=0,
                success=False,
                error=str(e)
            )
    
    def _determine_final_answer(self) -> str:
        """최종 답변 결정"""
        # 가장 마지막 성공한 단계의 결과를 사용
        for result in reversed(self.results):
            if result.success and not result.skipped and result.response.strip():
                return result.response
        
        # 컨텍스트에서 가능한 답변 찾기
        for key in ["final", "judge", "review", "draft", "answer"]:
            if key in self.context and self.context[key]:
                return self.context[key]
        
        return "No successful result found"
    
    def _step_result_to_dict(self, result: ContextualStepResult) -> Dict[str, Any]:
        """StepResult를 딕셔너리로 변환"""
        return {
            "step_id": result.step_id,
            "model": result.model,
            "output_key": result.output_key,
            "response": result.response[:100] + "..." if len(result.response) > 100 else result.response,
            "latency_ms": result.latency_ms,
            "eval_count": result.eval_count,
            "success": result.success,
            "skipped": result.skipped,
            "error": result.error
        }

# =============================================================================
# 헬퍼 함수들
# =============================================================================

def create_3stage_economic_pipeline() -> ContextualPipeline:
    """3단계 경제적 지능 파이프라인 생성"""
    
    def should_skip_review(context: Dict) -> bool:
        """리뷰 단계를 건너뛸지 결정"""
        draft = context.get("draft", "")
        query = context.get("query", "")
        
        # 간단한 질문이거나 짧은 답변이면 건너뛰기
        simple_indicators = ["what is", "who is", "when is", "where is", "2+2", "수도"]
        is_simple = any(indicator in query.lower() for indicator in simple_indicators)
        is_short = len(draft.split()) < 10
        
        return is_simple and is_short
    
    def should_skip_judge(context: Dict) -> bool:
        """판정 단계를 건너뛸지 결정"""
        review = context.get("review", "")
        draft = context.get("draft", "")
        
        # 리뷰가 없거나 초안과 매우 유사하면 건너뛰기
        if not review:
            return True
            
        # 간단한 유사도 체크
        review_words = set(review.lower().split())
        draft_words = set(draft.lower().split())
        
        if len(review_words) > 0:
            similarity = len(review_words & draft_words) / len(review_words | draft_words)
            return similarity > 0.8  # 80% 이상 유사하면 건너뛰기
            
        return False
    
    steps = [
        ContextualPipelineStep(
            model="qwen2:0.5b",
            prompt_template="Answer this question concisely: {query}",
            output_key="draft",
            temperature=0.7
        ),
        ContextualPipelineStep(
            model="gemma:2b", 
            prompt_template="Improve this draft answer:\n\nQuestion: {query}\nDraft: {draft}\n\nImproved answer:",
            output_key="review",
            temperature=0.5,
            promote_if=lambda ctx: not should_skip_review(ctx),
            required_keys=["draft"]
        ),
        ContextualPipelineStep(
            model="llama3:8b",
            prompt_template="Provide the final, highest quality answer:\n\nQuestion: {query}\nDraft: {draft}\nReview: {review}\n\nFinal answer:",
            output_key="final", 
            temperature=0.3,
            promote_if=lambda ctx: not should_skip_judge(ctx),
            required_keys=["draft", "review"]
        )
    ]
    
    return ContextualPipeline("3Stage-Economic-Intelligence", steps)

def run_3stage_with_context(llm_factory, query: str) -> Dict[str, Any]:
    """3단계 컨텍스트 파이프라인 실행 헬퍼"""
    pipeline = create_3stage_economic_pipeline()
    return pipeline.run(query, llm_factory)

def create_2stage_pipeline() -> ContextualPipeline:
    """2단계 파이프라인 생성"""
    steps = [
        ContextualPipelineStep(
            model="qwen2:0.5b",
            prompt_template="Answer briefly: {query}",
            output_key="draft",
            temperature=0.6
        ),
        ContextualPipelineStep(
            model="gemma:2b",
            prompt_template="Improve and finalize:\n\nQuestion: {query}\nDraft: {draft}\n\nFinal answer:",
            output_key="final",
            temperature=0.4,
            required_keys=["draft"]
        )
    ]
    
    return ContextualPipeline("2Stage-Draft-Review", steps)

# 사용 예시 및 테스트
def test_contextual_pipeline():
    """컨텍스트 파이프라인 테스트"""
    from src.llm.simple_llm import create_llm_auto
    
    print("🧪 Testing Contextual Pipeline...")
    
    # 3단계 테스트
    result = run_3stage_with_context(create_llm_auto, "What is the capital of South Korea?")
    
    print(f"Final answer: {result['final']}")
    print(f"Context keys: {list(result['context'].keys())}")
    print(f"Metrics: {result['metrics']}")
    
    return result

if __name__ == "__main__":
    test_contextual_pipeline()