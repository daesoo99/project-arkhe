# -*- coding: utf-8 -*-
"""
Project Arkhē - Experimental Pipeline with Information Theory
Shannon Entropy 기반 승급 정책 실험용 파이프라인

핵심 연구: draft_sampler(k) 훅 + 정보 이론 메트릭 기반 승급 결정
"""

import time
import json
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

from metrics.information_theory import InformationTheoryCalculator, PromotionPolicyEngine
from orchestrator.pipeline import ContextualPipelineStep, ContextualStepResult

@dataclass
class ExperimentalPipelineStep:
    """실험용 파이프라인 단계 - 다중 샘플링 지원"""
    model: str                              # 모델 ID
    prompt_template: str                    # 프롬프트 템플릿
    output_key: str                         # 결과 저장 키
    temperature: float = 0.7                # 온도 (다양성을 위해 기본값 높임)
    max_tokens: int = 512                   # 최대 토큰 수
    timeout: int = 120                      # 타임아웃 (초)
    num_samples: int = 3                    # 샘플 수 (k)
    promote_if: Optional[Callable[[Dict], bool]] = None  # 승급 조건 함수
    required_keys: List[str] = field(default_factory=list)  # 필요한 컨텍스트 키들
    
@dataclass
class ExperimentalStepResult:
    """실험용 단계 실행 결과 - 다중 샘플 + 메트릭"""
    step_id: int
    model: str
    output_key: str
    samples: List[str]                      # 모든 샘플들
    best_response: str                      # 선택된 최고 응답
    latency_ms: int
    eval_count: int = 0
    success: bool = True
    skipped: bool = False
    error: Optional[str] = None
    
    # 정보 이론 메트릭
    shannon_entropy: float = 0.0
    js_divergence: float = 0.0
    uncertainty_score: float = 0.0
    promotion_decision: Optional[Dict[str, Any]] = None

class ExperimentalPipeline:
    """Shannon Entropy 기반 승급 정책 실험용 파이프라인"""
    
    def __init__(self, name: str, steps: List[ExperimentalPipelineStep], 
                 tau1: float = 0.8, tau2: float = 1.0, 
                 log_file: str = None):
        self.name = name
        self.steps = steps
        self.context = {}
        self.results = []
        
        # 정보 이론 엔진
        self.info_calculator = InformationTheoryCalculator()
        self.promotion_engine = PromotionPolicyEngine(tau1=tau1, tau2=tau2)
        
        # 실험 로깅
        self.log_file = log_file or f"logs/experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
        
    def run(self, query: str, llm_factory=None, experiment_id: str = None) -> Dict[str, Any]:
        """
        실험용 파이프라인 실행
        
        Args:
            query: 초기 쿼리
            llm_factory: LLM 팩토리 함수
            experiment_id: 실험 식별자
            
        Returns:
            실행 결과 + 정보 이론 메트릭
        """
        if llm_factory is None:
            from llm.simple_llm import create_llm_auto
            llm_factory = create_llm_auto
            
        # 실험 시작
        experiment_start = time.time()
        self.context = {"query": query}
        self.results = []
        
        print(f"\n*** Experimental Pipeline: {self.name} ***")
        print(f"Query: {query}")
        print(f"Tau1: {self.promotion_engine.tau1}, Tau2: {self.promotion_engine.tau2}")
        
        executed_steps = 0
        skipped_steps = 0
        
        # 각 단계 실행
        for step_id, step in enumerate(self.steps):
            step_result = self._execute_experimental_step(step_id, step, llm_factory)
            self.results.append(step_result)
            
            # 로깅
            self._log_step_result(query, step_id, step_result, experiment_id)
            
            if step_result.skipped:
                skipped_steps += 1
                print(f"  Step {step_id + 1} ({step.model}) skipped by promotion policy")
                continue
                
            if step_result.success:
                executed_steps += 1
                # 컨텍스트에 결과 저장
                self.context[step.output_key] = step_result.best_response
                self.context[f"{step.output_key}_samples"] = step_result.samples
                self.context[f"{step.output_key}_entropy"] = step_result.shannon_entropy
                
                print(f"  Step {step_id + 1} ({step.model}) -> {step.output_key}")
                print(f"    Samples: {len(step_result.samples)}, Entropy: {step_result.shannon_entropy:.3f}")
                
                if step_result.promotion_decision:
                    decision = step_result.promotion_decision["decision"]
                    print(f"    Promotion: {decision}")
            else:
                print(f"  Step {step_id + 1} ({step.model}) failed: {step_result.error}")
        
        total_time = time.time() - experiment_start
        
        # 최종 결과 결정
        final_answer = self._determine_final_answer()
        
        # 메트릭 계산
        metrics = {
            "total_time_ms": int(total_time * 1000),
            "executed_steps": executed_steps,
            "skipped_steps": skipped_steps,
            "total_steps": len(self.steps),
            "promotion_rate": executed_steps / len(self.steps),
            "tau1": self.promotion_engine.tau1,
            "tau2": self.promotion_engine.tau2
        }
        
        result = {
            "final": final_answer,
            "context": dict(self.context),
            "metrics": metrics,
            "step_results": [self._step_result_to_dict(r) for r in self.results]
        }
        
        # 최종 결과 로깅
        self._log_final_result(query, result, experiment_id)
        
        return result
    
    def _execute_experimental_step(self, step_id: int, step: ExperimentalPipelineStep, 
                                 llm_factory) -> ExperimentalStepResult:
        """실험용 단계 실행 - 다중 샘플링 + 승급 정책"""
        
        # 승급 조건 체크 (기존 promote_if 또는 정보 이론 기반)
        if step.promote_if and not step.promote_if(self.context):
            return ExperimentalStepResult(
                step_id=step_id,
                model=step.model,
                output_key=step.output_key,
                samples=[],
                best_response="",
                latency_ms=0,
                success=True,
                skipped=True
            )
        
        # 필요한 키 체크
        missing_keys = [k for k in step.required_keys if k not in self.context]
        if missing_keys:
            return ExperimentalStepResult(
                step_id=step_id,
                model=step.model,
                output_key=step.output_key,
                samples=[],
                best_response="",
                latency_ms=0,
                success=False,
                error=f"Missing required keys: {missing_keys}"
            )
        
        try:
            # 프롬프트 생성
            formatted_prompt = step.prompt_template.format(**self.context)
            
            # 다중 샘플링 실행
            llm = llm_factory(step.model)
            start_time = time.time()
            
            samples = []
            total_tokens = 0
            
            for i in range(step.num_samples):
                response = llm.generate(
                    formatted_prompt,
                    temperature=step.temperature,
                    max_tokens=step.max_tokens
                )
                
                if isinstance(response, dict):
                    text = response.get("response", str(response))
                    tokens = response.get("eval_count", len(text.split()))
                else:
                    text = str(response)
                    tokens = len(text.split())
                
                samples.append(text)
                total_tokens += tokens
            
            latency = int((time.time() - start_time) * 1000)
            
            # 정보 이론 메트릭 계산
            metrics = self.info_calculator.calculate_all_metrics(samples)
            
            # 최고 응답 선택 (가장 긴 응답 또는 첫 번째)
            best_response = max(samples, key=len) if samples else ""
            
            # 승급 결정 (정보 이론 기반)
            promotion_decision = None
            if step.output_key == "draft":
                # Draft -> Review 승급 결정
                should_promote, decision_info = self.promotion_engine.should_promote_to_review(
                    samples, self.context
                )
                promotion_decision = decision_info
                
                # 다음 단계가 있고 승급하지 않으면 전체 파이프라인 조기 종료
                if not should_promote and step_id < len(self.steps) - 1:
                    # 다음 단계들을 모두 스킵하도록 컨텍스트에 표시
                    self.context["_early_termination"] = True
                    
            elif step.output_key == "review":
                # Review -> Judge 승급 결정
                draft_samples = self.context.get("draft_samples", [])
                should_promote, decision_info = self.promotion_engine.should_promote_to_judge(
                    samples, draft_samples, self.context
                )
                promotion_decision = decision_info
                
                if not should_promote and step_id < len(self.steps) - 1:
                    self.context["_early_termination"] = True
            
            return ExperimentalStepResult(
                step_id=step_id,
                model=step.model,
                output_key=step.output_key,
                samples=samples,
                best_response=best_response,
                latency_ms=latency,
                eval_count=total_tokens,
                success=True,
                shannon_entropy=metrics.shannon_entropy,
                js_divergence=metrics.js_divergence,
                uncertainty_score=metrics.uncertainty_score,
                promotion_decision=promotion_decision
            )
            
        except Exception as e:
            return ExperimentalStepResult(
                step_id=step_id,
                model=step.model,
                output_key=step.output_key,
                samples=[],
                best_response="",
                latency_ms=0,
                success=False,
                error=str(e)
            )
    
    def _determine_final_answer(self) -> str:
        """최종 답변 결정"""
        # 가장 마지막 성공한 단계의 결과를 사용
        for result in reversed(self.results):
            if result.success and not result.skipped and result.best_response.strip():
                return result.best_response
        
        # 컨텍스트에서 가능한 답변 찾기
        for key in ["final", "judge", "review", "draft"]:
            if key in self.context and self.context[key]:
                return self.context[key]
        
        return "No successful result found"
    
    def _step_result_to_dict(self, result: ExperimentalStepResult) -> Dict[str, Any]:
        """결과를 딕셔너리로 변환"""
        return {
            "step_id": result.step_id,
            "model": result.model,
            "output_key": result.output_key,
            "num_samples": len(result.samples),
            "best_response": result.best_response[:100] + "..." if len(result.best_response) > 100 else result.best_response,
            "latency_ms": result.latency_ms,
            "eval_count": result.eval_count,
            "success": result.success,
            "skipped": result.skipped,
            "error": result.error,
            "shannon_entropy": result.shannon_entropy,
            "js_divergence": result.js_divergence,
            "uncertainty_score": result.uncertainty_score,
            "promotion_decision": result.promotion_decision
        }
    
    def _log_step_result(self, query: str, step_id: int, result: ExperimentalStepResult, 
                        experiment_id: str = None):
        """단계별 결과 로깅"""
        log_entry = {
            "type": "step_result",
            "timestamp": datetime.now().isoformat(),
            "experiment_id": experiment_id,
            "query": query,
            "step_id": step_id,
            "model": result.model,
            "output_key": result.output_key,
            "num_samples": len(result.samples),
            "latency_ms": result.latency_ms,
            "eval_count": result.eval_count,
            "success": result.success,
            "skipped": result.skipped,
            "shannon_entropy": result.shannon_entropy,
            "js_divergence": result.js_divergence,
            "uncertainty_score": result.uncertainty_score,
            "promotion_decision": result.promotion_decision,
            "samples_preview": [s[:50] + "..." if len(s) > 50 else s for s in result.samples[:3]]
        }
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def _log_final_result(self, query: str, result: Dict[str, Any], experiment_id: str = None):
        """최종 결과 로깅"""
        log_entry = {
            "type": "final_result",
            "timestamp": datetime.now().isoformat(),
            "experiment_id": experiment_id,
            "query": query,
            "final_answer": result["final"],
            "metrics": result["metrics"],
            "pipeline_name": self.name
        }
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

def create_entropy_experiment_pipeline(tau1: float = 0.8, tau2: float = 1.0, 
                                     k_samples: int = 3) -> ExperimentalPipeline:
    """Shannon Entropy 실험용 3단계 파이프라인 생성"""
    
    def early_termination_check(context: Dict) -> bool:
        """조기 종료 체크"""
        return not context.get("_early_termination", False)
    
    steps = [
        ExperimentalPipelineStep(
            model="qwen2:0.5b",
            prompt_template="Answer this question concisely: {query}",
            output_key="draft",
            temperature=0.8,
            num_samples=k_samples
        ),
        ExperimentalPipelineStep(
            model="gemma:2b", 
            prompt_template="Improve this draft answer:\n\nQuestion: {query}\nDraft: {draft}\n\nImproved answer:",
            output_key="review",
            temperature=0.6,
            num_samples=max(2, k_samples - 1),
            promote_if=early_termination_check,
            required_keys=["draft"]
        ),
        ExperimentalPipelineStep(
            model="llama3:8b",
            prompt_template="Provide the final, highest quality answer:\n\nQuestion: {query}\nDraft: {draft}\nReview: {review}\n\nFinal answer:",
            output_key="final", 
            temperature=0.4,
            num_samples=max(1, k_samples - 2),
            promote_if=early_termination_check,
            required_keys=["draft", "review"]
        )
    ]
    
    return ExperimentalPipeline(
        f"EntropyExperiment_tau1_{tau1}_tau2_{tau2}_k_{k_samples}", 
        steps, tau1=tau1, tau2=tau2
    )