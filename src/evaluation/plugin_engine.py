# -*- coding: utf-8 -*-
"""
Project Arkhē - Plugin-based Evaluation Engine
플러그인 시스템을 사용한 통합 평가 엔진

CLAUDE.local 규칙 준수:
- 의존성 주입 완전 적용
- 플러그인 교체 가능 아키텍처
- 하드코딩 Zero Tolerance
"""

import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

from ..plugins.registry import get_plugin_registry, initialize_plugins
from ..plugins.interfaces import (
    TaskType, ScoringResult, AggregationResult, 
    IScorer, IAggregator, ScoringError, AggregationError
)

logger = logging.getLogger(__name__)

@dataclass
class EvaluationRequest:
    """평가 요청 데이터"""
    ground_truth: str
    response: str
    task_type: TaskType
    scorer_names: Optional[List[str]] = None
    aggregator_name: Optional[str] = None
    scoring_kwargs: Dict[str, Any] = None
    aggregation_kwargs: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.scoring_kwargs is None:
            self.scoring_kwargs = {}
        if self.aggregation_kwargs is None:
            self.aggregation_kwargs = {}

@dataclass
class EvaluationResult:
    """평가 결과 데이터"""
    request: EvaluationRequest
    individual_scores: List[ScoringResult]
    aggregated_result: AggregationResult
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.request.task_type.value,
            "ground_truth": self.request.ground_truth,
            "response": self.request.response,
            "individual_scores": [score.to_dict() for score in self.individual_scores],
            "aggregated_result": {
                "score": self.aggregated_result.aggregated_score,
                "method": self.aggregated_result.method,
                "confidence": self.aggregated_result.confidence,
                "metadata": self.aggregated_result.metadata
            },
            "metadata": self.metadata
        }

class PluginEvaluationEngine:
    """
    플러그인 기반 평가 엔진
    
    기능:
    - 플러그인 기반 채점기/집계기 사용
    - 동적 플러그인 교체
    - 다중 채점기 자동 선택
    - 설정 기반 평가 파이프라인
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 플러그인 및 평가 설정
        """
        self.config = config or {}
        self.plugin_registry = get_plugin_registry()
        self._initialize_plugins()
        
        # 기본 설정
        self.default_aggregator = self.config.get("default_aggregator", "weighted_average")
        self.auto_select_scorers = self.config.get("auto_select_scorers", True)
        self.fallback_scorer = self.config.get("fallback_scorer", "legacy_scorer")
    
    def _initialize_plugins(self):
        """플러그인 시스템 초기화"""
        if not self.plugin_registry._initialized:
            plugin_config = self.config.get("plugins", {})
            plugin_paths = self.config.get("plugin_paths", None)
            initialize_plugins(plugin_paths, plugin_config)
            
        logger.info(f"Plugin engine initialized with {len(self.plugin_registry.list_scorers())} scorers, "
                   f"{len(self.plugin_registry.list_aggregators())} aggregators")
    
    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        """
        플러그인 기반 평가 수행
        
        Args:
            request: 평가 요청
            
        Returns:
            평가 결과
        """
        try:
            # 채점기 선택
            scorers = self._select_scorers(request)
            if not scorers:
                raise ScoringError(f"No scorers available for task type: {request.task_type}")
            
            # 개별 채점 수행
            individual_scores = []
            for scorer in scorers:
                try:
                    score_result = scorer.score(
                        request.ground_truth,
                        request.response,
                        request.task_type,
                        **request.scoring_kwargs
                    )
                    individual_scores.append(score_result)
                    logger.debug(f"Scorer {scorer.name}: {score_result.score:.3f}")
                    
                except Exception as e:
                    logger.error(f"Scorer {scorer.name} failed: {e}")
                    # 오류 발생시 0점 처리
                    individual_scores.append(ScoringResult(
                        score=0.0,
                        method=f"{scorer.name}_error",
                        details=f"Scoring failed: {str(e)}",
                        metadata={"error": str(e), "scorer": scorer.name}
                    ))
            
            # 집계기 선택 및 집계 수행
            aggregator = self._select_aggregator(request)
            if not aggregator:
                raise AggregationError(f"No aggregator available: {request.aggregator_name}")
            
            aggregated_result = aggregator.aggregate(individual_scores, **request.aggregation_kwargs)
            
            # 결과 생성
            result = EvaluationResult(
                request=request,
                individual_scores=individual_scores,
                aggregated_result=aggregated_result,
                metadata={
                    "scorers_used": [scorer.name for scorer in scorers],
                    "aggregator_used": aggregator.name,
                    "engine_version": "1.0.0",
                    "plugin_info": {
                        "total_scorers": len(self.plugin_registry.list_scorers()),
                        "total_aggregators": len(self.plugin_registry.list_aggregators())
                    }
                }
            )
            
            logger.info(f"Evaluation completed: {result.aggregated_result.aggregated_score:.3f} "
                       f"(confidence: {result.aggregated_result.confidence:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def _select_scorers(self, request: EvaluationRequest) -> List[IScorer]:
        """평가 요청에 맞는 채점기들 선택"""
        if request.scorer_names:
            # 명시적으로 지정된 채점기들 사용
            scorers = []
            for name in request.scorer_names:
                scorer = self.plugin_registry.get_scorer(name)
                if scorer:
                    scorers.append(scorer)
                else:
                    logger.warning(f"Scorer not found: {name}")
            return scorers
        
        elif self.auto_select_scorers:
            # 태스크 타입에 맞는 채점기들 자동 선택
            scorers = self.plugin_registry.get_scorers_for_task(request.task_type)
            if not scorers and self.fallback_scorer:
                # fallback 채점기 사용
                fallback = self.plugin_registry.get_scorer(self.fallback_scorer)
                if fallback:
                    scorers = [fallback]
            return scorers
        
        else:
            # fallback 채점기만 사용
            fallback = self.plugin_registry.get_scorer(self.fallback_scorer)
            return [fallback] if fallback else []
    
    def _select_aggregator(self, request: EvaluationRequest) -> Optional[IAggregator]:
        """평가 요청에 맞는 집계기 선택"""
        aggregator_name = request.aggregator_name or self.default_aggregator
        return self.plugin_registry.get_aggregator(aggregator_name)
    
    def batch_evaluate(self, requests: List[EvaluationRequest]) -> List[EvaluationResult]:
        """배치 평가 수행"""
        results = []
        for request in requests:
            try:
                result = self.evaluate(request)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch evaluation failed for request: {e}")
                # 실패한 경우 빈 결과 추가
                results.append(EvaluationResult(
                    request=request,
                    individual_scores=[],
                    aggregated_result=AggregationResult(
                        aggregated_score=0.0,
                        method="evaluation_failed",
                        individual_scores=[],
                        confidence=0.0,
                        metadata={"error": str(e)}
                    ),
                    metadata={"evaluation_error": str(e)}
                ))
        
        logger.info(f"Batch evaluation completed: {len(results)} results")
        return results
    
    def get_available_scorers(self, task_type: Optional[TaskType] = None) -> Dict[str, Dict[str, Any]]:
        """사용 가능한 채점기 목록"""
        if task_type:
            scorers = self.plugin_registry.get_scorers_for_task(task_type)
            return {scorer.name: scorer.get_metadata() for scorer in scorers}
        else:
            return self.plugin_registry.list_scorers()
    
    def get_available_aggregators(self) -> Dict[str, Dict[str, Any]]:
        """사용 가능한 집계기 목록"""
        return self.plugin_registry.list_aggregators()
    
    def get_task_coverage(self) -> Dict[str, List[str]]:
        """태스크별 채점기 커버리지"""
        return self.plugin_registry.get_task_coverage()
    
    def reload_plugins(self) -> bool:
        """플러그인 재로드"""
        try:
            # TODO: 플러그인 재로드 구현
            logger.info("Plugin reload requested (not implemented yet)")
            return True
        except Exception as e:
            logger.error(f"Plugin reload failed: {e}")
            return False
    
    def validate_request(self, request: EvaluationRequest) -> List[str]:
        """평가 요청 검증 및 경고사항 반환"""
        warnings = []
        
        # 기본 입력 검증
        if not request.ground_truth or not request.response:
            warnings.append("Empty ground truth or response")
        
        # 채점기 존재 확인
        available_scorers = self.get_available_scorers(request.task_type)
        if not available_scorers:
            warnings.append(f"No scorers available for task type: {request.task_type.value}")
        
        # 집계기 존재 확인
        aggregator_name = request.aggregator_name or self.default_aggregator
        if not self.plugin_registry.get_aggregator(aggregator_name):
            warnings.append(f"Aggregator not found: {aggregator_name}")
        
        # 명시적 채점기들 존재 확인
        if request.scorer_names:
            for scorer_name in request.scorer_names:
                if not self.plugin_registry.get_scorer(scorer_name):
                    warnings.append(f"Scorer not found: {scorer_name}")
        
        return warnings

# 편의 함수들
def create_evaluation_engine(config: Optional[Dict[str, Any]] = None) -> PluginEvaluationEngine:
    """평가 엔진 생성 (편의 함수)"""
    return PluginEvaluationEngine(config)

def quick_evaluate(ground_truth: str, response: str, task_type: Union[str, TaskType], 
                  config: Optional[Dict[str, Any]] = None) -> float:
    """빠른 평가 (편의 함수) - 집계된 점수만 반환"""
    if isinstance(task_type, str):
        task_type = TaskType(task_type)
    
    engine = create_evaluation_engine(config)
    request = EvaluationRequest(ground_truth=ground_truth, response=response, task_type=task_type)
    result = engine.evaluate(request)
    
    return result.aggregated_result.aggregated_score