# -*- coding: utf-8 -*-
"""
Project Arkhē - Plugin-Experiment Integration Adapter
플러그인 시스템과 실험 프레임워크를 연결하는 어댑터

Phase 3: Plugin System + Experiment Framework Integration
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict

from ..registry.experiment_registry import ExperimentRegistry, ExperimentConfig, get_experiment_registry
from ..evaluation.plugin_engine import PluginEvaluationEngine, EvaluationRequest, EvaluationResult, create_evaluation_engine
from ..plugins.interfaces import TaskType

logger = logging.getLogger(__name__)

@dataclass
class IntegratedExperimentResult:
    """통합 실험 결과"""
    experiment_config: ExperimentConfig
    evaluation_results: List[EvaluationResult]
    plugin_metadata: Dict[str, Any]
    performance_metrics: Dict[str, float]
    total_score: float
    success_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "experiment": {
                "name": self.experiment_config.name,
                "description": self.experiment_config.description,
                "environment": self.experiment_config.environment,
                "metadata": self.experiment_config.metadata
            },
            "evaluation_summary": {
                "total_evaluations": len(self.evaluation_results),
                "total_score": self.total_score,
                "average_score": self.total_score / len(self.evaluation_results) if self.evaluation_results else 0.0,
                "success_rate": self.success_rate
            },
            "evaluation_results": [result.to_dict() for result in self.evaluation_results],
            "plugin_metadata": self.plugin_metadata,
            "performance_metrics": self.performance_metrics
        }

class PluginExperimentAdapter:
    """
    플러그인 시스템과 실험 프레임워크 통합 어댑터
    
    기능:
    - 실험 설정을 플러그인 평가 요청으로 변환
    - 플러그인 평가 결과를 실험 결과 형식으로 변환
    - 설정 기반 플러그인 선택 및 조정
    - 통합 메트릭 계산
    """
    
    def __init__(self, 
                 experiment_environment: str = "development",
                 plugin_config: Optional[Dict[str, Any]] = None):
        """
        Args:
            experiment_environment: 실험 환경 설정
            plugin_config: 플러그인 시스템 설정
        """
        self.experiment_registry = get_experiment_registry(experiment_environment)
        
        # 플러그인 설정 로드
        if plugin_config is None:
            plugin_config = self._load_plugin_config()
        
        self.plugin_engine = create_evaluation_engine(plugin_config)
        self.plugin_config = plugin_config
        
        logger.info(f"Initialized PluginExperimentAdapter with environment: {experiment_environment}")
    
    def _load_plugin_config(self) -> Dict[str, Any]:
        """플러그인 설정 파일 로드"""
        config_paths = [
            Path(__file__).parent.parent.parent / "config" / "plugin_config.json",
            Path.cwd() / "config" / "plugin_config.json",
            Path.cwd() / "plugin_config.json"
        ]
        
        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    logger.info(f"Loaded plugin config from: {config_path}")
                    return config
                except Exception as e:
                    logger.warning(f"Failed to load plugin config from {config_path}: {e}")
                    
        # 기본 설정 반환
        logger.info("Using default plugin configuration")
        return {
            "auto_select_scorers": True,
            "default_aggregator": "weighted_average"
        }
    
    def run_experiment(self, 
                      template_name: str,
                      test_data: List[Dict[str, Any]],
                      **experiment_overrides) -> IntegratedExperimentResult:
        """
        통합 실험 실행
        
        Args:
            template_name: 실험 템플릿 이름
            test_data: 테스트 데이터 (ground_truth, response, task_type 포함)
            **experiment_overrides: 실험 설정 오버라이드
            
        Returns:
            통합 실험 결과
        """
        # 1. 실험 설정 로드
        experiment_config = self.experiment_registry.get_experiment_config(
            template_name, **experiment_overrides
        )
        
        logger.info(f"Starting experiment: {experiment_config.name}")
        
        # 2. 테스트 데이터를 평가 요청으로 변환
        evaluation_requests = self._convert_test_data_to_requests(test_data, experiment_config)
        
        # 3. 플러그인 기반 평가 수행
        evaluation_results = []
        failed_evaluations = 0
        
        for request in evaluation_requests:
            try:
                result = self.plugin_engine.evaluate(request)
                evaluation_results.append(result)
            except Exception as e:
                logger.error(f"Evaluation failed for {request.task_type}: {e}")
                failed_evaluations += 1
                
        # 4. 통합 결과 생성
        return self._create_integrated_result(
            experiment_config, 
            evaluation_results, 
            failed_evaluations,
            len(evaluation_requests)
        )
    
    def _convert_test_data_to_requests(self, 
                                     test_data: List[Dict[str, Any]], 
                                     experiment_config: ExperimentConfig) -> List[EvaluationRequest]:
        """테스트 데이터를 평가 요청으로 변환"""
        requests = []
        
        for data in test_data:
            # 태스크 타입 변환
            task_type_str = data.get('task_type', 'fact')
            try:
                task_type = TaskType(task_type_str.lower())
            except ValueError:
                logger.warning(f"Unknown task type: {task_type_str}, using FACT")
                task_type = TaskType.FACT
            
            # 플러그인 설정에서 태스크별 설정 확인
            task_config = self.plugin_config.get('task_specific_configs', {}).get(task_type.value, {})
            
            # 평가 요청 생성
            request = EvaluationRequest(
                ground_truth=data['ground_truth'],
                response=data['response'],
                task_type=task_type,
                scorer_names=task_config.get('preferred_scorers'),
                aggregator_name=task_config.get('preferred_aggregator'),
                scoring_kwargs=data.get('scoring_kwargs', {}),
                aggregation_kwargs=data.get('aggregation_kwargs', {})
            )
            
            requests.append(request)
            
        logger.info(f"Converted {len(test_data)} test items to evaluation requests")
        return requests
    
    def _create_integrated_result(self, 
                                experiment_config: ExperimentConfig,
                                evaluation_results: List[EvaluationResult],
                                failed_count: int,
                                total_count: int) -> IntegratedExperimentResult:
        """통합 실험 결과 생성"""
        
        # 총점 및 성공률 계산
        total_score = sum(result.aggregated_result.aggregated_score for result in evaluation_results)
        success_rate = (total_count - failed_count) / total_count if total_count > 0 else 0.0
        
        # 플러그인 메타데이터 수집
        plugin_metadata = {
            "scorers_used": list(set(
                scorer for result in evaluation_results 
                for scorer in result.metadata.get('scorers_used', [])
            )),
            "aggregators_used": list(set(
                result.metadata.get('aggregator_used', '') 
                for result in evaluation_results
            )),
            "total_scorers_available": len(self.plugin_engine.get_available_scorers()),
            "total_aggregators_available": len(self.plugin_engine.get_available_aggregators()),
            "task_coverage": self.plugin_engine.get_task_coverage()
        }
        
        # 성능 메트릭 계산
        performance_metrics = self._calculate_performance_metrics(evaluation_results)
        
        return IntegratedExperimentResult(
            experiment_config=experiment_config,
            evaluation_results=evaluation_results,
            plugin_metadata=plugin_metadata,
            performance_metrics=performance_metrics,
            total_score=total_score,
            success_rate=success_rate
        )
    
    def _calculate_performance_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """성능 메트릭 계산"""
        if not results:
            return {}
        
        scores = [result.aggregated_result.aggregated_score for result in results]
        confidences = [result.aggregated_result.confidence for result in results]
        
        # 기본 통계
        metrics = {
            "mean_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "mean_confidence": sum(confidences) / len(confidences),
            "max_confidence": max(confidences),
            "min_confidence": min(confidences)
        }
        
        # 표준편차 계산
        if len(scores) > 1:
            mean_score = metrics["mean_score"]
            variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
            metrics["score_std"] = variance ** 0.5
        else:
            metrics["score_std"] = 0.0
            
        # 태스크별 성능
        task_scores = {}
        for result in results:
            task_type = result.request.task_type.value
            if task_type not in task_scores:
                task_scores[task_type] = []
            task_scores[task_type].append(result.aggregated_result.aggregated_score)
        
        for task_type, scores in task_scores.items():
            metrics[f"{task_type}_mean_score"] = sum(scores) / len(scores)
            metrics[f"{task_type}_count"] = len(scores)
        
        return metrics
    
    def run_parameter_sweep(self, 
                           sweep_name: str,
                           test_data: List[Dict[str, Any]]) -> List[IntegratedExperimentResult]:
        """매개변수 스위핑 실험 실행"""
        
        # 스위핑 실험 설정 생성
        sweep_configs = self.experiment_registry.generate_parameter_sweep(sweep_name)
        
        logger.info(f"Running parameter sweep '{sweep_name}' with {len(sweep_configs)} configurations")
        
        results = []
        for i, config in enumerate(sweep_configs):
            logger.info(f"Running sweep experiment {i+1}/{len(sweep_configs)}: {config.name}")
            
            try:
                # 개별 실험 실행 (오버라이드 없이 sweep config 그대로 사용)
                result = self.run_experiment(
                    config.metadata['template_name'],  # 원본 템플릿 이름
                    test_data
                )
                
                # sweep 메타데이터 추가
                result.experiment_config = config  # sweep config 사용
                results.append(result)
                
            except Exception as e:
                logger.error(f"Sweep experiment {config.name} failed: {e}")
                
        logger.info(f"Parameter sweep completed: {len(results)}/{len(sweep_configs)} successful")
        return results
    
    def validate_integration(self) -> Dict[str, Any]:
        """통합 시스템 검증"""
        validation_results = {
            "experiment_registry": True,
            "plugin_engine": True,
            "available_templates": [],
            "available_scorers": [],
            "available_aggregators": [],
            "warnings": []
        }
        
        try:
            # 실험 템플릿 확인
            templates = self.experiment_registry.list_available_templates()
            validation_results["available_templates"] = list(templates.keys())
            
            if not templates:
                validation_results["warnings"].append("No experiment templates available")
                
        except Exception as e:
            validation_results["experiment_registry"] = False
            validation_results["warnings"].append(f"Experiment registry error: {e}")
        
        try:
            # 플러그인 시스템 확인
            scorers = self.plugin_engine.get_available_scorers()
            aggregators = self.plugin_engine.get_available_aggregators()
            
            validation_results["available_scorers"] = list(scorers.keys())
            validation_results["available_aggregators"] = list(aggregators.keys())
            
            if not scorers:
                validation_results["warnings"].append("No scorers available")
            if not aggregators:
                validation_results["warnings"].append("No aggregators available")
                
        except Exception as e:
            validation_results["plugin_engine"] = False
            validation_results["warnings"].append(f"Plugin engine error: {e}")
        
        validation_results["integration_healthy"] = (
            validation_results["experiment_registry"] and 
            validation_results["plugin_engine"] and
            validation_results["available_templates"] and
            validation_results["available_scorers"] and
            validation_results["available_aggregators"]
        )
        
        return validation_results

# 편의 함수들
def create_plugin_experiment_adapter(environment: str = "development", 
                                    plugin_config_path: Optional[str] = None) -> PluginExperimentAdapter:
    """플러그인-실험 어댑터 생성"""
    plugin_config = None
    if plugin_config_path:
        with open(plugin_config_path, 'r', encoding='utf-8') as f:
            plugin_config = json.load(f)
    
    return PluginExperimentAdapter(environment, plugin_config)

def quick_experiment_run(template_name: str, 
                        test_data: List[Dict[str, Any]],
                        environment: str = "development") -> IntegratedExperimentResult:
    """빠른 실험 실행 (편의 함수)"""
    adapter = create_plugin_experiment_adapter(environment)
    return adapter.run_experiment(template_name, test_data)