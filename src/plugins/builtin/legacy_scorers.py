# -*- coding: utf-8 -*-
"""
Project Arkhē - Legacy Scorer Plugin
기존 src/utils/scorers.py를 플러그인 시스템에 통합
"""

from typing import List, Dict, Any
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.plugins.interfaces import IScorer, IScorerPlugin, TaskType, ScoringResult, PluginLoadError

logger = logging.getLogger(__name__)

class LegacyScorer(IScorer):
    """기존 채점 함수들을 플러그인 인터페이스로 래핑"""
    
    def __init__(self):
        self._name = "legacy_scorer"
        self._version = "1.0.0"
        self.description = "Legacy scoring functions from src/utils/scorers.py"
        
        try:
            # 기존 채점 함수들 import
            from src.utils.scorers import (
                score_fact, score_reason, score_summary, score_format, 
                score_code, score_korean, score_task
            )
            
            self._score_functions = {
                TaskType.FACT: score_fact,
                TaskType.REASON: score_reason,
                TaskType.SUMMARY: score_summary,
                TaskType.FORMAT: score_format,
                TaskType.CODE: score_code,
                TaskType.KOREAN: score_korean,
                TaskType.CREATIVE: score_summary,  # 창의적 문제는 요약 채점기
                TaskType.ANALYSIS: score_reason,   # 분석 문제는 추론 채점기
                TaskType.PHILOSOPHY: score_reason, # 철학 문제는 추론 채점기
                TaskType.PREDICTION: score_reason  # 예측 문제는 추론 채점기
            }
            
            self._score_task = score_task
            
        except ImportError as e:
            raise PluginLoadError(f"Could not import legacy scoring functions: {e}")
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def version(self) -> str:
        return self._version
    
    @property
    def supported_tasks(self) -> List[TaskType]:
        return list(self._score_functions.keys())
    
    def validate_input(self, ground_truth: str, response: str, task_type: TaskType) -> bool:
        """입력 데이터 기본 검증"""
        if not ground_truth or not response:
            return False
        if task_type not in self.supported_tasks:
            return False
        return True
    
    def score(self, ground_truth: str, response: str, task_type: TaskType, **kwargs) -> ScoringResult:
        """
        기존 채점 함수를 사용한 채점 수행
        """
        if not self.validate_input(ground_truth, response, task_type):
            return ScoringResult(
                score=0.0,
                method="validation_failed",
                details="Input validation failed",
                metadata={"task_type": task_type.value, "scorer": self.name}
            )
        
        try:
            # 기존 채점 함수 호출
            if task_type in self._score_functions:
                scoring_func = self._score_functions[task_type]
                result_dict = scoring_func(ground_truth, response, **kwargs)
            else:
                # 통합 채점 함수 사용
                result_dict = self._score_task(task_type.value, ground_truth, response, **kwargs)
            
            # 기존 dict 결과를 ScoringResult로 변환
            return ScoringResult(
                score=float(result_dict.get("score", 0.0)),
                method=result_dict.get("method", "unknown"),
                details=result_dict.get("details", ""),
                metadata={
                    "task_type": task_type.value,
                    "scorer": self.name,
                    "legacy_result": result_dict
                }
            )
            
        except Exception as e:
            logger.error(f"Legacy scoring error for {task_type.value}: {e}")
            return ScoringResult(
                score=0.0,
                method="scoring_error",
                details=f"Scoring failed: {str(e)}",
                metadata={"task_type": task_type.value, "scorer": self.name, "error": str(e)}
            )

class LegacyScorerPlugin(IScorerPlugin):
    """Legacy Scorer 플러그인"""
    
    def __init__(self):
        self._name = "legacy_scorer_plugin"
        self._version = "1.0.0"
        self._dependencies = []
        self.description = "Plugin wrapper for existing scoring functions"
        self.author = "Project Arkhē Team"
        self.license = "MIT"
        self._scorer = None
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def version(self) -> str:
        return self._version
    
    @property
    def dependencies(self) -> List[str]:
        return self._dependencies
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """플러그인 초기화"""
        try:
            self._scorer = LegacyScorer()
            logger.info(f"Initialized {self.name} v{self.version}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize {self.name}: {e}")
            return False
    
    def cleanup(self) -> bool:
        """플러그인 정리"""
        self._scorer = None
        logger.info(f"Cleaned up {self.name}")
        return True
    
    def get_scorer(self) -> IScorer:
        """채점기 인스턴스 반환"""
        if self._scorer is None:
            raise RuntimeError("Plugin not initialized")
        return self._scorer