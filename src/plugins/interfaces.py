# -*- coding: utf-8 -*-
"""
Project Arkhē - Plugin System Interfaces
플러그인 시스템의 핵심 인터페이스 정의

CLAUDE.local 규칙 준수:
- 인터페이스 우선 설계
- 느슨한 결합
- 의존성 주입 패턴
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum

class TaskType(Enum):
    """지원하는 태스크 타입"""
    FACT = "fact"
    REASON = "reason" 
    SUMMARY = "summary"
    FORMAT = "format"
    CODE = "code"
    KOREAN = "ko"
    CREATIVE = "creative"
    ANALYSIS = "analysis"
    PHILOSOPHY = "philosophy"
    PREDICTION = "prediction"

@dataclass
class ScoringResult:
    """채점 결과 데이터 클래스"""
    score: float  # 0.0 ~ 1.0
    method: str
    details: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "method": self.method,
            "details": self.details,
            "metadata": self.metadata
        }

@dataclass
class AggregationResult:
    """집계 결과 데이터 클래스"""
    aggregated_score: float
    method: str
    individual_scores: List[ScoringResult]
    confidence: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class IScorer(ABC):
    """채점기 인터페이스"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """채점기 이름"""
        pass
    
    @property
    @abstractmethod
    def supported_tasks(self) -> List[TaskType]:
        """지원하는 태스크 타입들"""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """채점기 버전"""
        pass
    
    @abstractmethod
    def score(self, ground_truth: str, response: str, task_type: TaskType, **kwargs) -> ScoringResult:
        """
        채점 수행
        
        Args:
            ground_truth: 정답
            response: 모델 응답
            task_type: 태스크 타입
            **kwargs: 추가 옵션
            
        Returns:
            채점 결과
        """
        pass
    
    @abstractmethod
    def validate_input(self, ground_truth: str, response: str, task_type: TaskType) -> bool:
        """
        입력 데이터 검증
        
        Args:
            ground_truth: 정답
            response: 모델 응답  
            task_type: 태스크 타입
            
        Returns:
            검증 성공 여부
        """
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """채점기 메타데이터 반환"""
        return {
            "name": self.name,
            "version": self.version,
            "supported_tasks": [task.value for task in self.supported_tasks],
            "description": getattr(self, 'description', 'No description')
        }

class IAggregator(ABC):
    """집계기 인터페이스"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """집계기 이름"""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """집계기 버전"""
        pass
    
    @abstractmethod
    def aggregate(self, scores: List[ScoringResult], **kwargs) -> AggregationResult:
        """
        점수 집계
        
        Args:
            scores: 개별 채점 결과 목록
            **kwargs: 집계 옵션
            
        Returns:
            집계 결과
        """
        pass
    
    @abstractmethod
    def validate_scores(self, scores: List[ScoringResult]) -> bool:
        """
        점수 유효성 검증
        
        Args:
            scores: 채점 결과 목록
            
        Returns:
            검증 성공 여부
        """
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """집계기 메타데이터 반환"""
        return {
            "name": self.name,
            "version": self.version,
            "description": getattr(self, 'description', 'No description')
        }

class IPlugin(ABC):
    """플러그인 기본 인터페이스"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """플러그인 이름"""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """플러그인 버전"""
        pass
    
    @property
    @abstractmethod
    def dependencies(self) -> List[str]:
        """의존성 목록"""
        pass
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        플러그인 초기화
        
        Args:
            config: 설정 딕셔너리
            
        Returns:
            초기화 성공 여부
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> bool:
        """
        플러그인 정리
        
        Returns:
            정리 성공 여부
        """
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """플러그인 메타데이터 반환"""
        return {
            "name": self.name,
            "version": self.version,
            "dependencies": self.dependencies,
            "description": getattr(self, 'description', 'No description'),
            "author": getattr(self, 'author', 'Unknown'),
            "license": getattr(self, 'license', 'Unknown')
        }

class IScorerPlugin(IPlugin):
    """채점기 플러그인 인터페이스"""
    
    @abstractmethod
    def get_scorer(self) -> IScorer:
        """채점기 인스턴스 반환"""
        pass

class IAggregatorPlugin(IPlugin):
    """집계기 플러그인 인터페이스"""
    
    @abstractmethod
    def get_aggregator(self) -> IAggregator:
        """집계기 인스턴스 반환"""
        pass

# 플러그인 발견을 위한 타입 힌트
PluginFactory = Callable[[], IPlugin]
ScorerFactory = Callable[[], IScorer]  
AggregatorFactory = Callable[[], IAggregator]

class PluginLoadError(Exception):
    """플러그인 로딩 오류"""
    pass

class PluginValidationError(Exception):
    """플러그인 검증 오류"""
    pass

class ScoringError(Exception):
    """채점 오류"""
    pass

class AggregationError(Exception):
    """집계 오류"""
    pass