# -*- coding: utf-8 -*-
"""
Project Arkhē - Standard Aggregators Plugin  
표준 점수 집계 알고리즘들
"""

from typing import List, Dict, Any
import logging
import statistics

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.plugins.interfaces import IAggregator, IAggregatorPlugin, ScoringResult, AggregationResult

logger = logging.getLogger(__name__)

class WeightedAverageAggregator(IAggregator):
    """가중 평균 집계기"""
    
    def __init__(self, weights: Dict[str, float] = None):
        self._name = "weighted_average"
        self._version = "1.0.0"
        self.description = "Weighted average aggregation with method-based weights"
        self._weights = weights or {}
        self._default_weight = 1.0
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def version(self) -> str:
        return self._version
    
    def validate_scores(self, scores: List[ScoringResult]) -> bool:
        """점수 유효성 검증"""
        if not scores:
            return False
        
        for score in scores:
            if not (0.0 <= score.score <= 1.0):
                return False
        
        return True
    
    def aggregate(self, scores: List[ScoringResult], **kwargs) -> AggregationResult:
        """가중 평균 집계"""
        if not self.validate_scores(scores):
            return AggregationResult(
                aggregated_score=0.0,
                method="validation_failed",
                individual_scores=scores,
                confidence=0.0,
                metadata={"error": "Score validation failed"}
            )
        
        # 가중치 계산
        weighted_sum = 0.0
        total_weight = 0.0
        
        for score_result in scores:
            method = score_result.method
            weight = self._weights.get(method, self._default_weight)
            
            weighted_sum += score_result.score * weight
            total_weight += weight
        
        if total_weight == 0:
            aggregated_score = 0.0
            confidence = 0.0
        else:
            aggregated_score = weighted_sum / total_weight
            
            # 신뢰도 계산 (점수 분산 기반)
            score_values = [s.score for s in scores]
            if len(score_values) > 1:
                score_std = statistics.stdev(score_values)
                confidence = max(0.0, 1.0 - score_std)  # 분산이 낮을수록 높은 신뢰도
            else:
                confidence = 1.0
        
        return AggregationResult(
            aggregated_score=aggregated_score,
            method=f"weighted_average_{len(scores)}",
            individual_scores=scores,
            confidence=confidence,
            metadata={
                "weights_used": {s.method: self._weights.get(s.method, self._default_weight) for s in scores},
                "total_weight": total_weight,
                "score_std": statistics.stdev([s.score for s in scores]) if len(scores) > 1 else 0.0
            }
        )

class MaxScoreAggregator(IAggregator):
    """최대값 집계기 (가장 관대한 채점)"""
    
    def __init__(self):
        self._name = "max_score"
        self._version = "1.0.0"
        self.description = "Takes the maximum score among all scorers"
    
    @property
    def name(self) -> str:
        return self._name
    
    @property 
    def version(self) -> str:
        return self._version
    
    def validate_scores(self, scores: List[ScoringResult]) -> bool:
        return bool(scores) and all(0.0 <= s.score <= 1.0 for s in scores)
    
    def aggregate(self, scores: List[ScoringResult], **kwargs) -> AggregationResult:
        """최대값 선택"""
        if not self.validate_scores(scores):
            return AggregationResult(
                aggregated_score=0.0,
                method="validation_failed",
                individual_scores=scores,
                confidence=0.0
            )
        
        max_score = max(s.score for s in scores)
        best_scorer = next(s for s in scores if s.score == max_score)
        
        # 신뢰도: 최고 점수가 다른 점수들보다 얼마나 일관되게 높은가
        sorted_scores = sorted([s.score for s in scores], reverse=True)
        if len(sorted_scores) > 1:
            confidence = max(0.0, (sorted_scores[0] - sorted_scores[1]))
        else:
            confidence = 1.0
        
        return AggregationResult(
            aggregated_score=max_score,
            method=f"max_score_from_{best_scorer.method}",
            individual_scores=scores,
            confidence=confidence,
            metadata={
                "best_scorer": best_scorer.method,
                "score_range": f"{min(s.score for s in scores):.3f}-{max_score:.3f}",
                "unanimous": len(set(s.score for s in scores)) == 1
            }
        )

class MedianAggregator(IAggregator):
    """중앙값 집계기 (로버스트 집계)"""
    
    def __init__(self):
        self._name = "median"
        self._version = "1.0.0"
        self.description = "Robust median aggregation, less sensitive to outliers"
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def version(self) -> str:
        return self._version
    
    def validate_scores(self, scores: List[ScoringResult]) -> bool:
        return bool(scores) and all(0.0 <= s.score <= 1.0 for s in scores)
    
    def aggregate(self, scores: List[ScoringResult], **kwargs) -> AggregationResult:
        """중앙값 집계"""
        if not self.validate_scores(scores):
            return AggregationResult(
                aggregated_score=0.0,
                method="validation_failed", 
                individual_scores=scores,
                confidence=0.0
            )
        
        score_values = [s.score for s in scores]
        median_score = statistics.median(score_values)
        
        # 신뢰도: 중앙값 주변의 집중도
        mad = statistics.median([abs(x - median_score) for x in score_values])  # Median Absolute Deviation
        confidence = max(0.0, 1.0 - mad * 2)  # MAD가 낮을수록 높은 신뢰도
        
        return AggregationResult(
            aggregated_score=median_score,
            method=f"median_{len(scores)}",
            individual_scores=scores,
            confidence=confidence,
            metadata={
                "median_absolute_deviation": mad,
                "score_distribution": {
                    "min": min(score_values),
                    "q1": statistics.quantiles(score_values, n=4)[0] if len(score_values) >= 4 else min(score_values),
                    "median": median_score,
                    "q3": statistics.quantiles(score_values, n=4)[2] if len(score_values) >= 4 else max(score_values),
                    "max": max(score_values)
                }
            }
        )

class ConsensusAggregator(IAggregator):
    """컨센서스 집계기 (합의 기반)"""
    
    def __init__(self, consensus_threshold: float = 0.1):
        self._name = "consensus"
        self._version = "1.0.0"  
        self.description = "Consensus-based aggregation, requires agreement within threshold"
        self._consensus_threshold = consensus_threshold
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def version(self) -> str:
        return self._version
    
    def validate_scores(self, scores: List[ScoringResult]) -> bool:
        return bool(scores) and all(0.0 <= s.score <= 1.0 for s in scores)
    
    def aggregate(self, scores: List[ScoringResult], **kwargs) -> AggregationResult:
        """컨센서스 기반 집계"""
        if not self.validate_scores(scores):
            return AggregationResult(
                aggregated_score=0.0,
                method="validation_failed",
                individual_scores=scores, 
                confidence=0.0
            )
        
        score_values = [s.score for s in scores]
        mean_score = statistics.mean(score_values)
        std_score = statistics.stdev(score_values) if len(score_values) > 1 else 0.0
        
        # 컨센서스 확인: 표준편차가 임계값 이하인가?
        has_consensus = std_score <= self._consensus_threshold
        
        if has_consensus:
            # 컨센서스가 있으면 평균값 사용
            aggregated_score = mean_score
            confidence = 1.0 - std_score / self._consensus_threshold  # 표준편차가 낮을수록 높은 신뢰도
            method = f"consensus_agreement_{len(scores)}"
        else:
            # 컨센서스가 없으면 중앙값 사용 (robust fallback)
            aggregated_score = statistics.median(score_values)
            confidence = 0.5  # 합의 실패로 중간 신뢰도
            method = f"consensus_fallback_median_{len(scores)}"
        
        return AggregationResult(
            aggregated_score=aggregated_score,
            method=method,
            individual_scores=scores,
            confidence=confidence,
            metadata={
                "has_consensus": has_consensus,
                "score_std": std_score,
                "threshold": self._consensus_threshold,
                "mean_score": mean_score,
                "agreement_level": "high" if std_score < self._consensus_threshold/2 else "low" if not has_consensus else "medium"
            }
        )

class StandardAggregatorsPlugin(IAggregatorPlugin):
    """표준 집계기들을 제공하는 플러그인"""
    
    def __init__(self):
        self._name = "standard_aggregators"
        self._version = "1.0.0"
        self._dependencies = []
        self.description = "Standard score aggregation algorithms"
        self.author = "Project Arkhē Team"
        self.license = "MIT"
        self._aggregators = {}
    
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
            # 가중 평균 집계기
            weights = config.get("weighted_average", {}).get("weights", {})
            self._aggregators["weighted_average"] = WeightedAverageAggregator(weights)
            
            # 기타 집계기들
            self._aggregators["max_score"] = MaxScoreAggregator()
            self._aggregators["median"] = MedianAggregator()
            
            # 컨센서스 집계기
            consensus_threshold = config.get("consensus", {}).get("threshold", 0.1)
            self._aggregators["consensus"] = ConsensusAggregator(consensus_threshold)
            
            logger.info(f"Initialized {self.name} with {len(self._aggregators)} aggregators")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.name}: {e}")
            return False
    
    def cleanup(self) -> bool:
        """플러그인 정리"""
        self._aggregators.clear()
        logger.info(f"Cleaned up {self.name}")
        return True
    
    def get_aggregator(self) -> IAggregator:
        """기본 집계기 반환 (하위 호환성)"""
        return self._aggregators.get("weighted_average")
    
    def get_aggregators(self) -> List[IAggregator]:
        """모든 집계기 반환"""
        return list(self._aggregators.values())
    
    def get_aggregator_by_name(self, name: str) -> IAggregator:
        """이름으로 집계기 반환"""
        return self._aggregators.get(name)