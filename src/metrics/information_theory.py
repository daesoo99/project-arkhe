# -*- coding: utf-8 -*-
"""
Project Arkhē - Information Theory Metrics
Shannon Entropy 기반 승급 정책 실험용 메트릭 계산

핵심 연구: 정보 이론 기반 경제적 지능 최적화
"""

import math
import numpy as np
from collections import Counter
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

@dataclass
class InformationMetrics:
    """정보 이론 메트릭 결과"""
    shannon_entropy: float
    js_divergence: float
    renyi_entropy: float
    mutual_information: float
    uncertainty_score: float  # 종합 불확실성 지표
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "shannon_entropy": self.shannon_entropy,
            "js_divergence": self.js_divergence, 
            "renyi_entropy": self.renyi_entropy,
            "mutual_information": self.mutual_information,
            "uncertainty_score": self.uncertainty_score
        }

class InformationTheoryCalculator:
    """Shannon Entropy 및 정보 이론 메트릭 계산기"""
    
    def __init__(self, smoothing: float = 1e-10):
        self.smoothing = smoothing
        
    def shannon_entropy(self, samples: List[str]) -> float:
        """Shannon Entropy 계산 - 다양성 측정"""
        if not samples:
            return 0.0
            
        # 텍스트 정규화 (공백 제거, 소문자 변환, 최대 200자)
        normalized = []
        for sample in samples:
            text = " ".join((sample or "").lower().split())[:200]
            normalized.append(text)
        
        # 빈도 계산
        counter = Counter(normalized)
        total = len(samples)
        
        # Shannon entropy: H(X) = -Σ p(x) log₂ p(x)
        entropy = 0.0
        for count in counter.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p + self.smoothing)
        
        return entropy
    
    def jensen_shannon_divergence(self, samples1: List[str], samples2: List[str]) -> float:
        """Jensen-Shannon Divergence - 두 분포간 차이"""
        if not samples1 or not samples2:
            return 1.0
            
        def get_word_distribution(samples):
            """단어 분포 계산"""
            all_words = []
            for sample in samples:
                words = (sample or "").lower().split()
                all_words.extend(words)
            
            counter = Counter(all_words)
            total = sum(counter.values()) or 1
            return {word: count/total for word, count in counter.items()}
        
        dist1 = get_word_distribution(samples1)
        dist2 = get_word_distribution(samples2)
        
        # 공통 단어 집합
        all_words = set(dist1.keys()) | set(dist2.keys())
        if not all_words:
            return 1.0
        
        # 분포 벡터 생성
        p = np.array([dist1.get(word, 0) for word in all_words])
        q = np.array([dist2.get(word, 0) for word in all_words])
        
        # Jensen-Shannon divergence 계산
        m = 0.5 * (p + q)
        
        def kl_divergence(a, b):
            return np.sum(a * np.log2((a + self.smoothing) / (b + self.smoothing)))
        
        js_div = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
        return min(js_div, 1.0)  # [0, 1] 범위로 클램핑
    
    def renyi_entropy(self, samples: List[str], alpha: float = 2.0) -> float:
        """Rényi Entropy - 일반화된 엔트로피 (α=2일 때 collision entropy)"""
        if not samples:
            return 0.0
            
        # 정규화된 텍스트로 빈도 계산
        normalized = [" ".join((s or "").lower().split())[:200] for s in samples]
        counter = Counter(normalized)
        total = len(samples)
        
        if alpha == 1.0:
            # α=1일 때는 Shannon entropy와 동일
            return self.shannon_entropy(samples)
        
        # Rényi entropy: H_α(X) = (1/(1-α)) log₂(Σ p(x)^α)
        sum_alpha = sum((count/total)**alpha for count in counter.values())
        
        if sum_alpha <= 0:
            return 0.0
            
        renyi_ent = (1/(1-alpha)) * math.log2(sum_alpha + self.smoothing)
        return renyi_ent
    
    def mutual_information(self, samples1: List[str], samples2: List[str]) -> float:
        """상호 정보량 - 두 변수간 의존성 측정"""
        if not samples1 or not samples2 or len(samples1) != len(samples2):
            return 0.0
        
        # 각 샘플을 카테고리로 변환 (길이 기반 간단 분류)
        def categorize(samples):
            categories = []
            for sample in samples:
                length = len((sample or "").split())
                if length <= 5:
                    categories.append("short")
                elif length <= 15:
                    categories.append("medium")
                else:
                    categories.append("long")
            return categories
        
        cat1 = categorize(samples1)
        cat2 = categorize(samples2)
        
        # 결합 분포와 주변 분포 계산
        joint_counter = Counter(zip(cat1, cat2))
        marginal1 = Counter(cat1)
        marginal2 = Counter(cat2)
        
        total = len(samples1)
        mi = 0.0
        
        # MI(X,Y) = Σ p(x,y) log₂(p(x,y) / (p(x)p(y)))
        for (x, y), joint_count in joint_counter.items():
            p_xy = joint_count / total
            p_x = marginal1[x] / total
            p_y = marginal2[y] / total
            
            if p_xy > 0 and p_x > 0 and p_y > 0:
                mi += p_xy * math.log2(p_xy / (p_x * p_y))
        
        return max(0, mi)  # MI는 항상 0 이상
    
    def calculate_all_metrics(self, samples: List[str], 
                            reference_samples: List[str] = None) -> InformationMetrics:
        """모든 정보 이론 메트릭을 한번에 계산"""
        
        # 기본 메트릭
        shannon_ent = self.shannon_entropy(samples)
        renyi_ent = self.renyi_entropy(samples, alpha=2.0)
        
        # 참조 샘플이 있으면 비교 메트릭 계산
        if reference_samples:
            js_div = self.jensen_shannon_divergence(samples, reference_samples)
            mi = self.mutual_information(samples, reference_samples)
        else:
            # 자기 자신과의 JS divergence (항상 0)
            js_div = 0.0
            mi = 0.0
        
        # 종합 불확실성 점수 (0~1, 높을수록 불확실)
        # Shannon entropy 정규화 + Renyi entropy 가중평균
        max_entropy = math.log2(len(samples)) if len(samples) > 1 else 1.0
        normalized_shannon = shannon_ent / max_entropy
        normalized_renyi = min(renyi_ent / max_entropy, 1.0)
        
        uncertainty_score = 0.7 * normalized_shannon + 0.3 * normalized_renyi
        
        return InformationMetrics(
            shannon_entropy=shannon_ent,
            js_divergence=js_div,
            renyi_entropy=renyi_ent,
            mutual_information=mi,
            uncertainty_score=uncertainty_score
        )

class PromotionPolicyEngine:
    """Shannon Entropy 기반 승급 정책 엔진"""
    
    def __init__(self, tau1: float = 0.8, tau2: float = 1.0, 
                 uncertainty_threshold: float = 0.6):
        self.tau1 = tau1  # draft → review 임계치
        self.tau2 = tau2  # review → judge 임계치  
        self.uncertainty_threshold = uncertainty_threshold
        self.calculator = InformationTheoryCalculator()
    
    def should_promote_to_review(self, draft_samples: List[str], 
                                context: Dict[str, Any] = None) -> Tuple[bool, Dict[str, Any]]:
        """초안 → 검토 승급 결정"""
        metrics = self.calculator.calculate_all_metrics(draft_samples)
        
        # 승급 조건들
        high_entropy = metrics.shannon_entropy > self.tau1
        high_uncertainty = metrics.uncertainty_score > self.uncertainty_threshold
        low_consensus = len(set(draft_samples)) > len(draft_samples) * 0.7  # 70% 이상 다름
        
        # 간단한 질문 감지 (휴리스틱)
        query = (context or {}).get("query", "")
        simple_patterns = ["what is", "who is", "수도", "2+2", "when is"]
        is_simple = any(pattern in query.lower() for pattern in simple_patterns)
        
        should_promote = (high_entropy or high_uncertainty or low_consensus) and not is_simple
        
        decision_info = {
            "metrics": metrics.to_dict(),
            "high_entropy": bool(high_entropy),
            "high_uncertainty": bool(high_uncertainty), 
            "low_consensus": bool(low_consensus),
            "is_simple_query": bool(is_simple),
            "tau1": float(self.tau1),
            "decision": "promote" if should_promote else "terminate"
        }
        
        return should_promote, decision_info
    
    def should_promote_to_judge(self, review_samples: List[str], 
                               draft_samples: List[str] = None,
                               context: Dict[str, Any] = None) -> Tuple[bool, Dict[str, Any]]:
        """검토 → 판정 승급 결정"""
        metrics = self.calculator.calculate_all_metrics(review_samples, draft_samples)
        
        # 승급 조건들
        high_entropy = metrics.shannon_entropy > self.tau2
        significant_change = metrics.js_divergence > 0.3 if draft_samples else True
        high_uncertainty = metrics.uncertainty_score > self.uncertainty_threshold
        
        should_promote = high_entropy or significant_change or high_uncertainty
        
        decision_info = {
            "metrics": metrics.to_dict(),
            "high_entropy": bool(high_entropy),
            "significant_change": bool(significant_change),
            "high_uncertainty": bool(high_uncertainty),
            "tau2": float(self.tau2),
            "decision": "promote" if should_promote else "terminate"
        }
        
        return should_promote, decision_info

def test_information_metrics():
    """정보 이론 메트릭 테스트"""
    print("=" * 60)
    print("INFORMATION THEORY METRICS TEST")
    print("=" * 60)
    
    calc = InformationTheoryCalculator()
    
    # 테스트 샘플들
    diverse_samples = [
        "서울은 대한민국의 수도입니다",
        "한국의 수도는 서울이다", 
        "Seoul is the capital of South Korea",
        "수도: 서울특별시"
    ]
    
    uniform_samples = [
        "서울입니다",
        "서울입니다", 
        "서울입니다",
        "서울입니다"
    ]
    
    # Shannon Entropy 비교
    diverse_entropy = calc.shannon_entropy(diverse_samples)
    uniform_entropy = calc.shannon_entropy(uniform_samples)
    
    print(f"Diverse samples entropy: {diverse_entropy:.3f}")
    print(f"Uniform samples entropy: {uniform_entropy:.3f}")
    
    # 전체 메트릭 계산
    metrics = calc.calculate_all_metrics(diverse_samples, uniform_samples)
    print(f"\nAll metrics: {metrics.to_dict()}")
    
    # 승급 정책 테스트
    policy = PromotionPolicyEngine(tau1=0.8, tau2=1.0)
    should_promote, info = policy.should_promote_to_review(diverse_samples, 
                                                          {"query": "대한민국의 수도는?"})
    
    print(f"\nPromotion decision: {should_promote}")
    print(f"Decision info: {info}")

if __name__ == "__main__":
    test_information_metrics()