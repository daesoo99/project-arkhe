#!/usr/bin/env python3
"""
Project Arkhē - Information Theory Module
Shannon Entropy 기반 정보 손실 측정 및 최적화
"""

import math
import re
from typing import List, Dict, Any, Tuple
from collections import Counter
import numpy as np
from dataclasses import dataclass

@dataclass
class EntropyMetrics:
    """엔트로피 측정 결과"""
    shannon_entropy: float
    unique_words: int
    total_words: int
    compression_ratio: float
    information_density: float

@dataclass
class InformationFlow:
    """단계별 정보 흐름 분석"""
    stage_name: str
    input_entropy: float
    output_entropy: float
    information_gain: float  # positive = gain, negative = loss
    compression_efficiency: float
    
class ShannonEntropyAnalyzer:
    """Shannon Entropy 기반 정보 분석기"""
    
    def __init__(self):
        self.stage_history = []
        
    def calculate_shannon_entropy(self, text: str) -> float:
        """텍스트의 Shannon Entropy 계산"""
        if not text or len(text) == 0:
            return 0.0
            
        # 단어 단위로 분석 (더 의미있는 정보 단위)
        words = self._preprocess_text(text)
        if len(words) == 0:
            return 0.0
            
        # 단어 빈도 계산
        word_counts = Counter(words)
        total_words = len(words)
        
        # Shannon Entropy 계산: H(X) = -Σ p(x) * log2(p(x))
        entropy = 0.0
        for word, count in word_counts.items():
            probability = count / total_words
            if probability > 0:
                entropy -= probability * math.log2(probability)
                
        return entropy
    
    def _preprocess_text(self, text: str) -> List[str]:
        """텍스트 전처리 - 의미있는 단어 추출"""
        # 소문자 변환 및 특수문자 제거
        text = re.sub(r'[^\w\s가-힣]', ' ', text.lower())
        words = text.split()
        
        # 불용어 제거 (간단한 버전)
        stopwords = {'의', '가', '이', '은', '는', '을', '를', '에', '에서', '과', '와', '그리고', '또한', '하지만',
                    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        meaningful_words = [word for word in words if len(word) > 1 and word not in stopwords]
        return meaningful_words
    
    def analyze_text_diversity(self, text: str) -> EntropyMetrics:
        """텍스트 다양성 종합 분석"""
        words = self._preprocess_text(text)
        unique_words = len(set(words))
        total_words = len(words)
        
        shannon_entropy = self.calculate_shannon_entropy(text)
        
        # 압축 비율 (유니크 단어 / 전체 단어)
        compression_ratio = unique_words / total_words if total_words > 0 else 0
        
        # 정보 밀도 (엔트로피 / 텍스트 길이)
        information_density = shannon_entropy / len(text) if len(text) > 0 else 0
        
        return EntropyMetrics(
            shannon_entropy=shannon_entropy,
            unique_words=unique_words,
            total_words=total_words,
            compression_ratio=compression_ratio,
            information_density=information_density
        )
    
    def track_information_flow(self, stage_name: str, input_text: str, output_text: str) -> InformationFlow:
        """단계별 정보 흐름 추적"""
        input_entropy = self.calculate_shannon_entropy(input_text)
        output_entropy = self.calculate_shannon_entropy(output_text)
        
        # 정보 획득/손실 계산
        information_gain = output_entropy - input_entropy
        
        # 압축 효율성 (출력 엔트로피 / 입력 엔트로피)
        compression_efficiency = output_entropy / input_entropy if input_entropy > 0 else 1.0
        
        flow = InformationFlow(
            stage_name=stage_name,
            input_entropy=input_entropy,
            output_entropy=output_entropy,
            information_gain=information_gain,
            compression_efficiency=compression_efficiency
        )
        
        self.stage_history.append(flow)
        return flow
    
    def analyze_pipeline_efficiency(self) -> Dict[str, Any]:
        """전체 파이프라인 정보 효율성 분석"""
        if not self.stage_history:
            return {"error": "No stage history available"}
            
        # 전체 정보 흐름 분석
        initial_entropy = self.stage_history[0].input_entropy
        final_entropy = self.stage_history[-1].output_entropy
        
        total_information_gain = final_entropy - initial_entropy
        total_stages = len(self.stage_history)
        
        # 단계별 정보 손실/획득
        information_changes = []
        for flow in self.stage_history:
            information_changes.append({
                "stage": flow.stage_name,
                "input_entropy": flow.input_entropy,
                "output_entropy": flow.output_entropy,
                "information_gain": flow.information_gain,
                "efficiency": flow.compression_efficiency
            })
        
        # 파이프라인 효율성 메트릭
        average_efficiency = sum(flow.compression_efficiency for flow in self.stage_history) / total_stages
        information_preservation_rate = final_entropy / initial_entropy if initial_entropy > 0 else 1.0
        
        return {
            "initial_entropy": initial_entropy,
            "final_entropy": final_entropy,
            "total_information_gain": total_information_gain,
            "information_preservation_rate": information_preservation_rate,
            "average_stage_efficiency": average_efficiency,
            "stage_analysis": information_changes,
            "pipeline_verdict": self._get_pipeline_verdict(information_preservation_rate, average_efficiency)
        }
    
    def _get_pipeline_verdict(self, preservation_rate: float, avg_efficiency: float) -> str:
        """파이프라인 성능 판정"""
        if preservation_rate > 1.2 and avg_efficiency > 1.1:
            return "EXCELLENT: Information gain with high efficiency"
        elif preservation_rate > 1.0 and avg_efficiency > 1.0:
            return "GOOD: Positive information flow"
        elif preservation_rate > 0.8:
            return "ACCEPTABLE: Minor information loss"
        else:
            return "POOR: Significant information loss detected"
    
    def suggest_optimization(self) -> List[str]:
        """정보 이론 기반 최적화 제안"""
        if not self.stage_history:
            return ["No data available for optimization"]
            
        suggestions = []
        
        # 정보 손실이 큰 단계 식별
        for flow in self.stage_history:
            if flow.information_gain < -1.0:  # 심각한 정보 손실
                suggestions.append(f"⚠️ {flow.stage_name} stage shows significant information loss (-{abs(flow.information_gain):.2f})")
                suggestions.append(f"   → Consider: Richer prompts, larger model, or better context preservation")
            
            if flow.compression_efficiency < 0.7:  # 비효율적 압축
                suggestions.append(f"📉 {flow.stage_name} stage has low efficiency ({flow.compression_efficiency:.2f})")
                suggestions.append(f"   → Consider: Simplify prompts or use more capable model")
        
        # 전체 파이프라인 제안
        analysis = self.analyze_pipeline_efficiency()
        if analysis["information_preservation_rate"] < 0.8:
            suggestions.append("🔧 Overall pipeline loses too much information")
            suggestions.append("   → Consider: Reduce compression, improve context passing")
        
        if analysis["average_stage_efficiency"] > 1.3:
            suggestions.append("✨ Pipeline shows good information amplification")
            suggestions.append("   → Consider: Apply this approach to other problem types")
        
        return suggestions if suggestions else ["✅ Pipeline shows good information efficiency"]

class EntropyBasedOptimizer:
    """엔트로피 기반 Multi-Agent 최적화"""
    
    def __init__(self, analyzer: ShannonEntropyAnalyzer):
        self.analyzer = analyzer
        
    def optimize_stage_prompts(self, current_prompt: str, target_entropy: float) -> str:
        """목표 엔트로피에 맞게 프롬프트 최적화"""
        current_entropy = self.analyzer.calculate_shannon_entropy(current_prompt)
        
        if current_entropy < target_entropy:
            # 엔트로피 증가 필요 - 더 다양한 관점 추가
            enhanced_prompt = f"""
{current_prompt}

다음 관점들을 추가로 고려하세요:
- 대안적 접근 방법은 무엇인가?
- 예상치 못한 결과나 부작용은?
- 다른 분야의 유사한 사례는?
- 창의적이고 혁신적인 아이디어는?
"""
            return enhanced_prompt.strip()
        else:
            # 엔트로피 적정 - 현재 프롬프트 유지
            return current_prompt
    
    def calculate_optimal_model_allocation(self, problem_complexity: float) -> Dict[str, str]:
        """문제 복잡도에 따른 최적 모델 배치"""
        
        # 엔트로피 기반 모델 선택 규칙
        if problem_complexity < 3.0:
            return {
                "draft": "qwen2:0.5b",    # 낮은 엔트로피 문제는 빠른 모델
                "review": None,           # Review 단계 생략
                "judge": "qwen2:7b"       # 적당한 성능 모델
            }
        elif problem_complexity < 7.0:
            return {
                "draft": "qwen2:7b",      # 중간 복잡도는 균형잡힌 모델
                "review": "qwen2:7b",     
                "judge": "llama3:8b"      # 최종은 강력한 모델
            }
        else:
            return {
                "draft": "qwen2:7b",      # 고복잡도는 모든 단계 강화
                "review": "llama3:8b",    
                "judge": "llama3:8b"      # 또는 더 큰 모델
            }

# 사용 예시
def demonstrate_entropy_analysis():
    """엔트로피 분석 데모"""
    analyzer = ShannonEntropyAnalyzer()
    
    # 샘플 텍스트들
    simple_text = "서울은 대한민국의 수도입니다."
    complex_text = """
    미래 도시의 교통 체증 해결을 위해서는 다층적 접근이 필요합니다. 
    첫째, 3차원 교통 시스템을 통해 지하-지상-공중을 연결한 입체적 교통망을 구축해야 합니다.
    둘째, AI 기반 예측 신호등 시스템으로 실시간 교통량을 분석하여 신호를 최적화해야 합니다.
    셋째, 개인용 드론 택시와 같은 새로운 교통수단을 도입하여 단거리 이동의 효율성을 높여야 합니다.
    """
    
    # 엔트로피 분석
    simple_metrics = analyzer.analyze_text_diversity(simple_text)
    complex_metrics = analyzer.analyze_text_diversity(complex_text)
    
    print("=== Shannon Entropy Analysis Demo ===")
    print(f"\nSimple text entropy: {simple_metrics.shannon_entropy:.2f}")
    print(f"Complex text entropy: {complex_metrics.shannon_entropy:.2f}")
    print(f"Entropy ratio: {complex_metrics.shannon_entropy / simple_metrics.shannon_entropy:.2f}x")
    
    # 정보 흐름 추적
    flow = analyzer.track_information_flow("Enhancement", simple_text, complex_text)
    print(f"\nInformation gain: {flow.information_gain:.2f}")
    print(f"Compression efficiency: {flow.compression_efficiency:.2f}")

if __name__ == "__main__":
    demonstrate_entropy_analysis()