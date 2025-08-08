"""
문제 복잡도 분석기
Problem Complexity Analyzer for Economic Intelligence
"""

import re
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class ComplexityMetrics:
    """복잡도 분석 결과"""
    score: float  # 1.0 (단순) ~ 10.0 (복잡)
    reasoning_depth: int  # 1 (기본) ~ 5 (깊은 추론)
    domain: str  # math, science, philosophy, etc.
    keywords: List[str]
    recommended_model: str

class ComplexityAnalyzer:
    """문제 복잡도를 분석하여 최적 모델을 추천"""
    
    def __init__(self):
        # 복잡도 지표들
        self.simple_patterns = [
            r'\d+\s*[\+\-\*/]\s*\d+',  # 간단한 수학
            r'수도.*어디',  # 수도 질문
            r'몇\s*(살|년|개|명)',  # 단순 사실 질문
            r'예.*아니오',  # Yes/No 질문
        ]
        
        self.complex_patterns = [
            r'왜.*때문',  # 인과관계 질문
            r'어떻게.*해야',  # 방법론 질문  
            r'비교.*분석',  # 비교 분석
            r'장단점',  # 장단점 분석
            r'철학.*윤리',  # 철학적 질문
            r'미래.*예측',  # 미래 예측
        ]
        
        self.domain_keywords = {
            'math': ['수학', '계산', '공식', '+', '-', '*', '/', '='],
            'science': ['과학', '물리', '화학', '생물', '실험'],
            'philosophy': ['철학', '윤리', '도덕', '가치', '의미'],
            'economics': ['경제', '시장', '투자', '금융', '비용'],
            'history': ['역사', '과거', '전쟁', '혁명', '시대'],
            'general': []  # 기본값
        }
        
        # 모델 선택 규칙
        self.model_selection = {
            (1.0, 3.0): "gemma:2b",  # 단순한 문제
            (3.0, 6.0): "gpt-3.5-turbo",  # 중간 복잡도
            (6.0, 8.0): "claude-3-5-sonnet",  # 높은 복잡도
            (8.0, 10.0): "gpt-4"  # 최고 복잡도
        }
    
    def analyze(self, problem: str) -> ComplexityMetrics:
        """문제를 분석하여 복잡도 메트릭스 반환"""
        problem_lower = problem.lower()
        
        # 1. 기본 복잡도 점수 계산
        base_score = self._calculate_base_score(problem_lower)
        
        # 2. 추론 깊이 분석
        reasoning_depth = self._analyze_reasoning_depth(problem_lower)
        
        # 3. 도메인 분류
        domain = self._classify_domain(problem_lower)
        
        # 4. 키워드 추출
        keywords = self._extract_keywords(problem_lower)
        
        # 5. 최종 복잡도 점수 (조정)
        final_score = self._adjust_complexity_score(base_score, domain, reasoning_depth)
        
        # 6. 모델 추천
        recommended_model = self._recommend_model(final_score)
        
        return ComplexityMetrics(
            score=final_score,
            reasoning_depth=reasoning_depth,
            domain=domain,
            keywords=keywords,
            recommended_model=recommended_model
        )
    
    def _calculate_base_score(self, problem: str) -> float:
        """기본 복잡도 점수 계산"""
        score = 5.0  # 기본값
        
        # 단순한 패턴 체크
        for pattern in self.simple_patterns:
            if re.search(pattern, problem):
                score -= 2.0
                break
        
        # 복잡한 패턴 체크
        for pattern in self.complex_patterns:
            if re.search(pattern, problem):
                score += 2.5
                break
        
        # 문장 길이 고려
        word_count = len(problem.split())
        if word_count > 50:
            score += 1.5
        elif word_count < 10:
            score -= 1.0
        
        # 질문 개수 고려 (복합 질문)
        question_marks = problem.count('?') + problem.count('？')
        if question_marks > 1:
            score += 1.0
        
        return max(1.0, min(10.0, score))
    
    def _analyze_reasoning_depth(self, problem: str) -> int:
        """추론 깊이 분석"""
        depth = 1
        
        # 추론 깊이 지표들
        if any(word in problem for word in ['왜', '어떻게', '원인']):
            depth += 1
        
        if any(word in problem for word in ['비교', '차이점', '장단점']):
            depth += 1
            
        if any(word in problem for word in ['예측', '미래', '가능성']):
            depth += 2
            
        if any(word in problem for word in ['철학', '윤리', '의미']):
            depth += 2
        
        return min(5, depth)
    
    def _classify_domain(self, problem: str) -> str:
        """도메인 분류"""
        max_matches = 0
        best_domain = 'general'
        
        for domain, keywords in self.domain_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in problem)
            if matches > max_matches:
                max_matches = matches
                best_domain = domain
        
        return best_domain
    
    def _extract_keywords(self, problem: str) -> List[str]:
        """키워드 추출 (간단한 버전)"""
        # 한글 단어 추출 (2자 이상)
        korean_words = re.findall(r'[가-힣]{2,}', problem)
        # 영어 단어 추출
        english_words = re.findall(r'[a-zA-Z]{3,}', problem)
        
        return list(set(korean_words + english_words))[:10]  # 최대 10개
    
    def _adjust_complexity_score(self, base_score: float, domain: str, depth: int) -> float:
        """도메인과 추론 깊이에 따른 복잡도 조정"""
        adjusted = base_score
        
        # 도메인별 조정
        domain_adjustments = {
            'math': -1.0,  # 수학은 상대적으로 단순
            'philosophy': +2.0,  # 철학은 복잡
            'science': +0.5,
            'economics': +1.0
        }
        
        adjusted += domain_adjustments.get(domain, 0)
        
        # 추론 깊이 조정
        adjusted += (depth - 1) * 0.8
        
        return max(1.0, min(10.0, adjusted))
    
    def _recommend_model(self, complexity_score: float) -> str:
        """복잡도에 따른 모델 추천"""
        for (min_score, max_score), model in self.model_selection.items():
            if min_score <= complexity_score < max_score:
                return model
        
        # 기본값
        return "gpt-3.5-turbo"

# 테스트 함수
def test_complexity_analyzer():
    """복잡도 분석기 테스트"""
    analyzer = ComplexityAnalyzer()
    
    test_problems = [
        "2 + 2는 무엇인가요?",  # 단순
        "프랑스의 수도는 어디인가요?",  # 단순
        "AI 규제의 장단점을 비교 분석해주세요.",  # 복잡
        "양자역학과 상대성이론의 철학적 의미는 무엇인가요?",  # 매우 복잡
        "기후 변화가 경제에 미치는 영향을 예측해보세요.",  # 복잡
    ]
    
    for problem in test_problems:
        metrics = analyzer.analyze(problem)
        print(f"\n문제: {problem}")
        print(f"복잡도: {metrics.score:.1f}/10.0")
        print(f"추론 깊이: {metrics.reasoning_depth}/5")
        print(f"도메인: {metrics.domain}")
        print(f"추천 모델: {metrics.recommended_model}")

if __name__ == "__main__":
    test_complexity_analyzer()