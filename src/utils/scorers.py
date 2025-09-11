"""
Project Arkhē 태스크별 채점기 시스템
전문적인 정답 판별 로직 구현
"""

import re
import json
import math
from typing import Dict, Any, Optional, List, Union

def normalize_number(text: str) -> Optional[float]:
    """숫자 정규화: 쉼표, 단위 제거 후 숫자 추출"""
    if not text:
        return None
    
    # 한글/영어 단위 제거 (km, 킬로미터, %, 퍼센트 등)
    units = r'(km|킬로미터|미터|m|%|퍼센트|원|달러|\$|개|명|시간|분|초|년|월|일)'
    text = re.sub(units, '', text, flags=re.IGNORECASE)
    
    # 쉼표 제거, 공백 정리
    text = re.sub(r'[,\s]+', '', text)
    
    # 숫자 패턴 추출 (정수, 소수, 분수 지원)
    number_patterns = [
        r'(\d+\.\d+)',  # 소수
        r'(\d+)',       # 정수
        r'(\d+)/(\d+)'  # 분수
    ]
    
    for pattern in number_patterns:
        match = re.search(pattern, text)
        if match:
            if '/' in pattern:  # 분수 처리
                num, den = match.groups()
                return float(num) / float(den)
            else:
                return float(match.group(1))
    
    return None

def score_fact(ground_truth: str, response: str, tolerance: float = 0.05) -> Dict[str, Any]:
    """
    사실 문제 채점: 숫자는 오차 허용, 텍스트는 포함 여부 확인
    
    Args:
        ground_truth: 정답
        response: 모델 응답
        tolerance: 숫자 오차 허용폭 (기본 5%)
    
    Returns:
        채점 결과 딕셔너리
    """
    if not ground_truth or not response:
        return {"score": 0, "method": "empty", "details": "Empty input"}
    
    gt_clean = ground_truth.strip().lower()
    resp_clean = response.strip().lower()
    
    # 1. 정확한 문자열 매치 (가장 확실한 경우)
    if gt_clean in resp_clean:
        return {"score": 1, "method": "exact_match", "details": f"Found '{gt_clean}' in response"}
    
    # 2. 숫자 비교 (오차 허용)
    gt_num = normalize_number(ground_truth)
    resp_num = normalize_number(response)
    
    if gt_num is not None and resp_num is not None:
        if gt_num == 0:  # 0으로 나누기 방지
            error_rate = abs(resp_num - gt_num)
        else:
            error_rate = abs(resp_num - gt_num) / abs(gt_num)
        
        if error_rate <= tolerance:
            return {
                "score": 1, 
                "method": "numeric_tolerance",
                "details": f"Numbers match within {tolerance:.1%}: {gt_num} vs {resp_num} (error: {error_rate:.1%})"
            }
        else:
            return {
                "score": 0,
                "method": "numeric_mismatch", 
                "details": f"Numbers don't match: {gt_num} vs {resp_num} (error: {error_rate:.1%})"
            }
    
    # 3. 키워드 부분 매치 (최후 수단)
    gt_words = set(re.findall(r'\w+', gt_clean))
    resp_words = set(re.findall(r'\w+', resp_clean))
    
    if gt_words:
        overlap = len(gt_words.intersection(resp_words)) / len(gt_words)
        if overlap >= 0.5:  # 50% 이상 키워드 일치
            return {
                "score": 0.5,
                "method": "keyword_partial",
                "details": f"Keyword overlap: {overlap:.1%}"
            }
    
    return {"score": 0, "method": "no_match", "details": "No meaningful match found"}

def score_reason(ground_truth: str, response: str, min_keywords: int = 2) -> Dict[str, Any]:
    """
    추론 문제 채점: 중간 단계 키워드 포함 여부로 체크리스트 채점
    
    Args:
        ground_truth: 정답 (키워드나 단계가 포함된 설명)
        response: 모델 응답
        min_keywords: 최소 키워드 매치 수
    
    Returns:
        채점 결과 딕셔너리
    """
    if not ground_truth or not response:
        return {"score": 0, "method": "empty", "details": "Empty input"}
    
    # 정답에서 핵심 키워드 추출
    gt_lower = ground_truth.lower()
    resp_lower = response.lower()
    
    # 숫자, 날짜, 핵심 동사/명사 추출
    key_patterns = [
        r'\b\d+\b',           # 숫자
        r'\b\d+월\b',         # 월
        r'\b\d+일\b',         # 일
        r'[가-힣]{2,}',       # 한글 키워드 (2글자 이상)
        r'[a-z]{3,}',         # 영어 키워드 (3글자 이상)
    ]
    
    gt_keywords = set()
    for pattern in key_patterns:
        gt_keywords.update(re.findall(pattern, gt_lower))
    
    # 응답에서 키워드 매치 확인
    matched_keywords = []
    for keyword in gt_keywords:
        if keyword in resp_lower:
            matched_keywords.append(keyword)
    
    match_ratio = len(matched_keywords) / len(gt_keywords) if gt_keywords else 0
    
    # 점수 계산
    if len(matched_keywords) >= min_keywords:
        score = min(1.0, match_ratio)  # 최대 1점
    else:
        score = match_ratio * 0.5  # 키워드 부족 시 감점
    
    return {
        "score": score,
        "method": "keyword_checklist",
        "details": f"Matched {len(matched_keywords)}/{len(gt_keywords)} keywords: {matched_keywords[:3]}"
    }

def score_summary(ground_truth: str, response: str) -> Dict[str, Any]:
    """
    요약 문제 채점: ROUGE-L 간이 계산 (토큰 집합 교집합)
    
    Args:
        ground_truth: 정답 요약
        response: 모델 응답 요약
    
    Returns:
        채점 결과 딕셔너리
    """
    if not ground_truth or not response:
        return {"score": 0, "method": "empty", "details": "Empty input"}
    
    # 토큰화 (한글/영어 단어 추출)
    def tokenize(text):
        # 한글 단어, 영어 단어, 숫자 추출
        tokens = re.findall(r'[가-힣]+|[a-zA-Z]+|\d+', text.lower())
        return set(tokens)
    
    gt_tokens = tokenize(ground_truth)
    resp_tokens = tokenize(response)
    
    if not gt_tokens:
        return {"score": 0, "method": "no_reference_tokens", "details": "No tokens in ground truth"}
    
    # Precision, Recall 계산
    intersection = gt_tokens.intersection(resp_tokens)
    
    precision = len(intersection) / len(resp_tokens) if resp_tokens else 0
    recall = len(intersection) / len(gt_tokens)
    
    # F1-Score (ROUGE-L 근사)
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    
    return {
        "score": f1_score,
        "method": "rouge_l_approximation",
        "details": f"P={precision:.2f}, R={recall:.2f}, F1={f1_score:.2f}, Common={len(intersection)}"
    }

def score_format(expected_schema: str, response: str, allow_extra_keys: bool = False) -> Dict[str, Any]:
    """
    형식 문제 채점: JSON 스키마 검증
    
    Args:
        expected_schema: 기대하는 JSON 스키마 (예: '{"name": "string", "age": "int"}')
        response: 모델 응답
        allow_extra_keys: 추가 키 허용 여부
    
    Returns:
        채점 결과 딕셔너리
    """
    try:
        # 응답에서 JSON 추출 시도
        json_match = re.search(r'\{[^}]*\}', response.strip())
        if not json_match:
            return {"score": 0, "method": "no_json_found", "details": "No JSON object found in response"}
        
        json_str = json_match.group(0)
        parsed_response = json.loads(json_str)
        
        # 기대 스키마 파싱
        try:
            expected_obj = json.loads(expected_schema)
        except:
            # 스키마가 예제 형태인 경우 (예: {"name": "Kim", "age": 26})
            expected_obj = json.loads(expected_schema)
        
        # 키 검증
        expected_keys = set(expected_obj.keys())
        response_keys = set(parsed_response.keys())
        
        missing_keys = expected_keys - response_keys
        extra_keys = response_keys - expected_keys
        
        # 점수 계산
        score = 0
        issues = []
        
        # 필수 키 검사
        if not missing_keys:
            score += 0.6  # 키 완전성 60%
        else:
            issues.append(f"Missing keys: {list(missing_keys)}")
        
        # 추가 키 검사
        if extra_keys and not allow_extra_keys:
            issues.append(f"Extra keys: {list(extra_keys)}")
        elif not extra_keys:
            score += 0.2  # 키 정확성 20%
        
        # 타입 검증 (간단한 경우만)
        type_correct = True
        for key, expected_val in expected_obj.items():
            if key in parsed_response:
                resp_val = parsed_response[key]
                if isinstance(expected_val, int) and not isinstance(resp_val, int):
                    type_correct = False
                    issues.append(f"Type mismatch for '{key}': expected int, got {type(resp_val).__name__}")
                elif isinstance(expected_val, str) and not isinstance(resp_val, str):
                    type_correct = False
                    issues.append(f"Type mismatch for '{key}': expected string, got {type(resp_val).__name__}")
        
        if type_correct:
            score += 0.2  # 타입 정확성 20%
        
        return {
            "score": min(1.0, score),
            "method": "json_schema_validation", 
            "details": f"Issues: {'; '.join(issues) if issues else 'All checks passed'}"
        }
        
    except json.JSONDecodeError as e:
        return {"score": 0, "method": "json_parse_error", "details": f"JSON parsing failed: {e}"}
    except Exception as e:
        return {"score": 0, "method": "validation_error", "details": f"Validation error: {e}"}

def score_code(ground_truth: str, response: str, safe_check: bool = True) -> Dict[str, Any]:
    """
    코드 문제 채점: exec 금지, 정규표현식으로 함수 시그니처/구조 검증
    
    Args:
        ground_truth: 정답 코드 (참조용)
        response: 모델이 생성한 코드
        safe_check: 안전성 검사 활성화 여부
    
    Returns:
        채점 결과 딕셔너리
    """
    if not response:
        return {"score": 0, "method": "empty_response", "details": "No code provided"}
    
    issues = []
    score = 0
    
    # 1. 안전성 검사 (위험한 코드 패턴 감지)
    if safe_check:
        dangerous_patterns = [
            r'\bexec\b', r'\beval\b', r'__import__', r'\bopen\b',
            r'subprocess', r'os\.system', r'input\(', r'raw_input\('
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return {"score": 0, "method": "unsafe_code", "details": f"Dangerous pattern detected: {pattern}"}
    
    # 2. 함수 정의 검사
    func_pattern = r'def\s+(\w+)\s*\([^)]*\):'
    funcs_in_response = re.findall(func_pattern, response)
    funcs_in_ground_truth = re.findall(func_pattern, ground_truth) if ground_truth else []
    
    if funcs_in_ground_truth and funcs_in_response:
        # 함수명 매치 검사
        if set(funcs_in_ground_truth).intersection(set(funcs_in_response)):
            score += 0.4  # 함수명 일치 40%
        else:
            issues.append(f"Function names don't match. Expected: {funcs_in_ground_truth}, Got: {funcs_in_response}")
    elif funcs_in_response:
        score += 0.2  # 함수 정의 존재 20%
    else:
        issues.append("No function definition found")
    
    # 3. 기본 구문 검사 (문법 오류 감지)
    try:
        compile(response, '<string>', 'exec')
        score += 0.3  # 문법 정확성 30%
    except SyntaxError as e:
        issues.append(f"Syntax error: {e}")
    
    # 4. 테스트 케이스/출력 검사
    if 'print(' in response:
        score += 0.1  # 출력 존재 10%
    
    # 5. 주석/독스트링 보너스
    if '#' in response or '"""' in response or "'''" in response:
        score += 0.1  # 문서화 보너스 10%
    
    # 6. 반환문 검사
    if 'return' in response:
        score += 0.1  # 반환문 존재 10%
    
    return {
        "score": min(1.0, score),
        "method": "code_structure_analysis",
        "details": f"Functions: {funcs_in_response}, Issues: {'; '.join(issues) if issues else 'None'}"
    }

def score_korean(ground_truth: str, response: str) -> Dict[str, Any]:
    """
    한국어 특화 문제 채점: 한국어 특성 고려한 유사도 계산
    
    Args:
        ground_truth: 정답
        response: 모델 응답
    
    Returns:
        채점 결과 딕셔너리
    """
    if not ground_truth or not response:
        return {"score": 0, "method": "empty", "details": "Empty input"}
    
    # 한국어 키워드 추출 (조사 제거)
    def extract_korean_content(text):
        # 조사, 어미 패턴 제거 (간단한 버전)
        particles = r'(은|는|이|가|을|를|에|에서|으로|로|의|와|과|도|만|까지|부터|께서|에게|한테)'
        text = re.sub(particles, '', text)
        
        # 한국어 단어 추출 (2글자 이상)
        korean_words = re.findall(r'[가-힣]{2,}', text)
        return set(korean_words)
    
    gt_words = extract_korean_content(ground_truth)
    resp_words = extract_korean_content(response)
    
    if not gt_words:
        return {"score": 0, "method": "no_korean_content", "details": "No Korean content in ground truth"}
    
    # Jaccard 유사도 계산
    intersection = gt_words.intersection(resp_words)
    union = gt_words.union(resp_words)
    
    jaccard_score = len(intersection) / len(union) if union else 0
    
    # 길이 페널티/보너스 (너무 짧거나 긴 답변 보정)
    len_ratio = len(response) / len(ground_truth) if ground_truth else 0
    if 0.5 <= len_ratio <= 2.0:  # 적절한 길이
        length_bonus = 1.0
    elif len_ratio < 0.5:  # 너무 짧음
        length_bonus = 0.8
    else:  # 너무 김
        length_bonus = 0.9
    
    final_score = jaccard_score * length_bonus
    
    return {
        "score": final_score,
        "method": "korean_content_similarity",
        "details": f"Jaccard={jaccard_score:.2f}, Length_ratio={len_ratio:.2f}, Common_words={len(intersection)}"
    }

# 통합 채점 함수
def score_task(task_type: str, ground_truth: str, response: str, **kwargs) -> Dict[str, Any]:
    """
    태스크 타입에 따른 통합 채점 함수
    
    Args:
        task_type: 태스크 타입 (fact, reason, summary, format, code, ko)
        ground_truth: 정답
        response: 모델 응답
        **kwargs: 각 채점기별 추가 옵션
    
    Returns:
        채점 결과 딕셔너리
    """
    scorers = {
        'fact': score_fact,
        'reason': score_reason,
        'summary': score_summary,
        'format': score_format,
        'code': score_code,
        'ko': score_korean,
        'creative': score_summary,  # 창의적 문제는 요약 채점기 사용
        'analysis': score_reason,   # 분석 문제는 추론 채점기 사용
        'philosophy': score_reason, # 철학 문제는 추론 채점기 사용
        'prediction': score_reason  # 예측 문제는 추론 채점기 사용
    }
    
    scorer_func = scorers.get(task_type, score_fact)  # 기본은 사실 채점기
    
    try:
        return scorer_func(ground_truth, response, **kwargs)
    except Exception as e:
        return {"score": 0, "method": "scorer_error", "details": f"Scoring error: {e}"}