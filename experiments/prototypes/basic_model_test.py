# -*- coding: utf-8 -*-
"""
기본 모델 성능 점검 테스트
각 모델의 수학 능력과 언어 이해력을 개별적으로 테스트
"""

import sys
import time
import json
from typing import List, Dict, Any
sys.path.append('.')

# Registry 시스템 사용으로 하드코딩 제거
from src.registry.model_registry import get_model_registry

def test_basic_arithmetic():
    """기본 사칙연산 테스트"""
    tests = [
        {"question": "What is 15 - 7?", "expected": "8"},
        {"question": "What is 25 * 0.20?", "expected": "5"},
        {"question": "What is 25 - 5?", "expected": "20"},
        {"question": "What is 240 / 140?", "expected": "1.714"},
        {"question": "What is 2 + 2?", "expected": "4"},
        {"question": "What is 36 / 4?", "expected": "9"},
    ]
    return tests

def test_word_problems():
    """간단한 word problem 테스트"""
    tests = [
        {"question": "Sarah has 15 apples. She gives away 7. How many are left?", "expected": "8"},
        {"question": "A shirt costs $25. With 20% discount, what is the final price?", "expected": "20"},
        {"question": "If 4 friends share 36 chocolates equally, how many does each get?", "expected": "9"},
    ]
    return tests

def test_multi_step():
    """다단계 계산 테스트"""
    tests = [
        {"question": "Rectangle is 12m by 8m. Perimeter is?", "expected": "40"},
        {"question": "Perimeter 40m, fence costs $3 per meter. Total cost?", "expected": "120"},
        {"question": "Two trains 240 miles apart, speeds 60 and 80 mph. Meeting time in hours?", "expected": "1.714"},
    ]
    return tests

def test_korean_understanding():
    """한국어 이해력 테스트"""
    tests = [
        {"question": "15에서 7을 빼면?", "expected": "8"},
        {"question": "25달러에서 20% 할인하면 최종 가격은?", "expected": "20"},
        {"question": "사과 15개에서 7개 주면 몇개 남아?", "expected": "8"},
    ]
    return tests

class BasicModelTester:
    """기본 모델 테스터 - Registry 기반 (하드코딩 제거)"""
    
    def __init__(self, environment: str = "development"):
        print(">>> 모델 로딩 중... (Registry 기반)")
        
        # Registry를 통한 설정 기반 모델 로딩
        self.registry = get_model_registry(environment)
        
        # 설정에서 정의된 모든 모델 로딩 (하드코딩 제거!)
        available_models = self.registry.list_available_models()
        self.models = {}
        
        # 각 티어별 모델을 이름으로 매핑
        for tier, model_name in available_models.items():
            self.models[model_name] = self.registry.get_model(tier)
        
        # 설정 정보 출력
        print("  사용 가능한 모델:")
        for model_name in self.models.keys():
            print(f"    - {model_name}")
        print(">>> 모든 모델 로딩 완료 (Registry 기반)")
    
    def extract_answer(self, response_text: str) -> str:
        """답변에서 숫자 추출"""
        import re
        
        # 다양한 패턴으로 숫자 찾기
        patterns = [
            r'답[은는]?\s*([+-]?\d+(?:\.\d+)?)',
            r'answer[is:]*\s*([+-]?\d+(?:\.\d+)?)',
            r'result[is:]*\s*([+-]?\d+(?:\.\d+)?)',
            r'=\s*([+-]?\d+(?:\.\d+)?)',
            r'([+-]?\d+(?:\.\d+)?)\s*$',
            r'([+-]?\d+(?:\.\d+)?)\s*[개달러원]',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # 마지막 숫자 찾기
        numbers = re.findall(r'([+-]?\d+(?:\.\d+)?)', response_text)
        if numbers:
            return numbers[-1]
        
        return ""
    
    def is_correct(self, predicted: str, expected: str) -> bool:
        """정답 여부 확인"""
        if not predicted:
            return False
        
        try:
            pred_num = float(predicted)
            exp_num = float(expected)
            return abs(pred_num - exp_num) < 0.01  # 소수점 오차 허용
        except:
            return predicted.strip().lower() == expected.strip().lower()
    
    def test_model(self, model_name: str, model, test_cases: List[Dict], test_type: str):
        """개별 모델 테스트"""
        print(f"\n=== {model_name} - {test_type} ===")
        
        results = []
        for i, test in enumerate(test_cases):
            question = test["question"]
            expected = test["expected"]
            
            # 다양한 프롬프트 스타일 테스트
            prompts = [
                f"{question}",
                f"Question: {question}\nAnswer:",
                f"{question}\n\nProvide only the numerical answer:",
                f"Solve: {question}\nFinal answer:"
            ]
            
            best_result = None
            for prompt_idx, prompt in enumerate(prompts):
                try:
                    start_time = time.time()
                    response = model.generate(prompt, temperature=0.1, max_tokens=100)
                    
                    if isinstance(response, dict):
                        answer_text = response.get("response", "").strip()
                    else:
                        answer_text = str(response).strip()
                    
                    predicted = self.extract_answer(answer_text)
                    correct = self.is_correct(predicted, expected)
                    time_ms = int((time.time() - start_time) * 1000)
                    
                    result = {
                        "question": question,
                        "expected": expected,
                        "predicted": predicted,
                        "correct": correct,
                        "full_response": answer_text,
                        "prompt_type": prompt_idx,
                        "time_ms": time_ms
                    }
                    
                    if correct or best_result is None:
                        best_result = result
                    
                    if correct:
                        break  # 정답이면 다음 프롬프트 시도 안함
                        
                except Exception as e:
                    result = {
                        "question": question,
                        "expected": expected,
                        "predicted": "",
                        "correct": False,
                        "full_response": f"ERROR: {str(e)}",
                        "prompt_type": prompt_idx,
                        "time_ms": 0
                    }
                    if best_result is None:
                        best_result = result
            
            results.append(best_result)
            
            # 실시간 출력
            status = "✅" if best_result["correct"] else "❌"
            print(f"  {status} {question[:40]}... → {best_result['predicted']} (expected: {expected})")
            if not best_result["correct"]:
                print(f"      전체 응답: {best_result['full_response'][:80]}...")
        
        # 요약 통계
        correct_count = sum(1 for r in results if r["correct"])
        accuracy = correct_count / len(results)
        avg_time = sum(r["time_ms"] for r in results) / len(results)
        
        print(f"  📊 정확도: {accuracy:.1%} ({correct_count}/{len(results)})")
        print(f"  ⏱️  평균 시간: {avg_time:.0f}ms")
        
        return results

def run_comprehensive_test(environment: str = "development"):
    """종합 테스트 실행 - Registry 기반"""
    print("=" * 80)
    print(">>> 기본 모델 성능 점검 테스트 (Registry 기반)")
    print(f">>> 환경: {environment}")
    print("=" * 80)
    
    tester = BasicModelTester(environment)
    
    # 테스트 케이스 준비
    test_suites = [
        ("기본 사칙연산", test_basic_arithmetic()),
        ("간단한 Word Problem", test_word_problems()),
        ("다단계 계산", test_multi_step()),
        ("한국어 이해력", test_korean_understanding())
    ]
    
    all_results = {}
    
    # 각 모델별로 모든 테스트 수행
    for model_name, model in tester.models.items():
        print(f"\n🤖 {model_name} 테스트 시작...")
        all_results[model_name] = {}
        
        for test_name, test_cases in test_suites:
            results = tester.test_model(model_name, model, test_cases, test_name)
            all_results[model_name][test_name] = results
    
    # 종합 분석
    print_comprehensive_analysis(all_results)
    
    # 결과 저장
    save_test_results(all_results)
    
    return all_results

def print_comprehensive_analysis(results: Dict):
    """종합 분석 출력"""
    print(f"\n" + "=" * 80)
    print("📊 종합 분석 결과")
    print("=" * 80)
    
    # 모델별 전체 성능 요약
    print(f"\n🎯 모델별 전체 성능:")
    for model_name in results.keys():
        total_correct = 0
        total_tests = 0
        total_time = 0
        
        for test_type, test_results in results[model_name].items():
            for result in test_results:
                total_correct += int(result["correct"])
                total_tests += 1
                total_time += result["time_ms"]
        
        accuracy = total_correct / total_tests if total_tests > 0 else 0
        avg_time = total_time / total_tests if total_tests > 0 else 0
        
        print(f"  {model_name}: {accuracy:.1%} ({total_correct}/{total_tests}) - {avg_time:.0f}ms")
    
    # 테스트 유형별 분석
    print(f"\n📋 테스트 유형별 분석:")
    test_types = list(next(iter(results.values())).keys())
    
    for test_type in test_types:
        print(f"\n  📝 {test_type}:")
        for model_name in results.keys():
            test_results = results[model_name][test_type]
            correct = sum(1 for r in test_results if r["correct"])
            total = len(test_results)
            accuracy = correct / total if total > 0 else 0
            print(f"    {model_name}: {accuracy:.1%} ({correct}/{total})")
    
    # 특별한 패턴 분석
    print(f"\n🔍 주요 발견사항:")
    
    # llama3:8b 기본 사칙연산 확인
    llama_basic = results.get("llama3:8b", {}).get("기본 사칙연산", [])
    if llama_basic:
        basic_correct = sum(1 for r in llama_basic if r["correct"])
        basic_total = len(llama_basic)
        print(f"  • llama3:8b 기본 사칙연산: {basic_correct}/{basic_total} ({basic_correct/basic_total:.1%})")
        
        if basic_correct < basic_total:
            print(f"    ⚠️  Judge 모델도 기본 계산에서 실수함!")
            failed_cases = [r for r in llama_basic if not r["correct"]]
            for case in failed_cases[:2]:  # 처음 2개만 표시
                print(f"    ❌ '{case['question']}' → '{case['predicted']}' (정답: {case['expected']})")
    
    # 한국어 vs 영어 성능 비교
    for model_name in results.keys():
        english_results = results[model_name].get("간단한 Word Problem", [])
        korean_results = results[model_name].get("한국어 이해력", [])
        
        if english_results and korean_results:
            eng_acc = sum(1 for r in english_results if r["correct"]) / len(english_results)
            kor_acc = sum(1 for r in korean_results if r["correct"]) / len(korean_results)
            
            print(f"  • {model_name} 언어별 성능: 영어 {eng_acc:.1%} vs 한국어 {kor_acc:.1%}")

def save_test_results(results: Dict):
    """테스트 결과 저장"""
    timestamp = int(time.time())
    filename = f"results/basic_model_test_{timestamp}.json"
    
    try:
        import os
        os.makedirs("results", exist_ok=True)
        
        # JSON 직렬화 가능한 형태로 변환
        serializable_results = {
            "test_info": {
                "timestamp": timestamp,
                "models_tested": list(results.keys()),
                "test_types": list(next(iter(results.values())).keys())
            },
            "results": results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 상세 결과 저장됨: {filename}")
        
    except Exception as e:
        print(f"❌ 결과 저장 실패: {e}")

if __name__ == "__main__":
    print(">>> 기본 모델 성능 점검을 시작합니다... (Registry 기반)")
    print(">>> 각 모델의 수학 능력과 언어 이해력을 개별 테스트합니다")
    print(">>> 예상 소요 시간: 5-10분")
    
    # 환경별 테스트 옵션
    print("\n>>> 테스트 환경 선택:")
    print("  1. development (빠른 테스트)")
    print("  2. test (중간 성능)")
    print("  3. production (고성능, 시간 오래 걸림)")
    
    choice = input("\n환경 선택 (1-3, 기본값=1): ").strip() or "1"
    environments = {"1": "development", "2": "test", "3": "production"}
    environment = environments.get(choice, "development")
    
    print(f"\n>>> {environment} 환경으로 테스트 시작...")
    results = run_comprehensive_test(environment)
    
    print(f"\n>>> 모든 테스트 완료! (Registry 기반)")
    print(f">>> 위 분석을 통해 각 모델의 강약점을 파악할 수 있습니다")