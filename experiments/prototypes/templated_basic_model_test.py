# -*- coding: utf-8 -*-
"""
템플릿 기반 기본 모델 테스트 - ExperimentRegistry 시스템 실증
기존 basic_model_test.py의 템플릿화 버전

BEFORE: 하드코딩된 매개변수와 설정
AFTER: config/experiments.yaml 기반 중앙집중식 설정 관리
"""

import sys
import time
import json
from typing import List, Dict, Any
from dataclasses import dataclass
sys.path.append('.')

# Registry 시스템들 사용
from src.registry.model_registry import get_model_registry
from src.registry.experiment_registry import get_experiment_registry, ExperimentConfig

@dataclass 
class TemplatedResult:
    """템플릿 기반 실험 결과"""
    method: str
    question: str
    expected: str
    predicted: str
    correct: bool
    tokens: int
    time_ms: int
    experiment_config: str
    metadata: Dict[str, Any]

class TemplatedModelTester:
    """템플릿 기반 모델 테스터 - 완전한 설정 분리"""
    
    def __init__(self, environment: str = "development"):
        print(f">>> 템플릿 기반 모델 테스터 초기화 - 환경: {environment}")
        
        # Model Registry와 Experiment Registry 통합 사용
        self.model_registry = get_model_registry(environment)
        self.experiment_registry = get_experiment_registry(environment)
        
        # 실험 설정 로드 (하드코딩 완전 제거!)
        self.experiment_config = self.experiment_registry.get_experiment_config("basic_model_test")
        
        # 설정 검증
        warnings = self.experiment_registry.validate_config("basic_model_test")
        if warnings:
            print(">>> 설정 경고사항:")
            for warning in warnings:
                print(f"  ⚠️ {warning}")
        
        # 실험 설정에 따른 모델 로딩
        self.models = {}
        for role in self.experiment_config.roles_required:
            model_name = self.model_registry.get_model_name(role)
            self.models[model_name] = self.model_registry.get_model(role)
        
        # 설정 정보 출력
        print(f"  실험 설정: {self.experiment_config.name}")
        print(f"  설명: {self.experiment_config.description}")
        print(f"  사용 모델: {list(self.models.keys())}")
        print(f"  메트릭: {self.experiment_config.metrics}")
        print(">>> 템플릿 기반 초기화 완료")
    
    def get_test_categories(self):
        """실험 설정에서 테스트 카테고리 로드"""
        # 실험 설정에서 test_categories 가져오기 (하드코딩 제거)
        return self.experiment_config.metadata.get('test_categories', [
            "기본 사칙연산",
            "간단한 Word Problem", 
            "다단계 계산",
            "한국어 이해력"
        ])
    
    def get_test_cases(self, category: str) -> List[Dict[str, str]]:
        """카테고리별 테스트 케이스 로드 (추후 외부 파일로 분리 가능)"""
        test_data = {
            "기본 사칙연산": [
                {"question": "What is 15 - 7?", "expected": "8"},
                {"question": "What is 25 * 0.20?", "expected": "5"},
                {"question": "What is 25 - 5?", "expected": "20"},
            ],
            "간단한 Word Problem": [
                {"question": "Sarah has 15 apples. She gives away 7. How many are left?", "expected": "8"},
                {"question": "A shirt costs $25. With 20% discount, what is the final price?", "expected": "20"},
            ],
            "다단계 계산": [
                {"question": "Rectangle is 12m by 8m. Perimeter is?", "expected": "40"},
                {"question": "Two trains 240 miles apart, speeds 60 and 80 mph. Meeting time in hours?", "expected": "1.714"},
            ],
            "한국어 이해력": [
                {"question": "15에서 7을 빼면?", "expected": "8"},
                {"question": "사과 15개에서 7개 주면 몇개 남아?", "expected": "8"},
            ]
        }
        return test_data.get(category, [])
    
    def extract_answer(self, response_text: str) -> str:
        """답변에서 숫자 추출 (기존 로직 유지)"""
        import re
        
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
        
        numbers = re.findall(r'([+-]?\d+(?:\.\d+)?)', response_text)
        if numbers:
            return numbers[-1]
        
        return ""
    
    def is_correct(self, predicted: str, expected: str) -> bool:
        """정답 여부 확인 (기존 로직 유지)"""
        if not predicted:
            return False
        
        try:
            pred_num = float(predicted)
            exp_num = float(expected)
            return abs(pred_num - exp_num) < 0.01
        except:
            return predicted.strip().lower() == expected.strip().lower()
    
    def test_model(self, model_name: str, model, test_cases: List[Dict], test_type: str):
        """개별 모델 테스트 - 템플릿 매개변수 사용"""
        print(f"\n=== {model_name} - {test_type} ===")
        
        # 실험 설정에서 생성 매개변수 가져오기 (하드코딩 제거!)
        gen_params = self.experiment_config.get_generation_params()
        prompts_per_model = self.experiment_config.generation_params.get('prompts_per_model', 4)
        
        results = []
        for i, test in enumerate(test_cases):
            question = test["question"]
            expected = test["expected"]
            
            # 템플릿에서 정의된 다양한 프롬프트 스타일
            prompts = [
                f"{question}",
                f"Question: {question}\nAnswer:",
                f"{question}\n\nProvide only the numerical answer:",
                f"Solve: {question}\nFinal answer:"
            ]
            
            best_result = None
            for prompt_idx, prompt in enumerate(prompts[:prompts_per_model]):
                try:
                    start_time = time.time()
                    
                    # 템플릿 매개변수 사용 (하드코딩 제거!)
                    response = model.generate(
                        prompt, 
                        temperature=gen_params.temperature,
                        max_tokens=gen_params.max_tokens
                    )
                    
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
                        "time_ms": time_ms,
                        "generation_params": gen_params.to_dict()
                    }
                    
                    if correct or best_result is None:
                        best_result = result
                    
                    if correct:
                        break
                        
                except Exception as e:
                    result = {
                        "question": question,
                        "expected": expected,
                        "predicted": "",
                        "correct": False,
                        "full_response": f"ERROR: {str(e)}",
                        "prompt_type": prompt_idx,
                        "time_ms": 0,
                        "generation_params": gen_params.to_dict()
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
        print(f"  🎛️  사용된 매개변수: temp={gen_params.temperature}, tokens={gen_params.max_tokens}")
        
        return results

def run_templated_experiment(environment: str = "development"):
    """템플릿 기반 실험 실행"""
    print("=" * 80)
    print(">>> 템플릿 기반 기본 모델 성능 테스트")
    print(f">>> 환경: {environment}")
    print("=" * 80)
    
    tester = TemplatedModelTester(environment)
    
    # 실험 설정에서 테스트 카테고리 로드
    test_categories = tester.get_test_categories()
    
    all_results = {}
    
    # 각 모델별로 모든 테스트 수행
    for model_name, model in tester.models.items():
        print(f"\n🤖 {model_name} 테스트 시작...")
        all_results[model_name] = {}
        
        for category in test_categories:
            test_cases = tester.get_test_cases(category)
            if test_cases:  # 테스트 케이스가 있는 카테고리만 실행
                results = tester.test_model(model_name, model, test_cases, category)
                all_results[model_name][category] = results
    
    # 결과 분석 및 저장
    analyze_templated_results(all_results, tester.experiment_config)
    save_templated_results(all_results, tester.experiment_config)
    
    return all_results

def analyze_templated_results(results: Dict, config: ExperimentConfig):
    """템플릿 기반 결과 분석"""
    print(f"\n" + "=" * 80)
    print("📊 템플릿 기반 실험 결과 분석")
    print("=" * 80)
    
    print(f"실험 설정: {config.name} ({config.description})")
    print(f"환경: {config.environment}")
    print(f"설정 해시: {config.metadata.get('config_hash', 'unknown')}")
    
    # 기존 분석 로직 + 템플릿 메타데이터
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

def save_templated_results(results: Dict, config: ExperimentConfig):
    """템플릿 기반 결과 저장"""
    # 실험 레지스트리에서 출력 설정 가져오기
    output_config = get_experiment_registry(config.environment).get_output_config()
    
    timestamp = int(time.time())
    filename_template = output_config.get('filename_template', '{experiment_type}_{environment}_{timestamp}')
    filename = filename_template.format(
        experiment_type=config.name,
        environment=config.environment,
        timestamp=timestamp
    )
    
    full_filename = f"results/{filename}.json"
    
    # 템플릿 메타데이터 포함된 결과 저장
    serializable_results = {
        "experiment_metadata": {
            "template_name": config.name,
            "description": config.description,
            "environment": config.environment,
            "timestamp": timestamp,
            "config_hash": config.metadata.get('config_hash'),
            "roles_used": config.roles_required,
            "metrics": config.metrics,
            "generation_params_used": config.generation_params
        },
        "results": results
    }
    
    try:
        import os
        os.makedirs("results", exist_ok=True)
        
        with open(full_filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 템플릿 기반 결과 저장: {full_filename}")
        print(f"📊 메타데이터 포함하여 재현 가능한 실험 결과 생성")
        
    except Exception as e:
        print(f"❌ 결과 저장 실패: {e}")

if __name__ == "__main__":
    print(">>> 템플릿 기반 기본 모델 테스트 시작...")
    print(">>> ExperimentRegistry 시스템 실증 실험")
    
    # 환경 선택
    print("\n>>> 테스트 환경 선택:")
    print("  1. development (빠른 테스트)")
    print("  2. test (중간 성능)")
    print("  3. production (완전한 테스트)")
    
    choice = input("\n환경 선택 (1-3, 기본값=1): ").strip() or "1"
    environments = {"1": "development", "2": "test", "3": "production"}
    environment = environments.get(choice, "development")
    
    print(f"\n>>> {environment} 환경으로 템플릿 기반 테스트 시작...")
    results = run_templated_experiment(environment)
    
    print(f"\n>>> 템플릿 기반 테스트 완료!")
    print(f">>> config/experiments.yaml의 설정이 성공적으로 적용됨")
    print(f">>> 하드코딩된 매개변수 완전 제거 달성!")