# -*- coding: utf-8 -*-
"""
Project Arkhē - 통합 실험 실행기
ExperimentRegistry 시스템을 사용한 통합 실험 관리

사용법:
python scripts/experiment_runner.py --template basic_model_test --env development
python scripts/experiment_runner.py --sweep temperature_sweep --env test
python scripts/experiment_runner.py --list-templates
"""

import argparse
import sys
import os
from pathlib import Path

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.registry.experiment_registry import get_experiment_registry
from src.registry.model_registry import get_model_registry

def list_templates(environment: str = "development"):
    """사용 가능한 템플릿 목록 출력"""
    registry = get_experiment_registry(environment)
    templates = registry.list_available_templates()
    
    print(f">>> 사용 가능한 실험 템플릿 ({environment} 환경):")
    print("=" * 60)
    for name, description in templates.items():
        print(f"TEMPLATE {name}")
        print(f"   설명: {description}")
        
        # 템플릿 상세 정보
        try:
            config = registry.get_experiment_config(name)
            print(f"   필요 역할: {', '.join(config.roles_required)}")
            print(f"   메트릭: {', '.join(config.metrics)}")
            
            # 설정 검증
            warnings = registry.validate_config(name)
            if warnings:
                print(f"   WARNING 경고: {len(warnings)}개 이슈")
            else:
                print(f"   OK 설정 검증 통과")
            
        except Exception as e:
            print(f"   ERROR 설정 오류: {e}")
        print()

def list_sweeps(environment: str = "development"):
    """사용 가능한 매개변수 스위핑 목록 출력"""
    registry = get_experiment_registry(environment)
    sweeps = registry.config.get('parameter_sweeps', {})
    
    print(f">>> 사용 가능한 매개변수 스위핑 ({environment} 환경):")
    print("=" * 60)
    for name, sweep_config in sweeps.items():
        print(f"SWEEP {name}")
        print(f"   설명: {sweep_config.get('description', 'No description')}")
        print(f"   기본 템플릿: {sweep_config['base_template']}")
        print(f"   스위핑 매개변수: {list(sweep_config['sweep_params'].keys())}")
        
        # 조합 수 계산
        total_combinations = 1
        for param_values in sweep_config['sweep_params'].values():
            total_combinations *= len(param_values)
        print(f"   총 실험 수: {total_combinations}")
        print()

def run_single_experiment(template_name: str, environment: str = "development", **overrides):
    """단일 실험 실행"""
    print(f">>> 실험 실행: {template_name} (환경: {environment})")
    
    registry = get_experiment_registry(environment)
    config = registry.get_experiment_config(template_name, **overrides)
    
    print(f">>> 실험 설정:")
    print(f"  이름: {config.name}")
    print(f"  설명: {config.description}") 
    print(f"  필요 역할: {config.roles_required}")
    print(f"  메트릭: {config.metrics}")
    
    # 실제 실험 파일 실행
    if template_name == "basic_model_test":
        # 템플릿 기반 실험 실행
        from experiments.prototypes.templated_basic_model_test import run_templated_experiment
        return run_templated_experiment(environment)
    
    elif template_name == "hierarchical_multiagent":
        # 계층적 멀티에이전트 실험 (추후 구현)
        print("⚠️ hierarchical_multiagent 템플릿 실행기 구현 예정")
        return None
    
    elif template_name == "benchmark_comparison":
        # 벤치마크 비교 실험 (추후 구현)
        print("⚠️ benchmark_comparison 템플릿 실행기 구현 예정")
        return None
    
    else:
        print(f"❌ 알 수 없는 템플릿: {template_name}")
        return None

def run_parameter_sweep(sweep_name: str, environment: str = "development"):
    """매개변수 스위핑 실험 실행"""
    print(f">>> 매개변수 스위핑 실행: {sweep_name} (환경: {environment})")
    
    registry = get_experiment_registry(environment)
    experiments = registry.generate_parameter_sweep(sweep_name)
    
    print(f">>> 총 {len(experiments)}개 실험 생성됨")
    
    # 각 실험 순차 실행 (실제로는 병렬 처리 가능)
    all_results = []
    for i, experiment_config in enumerate(experiments):
        print(f"\n>>> 실험 {i+1}/{len(experiments)} 시작: {experiment_config.name}")
        
        # 매개변수 정보 출력
        gen_params = experiment_config.get_generation_params()
        print(f"  매개변수: temp={gen_params.temperature}, tokens={gen_params.max_tokens}")
        print(f"  스위핑 조합: {experiment_config.metadata.get('sweep_combination', {})}")
        
        # 실제 실험 실행 (여기서는 시뮬레이션)
        # TODO: 실제 실험 코드 연결
        print(f"  ⚠️ 실험 실행 로직 구현 예정 (현재는 시뮬레이션)")
        
        # 결과 시뮬레이션
        simulated_result = {
            "experiment_name": experiment_config.name,
            "config_hash": experiment_config.metadata.get('config_hash'),
            "parameters": gen_params.to_dict(),
            "sweep_metadata": experiment_config.metadata,
            "status": "completed_simulation"
        }
        all_results.append(simulated_result)
        
        print(f"  ✅ 실험 {i+1} 완료")
    
    print(f"\n>>> 매개변수 스위핑 완료: {len(all_results)}개 실험 결과")
    return all_results

def validate_environment():
    """환경 설정 검증"""
    print(">>> 실험 환경 검증 중...")
    
    # ModelRegistry 검증
    try:
        model_registry = get_model_registry("development")
        models = model_registry.list_available_models()
        roles = model_registry.list_available_roles()
        print(f"OK ModelRegistry: {len(models)}개 모델, {len(roles)}개 역할")
    except Exception as e:
        print(f"ERROR ModelRegistry 오류: {e}")
        return False
    
    # ExperimentRegistry 검증
    try:
        experiment_registry = get_experiment_registry("development")
        templates = experiment_registry.list_available_templates()
        print(f"OK ExperimentRegistry: {len(templates)}개 템플릿")
    except Exception as e:
        print(f"ERROR ExperimentRegistry 오류: {e}")
        return False
    
    # 결과 디렉터리 확인
    results_dir = project_root / "results"
    if not results_dir.exists():
        results_dir.mkdir(parents=True, exist_ok=True)
        print("DIR results/ 디렉터리 생성")
    else:
        print("DIR results/ 디렉터리 존재")
    
    print(">>> 환경 검증 완료")
    return True

def main():
    parser = argparse.ArgumentParser(description="Project Arkhē 실험 실행기")
    
    # 실행 모드 선택
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list-templates", action="store_true", help="사용 가능한 템플릿 목록")
    group.add_argument("--list-sweeps", action="store_true", help="사용 가능한 매개변수 스위핑 목록")
    group.add_argument("--template", type=str, help="실행할 실험 템플릿 이름")
    group.add_argument("--sweep", type=str, help="실행할 매개변수 스위핑 이름")
    group.add_argument("--validate", action="store_true", help="환경 설정 검증")
    
    # 공통 옵션
    parser.add_argument("--env", "--environment", default="development", 
                       choices=["development", "test", "production"],
                       help="실행 환경 (기본값: development)")
    
    # 실험 오버라이드 옵션
    parser.add_argument("--temperature", type=float, help="온도 매개변수 오버라이드")
    parser.add_argument("--max-tokens", type=int, help="최대 토큰 수 오버라이드")
    
    args = parser.parse_args()
    
    # 환경 검증
    if not validate_environment():
        print("❌ 환경 검증 실패")
        return 1
    
    # 명령 실행
    if args.list_templates:
        list_templates(args.env)
        
    elif args.list_sweeps:
        list_sweeps(args.env)
        
    elif args.validate:
        print("OK 환경 검증 완료 - 실험 실행 준비됨")
        
    elif args.template:
        # 오버라이드 매개변수 준비
        overrides = {}
        if args.temperature is not None:
            overrides['generation_params'] = overrides.get('generation_params', {})
            overrides['generation_params']['temperature'] = args.temperature
        if args.max_tokens is not None:
            overrides['generation_params'] = overrides.get('generation_params', {})
            overrides['generation_params']['max_tokens'] = args.max_tokens
            
        result = run_single_experiment(args.template, args.env, **overrides)
        if result is not None:
            print("OK 실험 실행 완료")
        else:
            print("ERROR 실험 실행 실패")
            return 1
            
    elif args.sweep:
        result = run_parameter_sweep(args.sweep, args.env)
        if result:
            print("OK 매개변수 스위핑 완료")
        else:
            print("ERROR 매개변수 스위핑 실패")
            return 1
    
    return 0

if __name__ == "__main__":
    exit(main())