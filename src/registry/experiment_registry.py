# -*- coding: utf-8 -*-
"""
Project Arkhē - Experiment Configuration Registry
실험 설정 템플릿 중앙집중식 관리 및 매개변수 스위핑 지원

CLAUDE.local 규칙 준수:
- 하드코딩 Zero Tolerance
- 인터페이스 우선 설계  
- 느슨한 결합
"""

import os
import yaml
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
from copy import deepcopy
import time
import hashlib

@dataclass
class GenerationParams:
    """LLM 생성 매개변수"""
    temperature: float = 0.1
    max_tokens: int = 200
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens, 
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty
        }

@dataclass
class ExperimentConfig:
    """실험 설정 데이터클래스"""
    name: str
    description: str
    roles_required: List[str]
    generation_params: Dict[str, Any]
    metrics: List[str]
    environment: str = "development"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_generation_params(self, role: str = "default") -> GenerationParams:
        """역할별 생성 매개변수 반환"""
        if role in self.generation_params:
            params = self.generation_params[role]
        elif f"{role}_temperature" in self.generation_params:
            # 역할별 매개변수가 분리되어 있는 경우
            params = {
                "temperature": self.generation_params.get(f"{role}_temperature", 0.1),
                "max_tokens": self.generation_params.get(f"{role}_max_tokens", 200),
            }
        else:
            params = self.generation_params
            
        return GenerationParams(
            temperature=params.get("temperature", 0.1),
            max_tokens=params.get("max_tokens", 200),
            top_p=params.get("top_p", 1.0),
            frequency_penalty=params.get("frequency_penalty", 0.0),
            presence_penalty=params.get("presence_penalty", 0.0)
        )

class ExperimentRegistry:
    """
    실험 설정 템플릿 레지스트리
    
    기능:
    - YAML 기반 실험 템플릿 로드
    - 환경별 설정 오버라이드
    - 매개변수 스위핑 지원
    - 실험 메타데이터 관리
    """
    
    def __init__(self, config_path: Optional[str] = None, environment: str = "development"):
        """
        Args:
            config_path: experiments.yaml 경로 (None이면 자동 탐지)
            environment: 실행 환경 (development/test/production)
        """
        self.environment = environment
        self.config_path = config_path or self._find_config_path()
        self.config = self._load_config()
        self._templates_cache = {}
        
    def _find_config_path(self) -> str:
        """config/experiments.yaml 자동 탐지"""
        current = Path(__file__).parent
        
        # 프로젝트 루트까지 올라가면서 찾기
        for _ in range(5):
            config_path = current / "config" / "experiments.yaml"
            if config_path.exists():
                return str(config_path)
            current = current.parent
            
        # 환경변수에서 찾기
        env_path = os.getenv("ARKHE_EXPERIMENT_CONFIG")
        if env_path and Path(env_path).exists():
            return env_path
            
        raise FileNotFoundError("config/experiments.yaml not found. Please create it or set ARKHE_EXPERIMENT_CONFIG")
    
    def _load_config(self) -> Dict[str, Any]:
        """YAML 설정 로드 및 환경별 병합"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            # 환경별 오버라이드 적용
            if self.environment in config.get('environments', {}):
                env_config = config['environments'][self.environment]
                
                # defaults 병합
                if 'defaults' in env_config:
                    self._deep_update(config['defaults'], env_config['defaults'])
                    
                # experiment_templates 병합
                if 'experiment_templates' in env_config:
                    for template_name, template_overrides in env_config['experiment_templates'].items():
                        if template_name in config['experiment_templates']:
                            self._deep_update(config['experiment_templates'][template_name], template_overrides)
                            
            return config
        except Exception as e:
            raise RuntimeError(f"Failed to load experiment config: {e}")
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """딕셔너리 깊은 병합"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get_experiment_config(self, template_name: str, **overrides) -> ExperimentConfig:
        """
        실험 설정 반환
        
        Args:
            template_name: 실험 템플릿 이름
            **overrides: 런타임 오버라이드 매개변수
        """
        if template_name not in self.config.get('experiment_templates', {}):
            raise ValueError(f"Unknown experiment template: {template_name}")
            
        # 기본 설정과 템플릿 병합
        defaults = deepcopy(self.config.get('defaults', {}))
        template = deepcopy(self.config['experiment_templates'][template_name])
        
        # 설정 병합
        merged_config = defaults.copy()
        self._deep_update(merged_config, template)
        
        # 런타임 오버라이드 적용
        if overrides:
            self._deep_update(merged_config, overrides)
            
        # ExperimentConfig 생성
        return ExperimentConfig(
            name=template_name,
            description=merged_config.get('description', ''),
            roles_required=merged_config.get('roles_required', []),
            generation_params=merged_config.get('generation_params', {}),
            metrics=merged_config.get('metrics', []),
            environment=self.environment,
            metadata={
                'template_name': template_name,
                'timestamp': int(time.time()),
                'config_hash': self._get_config_hash(merged_config),
                'overrides': overrides
            }
        )
    
    def list_available_templates(self) -> Dict[str, str]:
        """사용 가능한 실험 템플릿 목록"""
        templates = {}
        for name, config in self.config.get('experiment_templates', {}).items():
            templates[name] = config.get('description', 'No description')
        return templates
    
    def generate_parameter_sweep(self, sweep_name: str) -> List[ExperimentConfig]:
        """
        매개변수 스위핑 실험 설정 생성
        
        Args:
            sweep_name: parameter_sweeps에 정의된 스위핑 이름
            
        Returns:
            다양한 매개변수 조합의 실험 설정 리스트
        """
        if sweep_name not in self.config.get('parameter_sweeps', {}):
            raise ValueError(f"Unknown parameter sweep: {sweep_name}")
            
        sweep_config = self.config['parameter_sweeps'][sweep_name]
        base_template = sweep_config['base_template']
        sweep_params = sweep_config['sweep_params']
        fixed_params = sweep_config.get('fixed_params', {})
        
        # 매개변수 조합 생성
        param_combinations = self._generate_combinations(sweep_params)
        
        # 각 조합에 대한 실험 설정 생성
        experiments = []
        for i, combination in enumerate(param_combinations):
            # fixed_params와 combination 병합
            merged_params = {**fixed_params, **combination}
            
            # 실험 설정 생성
            config = self.get_experiment_config(base_template, **merged_params)
            config.name = f"{base_template}_{sweep_name}_{i:03d}"
            config.metadata.update({
                'sweep_name': sweep_name,
                'sweep_combination': combination,
                'sweep_index': i
            })
            experiments.append(config)
            
        return experiments
    
    def _generate_combinations(self, sweep_params: Dict[str, List]) -> List[Dict]:
        """매개변수 조합 생성 (Cartesian product)"""
        import itertools
        
        param_names = list(sweep_params.keys())
        param_values = list(sweep_params.values())
        
        combinations = []
        for combo in itertools.product(*param_values):
            combinations.append(dict(zip(param_names, combo)))
            
        return combinations
    
    def get_output_config(self) -> Dict[str, Any]:
        """실험 결과 출력 설정 반환"""
        return self.config.get('output_config', {})
    
    def _get_config_hash(self, config: Dict[str, Any]) -> str:
        """설정의 해시값 계산 (재현성 보장)"""
        config_str = yaml.dump(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def validate_config(self, template_name: str) -> List[str]:
        """실험 설정 검증 및 경고사항 반환"""
        warnings = []
        
        try:
            config = self.get_experiment_config(template_name)
            
            # 필수 역할 존재 확인
            if not config.roles_required:
                warnings.append("No roles_required specified")
                
            # 메트릭 정의 확인
            if not config.metrics:
                warnings.append("No metrics defined")
                
            # 생성 매개변수 범위 확인
            gen_params = config.get_generation_params()
            if gen_params.temperature > 1.0 or gen_params.temperature < 0.0:
                warnings.append(f"Temperature {gen_params.temperature} outside normal range [0.0, 1.0]")
                
            if gen_params.max_tokens > 4000:
                warnings.append(f"Max tokens {gen_params.max_tokens} is very high, may cause slow/expensive calls")
                
        except Exception as e:
            warnings.append(f"Config validation error: {e}")
            
        return warnings

# 글로벌 실험 레지스트리 (싱글톤 패턴)
_global_experiment_registry = None

def get_experiment_registry(environment: str = "development") -> ExperimentRegistry:
    """글로벌 실험 레지스트리 반환"""
    global _global_experiment_registry
    if _global_experiment_registry is None or _global_experiment_registry.environment != environment:
        _global_experiment_registry = ExperimentRegistry(environment=environment)
    return _global_experiment_registry

# 편의 함수들
def get_experiment_config(template_name: str, environment: str = "development", **overrides) -> ExperimentConfig:
    """실험 설정 반환 (편의 함수)"""
    registry = get_experiment_registry(environment)
    return registry.get_experiment_config(template_name, **overrides)

def list_experiment_templates(environment: str = "development") -> Dict[str, str]:
    """실험 템플릿 목록 반환 (편의 함수)"""
    registry = get_experiment_registry(environment)
    return registry.list_available_templates()