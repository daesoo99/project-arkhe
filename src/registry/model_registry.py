# -*- coding: utf-8 -*-
"""
Project Arkhē - Model Registry
중앙집중식 모델 관리 및 하드코딩 제거

CLAUDE.local 규칙 준수:
- 하드코딩 Zero Tolerance
- 인터페이스 우선 설계  
- 느슨한 결합
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

# 상대 경로로 LLM 클래스들 import
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from llm.simple_llm import create_llm_auto, LLM

@dataclass
class ModelConfig:
    """모델 설정 데이터클래스"""
    name: str
    cost_factor: float
    context_window: int
    description: str
    provider: str

class ModelRegistry:
    """
    중앙집중식 모델 레지스트리
    
    기능:
    - YAML 설정 기반 모델 관리
    - 역할별 모델 자동 할당
    - 환경별 설정 오버라이드
    - 비용 요소 자동 계산
    """
    
    def __init__(self, config_path: Optional[str] = None, environment: str = "development"):
        """
        Args:
            config_path: models.yaml 경로 (None이면 자동 탐지)
            environment: 환경 (development/production/test)
        """
        self.environment = environment
        self.config_path = config_path or self._find_config_path()
        self.config = self._load_config()
        self._models_cache = {}
        
    def _find_config_path(self) -> str:
        """config/models.yaml 자동 탐지"""
        current = Path(__file__).parent
        
        # 프로젝트 루트까지 올라가면서 config/models.yaml 찾기
        for _ in range(5):  # 최대 5단계까지
            config_path = current / "config" / "models.yaml"
            if config_path.exists():
                return str(config_path)
            current = current.parent
            
        # 환경변수에서 찾기
        env_path = os.getenv("ARKHE_MODEL_CONFIG")
        if env_path and Path(env_path).exists():
            return env_path
            
        raise FileNotFoundError("config/models.yaml not found. Please create it or set ARKHE_MODEL_CONFIG")
    
    def _load_config(self) -> Dict[str, Any]:
        """YAML 설정 로드"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            # 환경별 오버라이드 적용
            if self.environment in config.get('environments', {}):
                env_config = config['environments'][self.environment]
                if env_config.get('roles'):
                    config['roles'].update(env_config['roles'])
                    
            return config
        except Exception as e:
            raise RuntimeError(f"Failed to load model config: {e}")
    
    def get_model_config(self, tier_or_role: str) -> ModelConfig:
        """
        모델 설정 반환
        
        Args:
            tier_or_role: 모델 티어(small/medium/large) 또는 역할(draft/review/judge)
        """
        # 역할이면 티어로 변환
        if tier_or_role in self.config.get('roles', {}):
            tier = self.config['roles'][tier_or_role]
        else:
            tier = tier_or_role
            
        if tier not in self.config.get('models', {}):
            raise ValueError(f"Unknown model tier/role: {tier_or_role}")
            
        model_data = self.config['models'][tier]
        return ModelConfig(
            name=model_data['name'],
            cost_factor=model_data['cost_factor'],
            context_window=model_data['context_window'],
            description=model_data['description'],
            provider=model_data['provider']
        )
    
    def get_model(self, tier_or_role: str) -> LLM:
        """
        LLM 인스턴스 반환 (캐시됨)
        
        Args:
            tier_or_role: 모델 티어 또는 역할
        """
        if tier_or_role not in self._models_cache:
            config = self.get_model_config(tier_or_role)
            self._models_cache[tier_or_role] = create_llm_auto(config.name)
            
        return self._models_cache[tier_or_role]
    
    def get_cost_factor(self, tier_or_role: str) -> float:
        """비용 요소 반환"""
        config = self.get_model_config(tier_or_role)
        return config.cost_factor
        
    def get_model_name(self, tier_or_role: str) -> str:
        """모델명 반환"""
        config = self.get_model_config(tier_or_role)
        return config.name
        
    def list_available_models(self) -> Dict[str, str]:
        """사용 가능한 모델 목록 반환"""
        models = {}
        for tier, data in self.config.get('models', {}).items():
            models[tier] = data['name']
        return models
        
    def list_available_roles(self) -> Dict[str, str]:
        """사용 가능한 역할 목록 반환"""
        roles = {}
        for role, tier in self.config.get('roles', {}).items():
            tier_data = self.config['models'][tier]
            roles[role] = tier_data['name']
        return roles
    
    def reload_config(self):
        """설정 다시 로드"""
        self.config = self._load_config()
        self._models_cache.clear()

# 글로벌 레지스트리 인스턴스 (싱글톤 패턴)
_global_registry = None

def get_model_registry(environment: str = "development") -> ModelRegistry:
    """글로벌 모델 레지스트리 반환"""
    global _global_registry
    if _global_registry is None:
        _global_registry = ModelRegistry(environment=environment)
    return _global_registry

def get_model_by_role(role: str, environment: str = "development") -> LLM:
    """역할별 모델 반환 (편의 함수)"""
    registry = get_model_registry(environment)
    return registry.get_model(role)

# 하위 호환성을 위한 별칭들 (기존 코드 전환시 사용)
def create_llm_by_role(role: str) -> LLM:
    """역할 기반 LLM 생성 (하위 호환성)"""
    return get_model_by_role(role)