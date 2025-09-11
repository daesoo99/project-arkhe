# -*- coding: utf-8 -*-
"""
Project Arkhē - Integration Module
시스템 통합 모듈 (Phase 3 완성)

플러그인 시스템 + 실험 프레임워크 + 모델 레지스트리 통합
"""

from .plugin_experiment_adapter import (
    PluginExperimentAdapter,
    IntegratedExperimentResult,
    create_plugin_experiment_adapter,
    quick_experiment_run
)

__all__ = [
    "PluginExperimentAdapter",
    "IntegratedExperimentResult", 
    "create_plugin_experiment_adapter",
    "quick_experiment_run"
]