# -*- coding: utf-8 -*-
"""
Project Arkhē - Plugin Registry
플러그인 등록, 발견, 관리 시스템

CLAUDE.local 규칙 준수:
- 의존성 주입 패턴
- 싱글톤 패턴 (글로벌 레지스트리)
- 인터페이스 기반 확장성
"""

import os
import importlib
import inspect
from typing import Dict, List, Type, Optional, Any, Union
from pathlib import Path
import logging

from .interfaces import (
    IPlugin, IScorer, IAggregator, IScorerPlugin, IAggregatorPlugin,
    TaskType, PluginLoadError, PluginValidationError
)

logger = logging.getLogger(__name__)

class PluginRegistry:
    """
    플러그인 중앙 레지스트리
    
    기능:
    - 플러그인 자동 발견 및 로딩
    - 의존성 관리
    - 런타임 플러그인 교체
    - 설정 기반 플러그인 선택
    """
    
    def __init__(self):
        self._scorers: Dict[str, IScorer] = {}
        self._aggregators: Dict[str, IAggregator] = {}
        self._plugins: Dict[str, IPlugin] = {}
        self._task_scorers: Dict[TaskType, List[str]] = {}
        self._initialized = False
        
    def initialize(self, plugin_paths: Optional[List[str]] = None, config: Optional[Dict[str, Any]] = None):
        """
        플러그인 레지스트리 초기화
        
        Args:
            plugin_paths: 플러그인 검색 경로 목록
            config: 플러그인별 설정
        """
        if self._initialized:
            return
            
        config = config or {}
        plugin_paths = plugin_paths or self._get_default_plugin_paths()
        
        logger.info(f"Initializing plugin registry with paths: {plugin_paths}")
        
        # 내장 플러그인 로드
        self._load_builtin_plugins(config)
        
        # 외부 플러그인 발견 및 로드
        for path in plugin_paths:
            self._discover_plugins(path, config)
            
        # 태스크별 채점기 매핑 구축
        self._build_task_mappings()
        
        self._initialized = True
        logger.info(f"Plugin registry initialized: {len(self._scorers)} scorers, {len(self._aggregators)} aggregators")
    
    def _get_default_plugin_paths(self) -> List[str]:
        """기본 플러그인 경로 반환"""
        current_dir = Path(__file__).parent
        return [
            str(current_dir / "scorers"),
            str(current_dir / "aggregators"),
            str(current_dir / "builtin")
        ]
    
    def _load_builtin_plugins(self, config: Dict[str, Any]):
        """내장 플러그인 로드"""
        try:
            # 기존 scorers.py를 래핑한 내장 채점기들
            from .builtin.legacy_scorers import LegacyScorerPlugin
            from .builtin.standard_aggregators import StandardAggregatorsPlugin
            
            # 레거시 채점기 플러그인 로드
            legacy_plugin = LegacyScorerPlugin()
            if legacy_plugin.initialize(config.get('legacy_scorer', {})):
                scorer = legacy_plugin.get_scorer()
                self.register_scorer(scorer)
                logger.info(f"Loaded builtin scorer: {scorer.name}")
            
            # 표준 집계기 플러그인 로드
            std_agg_plugin = StandardAggregatorsPlugin()
            if std_agg_plugin.initialize(config.get('standard_aggregators', {})):
                aggregators = std_agg_plugin.get_aggregators()
                for aggregator in aggregators:
                    self.register_aggregator(aggregator)
                    logger.info(f"Loaded builtin aggregator: {aggregator.name}")
                    
        except ImportError as e:
            logger.warning(f"Could not load builtin plugins: {e}")
    
    def _discover_plugins(self, plugin_path: str, config: Dict[str, Any]):
        """플러그인 디렉터리에서 플러그인 발견"""
        path = Path(plugin_path)
        if not path.exists():
            logger.debug(f"Plugin path does not exist: {plugin_path}")
            return
            
        for py_file in path.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
                
            try:
                self._load_plugin_file(py_file, config)
            except Exception as e:
                logger.error(f"Failed to load plugin {py_file}: {e}")
    
    def _load_plugin_file(self, file_path: Path, config: Dict[str, Any]):
        """플러그인 파일 로드"""
        module_name = file_path.stem
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        
        if spec is None or spec.loader is None:
            raise PluginLoadError(f"Cannot load module spec from {file_path}")
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 플러그인 클래스 찾기
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, IPlugin) and 
                obj is not IPlugin and
                not inspect.isabstract(obj)):
                
                try:
                    plugin = obj()
                    plugin_config = config.get(plugin.name, {})
                    
                    if plugin.initialize(plugin_config):
                        self._register_plugin(plugin)
                        logger.info(f"Loaded plugin: {plugin.name} v{plugin.version}")
                    else:
                        logger.warning(f"Plugin initialization failed: {plugin.name}")
                        
                except Exception as e:
                    logger.error(f"Failed to instantiate plugin {name}: {e}")
    
    def _register_plugin(self, plugin: IPlugin):
        """플러그인 등록"""
        self._plugins[plugin.name] = plugin
        
        # 채점기 플러그인 처리
        if isinstance(plugin, IScorerPlugin):
            scorer = plugin.get_scorer()
            self.register_scorer(scorer)
            
        # 집계기 플러그인 처리
        if isinstance(plugin, IAggregatorPlugin):
            aggregator = plugin.get_aggregator()
            self.register_aggregator(aggregator)
    
    def register_scorer(self, scorer: IScorer):
        """채점기 등록"""
        self._validate_scorer(scorer)
        self._scorers[scorer.name] = scorer
        logger.debug(f"Registered scorer: {scorer.name}")
    
    def register_aggregator(self, aggregator: IAggregator):
        """집계기 등록"""
        self._validate_aggregator(aggregator)
        self._aggregators[aggregator.name] = aggregator
        logger.debug(f"Registered aggregator: {aggregator.name}")
    
    def _validate_scorer(self, scorer: IScorer):
        """채점기 검증"""
        if not scorer.name or not scorer.version:
            raise PluginValidationError("Scorer must have name and version")
            
        if not scorer.supported_tasks:
            raise PluginValidationError("Scorer must support at least one task type")
    
    def _validate_aggregator(self, aggregator: IAggregator):
        """집계기 검증"""
        if not aggregator.name or not aggregator.version:
            raise PluginValidationError("Aggregator must have name and version")
    
    def _build_task_mappings(self):
        """태스크별 채점기 매핑 구축"""
        self._task_scorers.clear()
        
        for scorer_name, scorer in self._scorers.items():
            for task_type in scorer.supported_tasks:
                if task_type not in self._task_scorers:
                    self._task_scorers[task_type] = []
                self._task_scorers[task_type].append(scorer_name)
    
    def get_scorer(self, name: str) -> Optional[IScorer]:
        """이름으로 채점기 반환"""
        return self._scorers.get(name)
    
    def get_aggregator(self, name: str) -> Optional[IAggregator]:
        """이름으로 집계기 반환"""
        return self._aggregators.get(name)
    
    def get_scorers_for_task(self, task_type: TaskType) -> List[IScorer]:
        """태스크 타입에 대응하는 채점기들 반환"""
        scorer_names = self._task_scorers.get(task_type, [])
        return [self._scorers[name] for name in scorer_names if name in self._scorers]
    
    def list_scorers(self) -> Dict[str, Dict[str, Any]]:
        """등록된 채점기 목록과 메타데이터 반환"""
        return {name: scorer.get_metadata() for name, scorer in self._scorers.items()}
    
    def list_aggregators(self) -> Dict[str, Dict[str, Any]]:
        """등록된 집계기 목록과 메타데이터 반환"""
        return {name: agg.get_metadata() for name, agg in self._aggregators.items()}
    
    def list_plugins(self) -> Dict[str, Dict[str, Any]]:
        """등록된 플러그인 목록과 메타데이터 반환"""
        return {name: plugin.get_metadata() for name, plugin in self._plugins.items()}
    
    def get_task_coverage(self) -> Dict[str, List[str]]:
        """태스크별 사용 가능한 채점기 매핑 반환"""
        return {task.value: scorers for task, scorers in self._task_scorers.items()}
    
    def reload_plugin(self, plugin_name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """플러그인 재로드"""
        if plugin_name not in self._plugins:
            return False
            
        # 기존 플러그인 정리
        old_plugin = self._plugins[plugin_name]
        old_plugin.cleanup()
        
        # 관련 채점기/집계기 제거
        if isinstance(old_plugin, IScorerPlugin):
            scorer = old_plugin.get_scorer()
            if scorer.name in self._scorers:
                del self._scorers[scorer.name]
                
        if isinstance(old_plugin, IAggregatorPlugin):
            aggregator = old_plugin.get_aggregator()
            if aggregator.name in self._aggregators:
                del self._aggregators[aggregator.name]
        
        del self._plugins[plugin_name]
        
        # 태스크 매핑 재구축
        self._build_task_mappings()
        
        logger.info(f"Reloaded plugin: {plugin_name}")
        return True
    
    def shutdown(self):
        """플러그인 레지스트리 종료"""
        logger.info("Shutting down plugin registry")
        
        for plugin in self._plugins.values():
            try:
                plugin.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up plugin {plugin.name}: {e}")
        
        self._scorers.clear()
        self._aggregators.clear()
        self._plugins.clear()
        self._task_scorers.clear()
        self._initialized = False

# 글로벌 플러그인 레지스트리 (싱글톤 패턴)
_global_plugin_registry: Optional[PluginRegistry] = None

def get_plugin_registry() -> PluginRegistry:
    """글로벌 플러그인 레지스트리 반환"""
    global _global_plugin_registry
    if _global_plugin_registry is None:
        _global_plugin_registry = PluginRegistry()
    return _global_plugin_registry

def initialize_plugins(plugin_paths: Optional[List[str]] = None, config: Optional[Dict[str, Any]] = None):
    """플러그인 시스템 초기화 (편의 함수)"""
    registry = get_plugin_registry()
    registry.initialize(plugin_paths, config)