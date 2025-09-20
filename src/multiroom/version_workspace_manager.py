#!/usr/bin/env python3
"""
Version-Based Workspace System
하나의 채팅방 내에서 버전별 작업공간을 관리하는 시스템
Git 브랜치와 유사한 개념으로 대화 컨텍스트를 버전별로 분리 관리
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from enum import Enum
import uuid
import time
import json
from datetime import datetime


class ContextPriority(Enum):
    """컨텍스트 우선순위 레벨"""
    PERSISTENT = "persistent"    # 절대 안 바뀌는 핵심 규칙 (최고 우선순위)
    REFERENCE = "reference"      # 과거 중요 정보 (필요시만 참조)
    ACTIVE = "active"           # 현재 진행중인 대화 (실시간 우선순위)
    ARCHIVED = "archived"       # 완료된 과거 대화 (검색시만 사용)


class WorkspaceType(Enum):
    """워크스페이스 타입"""
    RESEARCH = "research"      # 연구/분석 모드
    CREATIVE = "creative"      # 창작/아이디어 모드
    DEBUG = "debug"           # 디버깅/문제해결 모드
    DESIGN = "design"         # 설계/아키텍처 모드
    LEARNING = "learning"     # 학습/교육 모드
    CUSTOM = "custom"         # 커스텀 모드


@dataclass
class WorkspaceConcept:
    """각 워크스페이스의 고유 컨셉/규칙"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    rules: List[str] = field(default_factory=list)
    personality: str = ""  # AI 성격/어조
    focus_areas: List[str] = field(default_factory=list)  # 집중 영역
    constraints: List[str] = field(default_factory=list)  # 제약사항
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "rules": self.rules,
            "personality": self.personality,
            "focus_areas": self.focus_areas,
            "constraints": self.constraints,
            "created_at": self.created_at
        }


@dataclass
class Message:
    """메시지 객체 - 우선순위 기반 컨텍스트 분리"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    role: str = "user"  # user, assistant, system
    timestamp: float = field(default_factory=time.time)
    workspace_version: str = "main"

    # 🔥 핵심 추가: 컨텍스트 우선순위
    priority: ContextPriority = ContextPriority.ACTIVE

    # 메시지 분류 메타데이터
    is_rule: bool = False           # 규칙/원칙 관련 메시지인지
    is_decision: bool = False       # 중요한 결정사항인지
    expires_at: Optional[float] = None  # 언제까지 유효한지 (임시 정보)
    tags: Set[str] = field(default_factory=set)  # 검색용 태그

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "role": self.role,
            "timestamp": self.timestamp,
            "workspace_version": self.workspace_version,
            "priority": self.priority.value,
            "is_rule": self.is_rule,
            "is_decision": self.is_decision,
            "expires_at": self.expires_at,
            "tags": list(self.tags),
            "metadata": self.metadata
        }

    def is_expired(self) -> bool:
        """메시지가 만료되었는지 확인"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def age_hours(self) -> float:
        """메시지 나이 (시간 단위)"""
        return (time.time() - self.timestamp) / 3600


@dataclass
class Workspace:
    """버전별 작업공간"""
    version: str = "v1"  # v1, v2, v3, ...
    name: str = ""
    workspace_type: WorkspaceType = WorkspaceType.CUSTOM
    concept: WorkspaceConcept = field(default_factory=WorkspaceConcept)

    # 워크스페이스별 메시지 (메인 스레드 + 워크스페이스별 추가 메시지)
    messages: List[Message] = field(default_factory=list)

    # 워크스페이스 상태
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    is_active: bool = True

    # 연관성
    parent_version: Optional[str] = None  # 어느 버전에서 파생되었는지
    related_versions: Set[str] = field(default_factory=set)  # 관련 버전들

    def add_message(self, message: Message) -> None:
        """메시지 추가 및 활동 시간 업데이트"""
        message.workspace_version = self.version
        self.messages.append(message)
        self.last_active = time.time()

    def get_context_prompt(self) -> str:
        """워크스페이스별 컨텍스트 프롬프트 생성"""
        context = f"""
=== {self.name} (Version {self.version}) ===
컨셉: {self.concept.title}
설명: {self.concept.description}
성격: {self.concept.personality}

규칙:
{chr(10).join(f"- {rule}" for rule in self.concept.rules)}

집중 영역:
{chr(10).join(f"- {area}" for area in self.concept.focus_areas)}

제약사항:
{chr(10).join(f"- {constraint}" for constraint in self.concept.constraints)}
========================================
"""
        return context.strip()


class VersionWorkspaceManager:
    """버전별 워크스페이스 관리자"""

    def __init__(self):
        self.main_thread: List[Message] = []  # 메인 대화 스레드
        self.workspaces: Dict[str, Workspace] = {}  # version -> Workspace
        self.current_version: str = "main"
        self.session_id: str = str(uuid.uuid4())

        # 기본 워크스페이스들 생성
        self._create_default_workspaces()

    def _create_default_workspaces(self) -> None:
        """기본 워크스페이스 생성"""

        # v1: Research Mode
        research_concept = WorkspaceConcept(
            title="Arkhe Research Mode",
            description="Focus on Project Arkhe LLM multi-agent system research",
            rules=[
                "Strictly follow CLAUDE.local rules",
                "Absolutely no hardcoding",
                "Must record experiment results in EXPERIMENT_LOG.md",
                "Prefer extending existing structure over creating new files"
            ],
            personality="Systematic and rigorous researcher tone",
            focus_areas=["Multi-Agent systems", "Shannon Entropy", "Experiment design", "Code quality"],
            constraints=["No root level file creation", "No backup file creation"]
        )

        self.create_workspace("v1_research", "Arkhe Research Mode",
                            WorkspaceType.RESEARCH, research_concept)

        # v2: Creative Mode
        creative_concept = WorkspaceConcept(
            title="Creative Development Mode",
            description="Focus on creative and experimental idea exploration",
            rules=[
                "Unrestricted free idea exploration",
                "Experimental spirit without fear of failure",
                "Innovative approach not bound by existing frameworks"
            ],
            personality="Creative and flexible innovator tone",
            focus_areas=["New architectures", "Creative algorithms", "Experimental features"],
            constraints=["Maintain only basic security rules"]
        )

        self.create_workspace("v2_creative", "Creative Development Mode",
                            WorkspaceType.CREATIVE, creative_concept)

    def create_workspace(self, version: str, name: str,
                        workspace_type: WorkspaceType,
                        concept: WorkspaceConcept,
                        parent_version: Optional[str] = None) -> Workspace:
        """새 워크스페이스 생성"""

        workspace = Workspace(
            version=version,
            name=name,
            workspace_type=workspace_type,
            concept=concept,
            parent_version=parent_version
        )

        self.workspaces[version] = workspace
        return workspace

    def switch_workspace(self, version: str) -> Optional[Workspace]:
        """워크스페이스 전환"""
        if version == "main":
            self.current_version = "main"
            return None

        if version in self.workspaces:
            self.current_version = version
            return self.workspaces[version]

        return None

    def get_current_workspace(self) -> Optional[Workspace]:
        """현재 활성 워크스페이스 반환"""
        if self.current_version == "main":
            return None
        return self.workspaces.get(self.current_version)

    def add_message(self, content: str, role: str = "user") -> None:
        """메시지 추가 (현재 활성 워크스페이스에)"""
        message = Message(content=content, role=role)

        if self.current_version == "main":
            message.workspace_version = "main"
            self.main_thread.append(message)
        else:
            current_workspace = self.workspaces.get(self.current_version)
            if current_workspace:
                current_workspace.add_message(message)

    def get_smart_context(self, max_tokens: int = 2000) -> str:
        """🧠 스마트 컨텍스트: 우선순위 기반으로 정보 혼재 방지"""
        context_parts = []
        current_workspace = self.get_current_workspace()

        # 1. 🏛️ PERSISTENT CONTEXT (항상 포함, 최고 우선순위)
        if current_workspace:
            persistent_msgs = [msg for msg in current_workspace.messages
                             if msg.priority == ContextPriority.PERSISTENT]
            if persistent_msgs:
                context_parts.append("=== CORE PRINCIPLES (ALWAYS ACTIVE) ===")
                for msg in persistent_msgs:
                    context_parts.append(f"{msg.role}: {msg.content}")
                context_parts.append("")

        # 2. 🔥 ACTIVE CONTEXT (현재 진행중인 대화만!)
        context_parts.append("=== CURRENT CONVERSATION ===")

        # 메인 스레드에서 최근 ACTIVE 메시지만
        recent_main_active = [msg for msg in self.main_thread[-10:]
                            if msg.priority == ContextPriority.ACTIVE and not msg.is_expired()]
        for msg in recent_main_active[-3:]:  # 최근 3개만
            context_parts.append(f"{msg.role}: {msg.content}")

        # 현재 워크스페이스에서 최근 ACTIVE 메시지만
        if current_workspace:
            recent_workspace_active = [msg for msg in current_workspace.messages
                                     if msg.priority == ContextPriority.ACTIVE and not msg.is_expired()]
            for msg in recent_workspace_active[-5:]:  # 최근 5개만
                context_parts.append(f"{msg.role}: {msg.content}")
            context_parts.append("")

        # 3. 📚 REFERENCE CONTEXT (필요시만 간략히 요약)
        reference_msgs = []
        if current_workspace:
            reference_msgs = [msg for msg in current_workspace.messages
                            if msg.priority == ContextPriority.REFERENCE and msg.is_decision]

        if reference_msgs:
            context_parts.append("=== IMPORTANT PAST DECISIONS (REFERENCE ONLY) ===")
            for msg in reference_msgs[-3:]:  # 최근 중요 결정 3개만
                # 요약된 형태로만 제공
                summary = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                age = int(msg.age_hours())
                context_parts.append(f"[{age}h ago] {summary}")

        return "\n".join(context_parts)

    def add_message_smart(self, content: str, role: str = "user",
                         priority: ContextPriority = ContextPriority.ACTIVE,
                         is_rule: bool = False, is_decision: bool = False,
                         expires_hours: Optional[float] = None,
                         tags: Set[str] = None) -> None:
        """스마트 메시지 추가 - 우선순위와 만료시간 자동 분류"""

        message = Message(
            content=content,
            role=role,
            priority=priority,
            is_rule=is_rule,
            is_decision=is_decision,
            tags=tags or set()
        )

        # 만료시간 설정
        if expires_hours:
            message.expires_at = time.time() + (expires_hours * 3600)

        # 자동 우선순위 판단
        if is_rule or "절대" in content or "반드시" in content or "금지" in content:
            message.priority = ContextPriority.PERSISTENT
            message.is_rule = True
        elif "결정" in content or "DECISION" in content or is_decision:
            message.priority = ContextPriority.REFERENCE
            message.is_decision = True
        elif message.age_hours() > 24:  # 24시간 지난 메시지
            message.priority = ContextPriority.REFERENCE

        # 워크스페이스별로 추가
        if self.current_version == "main":
            message.workspace_version = "main"
            self.main_thread.append(message)
        else:
            current_workspace = self.workspaces.get(self.current_version)
            if current_workspace:
                current_workspace.add_message(message)

    def auto_archive_old_messages(self, hours_threshold: float = 48) -> None:
        """오래된 ACTIVE 메시지를 자동으로 REFERENCE나 ARCHIVED로 변경"""
        current_time = time.time()

        # 메인 스레드 정리
        for msg in self.main_thread:
            if msg.priority == ContextPriority.ACTIVE and msg.age_hours() > hours_threshold:
                if msg.is_decision or msg.is_rule:
                    msg.priority = ContextPriority.REFERENCE
                else:
                    msg.priority = ContextPriority.ARCHIVED

        # 모든 워크스페이스 정리
        for workspace in self.workspaces.values():
            for msg in workspace.messages:
                if msg.priority == ContextPriority.ACTIVE and msg.age_hours() > hours_threshold:
                    if msg.is_decision or msg.is_rule:
                        msg.priority = ContextPriority.REFERENCE
                    else:
                        msg.priority = ContextPriority.ARCHIVED

    def list_workspaces(self) -> Dict[str, Dict[str, Any]]:
        """워크스페이스 목록 반환"""
        workspaces_info = {}

        for version, workspace in self.workspaces.items():
            workspaces_info[version] = {
                "name": workspace.name,
                "type": workspace.workspace_type.value,
                "concept_title": workspace.concept.title,
                "is_active": workspace.is_active,
                "message_count": len(workspace.messages),
                "last_active": workspace.last_active,
                "created_at": workspace.created_at
            }

        return workspaces_info

    def clone_workspace(self, source_version: str, new_version: str,
                       new_name: str, concept_changes: Dict[str, Any] = None) -> Optional[Workspace]:
        """기존 워크스페이스를 복제하여 새 버전 생성"""
        if source_version not in self.workspaces:
            return None

        source_workspace = self.workspaces[source_version]

        # 기존 컨셉 복제
        new_concept = WorkspaceConcept(
            title=source_workspace.concept.title,
            description=source_workspace.concept.description,
            rules=source_workspace.concept.rules.copy(),
            personality=source_workspace.concept.personality,
            focus_areas=source_workspace.concept.focus_areas.copy(),
            constraints=source_workspace.concept.constraints.copy()
        )

        # 변경사항 적용
        if concept_changes:
            for key, value in concept_changes.items():
                if hasattr(new_concept, key):
                    setattr(new_concept, key, value)

        return self.create_workspace(
            new_version, new_name, source_workspace.workspace_type,
            new_concept, parent_version=source_version
        )

    def export_session(self) -> Dict[str, Any]:
        """전체 세션 데이터 내보내기"""
        return {
            "session_id": self.session_id,
            "main_thread": [msg.to_dict() for msg in self.main_thread],
            "workspaces": {
                version: {
                    "version": ws.version,
                    "name": ws.name,
                    "workspace_type": ws.workspace_type.value,
                    "concept": ws.concept.to_dict(),
                    "messages": [msg.to_dict() for msg in ws.messages],
                    "created_at": ws.created_at,
                    "last_active": ws.last_active,
                    "parent_version": ws.parent_version,
                    "related_versions": list(ws.related_versions)
                }
                for version, ws in self.workspaces.items()
            },
            "current_version": self.current_version,
            "exported_at": time.time()
        }


# 사용 예시 및 테스트 코드
if __name__ == "__main__":
    # 워크스페이스 매니저 생성
    manager = VersionWorkspaceManager()

    # Start conversation in main thread
    manager.add_message("Hello! I want to talk about Project Arkhe", "user")
    manager.add_message("Yes! Let's talk about the Arkhe project.", "assistant")

    # Switch to research mode
    research_workspace = manager.switch_workspace("v1_research")
    manager.add_message("I want to improve Multi-Agent system performance", "user")
    manager.add_message("Let's approach this systematically in research mode. Checking EXPERIMENT_LOG...", "assistant")

    # Create new debug mode
    debug_concept = WorkspaceConcept(
        title="Debugging Focus Mode",
        description="Focus on code error resolution and performance optimization",
        rules=["Step-by-step debugging", "Log analysis first", "Reproducible testing"],
        personality="Calm and logical problem solver tone"
    )

    debug_workspace = manager.create_workspace(
        "v3_debug", "Debug Focus Mode",
        WorkspaceType.DEBUG, debug_concept
    )

    # 워크스페이스 목록 확인
    print("Available Workspaces:")
    for version, info in manager.list_workspaces().items():
        print(f"  {version}: {info['name']} ({info['concept_title']})")

    # 전체 컨텍스트 출력
    print("\nCurrent Context:")
    print(manager.get_full_context())