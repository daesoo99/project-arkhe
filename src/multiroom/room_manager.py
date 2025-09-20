#!/usr/bin/env python3
"""
Multi-Room Chat System - Room Management Core
컨텍스트 인식 멀티룸 채팅 시스템의 핵심 룸 관리자
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from enum import Enum
import uuid
import time
import json
from datetime import datetime


class RoomType(Enum):
    """룸 타입 정의"""
    PROJECT = "project"
    TOPIC = "topic"
    TIME_BASED = "time_based"
    CONTEXT = "context"


class MessageType(Enum):
    """메시지 타입"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    ROOM_SWITCH = "room_switch"


@dataclass
class Message:
    """채팅 메시지 모델"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    message_type: MessageType = MessageType.USER
    timestamp: float = field(default_factory=time.time)
    room_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "message_type": self.message_type.value,
            "timestamp": self.timestamp,
            "room_id": self.room_id,
            "metadata": self.metadata
        }


@dataclass
class RoomPrinciple:
    """룸별 원칙/규칙 정의"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    priority: int = 1  # 1(highest) - 5(lowest)
    active: bool = True
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "active": self.active,
            "created_at": self.created_at
        }


@dataclass
class Room:
    """채팅 룸 모델"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    room_type: RoomType = RoomType.TOPIC
    description: str = ""
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    
    # 메시지 히스토리
    messages: List[Message] = field(default_factory=list)
    
    # 룸별 고유 원칙들
    principles: List[RoomPrinciple] = field(default_factory=list)
    
    # 메타데이터
    tags: Set[str] = field(default_factory=set)
    parent_room_id: Optional[str] = None
    child_room_ids: Set[str] = field(default_factory=set)
    
    def add_message(self, message: Message) -> None:
        """메시지 추가 및 활동 시간 업데이트"""
        message.room_id = self.id
        self.messages.append(message)
        self.last_activity = time.time()
    
    def add_principle(self, principle: RoomPrinciple) -> None:
        """원칙 추가"""
        self.principles.append(principle)
    
    def get_active_principles(self) -> List[RoomPrinciple]:
        """활성화된 원칙들 반환"""
        return [p for p in self.principles if p.active]
    
    def get_recent_messages(self, limit: int = 10) -> List[Message]:
        """최근 메시지 반환"""
        return self.messages[-limit:] if self.messages else []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "room_type": self.room_type.value,
            "description": self.description,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "message_count": len(self.messages),
            "principles": [p.to_dict() for p in self.principles],
            "tags": list(self.tags),
            "parent_room_id": self.parent_room_id,
            "child_room_ids": list(self.child_room_ids)
        }


class RoomManager:
    """룸 관리자 - 멀티룸 시스템의 핵심"""
    
    def __init__(self):
        self.rooms: Dict[str, Room] = {}
        self.current_room_id: Optional[str] = None
        
        # 기본 룸들 생성
        self._create_default_rooms()
    
    def _create_default_rooms(self) -> None:
        """기본 룸들 자동 생성"""
        
        # Main Hub Room
        main_hub = Room(
            name="Main Hub",
            room_type=RoomType.CONTEXT,
            description="메인 허브 - 모든 대화의 중심점"
        )
        main_hub.add_principle(RoomPrinciple(
            title="허브 규칙",
            description="이곳은 다른 룸으로 이동하기 위한 중앙 허브입니다."
        ))
        
        # Project Arkhe Room
        arkhe_room = Room(
            name="Project-Arkhe",
            room_type=RoomType.PROJECT,
            description="Project Arkhe 연구 및 개발 전용 룸"
        )
        arkhe_room.add_principle(RoomPrinciple(
            title="새 파일 생성 금지",
            description="기존 파일 수정을 우선시하며 새 파일 생성을 최소화합니다.",
            priority=1
        ))
        arkhe_room.add_principle(RoomPrinciple(
            title="하드코딩 Zero Tolerance",
            description="모든 설정값은 config 파일이나 환경변수를 통해 관리합니다.",
            priority=1
        ))
        arkhe_room.add_principle(RoomPrinciple(
            title="TodoWrite 복잡한 작업시 필수",
            description="복잡한 작업은 반드시 TodoWrite로 계획하고 진행합니다.",
            priority=2
        ))
        arkhe_room.tags.add("research")
        arkhe_room.tags.add("multi-agent")
        arkhe_room.tags.add("llm")
        
        # Coding Help Room
        coding_room = Room(
            name="Coding-Help",
            room_type=RoomType.TOPIC,
            description="프로그래밍 문제 해결 및 코딩 도움"
        )
        coding_room.add_principle(RoomPrinciple(
            title="코드 설명보다 구현 우선",
            description="긴 설명보다는 실제 작동하는 코드 구현을 우선시합니다.",
            priority=1
        ))
        coding_room.add_principle(RoomPrinciple(
            title="에러 해결시 원인 분석 필수",
            description="에러 발생시 근본 원인을 분석하고 해결방안을 제시합니다.",
            priority=1
        ))
        coding_room.tags.add("programming")
        coding_room.tags.add("debugging")
        
        # 룸 등록
        self.rooms[main_hub.id] = main_hub
        self.rooms[arkhe_room.id] = arkhe_room
        self.rooms[coding_room.id] = coding_room
        
        # Main Hub를 현재 룸으로 설정
        self.current_room_id = main_hub.id
    
    def create_room(self, name: str, room_type: RoomType, description: str = "", 
                   parent_room_id: Optional[str] = None) -> Room:
        """새 룸 생성"""
        room = Room(name=name, room_type=room_type, description=description, 
                   parent_room_id=parent_room_id)
        
        self.rooms[room.id] = room
        
        # 부모-자식 관계 설정
        if parent_room_id and parent_room_id in self.rooms:
            self.rooms[parent_room_id].child_room_ids.add(room.id)
        
        return room
    
    def get_room(self, room_id: str) -> Optional[Room]:
        """룸 조회"""
        return self.rooms.get(room_id)
    
    def get_current_room(self) -> Optional[Room]:
        """현재 활성 룸 반환"""
        if self.current_room_id:
            return self.rooms.get(self.current_room_id)
        return None
    
    def switch_room(self, room_id: str) -> bool:
        """룸 전환"""
        if room_id in self.rooms:
            old_room_id = self.current_room_id
            self.current_room_id = room_id
            
            # 룸 전환 메시지 생성
            if old_room_id:
                switch_message = Message(
                    content=f"Room switched from {self.rooms[old_room_id].name} to {self.rooms[room_id].name}",
                    message_type=MessageType.ROOM_SWITCH,
                    metadata={"from_room": old_room_id, "to_room": room_id}
                )
                self.rooms[room_id].add_message(switch_message)
            
            return True
        return False
    
    def add_message_to_current_room(self, content: str, message_type: MessageType = MessageType.USER) -> Optional[Message]:
        """현재 룸에 메시지 추가"""
        current_room = self.get_current_room()
        if current_room:
            message = Message(content=content, message_type=message_type)
            current_room.add_message(message)
            return message
        return None
    
    def search_rooms(self, query: str) -> List[Room]:
        """룸 검색 (이름, 설명, 태그 기준)"""
        query_lower = query.lower()
        results = []
        
        for room in self.rooms.values():
            if (query_lower in room.name.lower() or 
                query_lower in room.description.lower() or
                any(query_lower in tag.lower() for tag in room.tags)):
                results.append(room)
        
        return results
    
    def get_rooms_by_type(self, room_type: RoomType) -> List[Room]:
        """타입별 룸 목록 반환"""
        return [room for room in self.rooms.values() if room.room_type == room_type]
    
    def get_room_statistics(self) -> Dict[str, Any]:
        """룸 통계 정보"""
        stats = {
            "total_rooms": len(self.rooms),
            "rooms_by_type": {},
            "total_messages": 0,
            "most_active_room": None,
            "least_active_room": None
        }
        
        # 타입별 룸 수
        for room_type in RoomType:
            stats["rooms_by_type"][room_type.value] = len(self.get_rooms_by_type(room_type))
        
        # 활동량 통계
        if self.rooms:
            rooms_with_activity = [(room, len(room.messages)) for room in self.rooms.values()]
            rooms_with_activity.sort(key=lambda x: x[1], reverse=True)
            
            stats["total_messages"] = sum(len(room.messages) for room in self.rooms.values())
            
            if rooms_with_activity:
                stats["most_active_room"] = {
                    "name": rooms_with_activity[0][0].name,
                    "message_count": rooms_with_activity[0][1]
                }
                stats["least_active_room"] = {
                    "name": rooms_with_activity[-1][0].name,
                    "message_count": rooms_with_activity[-1][1]
                }
        
        return stats
    
    def export_room_data(self, room_id: str) -> Optional[Dict[str, Any]]:
        """룸 데이터 export (백업용)"""
        room = self.get_room(room_id)
        if room:
            return {
                "room": room.to_dict(),
                "messages": [msg.to_dict() for msg in room.messages],
                "export_timestamp": time.time()
            }
        return None
    
    def get_room_summary(self, room_id: str) -> Optional[Dict[str, Any]]:
        """룸 요약 정보"""
        room = self.get_room(room_id)
        if room:
            recent_messages = room.get_recent_messages(5)
            active_principles = room.get_active_principles()
            
            return {
                "room_info": room.to_dict(),
                "recent_activity": {
                    "message_count": len(room.messages),
                    "last_message_time": recent_messages[-1].timestamp if recent_messages else None,
                    "recent_messages": [msg.content[:100] + "..." if len(msg.content) > 100 else msg.content 
                                     for msg in recent_messages]
                },
                "active_principles": [
                    {"title": p.title, "description": p.description, "priority": p.priority}
                    for p in active_principles
                ]
            }
        return None


# 사용 예시 및 테스트
def demonstrate_room_manager():
    """룸 매니저 데모"""
    print("=== Multi-Room Chat System Demo ===")
    
    # 룸 매니저 초기화
    manager = RoomManager()
    
    # 현재 룸 확인
    current = manager.get_current_room()
    print(f"Current Room: {current.name}")
    
    # 메시지 추가
    manager.add_message_to_current_room("안녕하세요! 멀티룸 시스템을 테스트합니다.")
    manager.add_message_to_current_room("Project Arkhe 방으로 이동하고 싶습니다.", MessageType.USER)
    
    # Arkhe 룸 찾기
    arkhe_rooms = manager.search_rooms("arkhe")
    if arkhe_rooms:
        arkhe_room = arkhe_rooms[0]
        print(f"Found Arkhe room: {arkhe_room.name}")
        
        # 룸 전환
        if manager.switch_room(arkhe_room.id):
            print(f"Switched to: {manager.get_current_room().name}")
            
            # Arkhe 룸에 메시지 추가
            manager.add_message_to_current_room("Shannon Entropy 최적화 작업을 시작합니다.")
            
            # Arkhe 룸의 원칙들 확인
            principles = arkhe_room.get_active_principles()
            print(f"Active principles in {arkhe_room.name}:")
            for p in principles:
                print(f"  - {p.title}: {p.description}")
    
    # 새 룸 생성
    brainstorm_room = manager.create_room(
        name="Brainstorming",
        room_type=RoomType.TOPIC,
        description="창의적 아이디어 생성 및 브레인스토밍"
    )
    brainstorm_room.add_principle(RoomPrinciple(
        title="판단 금지",
        description="브레인스토밍 중에는 아이디어에 대한 판단을 하지 않습니다."
    ))
    
    # 통계 정보
    stats = manager.get_room_statistics()
    print(f"\nRoom Statistics:")
    print(f"  Total Rooms: {stats['total_rooms']}")
    print(f"  Total Messages: {stats['total_messages']}")
    if stats['most_active_room']:
        print(f"  Most Active: {stats['most_active_room']['name']} ({stats['most_active_room']['message_count']} messages)")


if __name__ == "__main__":
    demonstrate_room_manager()