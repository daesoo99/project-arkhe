"""
Project Arkhē - MultiRoom Chat Extension
멀티룸 채팅 시스템: 컨텍스트 희석 해결 및 원칙 보존 연구 모듈

이 모듈은 Project Arkhē의 확장으로, 긴 대화에서 발생하는 
컨텍스트 희석과 원칙 손실 문제를 해결하기 위한 연구입니다.

주요 구성요소:
- room_manager: 룸 생성/관리 및 메시지 라우팅
- conversation_router: AI 기반 주제 분류 및 룸 추천
- context_preservor: 원칙 보존 및 컨텍스트 연결
- multiroom_experiments: 실험 및 성능 측정
"""

from .room_manager import (
    Room, 
    RoomManager, 
    RoomType, 
    Message, 
    MessageType, 
    RoomPrinciple
)

__version__ = "0.1.0"
__author__ = "Project Arkhē Research Team"
__description__ = "Multi-Room Chat System for Context Preservation"

# MultiRoom 모듈 식별자
MULTIROOM_MODULE_ID = "arkhe_multiroom_v1"
RESEARCH_FOCUS = "context_preservation_and_principle_adherence"