# Context-Aware Multi-Room Chat System
**지능형 대화 컨텍스트 관리 및 원칙 보존 시스템**

---

## 🎯 프로젝트 개요

### 핵심 문제
- **컨텍스트 희석**: 긴 대화에서 초기 설정과 원칙들이 점진적으로 무시됨
- **주제 혼재**: 여러 주제가 한 대화에 뒤섞여 집중도와 일관성 저하
- **원칙 손실**: 대화 초반에 정한 중요한 규칙들이 나중에 잊혀짐
- **맥락 단절**: 대화가 길어지면서 전체적인 흐름과 목적 상실

### 핵심 해결책
**주제별, 시간별, 목적별로 자동 분리되는 지능형 채팅룸 시스템**

---

## 🏗️ 시스템 아키텍처

### 1. **Room Management Layer**
```
Main Hub
├── 📋 Project Rooms (프로젝트별)
│   ├── 🔬 Project-Arkhe
│   ├── 💼 Job-Search  
│   └── 🎨 Creative-Projects
├── 📚 Topic Rooms (주제별)
│   ├── 💻 Coding-Help
│   ├── 📖 Learning-Discussion
│   └── 💡 Brainstorming
├── ⏰ Time-Based Rooms (시간별)
│   ├── 🌅 Daily-Planning
│   ├── 📅 Weekly-Review
│   └── 🎯 Session-Focused
└── 🧠 Context Rooms (컨텍스트별)
    ├── 📝 Rule-Setting
    ├── 🔄 Follow-up
    └── 🚨 Urgent-Issues
```

### 2. **Intelligent Router System**
```python
class ConversationRouter:
    def analyze_message(self, message: str) -> RoomRecommendation:
        # 메시지 분석
        - 주제 분류 (NLP 기반)
        - 시급성 판단
        - 연관 프로젝트 식별
        - 컨텍스트 연속성 체크
        
    def suggest_room_switch(self, current_room: str, message: str) -> bool:
        # 방 전환 필요성 판단
        - 주제 변화 감지
        - 컨텍스트 불일치 체크
        - 원칙 충돌 감지
```

### 3. **Context Preservation Engine**
```python
class ContextManager:
    def preserve_principles(self, room_id: str) -> List[Principle]:
        # 각 방별 고유 원칙 관리
        - 설정된 규칙 저장
        - 우선순위 관리  
        - 충돌 해결 메커니즘
        
    def maintain_thread_continuity(self, messages: List[Message]) -> Context:
        # 대화 연속성 유지
        - 이전 메시지 요약
        - 핵심 결정사항 추출
        - 미해결 이슈 추적
```

---

## 🎯 핵심 기능들

### **1. 🔄 자동 룸 라우팅**
- **주제 변화 감지**: "이제 다른 프로젝트 얘기해보자" → 새 방 제안
- **컨텍스트 전환**: "아까 정한 규칙 무시하고..." → 원칙 방으로 이동 제안
- **시간 기반 분리**: 일일 계획 vs 장기 전략 자동 분류

### **2. 📝 원칙 보존 시스템**
```python
# 각 방별 고유 원칙 설정
room_principles = {
    "project_arkhe": [
        "새 파일 생성 금지, 기존 파일 수정 우선",
        "하드코딩 zero tolerance", 
        "TodoWrite 복잡한 작업시 필수"
    ],
    "coding_help": [
        "코드 설명보다 구현 우선",
        "에러 해결시 원인 분석 필수"
    ]
}
```

### **3. 🧠 컨텍스트 연결 시스템**
- **Cross-Room References**: "아르케 방에서 했던 실험처럼..."
- **Principle Inheritance**: 상위 방의 원칙을 하위 방에 상속
- **Context Bridging**: 관련 방들 간의 정보 연결

### **4. 📊 대화 분석 & 인사이트**
```python
class ConversationAnalytics:
    def analyze_room_efficiency(self) -> RoomStats:
        - 각 방별 목표 달성도
        - 원칙 준수율
        - 대화 집중도 점수
        
    def suggest_room_optimization(self) -> List[Suggestion]:
        - 비효율적 방 통합 제안
        - 새 방 필요성 감지
        - 원칙 업데이트 제안
```

---

## 🛠️ 기술 스택

### **Backend**
```python
# FastAPI + WebSocket for Real-time
from fastapi import FastAPI, WebSocket
from sqlalchemy import create_engine  # Room & Message Storage
from transformers import pipeline    # Topic Classification
import chromadb                     # Context Vector Storage
```

### **Frontend**  
```javascript
// React + Socket.IO for Real-time UI
import { useState, useEffect } from 'react';
import io from 'socket.io-client';
```

### **AI Components**
- **Topic Classifier**: BERT 기반 주제 분류
- **Context Analyzer**: 대화 흐름 분석
- **Principle Extractor**: 규칙 자동 추출
- **Room Recommender**: 최적 방 추천

---

## 🎯 사용 시나리오

### **시나리오 1: 프로젝트 전환**
```
User: "아르케 프로젝트 얘기 그만하고 이제 취업 준비 상담해줘"
System: 💼 "Job-Search" 방으로 이동하시겠습니까? 
        📋 현재 대화의 핵심 내용을 "Project-Arkhe" 방에 요약 저장했습니다.
```

### **시나리오 2: 원칙 충돌 감지**
```  
User: "새로운 파일 만들어줘"
System: ⚠️ "Project-Arkhe" 방 원칙 위반 감지: "새 파일 생성 금지"
        🔄 기존 파일 수정으로 진행하거나, 📝 원칙 재검토 방으로 이동하시겠습니까?
```

### **시나리오 3: 컨텍스트 연결**
```
User: "이전에 얘기했던 Shannon Entropy 방법을 여기서도 써볼까?"
System: 🔗 "Project-Arkhe" 방의 관련 내용을 가져왔습니다:
        • Shannon Entropy 측정 시스템 구현됨
        • 엔트로피 균형 파이프라인 성공
        • 정보 이론 기반 최적화 완료
```

---

## 📊 예상 효과

### **✅ 해결되는 문제들**
1. **컨텍스트 희석 방지**: 각 방별 명확한 목적과 원칙 유지
2. **주제 집중도 향상**: 관련 없는 내용 자동 분리
3. **원칙 일관성**: 설정된 규칙의 지속적 준수
4. **대화 효율성**: 목적에 맞는 최적화된 상호작용

### **📈 부가 가치**
- **학습 패턴 분석**: 어떤 주제에서 가장 생산적인지 파악
- **원칙 진화**: 실제 사용 패턴 기반 원칙 개선 제안
- **크로스 프로젝트 인사이트**: 다른 방의 성공 패턴 적용

---

## 🚀 개발 단계

### **Phase 1: 프로토타입** (1-2주)
- [x] 기본 멀티룸 구조 설계
- [ ] 간단한 주제 분류 시스템
- [ ] 기본 원칙 저장/불러오기
- [ ] 웹 기반 채팅 UI

### **Phase 2: 지능화** (2-3주)  
- [ ] NLP 기반 자동 주제 분류
- [ ] 컨텍스트 연결 시스템
- [ ] 원칙 충돌 감지
- [ ] 방 전환 추천 엔진

### **Phase 3: 고도화** (3-4주)
- [ ] 대화 분석 & 인사이트
- [ ] 크로스룸 레퍼런스
- [ ] 원칙 자동 업데이트
- [ ] 모바일 앱 지원

---

## 💡 확장 가능성

### **개인용 → 팀용**
- 팀 프로젝트별 방 관리
- 역할별 접근 권한
- 협업 원칙 공유

### **일반 채팅 → 전문 도구**
- 고객 상담 도구로 확장
- 교육용 플랫폼 적용
- 기업 내부 커뮤니케이션

---

이 프로젝트는 **AI와 실제 생산성**을 결합한 정말 실용적인 솔루션이에요! 🎯