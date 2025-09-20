#!/usr/bin/env python3
"""
무한 컨텍스트 관리자 - 진짜 근본 해결책
AI 모델의 컨텍스트 윈도우 한계를 우회하는 시스템
"""

import json
import time
import sqlite3
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta


@dataclass
class CompressedMemory:
    """압축된 메모리 객체"""
    id: str
    summary: str  # AI가 생성한 요약
    importance: float  # 0.0 ~ 1.0
    timestamp: float
    original_length: int
    compressed_length: int
    tags: List[str]
    context_type: str  # persistent, reference, archived
    workspace_version: str


@dataclass
class ContextWindow:
    """현재 컨텍스트 윈도우"""
    persistent_rules: List[str]  # 절대 제거되지 않음
    recent_active: List[str]     # 최근 활성 대화
    relevant_memories: List[CompressedMemory]  # 검색된 관련 기억
    total_tokens: int
    max_tokens: int = 8000  # 안전 여유분


class InfiniteContextManager:
    """무한 컨텍스트 관리자 - 장기 기억 + 압축 + 검색"""

    def __init__(self, db_path: str = "infinite_context.db", max_context_tokens: int = 8000):
        self.db_path = db_path
        self.max_context_tokens = max_context_tokens
        self.current_workspace = "main"

        # 데이터베이스 초기화
        self._init_database()

        # 현재 활성 컨텍스트
        self.persistent_rules: List[str] = []
        self.active_conversations: List[Dict] = []

    def _init_database(self):
        """SQLite 데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 압축된 기억 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS compressed_memories (
                id TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                importance REAL NOT NULL,
                timestamp REAL NOT NULL,
                original_length INTEGER NOT NULL,
                compressed_length INTEGER NOT NULL,
                tags TEXT NOT NULL,  -- JSON array
                context_type TEXT NOT NULL,
                workspace_version TEXT NOT NULL,
                embedding BLOB  -- 벡터 임베딩 (향후 시맨틱 검색용)
            )
        ''')

        # 영구 규칙 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS persistent_rules (
                id TEXT PRIMARY KEY,
                rule_text TEXT NOT NULL,
                workspace_version TEXT NOT NULL,
                created_at REAL NOT NULL,
                importance REAL NOT NULL
            )
        ''')

        # 검색 인덱스
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON compressed_memories(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_importance ON compressed_memories(importance)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_workspace ON compressed_memories(workspace_version)')

        conn.commit()
        conn.close()

    def add_persistent_rule(self, rule: str, workspace: str = "main", importance: float = 1.0):
        """영구 규칙 추가 (절대 삭제되지 않음)"""
        rule_id = hashlib.md5(f"{rule}_{workspace}".encode()).hexdigest()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO persistent_rules
            (id, rule_text, workspace_version, created_at, importance)
            VALUES (?, ?, ?, ?, ?)
        ''', (rule_id, rule, workspace, time.time(), importance))

        conn.commit()
        conn.close()

    def add_conversation(self, content: str, role: str = "user", importance: float = 0.5):
        """일반 대화 추가"""
        conversation = {
            "content": content,
            "role": role,
            "timestamp": time.time(),
            "importance": importance,
            "workspace": self.current_workspace
        }

        self.active_conversations.append(conversation)

        # 컨텍스트 크기 체크 및 압축
        self._check_and_compress()

    def _estimate_tokens(self, text: str) -> int:
        """토큰 수 추정 (단어 수 * 1.3)"""
        return int(len(text.split()) * 1.3)

    def _check_and_compress(self):
        """컨텍스트 크기 체크 및 필요시 압축"""
        current_tokens = self._calculate_current_tokens()

        if current_tokens > self.max_context_tokens:
            print(f"Context limit exceeded ({current_tokens} > {self.max_context_tokens}). Compressing...")
            self._compress_old_conversations()

    def _calculate_current_tokens(self) -> int:
        """현재 컨텍스트 토큰 수 계산"""
        total = 0

        # 영구 규칙
        for rule in self.persistent_rules:
            total += self._estimate_tokens(rule)

        # 활성 대화
        for conv in self.active_conversations:
            total += self._estimate_tokens(conv["content"])

        return total

    def _compress_old_conversations(self):
        """오래된 대화들을 AI로 압축하여 데이터베이스에 저장"""

        if len(self.active_conversations) < 10:
            return  # 너무 적으면 압축하지 않음

        # 압축할 대화들 선별 (오래되고 중요도 낮은 것들)
        cutoff_time = time.time() - (24 * 3600)  # 24시간 전

        to_compress = []
        to_keep = []

        for conv in self.active_conversations:
            if conv["timestamp"] < cutoff_time and conv["importance"] < 0.7:
                to_compress.append(conv)
            else:
                to_keep.append(conv)

        if len(to_compress) < 5:
            # 시간 기준으로 부족하면 중요도로 선별
            sorted_convs = sorted(self.active_conversations, key=lambda x: x["importance"])
            to_compress = sorted_convs[:len(sorted_convs)//3]  # 하위 1/3 압축
            to_keep = sorted_convs[len(sorted_convs)//3:]

        # 압축 실행
        if to_compress:
            compressed = self._compress_conversations(to_compress)
            self._save_compressed_memory(compressed)

            # 활성 대화 업데이트
            self.active_conversations = to_keep

            print(f"Compressed {len(to_compress)} conversations to database")

    def _compress_conversations(self, conversations: List[Dict]) -> CompressedMemory:
        """대화들을 AI 요약으로 압축"""

        # 대화 내용 조합
        full_text = "\n".join([f"{c['role']}: {c['content']}" for c in conversations])

        # 간단한 압축 (실제로는 AI 모델 사용해야 함)
        summary = self._simple_summarize(full_text)

        # 태그 추출
        tags = self._extract_tags(full_text)

        # 압축된 메모리 객체 생성
        memory_id = hashlib.md5(f"{full_text}_{time.time()}".encode()).hexdigest()

        return CompressedMemory(
            id=memory_id,
            summary=summary,
            importance=max(c["importance"] for c in conversations),
            timestamp=time.time(),
            original_length=len(full_text),
            compressed_length=len(summary),
            tags=tags,
            context_type="archived",
            workspace_version=self.current_workspace
        )

    def _simple_summarize(self, text: str) -> str:
        """간단한 요약 (실제로는 AI 모델 사용)"""
        lines = text.split('\n')

        # 키워드 기반 요약
        important_lines = []
        keywords = ["결정", "중요", "문제", "해결", "규칙", "정책", "DECISION", "ERROR", "SUCCESS"]

        for line in lines:
            if any(keyword in line for keyword in keywords):
                important_lines.append(line)

        if important_lines:
            summary = " | ".join(important_lines[:3])  # 최대 3개 라인
        else:
            summary = f"General conversation with {len(lines)} messages"

        return summary[:200]  # 최대 200자

    def _extract_tags(self, text: str) -> List[str]:
        """텍스트에서 태그 추출"""
        tags = []

        # 기술 관련 태그
        tech_keywords = {
            "python": "programming",
            "javascript": "programming",
            "react": "frontend",
            "api": "backend",
            "database": "backend",
            "bug": "debugging",
            "error": "debugging",
            "test": "testing",
            "deploy": "deployment"
        }

        text_lower = text.lower()
        for keyword, tag in tech_keywords.items():
            if keyword in text_lower:
                tags.append(tag)

        # 프로젝트 관련 태그
        if "arkhe" in text_lower or "arkhē" in text_lower:
            tags.append("project-arkhe")

        return list(set(tags))  # 중복 제거

    def _save_compressed_memory(self, memory: CompressedMemory):
        """압축된 기억을 데이터베이스에 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO compressed_memories
            (id, summary, importance, timestamp, original_length, compressed_length,
             tags, context_type, workspace_version, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            memory.id, memory.summary, memory.importance, memory.timestamp,
            memory.original_length, memory.compressed_length,
            json.dumps(memory.tags), memory.context_type, memory.workspace_version,
            None  # 임베딩은 나중에 구현
        ))

        conn.commit()
        conn.close()

    def search_relevant_memories(self, query: str, limit: int = 5) -> List[CompressedMemory]:
        """쿼리와 관련된 기억 검색"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 간단한 키워드 기반 검색 (실제로는 시맨틱 검색 사용해야 함)
        query_words = query.lower().split()

        # 태그 기반 검색
        relevant_memories = []

        cursor.execute('''
            SELECT id, summary, importance, timestamp, original_length, compressed_length,
                   tags, context_type, workspace_version
            FROM compressed_memories
            WHERE workspace_version = ?
            ORDER BY importance DESC, timestamp DESC
            LIMIT ?
        ''', (self.current_workspace, limit * 2))

        rows = cursor.fetchall()

        for row in rows:
            memory = CompressedMemory(
                id=row[0], summary=row[1], importance=row[2], timestamp=row[3],
                original_length=row[4], compressed_length=row[5],
                tags=json.loads(row[6]), context_type=row[7], workspace_version=row[8]
            )

            # 관련성 점수 계산
            relevance_score = self._calculate_relevance(query, memory)
            if relevance_score > 0.3:  # 임계값
                relevant_memories.append(memory)

        conn.close()

        # 관련성과 중요도로 정렬
        relevant_memories.sort(key=lambda m: m.importance, reverse=True)
        return relevant_memories[:limit]

    def _calculate_relevance(self, query: str, memory: CompressedMemory) -> float:
        """쿼리와 메모리의 관련성 점수 계산"""
        query_words = set(query.lower().split())
        memory_words = set(memory.summary.lower().split())

        # 단어 겹침 기반 점수
        intersection = query_words & memory_words
        union = query_words | memory_words

        jaccard_score = len(intersection) / len(union) if union else 0

        # 태그 기반 추가 점수
        tag_bonus = 0
        for tag in memory.tags:
            if tag.lower() in query.lower():
                tag_bonus += 0.2

        return min(1.0, jaccard_score + tag_bonus)

    def get_infinite_context(self, current_query: str) -> ContextWindow:
        """무한 컨텍스트 생성 - 토큰 한계 내에서 최적화"""

        # 1. 영구 규칙 로드
        persistent_rules = self._load_persistent_rules()

        # 2. 현재 쿼리와 관련된 기억 검색
        relevant_memories = self.search_relevant_memories(current_query)

        # 3. 최근 활성 대화
        recent_active = [conv["content"] for conv in self.active_conversations[-10:]]

        # 4. 토큰 수 계산 및 조정
        total_tokens = 0
        total_tokens += sum(self._estimate_tokens(rule) for rule in persistent_rules)
        total_tokens += sum(self._estimate_tokens(text) for text in recent_active)
        total_tokens += sum(self._estimate_tokens(mem.summary) for mem in relevant_memories)

        # 토큰 초과시 조정
        if total_tokens > self.max_context_tokens:
            # 관련 기억 줄이기
            while total_tokens > self.max_context_tokens and relevant_memories:
                removed = relevant_memories.pop()
                total_tokens -= self._estimate_tokens(removed.summary)

            # 여전히 초과시 최근 대화 줄이기
            while total_tokens > self.max_context_tokens and len(recent_active) > 5:
                removed = recent_active.pop(0)
                total_tokens -= self._estimate_tokens(removed)

        return ContextWindow(
            persistent_rules=persistent_rules,
            recent_active=recent_active,
            relevant_memories=relevant_memories,
            total_tokens=total_tokens,
            max_tokens=self.max_context_tokens
        )

    def _load_persistent_rules(self) -> List[str]:
        """영구 규칙 로드"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT rule_text FROM persistent_rules
            WHERE workspace_version = ?
            ORDER BY importance DESC
        ''', (self.current_workspace,))

        rules = [row[0] for row in cursor.fetchall()]
        conn.close()

        return rules

    def format_context_for_ai(self, context: ContextWindow, current_query: str) -> str:
        """AI가 사용할 최종 컨텍스트 포맷팅"""
        parts = []

        # 영구 규칙
        if context.persistent_rules:
            parts.append("=== CORE PRINCIPLES (ALWAYS ACTIVE) ===")
            for rule in context.persistent_rules:
                parts.append(f"- {rule}")
            parts.append("")

        # 관련 기억들
        if context.relevant_memories:
            parts.append("=== RELEVANT PAST CONTEXT ===")
            for memory in context.relevant_memories:
                age_days = (time.time() - memory.timestamp) / (24 * 3600)
                parts.append(f"[{age_days:.0f}d ago, importance:{memory.importance:.1f}] {memory.summary}")
            parts.append("")

        # 최근 활성 대화
        if context.recent_active:
            parts.append("=== RECENT CONVERSATION ===")
            for msg in context.recent_active:
                parts.append(msg)
            parts.append("")

        # 현재 쿼리
        parts.append("=== CURRENT QUESTION ===")
        parts.append(current_query)

        # 토큰 정보 (디버깅용)
        parts.append(f"\n[Context tokens: {context.total_tokens}/{context.max_tokens}]")

        return "\n".join(parts)

    def get_memory_stats(self) -> Dict[str, Any]:
        """메모리 통계 정보"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM compressed_memories')
        total_memories = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM persistent_rules')
        total_rules = cursor.fetchone()[0]

        cursor.execute('SELECT SUM(original_length), SUM(compressed_length) FROM compressed_memories')
        sizes = cursor.fetchone()
        original_size = sizes[0] or 0
        compressed_size = sizes[1] or 0

        compression_ratio = compressed_size / original_size if original_size > 0 else 0

        conn.close()

        return {
            "total_memories": total_memories,
            "total_rules": total_rules,
            "active_conversations": len(self.active_conversations),
            "original_size": original_size,
            "compressed_size": compressed_size,
            "compression_ratio": compression_ratio,
            "db_file_size": Path(self.db_path).stat().st_size if Path(self.db_path).exists() else 0
        }


# 사용 예시
if __name__ == "__main__":
    manager = InfiniteContextManager()

    # 영구 규칙 설정
    manager.add_persistent_rule("하드코딩 절대 금지", importance=1.0)
    manager.add_persistent_rule("실험 결과는 EXPERIMENT_LOG.md에 기록", importance=0.9)

    # 일반 대화들 추가 (시간이 지나면 자동 압축됨)
    conversations = [
        "Multi-Agent 성능 테스트 결과",
        "Shannon Entropy 최적화 논의",
        "Registry 패턴 적용 완료",
        "버그 수정: import 경로 문제",
        "새로운 실험 설계 검토"
    ]

    for i, conv in enumerate(conversations):
        manager.add_conversation(f"{conv} - 상세 내용 {i}", importance=0.6)

    # 현재 쿼리로 무한 컨텍스트 생성
    query = "Python 함수 작성 도움이 필요해"
    context = manager.get_infinite_context(query)
    formatted = manager.format_context_for_ai(context, query)

    print("=== INFINITE CONTEXT DEMO ===")
    print(formatted)

    print("\n=== MEMORY STATS ===")
    stats = manager.get_memory_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")