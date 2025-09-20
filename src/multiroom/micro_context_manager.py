#!/usr/bin/env python3
"""
마이크로 컨텍스트 관리자 - 극한의 압축 전략
AI가 현재 질문에 꼭 필요한 최소한의 정보만 제공
"""

import json
import time
import sqlite3
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ContextRelevance(Enum):
    """컨텍스트 관련성 레벨"""
    CRITICAL = "critical"      # 현재 질문에 필수적
    HELPFUL = "helpful"        # 도움이 되지만 필수는 아님
    IRRELEVANT = "irrelevant"  # 현재 질문과 무관


@dataclass
class MicroRule:
    """마이크로 규칙 - 최대 한 줄"""
    text: str  # 최대 50자
    trigger_keywords: List[str]  # 언제 활성화될지
    importance: float


@dataclass
class MicroMemory:
    """마이크로 메모리 - 핵심만"""
    summary: str  # 최대 30자
    trigger_keywords: List[str]
    importance: float
    timestamp: float


class MicroContextManager:
    """극한 압축 컨텍스트 관리자"""

    def __init__(self, max_context_chars: int = 300):
        self.max_context_chars = max_context_chars
        self.micro_rules: List[MicroRule] = []
        self.micro_memories: List[MicroMemory] = []

        # 기본 규칙들 설정
        self._setup_default_rules()

    def _setup_default_rules(self):
        """기본 마이크로 규칙들"""
        default_rules = [
            MicroRule(
                text="No hardcoding, use config",
                trigger_keywords=["code", "function", "implement", "write"],
                importance=1.0
            ),
            MicroRule(
                text="Log experiments to EXPERIMENT_LOG.md",
                trigger_keywords=["experiment", "test", "result"],
                importance=0.9
            ),
            MicroRule(
                text="Use Registry pattern for models",
                trigger_keywords=["model", "registry", "pattern"],
                importance=0.8
            )
        ]

        self.micro_rules.extend(default_rules)

    def add_conversation(self, content: str, role: str = "user"):
        """대화 추가 - 즉시 마이크로 메모리로 압축"""

        # 핵심 키워드 추출
        keywords = self._extract_keywords(content)

        # 초압축 요약 생성 (최대 30자)
        micro_summary = self._create_micro_summary(content)

        # 중요도 계산
        importance = self._calculate_importance(content, keywords)

        # 마이크로 메모리 생성
        memory = MicroMemory(
            summary=micro_summary,
            trigger_keywords=keywords,
            importance=importance,
            timestamp=time.time()
        )

        self.micro_memories.append(memory)

        # 메모리 관리 (최대 50개 유지)
        if len(self.micro_memories) > 50:
            # 중요도 낮은 것부터 제거
            self.micro_memories.sort(key=lambda x: x.importance, reverse=True)
            self.micro_memories = self.micro_memories[:50]

    def _extract_keywords(self, content: str) -> List[str]:
        """핵심 키워드만 추출"""

        # 기술 키워드
        tech_keywords = [
            "python", "javascript", "react", "api", "database",
            "function", "class", "bug", "error", "test", "deploy",
            "experiment", "model", "registry", "pattern", "config"
        ]

        content_lower = content.lower()
        found_keywords = []

        for keyword in tech_keywords:
            if keyword in content_lower:
                found_keywords.append(keyword)

        return found_keywords[:3]  # 최대 3개만

    def _create_micro_summary(self, content: str) -> str:
        """초압축 요약 생성 (최대 30자)"""

        # 패턴 기반 압축
        if "function" in content.lower() and "python" in content.lower():
            return "Python func request"
        elif "bug" in content.lower() or "error" in content.lower():
            return "Error/bug discussion"
        elif "experiment" in content.lower():
            return "Experiment result"
        elif "test" in content.lower():
            return "Testing topic"
        elif "deploy" in content.lower():
            return "Deployment issue"
        else:
            # 첫 단어들로 요약
            words = content.split()[:3]
            summary = " ".join(words)
            return summary[:30]

    def _calculate_importance(self, content: str, keywords: List[str]) -> float:
        """중요도 계산"""
        importance = 0.5  # 기본값

        # 키워드 기반 가산점
        importance += len(keywords) * 0.1

        # 특별 키워드 보너스
        high_value_words = ["error", "bug", "decision", "important", "critical"]
        for word in high_value_words:
            if word in content.lower():
                importance += 0.2

        # 길이 기반 조정 (너무 짧으면 중요도 감소)
        if len(content) < 20:
            importance -= 0.1

        return min(1.0, max(0.1, importance))

    def get_micro_context(self, current_query: str) -> str:
        """현재 질문에 맞는 극소 컨텍스트 생성"""

        query_keywords = self._extract_keywords(current_query)

        # 1. 관련 규칙 선별 (최대 2개)
        relevant_rules = []
        for rule in self.micro_rules:
            if any(keyword in query_keywords for keyword in rule.trigger_keywords):
                relevant_rules.append(rule)

        relevant_rules.sort(key=lambda x: x.importance, reverse=True)
        relevant_rules = relevant_rules[:2]  # 최대 2개

        # 2. 관련 메모리 선별 (최대 3개)
        relevant_memories = []
        for memory in self.micro_memories:
            if any(keyword in query_keywords for keyword in memory.trigger_keywords):
                relevance_score = len(set(memory.trigger_keywords) & set(query_keywords))
                relevant_memories.append((memory, relevance_score))

        # 관련성과 중요도로 정렬
        relevant_memories.sort(key=lambda x: (x[1], x[0].importance), reverse=True)
        relevant_memories = [mem for mem, score in relevant_memories[:3]]

        # 3. 마이크로 컨텍스트 조합
        context_parts = []

        # 규칙 (한 줄씩)
        if relevant_rules:
            rules_text = " | ".join([rule.text for rule in relevant_rules])
            context_parts.append(f"Rules: {rules_text}")

        # 메모리 (한 줄로)
        if relevant_memories:
            memories_text = " | ".join([mem.summary for mem in relevant_memories])
            context_parts.append(f"Context: {memories_text}")

        # 현재 질문
        context_parts.append(f"Q: {current_query}")

        # 길이 체크 및 조정
        full_context = " // ".join(context_parts)

        if len(full_context) > self.max_context_chars:
            # 메모리부터 줄이기
            while len(full_context) > self.max_context_chars and relevant_memories:
                relevant_memories.pop()
                if relevant_memories:
                    memories_text = " | ".join([mem.summary for mem in relevant_memories])
                    context_parts[1] = f"Context: {memories_text}"
                else:
                    context_parts.pop(1)  # 메모리 섹션 전체 제거
                full_context = " // ".join(context_parts)

            # 여전히 길면 규칙도 줄이기
            if len(full_context) > self.max_context_chars and relevant_rules:
                relevant_rules = relevant_rules[:1]
                rules_text = relevant_rules[0].text
                context_parts[0] = f"Rules: {rules_text}"
                full_context = " // ".join(context_parts)

        return full_context

    def add_micro_rule(self, text: str, keywords: List[str], importance: float = 0.5):
        """새 마이크로 규칙 추가"""
        if len(text) > 50:
            text = text[:47] + "..."  # 강제 압축

        rule = MicroRule(
            text=text,
            trigger_keywords=keywords,
            importance=importance
        )

        self.micro_rules.append(rule)

    def get_stats(self) -> Dict:
        """통계 정보"""
        total_rules_chars = sum(len(rule.text) for rule in self.micro_rules)
        total_memories_chars = sum(len(mem.summary) for mem in self.micro_memories)

        return {
            "total_rules": len(self.micro_rules),
            "total_memories": len(self.micro_memories),
            "rules_chars": total_rules_chars,
            "memories_chars": total_memories_chars,
            "max_context_chars": self.max_context_chars
        }

    def simulate_long_conversation(self):
        """긴 대화 시뮬레이션"""

        # 1000개의 다양한 대화 시뮬레이션
        sample_conversations = [
            "Python function to calculate average needed",
            "Bug found in import path, need fix",
            "Shannon Entropy experiment results are good",
            "Registry pattern implementation complete",
            "Multi-Agent vs Single comparison done",
            "Database connection error occurred",
            "React component optimization required",
            "API endpoint design discussion",
            "Test cases for new feature written",
            "Deployment pipeline configuration issue"
        ]

        print("=== SIMULATING LONG CONVERSATION ===")

        for i in range(100):  # 100개 대화 추가
            content = sample_conversations[i % len(sample_conversations)]
            content += f" (conversation {i+1})"
            self.add_conversation(content)

            if (i + 1) % 20 == 0:
                stats = self.get_stats()
                print(f"After {i+1} conversations: {stats['total_memories']} memories, "
                      f"{stats['memories_chars']} chars")

        # 다양한 질문으로 컨텍스트 테스트
        test_queries = [
            "Write a Python function for me",
            "How to fix import errors?",
            "What are the experiment results?",
            "Deploy the application",
            "Test the new feature"
        ]

        print(f"\n=== MICRO CONTEXT RESULTS ===")

        for query in test_queries:
            context = self.get_micro_context(query)
            print(f"\nQuery: {query}")
            print(f"Context ({len(context)} chars): {context}")


# 실전 테스트
if __name__ == "__main__":
    manager = MicroContextManager(max_context_chars=200)  # 200자 제한

    # 긴 대화 시뮬레이션
    manager.simulate_long_conversation()

    print(f"\n=== FINAL STATS ===")
    stats = manager.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

    print(f"\n=== EXTREME COMPRESSION DEMO ===")

    # 극한 테스트: 엄청 복잡한 질문
    complex_query = "I need help writing a Python function that connects to database and handles errors"
    context = manager.get_micro_context(complex_query)

    print(f"Complex Query: {complex_query}")
    print(f"Micro Context ({len(context)} chars): {context}")
    print(f"Compression ratio: {len(context)/len(complex_query)*100:.1f}%")