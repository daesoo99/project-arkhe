#!/usr/bin/env python3
"""
Version Workspace System Demo
사용자가 원한 "버전별 방" 개념의 실제 사용 예시
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from multiroom.version_workspace_manager import VersionWorkspaceManager, WorkspaceConcept, WorkspaceType


def demo_version_workspace_system():
    """버전별 워크스페이스 시스템 데모"""

    print("=== Version-Based Workspace System Demo ===\n")

    # 1. 매니저 생성 (기본 v1_research, v2_creative 포함)
    manager = VersionWorkspaceManager()

    print("Step 1: Starting in MAIN thread")
    print("Current workspace: MAIN")

    # 메인에서 대화 시작
    manager.add_message("I need help with both research and creative tasks", "user")

    print("\nStep 2: Switch to Research Mode (v1)")

    # 연구 모드로 전환
    manager.switch_workspace("v1_research")
    print(f"Current workspace: {manager.current_version}")

    # 연구 모드에서 작업
    manager.add_message("Let's analyze the Multi-Agent performance data", "user")
    manager.add_message("Following research protocol. Checking EXPERIMENT_LOG first...", "assistant")

    print("Research mode personality: Systematic and rigorous researcher tone")
    print("Research rules: No hardcoding, must log experiments, etc.")

    print("\nStep 3: Create new Custom Workspace (v4)")

    # 새로운 커스텀 워크스페이스 생성
    learning_concept = WorkspaceConcept(
        title="Interactive Learning Mode",
        description="Focus on teaching and explaining complex concepts",
        rules=[
            "Use simple language and examples",
            "Break down complex topics step by step",
            "Encourage questions and interaction"
        ],
        personality="Patient and encouraging teacher tone",
        focus_areas=["Educational explanations", "Interactive learning", "Concept visualization"],
        constraints=["Avoid overwhelming technical jargon"]
    )

    learning_workspace = manager.create_workspace(
        "v4_learning", "Interactive Learning Mode",
        WorkspaceType.LEARNING, learning_concept
    )

    print("Created v4_learning with teacher personality")

    print("\nStep 4: Switch between workspaces")

    # 창작 모드로 전환
    manager.switch_workspace("v2_creative")
    print(f"Switched to: {manager.current_version}")
    manager.add_message("Let's brainstorm innovative architecture ideas", "user")
    print("Creative mode: No constraints, experimental spirit!")

    # 학습 모드로 전환
    manager.switch_workspace("v4_learning")
    print(f"Switched to: {manager.current_version}")
    manager.add_message("Can you explain Shannon Entropy in simple terms?", "user")
    print("Learning mode: Patient teacher tone, simple explanations")

    print("\nStep 5: Workspace Overview")

    # 전체 워크스페이스 목록
    workspaces = manager.list_workspaces()
    print("\nAvailable Workspaces:")
    for version, info in workspaces.items():
        print(f"  {version}: {info['name']}")
        print(f"    Type: {info['type']}")
        print(f"    Messages: {info['message_count']}")
        print(f"    Concept: {info['concept_title']}")
        print()

    print("Step 6: Context Integration")

    # 다시 연구 모드로 돌아가서 전체 컨텍스트 확인
    manager.switch_workspace("v1_research")

    print("\nFull Context (Main + Current Workspace):")
    print("-" * 50)
    context = manager.get_full_context()
    # 한글 부분 제거한 깔끔한 출력
    clean_context = context.replace("����", "Title").replace("��Ģ", "Rules").replace("���� ����", "Focus Areas").replace("�������", "Constraints")
    print(clean_context[:1000] + "..." if len(clean_context) > 1000 else clean_context)

    print("\n=== Demo Complete ===")
    print("\nKey Benefits:")
    print("✅ Single chat session with multiple personalities")
    print("✅ Each version maintains its unique concept over time")
    print("✅ Easy switching between different work modes")
    print("✅ Context preservation across versions")
    print("✅ Unlimited custom workspace creation")


def interactive_demo():
    """간단한 대화형 데모"""
    manager = VersionWorkspaceManager()

    print("\n=== Interactive Demo ===")
    print("Available commands:")
    print("  /switch <version>  - Switch workspace (v1_research, v2_creative)")
    print("  /list             - List all workspaces")
    print("  /context          - Show current context")
    print("  /quit             - Exit demo")
    print()

    while True:
        current = manager.current_version
        user_input = input(f"[{current}] > ").strip()

        if user_input == "/quit":
            break
        elif user_input == "/list":
            workspaces = manager.list_workspaces()
            for version, info in workspaces.items():
                marker = " <- CURRENT" if version == current else ""
                print(f"  {version}: {info['name']}{marker}")
        elif user_input.startswith("/switch "):
            new_version = user_input.split()[1]
            if manager.switch_workspace(new_version):
                print(f"Switched to {new_version}")
            else:
                print(f"Workspace {new_version} not found")
        elif user_input == "/context":
            context = manager.get_full_context()
            print(context[:500] + "..." if len(context) > 500 else context)
        else:
            manager.add_message(user_input, "user")
            # 실제 AI 응답 대신 워크스페이스별 모의 응답
            workspace = manager.get_current_workspace()
            if workspace:
                response = f"[{workspace.concept.personality}] Processing your request..."
            else:
                response = "[Main mode] General response..."
            manager.add_message(response, "assistant")
            print(response)


if __name__ == "__main__":
    # 기본 데모 실행
    demo_version_workspace_system()

    # 대화형 데모 (선택사항)
    if input("\nRun interactive demo? (y/n): ").lower() == 'y':
        interactive_demo()