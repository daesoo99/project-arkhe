#!/usr/bin/env python3
"""
Project Arkhē - MultiRoom Conversation Experiment
멀티룸 대화 시스템 효과성 검증 실험

연구 가설: 
주제별/컨텍스트별 룸 분리가 대화 품질과 원칙 준수율을 향상시킨다

실험 설계:
A그룹: 기존 단일 대화 방식 (baseline)
B그룹: 멀티룸 시뮬레이션 (treatment)

측정 지표:
- 원칙 준수율 (Principle Adherence Rate)
- 주제 집중도 (Topic Coherence Score) 
- 컨텍스트 보존율 (Context Preservation Rate)
- Shannon Entropy 기반 정보 일관성
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

# Add src to path for Project Arkhē infrastructure
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Project Arkhē 기존 인프라 활용
from llm.simple_llm import create_llm_auto
from utils.information_theory import ShannonEntropyAnalyzer
from utils.scorers import score_task

# MultiRoom 확장 모듈
from multiroom import RoomManager, Room, RoomType, Message, MessageType, RoomPrinciple


@dataclass
class ConversationScenario:
    """대화 시나리오 정의"""
    id: str
    name: str
    description: str
    initial_principles: List[str]
    conversation_flow: List[Dict[str, str]]  # [{"user": "...", "context": "..."}]
    expected_room_switches: List[str]


class MultiRoomExperimentFramework:
    """멀티룸 대화 실험 프레임워크"""
    
    def __init__(self, llm_factory):
        self.llm_factory = llm_factory
        self.entropy_analyzer = ShannonEntropyAnalyzer()
        
        # 실험 시나리오들
        self.scenarios = self._create_test_scenarios()
    
    def _create_test_scenarios(self) -> List[ConversationScenario]:
        """실험용 대화 시나리오 생성"""
        
        scenarios = []
        
        # 시나리오 1: 프로젝트 간 전환
        project_switch = ConversationScenario(
            id="project_switch",
            name="프로젝트 간 컨텍스트 전환",
            description="Arkhē 프로젝트에서 다른 프로젝트로 주제 변경시 원칙 보존",
            initial_principles=[
                "새 파일 생성 금지, 기존 파일 수정 우선",
                "하드코딩 zero tolerance",
                "TodoWrite 복잡한 작업시 필수"
            ],
            conversation_flow=[
                {"user": "Arkhē 프로젝트에서 Shannon Entropy 최적화 작업 중입니다.", "context": "project_arkhe"},
                {"user": "이제 새로운 웹 프로젝트를 시작하고 싶은데 처음부터 만들어주세요.", "context": "new_project"},
                {"user": "HTML, CSS, JS 파일을 각각 생성해주세요.", "context": "new_project"},
                {"user": "다시 Arkhē로 돌아가서 entropy 실험 결과를 분석해주세요.", "context": "project_arkhe"}
            ],
            expected_room_switches=["project_arkhe", "new_project", "new_project", "project_arkhe"]
        )
        
        # 시나리오 2: 원칙 충돌 감지
        principle_conflict = ConversationScenario(
            id="principle_conflict", 
            name="원칙 충돌 감지 및 대응",
            description="설정된 원칙과 상충하는 요청시 충돌 감지 능력",
            initial_principles=[
                "새 파일 생성 금지, 기존 파일 수정 우선",
                "하드코딩 zero tolerance",
                "코드 설명보다 구현 우선"
            ],
            conversation_flow=[
                {"user": "파이썬 함수 하나 만들어주세요.", "context": "coding"},
                {"user": "새로운 config.py 파일을 생성해서 설정값들을 넣어주세요.", "context": "coding"},
                {"user": "변수값을 코드에 직접 하드코딩해서 넣어주세요.", "context": "coding"},
                {"user": "이 함수가 어떻게 작동하는지 자세히 설명해주세요.", "context": "coding"}
            ],
            expected_room_switches=["coding", "coding", "coding", "coding"]
        )
        
        # 시나리오 3: 장기 대화 컨텍스트 보존
        long_conversation = ConversationScenario(
            id="long_context",
            name="장기 대화 컨텍스트 보존", 
            description="긴 대화에서 초기 설정과 중간 결정사항 보존",
            initial_principles=[
                "모든 실험은 재현 가능하게 기록",
                "결과는 Shannon Entropy로 평가",
                "실패 케이스도 학습 자산으로 보관"
            ],
            conversation_flow=[
                {"user": "새로운 Multi-Agent 실험을 설계해주세요.", "context": "experiment_design"},
                {"user": "실험에 사용할 평가 지표는 뭐가 좋을까요?", "context": "experiment_design"},
                {"user": "잠깐, 점심 메뉴 추천해주세요.", "context": "daily_life"},
                {"user": "날씨가 좋네요. 산책하고 싶어요.", "context": "daily_life"},
                {"user": "아까 설계한 실험을 실행해주세요. 평가 지표는 어떻게 하기로 했었죠?", "context": "experiment_design"}
            ],
            expected_room_switches=["experiment", "experiment", "daily", "daily", "experiment"]
        )
        
        scenarios.extend([project_switch, principle_conflict, long_conversation])
        return scenarios
    
    def run_baseline_conversation(self, scenario: ConversationScenario) -> Dict[str, Any]:
        """기존 단일 대화 방식 (Baseline)"""
        
        print(f"  Running Baseline: {scenario.name}")
        
        llm = self.llm_factory("qwen2:7b")
        
        # 초기 원칙 설정
        initial_context = f"""
다음 원칙들을 대화 전체에서 반드시 준수해주세요:

{chr(10).join(f"- {principle}" for principle in scenario.initial_principles)}

이 원칙들을 절대 잊지 마시고 모든 응답에서 지켜주세요.
"""
        
        conversation_history = [initial_context]
        responses = []
        principle_violations = 0
        
        for turn in scenario.conversation_flow:
            # 전체 대화 히스토리와 함께 요청
            full_prompt = "\n".join(conversation_history) + f"\n\nUser: {turn['user']}\n\nAssistant:"
            
            try:
                response_dict = llm.generate(full_prompt)
                response = response_dict.get('response', str(response_dict)) if isinstance(response_dict, dict) else str(response_dict)
                
                responses.append(response)
                conversation_history.append(f"User: {turn['user']}")
                conversation_history.append(f"Assistant: {response}")
                
                # 원칙 위반 체크 (간단한 키워드 기반)
                violations = self._check_principle_violations(response, scenario.initial_principles, turn['user'])
                principle_violations += violations
                
            except Exception as e:
                print(f"    Error in baseline: {e}")
                responses.append(f"Error: {e}")
        
        # 전체 대화 분석
        full_conversation = "\n".join(conversation_history)
        final_entropy = self.entropy_analyzer.calculate_shannon_entropy(full_conversation)
        
        return {
            "scenario_id": scenario.id,
            "method": "baseline",
            "responses": responses,
            "principle_violations": principle_violations,
            "total_turns": len(scenario.conversation_flow),
            "adherence_rate": max(0, 1 - (principle_violations / len(scenario.conversation_flow))),
            "final_entropy": final_entropy,
            "conversation_length": len(full_conversation),
            "context_preservation_score": self._calculate_context_preservation(conversation_history, scenario)
        }
    
    def run_multiroom_simulation(self, scenario: ConversationScenario) -> Dict[str, Any]:
        """멀티룸 시뮬레이션 (Treatment)"""
        
        print(f"  Running MultiRoom: {scenario.name}")
        
        # 룸 매니저 초기화
        room_manager = RoomManager()
        
        # 시나리오별 전용 룸 생성
        scenario_rooms = self._setup_scenario_rooms(room_manager, scenario)
        
        llm = self.llm_factory("qwen2:7b")
        responses = []
        principle_violations = 0
        room_switches = []
        
        for i, turn in enumerate(scenario.conversation_flow):
            # 적절한 룸 선택 (지능형 라우팅 시뮬레이션)
            target_room = self._intelligent_room_routing(turn, scenario_rooms, scenario.expected_room_switches[i])
            
            if room_manager.current_room_id != target_room.id:
                room_manager.switch_room(target_room.id)
                room_switches.append(f"Switched to: {target_room.name}")
            
            # 현재 룸의 컨텍스트와 원칙으로 프롬프트 구성
            room_context = self._build_room_context(target_room)
            
            contextual_prompt = f"""
{room_context}

Recent conversation in this room:
{self._get_room_conversation_summary(target_room)}

User: {turn['user']}
"""
            
            try:
                response_dict = llm.generate(contextual_prompt)
                response = response_dict.get('response', str(response_dict)) if isinstance(response_dict, dict) else str(response_dict)
                
                responses.append(response)
                
                # 룸에 메시지 추가
                target_room.add_message(Message(content=turn['user'], message_type=MessageType.USER))
                target_room.add_message(Message(content=response, message_type=MessageType.ASSISTANT))
                
                # 원칙 위반 체크 (룸별 원칙 기준)
                violations = self._check_room_principle_violations(response, target_room, turn['user'])
                principle_violations += violations
                
            except Exception as e:
                print(f"    Error in multiroom: {e}")
                responses.append(f"Error: {e}")
        
        # 전체 대화 분석
        all_messages = []
        for room in scenario_rooms.values():
            for msg in room.messages:
                all_messages.append(f"{msg.message_type.value}: {msg.content}")
        
        full_conversation = "\n".join(all_messages)
        final_entropy = self.entropy_analyzer.calculate_shannon_entropy(full_conversation)
        
        return {
            "scenario_id": scenario.id,
            "method": "multiroom",
            "responses": responses,
            "principle_violations": principle_violations,
            "total_turns": len(scenario.conversation_flow),
            "adherence_rate": max(0, 1 - (principle_violations / len(scenario.conversation_flow))),
            "final_entropy": final_entropy,
            "conversation_length": len(full_conversation),
            "room_switches": room_switches,
            "rooms_used": list(scenario_rooms.keys()),
            "context_preservation_score": self._calculate_multiroom_context_preservation(scenario_rooms, scenario)
        }
    
    def _setup_scenario_rooms(self, room_manager: RoomManager, scenario: ConversationScenario) -> Dict[str, Room]:
        """시나리오별 전용 룸 설정"""
        
        rooms = {}
        
        if scenario.id == "project_switch":
            # Project Arkhē 룸 (기존에 있으니 찾아서 사용)
            arkhe_rooms = room_manager.search_rooms("arkhe")
            if arkhe_rooms:
                rooms["project_arkhe"] = arkhe_rooms[0]
            
            # 새 프로젝트 룸 생성
            new_project_room = room_manager.create_room(
                name="New-Web-Project",
                room_type=RoomType.PROJECT,
                description="새로운 웹 프로젝트 개발"
            )
            new_project_room.add_principle(RoomPrinciple(
                title="창의적 자유",
                description="새 프로젝트에서는 기존 제약없이 자유롭게 개발할 수 있습니다."
            ))
            rooms["new_project"] = new_project_room
        
        elif scenario.id == "principle_conflict":
            # 코딩 룸 생성
            coding_room = room_manager.create_room(
                name="Coding-Help",
                room_type=RoomType.TOPIC,
                description="프로그래밍 문제 해결 및 코딩 도움"
            )
            for principle_text in scenario.initial_principles:
                coding_room.add_principle(RoomPrinciple(
                    title=principle_text.split(',')[0],
                    description=principle_text
                ))
            rooms["coding"] = coding_room
        
        elif scenario.id == "long_context":
            # 실험 설계 룸
            experiment_room = room_manager.create_room(
                name="Experiment-Design",
                room_type=RoomType.TOPIC,
                description="연구 실험 설계 및 분석"
            )
            for principle_text in scenario.initial_principles:
                experiment_room.add_principle(RoomPrinciple(
                    title=principle_text.split(',')[0] if ',' in principle_text else principle_text[:20],
                    description=principle_text
                ))
            rooms["experiment"] = experiment_room
            
            # 일상 대화 룸
            daily_room = room_manager.create_room(
                name="Daily-Life",
                room_type=RoomType.TOPIC,
                description="일상적인 대화와 잡담"
            )
            rooms["daily"] = daily_room
        
        return rooms
    
    def _intelligent_room_routing(self, turn: Dict[str, str], rooms: Dict[str, Room], expected_room: str) -> Room:
        """지능형 룸 라우팅 시뮬레이션"""
        
        # 현재는 expected_room을 사용하지만, 실제로는 NLP 기반 분석
        user_message = turn['user'].lower()
        
        # 키워드 기반 간단한 라우팅
        if "arkhe" in user_message or "entropy" in user_message or "실험" in user_message:
            return rooms.get("project_arkhe", list(rooms.values())[0])
        elif "새" in user_message and "프로젝트" in user_message or "html" in user_message or "웹" in user_message:
            return rooms.get("new_project", list(rooms.values())[0])
        
        # 기본값: expected room 사용
        return rooms.get(expected_room, list(rooms.values())[0])
    
    def _build_room_context(self, room: Room) -> str:
        """룸별 컨텍스트 구성"""
        
        principles_text = ""
        active_principles = room.get_active_principles()
        if active_principles:
            principles_text = "현재 룸의 원칙들:\n" + "\n".join(
                f"- {p.title}: {p.description}" for p in active_principles
            ) + "\n\n"
        
        return f"""
=== {room.name} 룸 ===
{room.description}

{principles_text}위 원칙들을 반드시 준수하여 응답해주세요.
"""
    
    def _get_room_conversation_summary(self, room: Room) -> str:
        """룸의 최근 대화 요약"""
        recent_messages = room.get_recent_messages(3)
        if not recent_messages:
            return "(이 룸에서의 첫 대화입니다)"
        
        summary = []
        for msg in recent_messages:
            summary.append(f"{msg.message_type.value}: {msg.content[:100]}...")
        
        return "\n".join(summary)
    
    def _check_principle_violations(self, response: str, principles: List[str], user_request: str) -> int:
        """원칙 위반 검사 (키워드 기반)"""
        violations = 0
        response_lower = response.lower()
        user_lower = user_request.lower()
        
        # 개선된 키워드 기반 검사
        for principle in principles:
            if "새 파일 생성 금지" in principle:
                # 새 파일 생성 요청에 대해 실제로 생성하려고 하는지 체크
                if any(keyword in user_lower for keyword in ["새로운", "새", "생성", "만들", "create"]) and \
                   any(keyword in user_lower for keyword in ["파일", "file"]):
                    if any(keyword in response_lower for keyword in ["생성하", "만들어", "create", "새 파일"]):
                        violations += 1
                        print(f"    Violation detected: 새 파일 생성 - '{user_request[:50]}...'")
            
            elif "하드코딩 zero tolerance" in principle:
                # 하드코딩 요청에 대해 실제로 하드코딩하려고 하는지 체크
                if any(keyword in user_lower for keyword in ["하드코딩", "직접", "고정값", "hardcode"]):
                    if any(keyword in response_lower for keyword in ["하드코딩", "직접 입력", "고정값", "="]):
                        violations += 1
                        print(f"    Violation detected: 하드코딩 - '{user_request[:50]}...'")
            
            elif "코드 설명보다 구현 우선" in principle:
                # 설명 요청에 대해 장황한 설명을 하는지 체크
                if any(keyword in user_lower for keyword in ["설명", "어떻게", "explain"]):
                    if len(response) > 500 and "코드" not in response_lower[:200]:  # 긴 설명이면서 코드가 앞에 없음
                        violations += 1
                        print(f"    Violation detected: 설명 우선 - '{user_request[:50]}...'")
        
        return violations
    
    def _check_room_principle_violations(self, response: str, room: Room, user_request: str) -> int:
        """룸별 원칙 위반 검사"""
        violations = 0
        response_lower = response.lower()
        
        for principle in room.get_active_principles():
            if "새 파일 생성 금지" in principle.description:
                if "새 파일" in response_lower or "파일을 생성" in response_lower:
                    violations += 1
            elif "창의적 자유" in principle.title:
                # 창의적 자유 룸에서는 제약이 적음
                pass
        
        return violations
    
    def _calculate_context_preservation(self, conversation_history: List[str], scenario: ConversationScenario) -> float:
        """컨텍스트 보존 점수 계산"""
        
        # 마지막 응답에서 초기 원칙들이 얼마나 언급되는지 체크
        if not conversation_history:
            return 0.0
        
        last_response = conversation_history[-1].lower()
        principle_mentions = 0
        
        for principle in scenario.initial_principles:
            key_terms = principle.lower().split()[:3]  # 첫 3단어로 키워드 추출
            if any(term in last_response for term in key_terms):
                principle_mentions += 1
        
        return principle_mentions / len(scenario.initial_principles) if scenario.initial_principles else 0.0
    
    def _calculate_multiroom_context_preservation(self, rooms: Dict[str, Room], scenario: ConversationScenario) -> float:
        """멀티룸 컨텍스트 보존 점수"""
        
        preservation_scores = []
        
        for room in rooms.values():
            if room.messages:
                # 각 룸에서 해당 룸의 원칙들이 얼마나 유지되는지 체크
                room_principles = [p.description for p in room.get_active_principles()]
                if room_principles:
                    last_message = room.messages[-1].content.lower()
                    mentions = sum(1 for principle in room_principles 
                                 if any(term in last_message for term in principle.lower().split()[:2]))
                    preservation_scores.append(mentions / len(room_principles))
        
        return sum(preservation_scores) / len(preservation_scores) if preservation_scores else 0.0
    
    def run_experiment(self, scenario_ids: List[str] = None) -> Dict[str, Any]:
        """실험 실행"""
        
        if scenario_ids is None:
            scenarios_to_run = self.scenarios
        else:
            scenarios_to_run = [s for s in self.scenarios if s.id in scenario_ids]
        
        results = []
        
        print("=== MultiRoom Conversation Experiment ===")
        print(f"Testing {len(scenarios_to_run)} scenarios")
        
        for scenario in scenarios_to_run:
            print(f"\n--- Scenario: {scenario.name} ---")
            
            # Baseline 실험
            baseline_result = self.run_baseline_conversation(scenario)
            
            # MultiRoom 실험
            multiroom_result = self.run_multiroom_simulation(scenario)
            
            # 비교 분석
            comparison = self._compare_results(baseline_result, multiroom_result)
            
            results.append({
                "scenario": scenario,
                "baseline": baseline_result,
                "multiroom": multiroom_result,
                "comparison": comparison
            })
            
            print(f"  Baseline Adherence: {baseline_result['adherence_rate']:.2f}")
            print(f"  MultiRoom Adherence: {multiroom_result['adherence_rate']:.2f}")
            print(f"  Improvement: {comparison['adherence_improvement']:.2f}")
        
        # 전체 결과 분석
        overall_analysis = self._analyze_overall_results(results)
        
        return {
            "experiment_timestamp": time.time(),
            "results": results,
            "overall_analysis": overall_analysis
        }
    
    def _compare_results(self, baseline: Dict[str, Any], multiroom: Dict[str, Any]) -> Dict[str, Any]:
        """결과 비교"""
        
        return {
            "adherence_improvement": multiroom['adherence_rate'] - baseline['adherence_rate'],
            "entropy_difference": multiroom['final_entropy'] - baseline['final_entropy'],
            "context_preservation_improvement": multiroom['context_preservation_score'] - baseline['context_preservation_score'],
            "violation_reduction": baseline['principle_violations'] - multiroom['principle_violations']
        }
    
    def _analyze_overall_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """전체 결과 분석"""
        
        avg_adherence_improvement = sum(r['comparison']['adherence_improvement'] for r in results) / len(results)
        avg_entropy_difference = sum(r['comparison']['entropy_difference'] for r in results) / len(results)
        avg_context_improvement = sum(r['comparison']['context_preservation_improvement'] for r in results) / len(results)
        
        # 결론 도출
        if avg_adherence_improvement > 0.1 and avg_context_improvement > 0.1:
            conclusion = "MultiRoom system shows significant improvement in principle adherence and context preservation"
        elif avg_adherence_improvement > 0.05:
            conclusion = "MultiRoom system shows moderate improvement"
        else:
            conclusion = "MultiRoom system shows no significant advantage over baseline"
        
        return {
            "avg_adherence_improvement": avg_adherence_improvement,
            "avg_entropy_difference": avg_entropy_difference,
            "avg_context_improvement": avg_context_improvement,
            "conclusion": conclusion,
            "recommendation": "Implement MultiRoom system" if avg_adherence_improvement > 0.1 else "Need further optimization"
        }


def main():
    """MultiRoom 실험 메인 함수"""
    
    print("Project Arkhe - MultiRoom Conversation Experiment")
    print("=" * 60)
    
    # LLM Factory 설정
    llm_factory = create_llm_auto
    
    # 실험 프레임워크 초기화
    experiment = MultiRoomExperimentFramework(llm_factory)
    
    # 실험 실행
    results = experiment.run_experiment()
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS")
    print("=" * 60)
    
    overall = results['overall_analysis']
    print(f"Average Adherence Improvement: {overall['avg_adherence_improvement']:+.3f}")
    print(f"Average Entropy Difference: {overall['avg_entropy_difference']:+.3f}")
    print(f"Average Context Improvement: {overall['avg_context_improvement']:+.3f}")
    print(f"\nConclusion: {overall['conclusion']}")
    print(f"Recommendation: {overall['recommendation']}")
    
    # 결과 저장
    timestamp = int(time.time())
    output_file = Path(__file__).parent.parent / "results" / f"multiroom_experiment_{timestamp}.json"
    output_file.parent.mkdir(exist_ok=True)
    
    # JSON 직렬화용 데이터 정리
    json_results = {
        "experiment_info": {
            "name": "MultiRoom Conversation Experiment",
            "timestamp": results['experiment_timestamp'],
            "framework": "Project Arkhe Extension"
        },
        "scenarios_tested": len(results['results']),
        "overall_analysis": overall,
        "detailed_results": [
            {
                "scenario_id": r['scenario'].id,
                "scenario_name": r['scenario'].name,
                "baseline_adherence": r['baseline']['adherence_rate'],
                "multiroom_adherence": r['multiroom']['adherence_rate'],
                "improvement": r['comparison']['adherence_improvement']
            }
            for r in results['results']
        ]
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved: {output_file}")


if __name__ == "__main__":
    main()