#!/usr/bin/env python3
"""
통합 실험 런처 - Project Arkhē
모든 실험을 통합하여 즉시 실행 가능한 벤치마크 시스템
"""

import json
import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.hierarchy import CostTracker, Mediator, IndependentThinker
from src.agents.economic_agent import EconomicAgent, FixedModelAgent  
from src.agents.integrated_arkhe import IntegratedArkheAgent, ArkheSystemFactory, TraditionalAgent

def load_tasks(tasks_file="prompts/tasks.jsonl"):
    """태스크 파일에서 테스트 문제들 로드"""
    tasks = []
    tasks_path = project_root / tasks_file
    
    if not tasks_path.exists():
        print(f"⚠️  태스크 파일을 찾을 수 없습니다: {tasks_path}")
        return _get_default_tasks()
    
    try:
        with open(tasks_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    task = json.loads(line)
                    tasks.append((task["id"], task["prompt"], task["complexity"], task["category"]))
        print(f"✅ {len(tasks)}개의 태스크를 로드했습니다.")
        return tasks
    except Exception as e:
        print(f"⚠️  태스크 파일 로드 오류: {e}")
        return _get_default_tasks()

def _get_default_tasks():
    """기본 테스트 태스크들"""
    return [
        ("simple_fact", "프랑스의 수도는 어디인가요?", 2.0, "간단"),
        ("basic_math", "2 + 2는 무엇인가요?", 1.5, "간단"),
        ("analysis", "AI 규제가 필요한 이유를 설명해주세요.", 6.0, "중간"),
        ("complex_policy", "AI 규제 정책 수립 시 고려해야 할 핵심 요소들을 분석해주세요.", 8.5, "복잡"),
        ("creative", "도시 교통 체증 문제를 해결할 혁신적 아이디어를 제안해주세요.", 8.0, "창의")
    ]

def run_basic_hierarchy_test(tasks):
    """기본 계층 구조 테스트 (현재 동작하는 시스템)"""
    print("\\n🔸 기본 계층 구조 테스트 (Ollama gemma:2b)")
    print("="*60)
    
    cost_tracker = CostTracker()
    
    # 3개의 독립 에이전트 생성
    thinkers = [
        IndependentThinker(f"Agent_{i+1}", cost_tracker, 'gemma:2b') 
        for i in range(3)
    ]
    
    mediator = Mediator(thinkers, cost_tracker)
    
    results = []
    for task_id, prompt, complexity, category in tasks:
        print(f"\\n📝 [{category}] {task_id}: {prompt[:50]}...")
        
        result = mediator.solve_problem(prompt)
        results.append({
            "task_id": task_id,
            "category": category,
            "complexity": complexity,
            "final_answer": result["final_answer"][:100] + "..." if len(result["final_answer"]) > 100 else result["final_answer"],
            "diversity": result["shannon_entropy"],
            "contradictions": result["contradiction_report"]
        })
        
        print(f"  답변: {result['final_answer'][:80]}...")
        print(f"  다양성: {result['shannon_entropy']:.2f}")
    
    total_cost = cost_tracker.get_total_cost()
    print(f"\\n💰 총 비용: ${total_cost:.6f}")
    
    return {
        "system": "Basic_Hierarchy",
        "total_cost": total_cost,
        "results": results
    }

def run_economic_intelligence_test(tasks):
    """경제적 지능 테스트 (동적 모델 선택)"""
    print("\\n🔸 경제적 지능 테스트 (동적 모델 선택)")
    print("="*60)
    
    # Control Group: 고정 모델
    control_cost_tracker = CostTracker()
    control_agents = [FixedModelAgent(f"Control_{i+1}", control_cost_tracker) for i in range(3)]
    control_mediator = Mediator(control_agents, control_cost_tracker)
    
    # Test Group: 동적 모델 선택
    test_cost_tracker = CostTracker()
    test_agents = [EconomicAgent(f"Economic_{i+1}", test_cost_tracker) for i in range(3)]
    test_mediator = Mediator(test_agents, test_cost_tracker)
    
    print("\\n📊 Control Group (고정 모델) vs Test Group (동적 선택)")
    
    control_results = []
    test_results = []
    
    for task_id, prompt, complexity, category in tasks:
        print(f"\\n📝 [{category}] {task_id}")
        
        try:
            # Control Group 실행
            control_result = control_mediator.solve_problem(prompt)
            control_results.append(control_result["final_answer"][:50] + "...")
            
            # Test Group 실행  
            test_result = test_mediator.solve_problem(prompt)
            test_results.append(test_result["final_answer"][:50] + "...")
            
            print(f"  Control: {control_result['final_answer'][:60]}...")
            print(f"  Economic: {test_result['final_answer'][:60]}...")
            
        except Exception as e:
            print(f"  ⚠️ 오류: {e}")
            control_results.append(f"Error: {e}")
            test_results.append(f"Error: {e}")
    
    control_cost = control_cost_tracker.get_total_cost()
    test_cost = test_cost_tracker.get_total_cost()
    
    print(f"\\n💰 Control Group 비용: ${control_cost:.6f}")
    print(f"💰 Test Group 비용: ${test_cost:.6f}")
    
    if control_cost > 0:
        savings = ((control_cost - test_cost) / control_cost) * 100
        print(f"💡 비용 절감: {savings:.1f}%")
    
    return {
        "control_cost": control_cost,
        "test_cost": test_cost,
        "control_results": control_results,
        "test_results": test_results
    }

def run_integrated_arkhe_test(tasks):
    """통합 Arkhē 시스템 테스트"""
    print("\\n🔸 통합 Arkhē 시스템 테스트")
    print("="*60)
    
    # 다양한 설정 테스트
    configs = {
        "Traditional": None,
        "Basic_Arkhe": ArkheSystemFactory.create_basic_config(),
        "Advanced_Arkhe": ArkheSystemFactory.create_advanced_config(),
        "Full_Arkhe": ArkheSystemFactory.create_full_config()
    }
    
    results = {}
    
    for config_name, config in configs.items():
        print(f"\\n🧪 {config_name} 설정 테스트")
        cost_tracker = CostTracker()
        
        try:
            if config is None:
                # Traditional 방식
                agent = TraditionalAgent("Traditional", cost_tracker)
            else:
                agent = IntegratedArkheAgent("Arkhe", cost_tracker, config)
            
            config_results = []
            for task_id, prompt, complexity, category in tasks[:3]:  # 처음 3개만 테스트
                try:
                    result = agent.solve(prompt)
                    config_results.append(result[:80] + "...")
                    print(f"  [{category}] {result[:60]}...")
                except Exception as e:
                    print(f"  ⚠️ 오류: {e}")
                    config_results.append(f"Error: {e}")
            
            results[config_name] = {
                "cost": cost_tracker.get_total_cost(),
                "results": config_results
            }
            
            print(f"  💰 비용: ${cost_tracker.get_total_cost():.6f}")
            
        except Exception as e:
            print(f"  ❌ {config_name} 설정 실패: {e}")
            results[config_name] = {"cost": 0, "results": [], "error": str(e)}
    
    return results

def print_summary(basic_result, economic_result, integrated_result):
    """실험 결과 요약"""
    print("\\n" + "="*80)
    print("🏁 실험 결과 요약")
    print("="*80)
    
    print(f"\\n🔸 기본 계층 시스템:")
    print(f"  비용: ${basic_result['total_cost']:.6f}")
    print(f"  처리한 태스크: {len(basic_result['results'])}개")
    
    print(f"\\n🔸 경제적 지능 비교:")
    print(f"  Control Group: ${economic_result['control_cost']:.6f}")
    print(f"  Economic Group: ${economic_result['test_cost']:.6f}")
    
    print(f"\\n🔸 통합 Arkhē 시스템:")
    for config_name, result in integrated_result.items():
        cost = result.get('cost', 0)
        error = result.get('error', '')
        if error:
            print(f"  {config_name}: 실패 ({error})")
        else:
            print(f"  {config_name}: ${cost:.6f}")

def main():
    """메인 실행 함수"""
    print("Project Arkhe 통합 실험 시작")
    print("="*80)
    
    # 태스크 로드
    tasks = load_tasks()
    
    print(f"\\n📋 실험 설정:")
    print(f"  태스크 수: {len(tasks)}개")
    print(f"  카테고리: {set(task[3] for task in tasks)}")
    
    # 실험 실행
    try:
        basic_result = run_basic_hierarchy_test(tasks)
        economic_result = run_economic_intelligence_test(tasks)
        integrated_result = run_integrated_arkhe_test(tasks)
        
        # 결과 요약
        print_summary(basic_result, economic_result, integrated_result)
        
    except KeyboardInterrupt:
        print("\\n⏸️  실험이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\\n❌ 실험 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()