#!/usr/bin/env python3
"""
통합 실험 런처 - Project Arkhe (간단 버전)
Windows 콘솔 호환을 위한 ASCII 버전
"""

import json
import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.hierarchy import CostTracker, Mediator, IndependentThinker

def load_tasks(tasks_file="prompts/tasks.jsonl"):
    """태스크 파일에서 테스트 문제들 로드 (새로운 구조화된 형식)"""
    tasks = []
    tasks_path = project_root / tasks_file
    
    if not tasks_path.exists():
        print(f"태스크 파일을 찾을 수 없습니다: {tasks_path}")
        return _get_default_tasks()
    
    try:
        with open(tasks_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    task = json.loads(line)
                    # 새로운 형식: id, type, prompt, answer, complexity
                    tasks.append((
                        task["id"], 
                        task["prompt"], 
                        task.get("complexity", 5.0), 
                        task["type"],  # type을 category로 사용
                        task.get("answer", "")  # 정답 추가
                    ))
        print(f"성공: {len(tasks)}개의 구조화된 태스크를 로드했습니다.")
        print(f"태스크 유형: {set(task[3] for task in tasks)}")
        return tasks
    except Exception as e:
        print(f"태스크 파일 로드 오류: {e}")
        return _get_default_tasks()

def _get_default_tasks():
    """기본 테스트 태스크들 (새로운 형식)"""
    return [
        ("fact-kr-002", "프랑스의 수도는 어디인가요? 도시명만 답하세요.", 1.5, "fact", "파리"),
        ("reason-002", "2 + 2는 무엇인가요? 계산 과정을 한 줄로 설명하세요.", 1.0, "reason", "4. 두 개의 2를 더하면 4가 됩니다."),
        ("format-001", "다음 스키마와 정확히 일치하는 JSON만 출력: {\"name\":string,\"age\":int}. 값: name=Kim, age=26", 3.5, "format", "{\"name\":\"Kim\",\"age\":26}"),
        ("creative-001", "도시 교통 체증 문제를 해결할 혁신적 아이디어 3가지를 제안해주세요.", 8.0, "creative", "1) 3차원 교통시스템 2) AI 예측 신호등 3) 개인용 드론 택시")
    ]

def run_basic_hierarchy_test(tasks):
    """기본 계층 구조 테스트 (현재 동작하는 시스템)"""
    print("\n== 기본 계층 구조 테스트 (Ollama gemma:2b) ==")
    print("="*60)
    
    cost_tracker = CostTracker()
    
    # 3개의 독립 에이전트 생성
    thinkers = [
        IndependentThinker(f"Agent_{i+1}", cost_tracker, 'gemma:2b') 
        for i in range(3)
    ]
    
    mediator = Mediator(thinkers, cost_tracker)
    
    results = []
    for task_id, prompt, complexity, category, expected_answer in tasks:
        print(f"\n[{category}] {task_id}: {prompt[:50]}...")
        
        try:
            result = mediator.solve_problem(prompt)
            final_answer = result["final_answer"]
            
            # 간단한 정확도 평가
            accuracy = 0
            if category == "fact" and expected_answer:
                accuracy = 1 if expected_answer.lower() in final_answer.lower() else 0
            elif category == "format" and expected_answer:
                # JSON 형식 체크
                try:
                    import json as json_lib
                    parsed = json_lib.loads(final_answer.strip())
                    expected_parsed = json_lib.loads(expected_answer)
                    accuracy = 1 if parsed == expected_parsed else 0
                except:
                    accuracy = 0
            
            results.append({
                "task_id": task_id,
                "category": category,
                "complexity": complexity,
                "final_answer": final_answer[:100] + "..." if len(final_answer) > 100 else final_answer,
                "expected_answer": expected_answer,
                "accuracy": accuracy,
                "diversity": result["shannon_entropy"],
                "contradictions": result["contradiction_report"]
            })
            
            print(f"  답변: {final_answer[:60]}...")
            print(f"  기대: {expected_answer[:60]}...")
            print(f"  정확도: {accuracy}, 다양성: {result['shannon_entropy']:.2f}")
            
        except Exception as e:
            print(f"  오류 발생: {e}")
            results.append({
                "task_id": task_id,
                "category": category,
                "accuracy": 0,
                "error": str(e)
            })
    
    total_cost = cost_tracker.get_total_cost()
    print(f"\n총 비용: ${total_cost:.6f}")
    
    return {
        "system": "Basic_Hierarchy",
        "total_cost": total_cost,
        "results": results
    }

def print_summary(basic_result):
    """실험 결과 요약"""
    print("\n" + "="*80)
    print("실험 결과 요약")
    print("="*80)
    
    print(f"\n기본 계층 시스템:")
    print(f"  비용: ${basic_result['total_cost']:.6f}")
    print(f"  처리한 태스크: {len(basic_result['results'])}개")
    
    # 성공/실패 통계
    success_count = sum(1 for r in basic_result['results'] if 'error' not in r)
    error_count = len(basic_result['results']) - success_count
    
    # 정확도 통계
    accuracy_scores = [r.get('accuracy', 0) for r in basic_result['results'] if 'accuracy' in r]
    avg_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0
    
    print(f"  성공: {success_count}개, 실패: {error_count}개")
    print(f"  평균 정확도: {avg_accuracy:.2%}")
    
    # 카테고리별 결과
    categories = {}
    for r in basic_result['results']:
        cat = r.get('category', 'Unknown')
        if cat not in categories:
            categories[cat] = {'success': 0, 'error': 0, 'accuracy': []}
        
        if 'error' in r:
            categories[cat]['error'] += 1
        else:
            categories[cat]['success'] += 1
            if 'accuracy' in r:
                categories[cat]['accuracy'].append(r['accuracy'])
    
    print(f"  카테고리별 결과:")
    for cat, stats in categories.items():
        acc_avg = sum(stats['accuracy']) / len(stats['accuracy']) if stats['accuracy'] else 0
        print(f"    {cat}: 성공 {stats['success']}, 실패 {stats['error']}, 정확도 {acc_avg:.2%}")

def main():
    """메인 실행 함수"""
    print("Project Arkhe 통합 실험 시작")
    print("="*80)
    
    # 태스크 로드
    tasks = load_tasks()
    
    print(f"\n실험 설정:")
    print(f"  태스크 수: {len(tasks)}개")
    print(f"  카테고리: {set(task[3] for task in tasks)}")
    
    # 실험 실행
    try:
        basic_result = run_basic_hierarchy_test(tasks)
        
        # 결과 요약
        print_summary(basic_result)
        
        print("\n실험 완료!")
        
    except KeyboardInterrupt:
        print("\n실험이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n실험 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()