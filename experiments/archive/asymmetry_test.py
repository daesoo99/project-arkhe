#!/usr/bin/env python3
"""
정보 비대칭 실험 래퍼
정보 격리가 편향 감소에 미치는 영향 테스트
"""

import sys
import json
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.hierarchy import CostTracker, Mediator, IndependentThinker

def run_information_asymmetry_test():
    """정보 비대칭 효과 테스트"""
    print("=" * 60)
    print("Project Arkhe - 정보 비대칭 실험")
    print("=" * 60)
    print("독립적 사고 vs 정보 공유가 답변 다양성에 미치는 영향")
    print()
    
    # 편향이 있을 수 있는 질문들 (주관적/논란의 여지가 있는 주제)
    controversial_tasks = [
        ("ai_regulation", "AI 규제가 필요한가요? 찬성/반대 이유를 제시하세요.", "opinion"),
        ("climate_priority", "기후변화 대응과 경제발전 중 무엇이 우선되어야 할까요?", "opinion"),
        ("remote_work", "원격근무가 전통적 사무실 근무보다 효율적인가요?", "opinion"),
        ("nuclear_energy", "원자력 에너지 사용에 대한 입장을 제시하세요.", "opinion")
    ]
    
    print(f"테스트 질문 수: {len(controversial_tasks)}개")
    print("각 질문에 대해 독립 에이전트들의 답변 다양성을 측정합니다.")
    print()
    
    cost_tracker = CostTracker()
    
    # 5개의 독립 에이전트 (더 많은 다양성 확보)
    thinkers = [
        IndependentThinker(f"Independent_Agent_{i+1}", cost_tracker, 'gemma:2b')
        for i in range(5)
    ]
    
    mediator = Mediator(thinkers, cost_tracker)
    
    results = []
    
    for task_id, prompt, task_type in controversial_tasks:
        print(f"\n[{task_type}] {task_id}")
        print(f"질문: {prompt}")
        print("-" * 40)
        
        try:
            result = mediator.solve_problem(prompt)
            
            # 각 에이전트의 답변을 개별 출력
            for i, response in enumerate(result["all_responses"], 1):
                print(f"Agent {i}: {response[:100]}...")
            
            diversity = result["shannon_entropy"]
            contradictions = result["contradiction_report"]
            
            print(f"\n다양성 지수 (Shannon Entropy): {diversity:.2f}")
            print(f"모순 감지: {contradictions}")
            
            # 높은 다양성 = 정보 비대칭 효과가 편향 감소에 기여
            if diversity > 1.0:
                print("높은 다양성 - 정보 비대칭 효과 확인")
            elif diversity > 0.5:
                print("중간 다양성 - 부분적 효과")
            else:
                print("낮은 다양성 - 집단사고 위험")
            
            results.append({
                "task_id": task_id,
                "task_type": task_type,
                "diversity": diversity,
                "contradictions": contradictions,
                "responses_count": len(result["all_responses"])
            })
            
        except Exception as e:
            print(f"오류 발생: {e}")
            results.append({
                "task_id": task_id,
                "error": str(e)
            })
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("정보 비대칭 실험 결과 요약")
    print("=" * 60)
    
    successful_results = [r for r in results if 'error' not in r]
    if successful_results:
        avg_diversity = sum(r["diversity"] for r in successful_results) / len(successful_results)
        high_diversity_count = sum(1 for r in successful_results if r["diversity"] > 1.0)
        
        print(f"평균 다양성 지수: {avg_diversity:.2f}")
        print(f"높은 다양성 달성: {high_diversity_count}/{len(successful_results)}개")
        print(f"정보 비대칭 효과: {'확인됨' if avg_diversity > 1.0 else '부분적' if avg_diversity > 0.5 else '미흡'}")
        
        total_cost = cost_tracker.get_total_cost()
        print(f"총 비용: ${total_cost:.6f}")
    
    return results

def main():
    """메인 실행 함수"""
    try:
        results = run_information_asymmetry_test()
        
        # 결과를 JSON 파일로 저장 (선택사항)
        results_file = project_root / "results" / "asymmetry_test_results.json"
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n상세 결과 저장됨: {results_file}")
        
    except Exception as e:
        print(f"실험 실행 중 오류: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())