"""
4단계 통합 실험: Project Arkhē 완전체 테스트
다양한 설정으로 최적 조합 찾기
"""

from typing import Dict, List
from src.agents.hierarchy import CostTracker, Mediator
from src.agents.integrated_arkhe import (
    IntegratedArkheAgent, ArkheSystemFactory, TraditionalAgent,
    ArkheLevelConfig
)

def run_integrated_arkhe_experiment():
    """통합된 Arkhē 시스템 종합 실험"""
    print("=== Integrated Project Arkhe Experiment ===\\n")
    
    # 다양한 복잡도와 유형의 테스트 문제들
    test_problems = [
        # 단순한 사실 문제들
        ("Simple Fact", "프랑스의 수도는 어디인가요?", 2.0, "간단"),
        ("Basic Knowledge", "태양계에서 가장 큰 행성은 무엇인가요?", 2.5, "간단"),
        
        # 중간 복잡도 분석 문제들  
        ("Analysis", "원격근무가 기업 생산성에 미치는 영향을 분석해주세요.", 6.5, "중간"),
        ("Comparison", "재생에너지와 화석연료의 장단점을 비교해주세요.", 7.0, "중간"),
        
        # 고복잡도 종합 문제들
        ("Complex Policy", "AI 규제 정책 수립 시 고려해야 할 핵심 요소들을 다각도로 분석해주세요.", 8.5, "복잡"),
        ("Future Prediction", "2030년대 메타버스 기술이 교육과 업무에 미칠 변화를 예측하고 대응방안을 제시해주세요.", 9.0, "복잡"),
        
        # 창의적 문제 해결
        ("Creative Problem", "도시 교통 체증 문제를 해결할 혁신적 아이디어 3가지를 제안해주세요.", 8.0, "창의"),
        
        # 철학적/윤리적 문제
        ("Philosophical", "인공지능이 인간의 일자리를 대체하는 것에 대한 윤리적 관점을 제시해주세요.", 8.8, "철학")
    ]
    
    # 테스트할 Arkhē 시스템 설정들
    arkhe_configs = {
        "Traditional": None,  # 기존 방식
        "Basic_Arkhe": ArkheSystemFactory.create_basic_config(),
        "Advanced_Arkhe": ArkheSystemFactory.create_advanced_config(), 
        "Full_Arkhe": ArkheSystemFactory.create_full_config(),
        "Optimized_Arkhe": ArkheSystemFactory.create_optimized_config()
    }
    
    print(f"Testing {len(arkhe_configs)} different configurations with {len(test_problems)} problems...\\n")
    
    # 결과 저장
    results = {config_name: {
        'total_cost': 0.0,
        'total_agents': 0,
        'problem_results': [],
        'method_distribution': {}
    } for config_name in arkhe_configs.keys()}
    
    for prob_idx, (category, problem, expected_complexity, difficulty) in enumerate(test_problems):
        print(f"{'='*120}")
        print(f"Problem {prob_idx + 1}/{len(test_problems)}: {category} ({difficulty})")
        print(f"Question: {problem}")
        print(f"Expected Complexity: {expected_complexity}")
        print(f"{'='*120}")
        
        for config_name, config in arkhe_configs.items():
            print(f"\\n[{config_name.upper()}]")
            
            # 비용 추적기 생성
            cost_tracker = CostTracker()
            
            try:
                if config_name == "Traditional":
                    # 전통적 방식
                    agent = TraditionalAgent("Traditional_Agent", cost_tracker)
                    result = agent.solve(problem)
                else:
                    # Arkhē 방식
                    agent = IntegratedArkheAgent(f"Arkhe_{config_name}", cost_tracker, config)
                    result = agent.solve(problem)
                
                # 비용 계산
                total_cost = cost_tracker.get_total_cost()
                
                # 결과 저장
                results[config_name]['total_cost'] += total_cost
                results[config_name]['total_agents'] += result.get('agents_used', 1)
                
                method_used = result.get('method', 'unknown')
                if method_used not in results[config_name]['method_distribution']:
                    results[config_name]['method_distribution'][method_used] = 0
                results[config_name]['method_distribution'][method_used] += 1
                
                results[config_name]['problem_results'].append({
                    'category': category,
                    'cost': total_cost,
                    'agents_used': result.get('agents_used', 1),
                    'method': method_used,
                    'complexity_score': result.get('complexity_score', 0),
                    'diversity_score': result.get('diversity_score', 0),
                    'confidence_score': result.get('confidence_score', 0)
                })
                
                # 결과 출력
                print(f"Method: {method_used}")
                print(f"Final Answer: {result['final_answer'][:100]}...")
                print(f"Cost: ${total_cost:.6f}")
                print(f"Agents Used: {result.get('agents_used', 1)}")
                
                if 'complexity_score' in result and result['complexity_score'] > 0:
                    print(f"Analyzed Complexity: {result['complexity_score']:.1f}")
                if 'recommended_model' in result:
                    print(f"Recommended Model: {result['recommended_model']}")
                if 'diversity_score' in result and result['diversity_score'] > 0:
                    print(f"Diversity Score: {result['diversity_score']:.3f}")
                
            except Exception as e:
                print(f"ERROR in {config_name}: {e}")
                # 오류 발생시 기본값으로 채움
                results[config_name]['problem_results'].append({
                    'category': category,
                    'cost': 0.001,  # 기본 비용
                    'agents_used': 1,
                    'method': 'error',
                    'complexity_score': 0,
                    'diversity_score': 0,
                    'confidence_score': 0
                })
        
        print()  # 문제 구분용 빈줄
    
    # 최종 종합 분석
    print("\\n" + "="*120)
    print("COMPREHENSIVE ANALYSIS - Integrated Project Arkhe")
    print("="*120)
    
    # 전체 성능 비교
    print("\\nOVERALL PERFORMANCE COMPARISON")
    print("-" * 80)
    
    baseline_cost = results['Traditional']['total_cost']
    baseline_agents = results['Traditional']['total_agents']
    
    for config_name, data in results.items():
        total_cost = data['total_cost']
        total_agents = data['total_agents']
        avg_agents = total_agents / len(test_problems)
        
        cost_change = ((total_cost - baseline_cost) / baseline_cost * 100) if baseline_cost > 0 else 0
        agent_ratio = total_agents / baseline_agents if baseline_agents > 0 else 1
        
        print(f"{config_name:15} | Cost: ${total_cost:.6f} ({cost_change:+6.1f}%) | Agents: {avg_agents:.1f}x ({agent_ratio:.1f}x)")
    
    # 문제 복잡도별 분석
    print("\\nPERFORMANCE BY PROBLEM COMPLEXITY")
    print("-" * 80)
    
    complexity_groups = {
        "간단": [r for config_data in results.values() for r in config_data['problem_results'] if any(p[3] == "간단" for p in test_problems if p[0] == r['category'])],
        "중간": [r for config_data in results.values() for r in config_data['problem_results'] if any(p[3] == "중간" for p in test_problems if p[0] == r['category'])],
        "복잡": [r for config_data in results.values() for r in config_data['problem_results'] if any(p[3] in ["복잡", "창의", "철학"] for p in test_problems if p[0] == r['category'])]
    }
    
    for complexity, group_results in complexity_groups.items():
        if group_results:
            config_groups = {}
            for result in group_results:
                # 결과에서 설정 이름 추출 (간단한 방법)
                for config_name, config_data in results.items():
                    if result in config_data['problem_results']:
                        if config_name not in config_groups:
                            config_groups[config_name] = []
                        config_groups[config_name].append(result)
                        break
            
            print(f"\\n{complexity} 문제들:")
            for config_name, config_results in config_groups.items():
                if config_results:
                    avg_cost = sum(r['cost'] for r in config_results) / len(config_results)
                    avg_agents = sum(r['agents_used'] for r in config_results) / len(config_results)
                    print(f"  {config_name:13}: ${avg_cost:.6f} avg, {avg_agents:.1f} agents")
    
    # 방법론 사용 분석
    print("\\n🔧 METHODOLOGY USAGE ANALYSIS")
    print("-" * 80)
    
    for config_name, data in results.items():
        if data['method_distribution']:
            methods = ", ".join([f"{method}({count})" for method, count in data['method_distribution'].items()])
            print(f"{config_name:15}: {methods}")
    
    # 최적 설정 추천
    print("\\n🏆 OPTIMAL CONFIGURATION RECOMMENDATION")
    print("-" * 80)
    
    # 비용 효율성 계산
    efficiency_scores = {}
    for config_name, data in results.items():
        total_cost = data['total_cost']
        total_agents = data['total_agents']
        
        if total_cost > 0:
            # 간단한 효율성 점수: 낮은 비용, 적절한 에이전트 사용
            cost_penalty = total_cost / baseline_cost if baseline_cost > 0 else 1
            agent_penalty = (total_agents / baseline_agents) if baseline_agents > 0 else 1
            efficiency_scores[config_name] = 1.0 / (cost_penalty * agent_penalty)
        else:
            efficiency_scores[config_name] = 0
    
    # 효율성 순으로 정렬
    ranked_configs = sorted(efficiency_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("Efficiency Ranking (Higher is Better):")
    for i, (config_name, score) in enumerate(ranked_configs, 1):
        total_cost = results[config_name]['total_cost']
        cost_change = ((total_cost - baseline_cost) / baseline_cost * 100) if baseline_cost > 0 else 0
        
        if i == 1:
            status = "🥇 BEST"
        elif i == 2:
            status = "🥈 GOOD" 
        elif i == len(ranked_configs):
            status = "❌ WORST"
        else:
            status = "⚪ OK"
        
        print(f"{i}. {config_name:15} | Score: {score:.3f} | Cost Change: {cost_change:+6.1f}% | {status}")
    
    # 최종 권고사항
    print("\\n💡 FINAL RECOMMENDATIONS")
    print("-" * 80)
    
    best_config = ranked_configs[0][0]
    best_cost_change = ((results[best_config]['total_cost'] - baseline_cost) / baseline_cost * 100) if baseline_cost > 0 else 0
    
    if best_config == "Traditional":
        print("⚠️  Traditional approach remains most efficient")
        print("   → Project Arkhē needs further optimization")
    else:
        print(f"✅ {best_config} is the optimal configuration")
        print(f"   → Achieves {abs(best_cost_change):.1f}% {'cost reduction' if best_cost_change < 0 else 'cost increase'}")
        print(f"   → Uses {results[best_config]['total_agents'] / baseline_agents:.1f}x agents on average")
    
    # 구체적 사용 시나리오 제안
    print("\\n📋 USAGE SCENARIOS")
    print("-" * 80)
    
    simple_problems_winner = min(results.items(), key=lambda x: sum(r['cost'] for r in x[1]['problem_results'] if any(p[3] == "간단" for p in test_problems if p[0] == r['category'])))
    complex_problems_winner = min(results.items(), key=lambda x: sum(r['cost'] for r in x[1]['problem_results'] if any(p[3] in ["복잡", "창의", "철학"] for p in test_problems if p[0] == r['category'])))
    
    print(f"For Simple Problems: Use {simple_problems_winner[0]}")
    print(f"For Complex Problems: Use {complex_problems_winner[0]}")
    print(f"General Purpose: Use {best_config}")
    
    return results, ranked_configs

if __name__ == "__main__":
    run_integrated_arkhe_experiment()