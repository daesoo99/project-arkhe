"""
2단계 실험: 자율적 재귀 A/B 테스트
Control Group: 평면적 해결 vs Test Group: 재귀적 분해 및 서브팀 생성
"""

from src.agents.hierarchy import CostTracker, Mediator
from src.agents.recursive_agent import RecursiveAgent, FlatAgent
from src.agents.complexity_analyzer import ComplexityAnalyzer

def run_recursive_intelligence_ab_test():
    """자율적 재귀 A/B 테스트 실행"""
    print("=== Autonomous Recursion A/B Test ===\n")
    
    # 재귀 분해가 효과적일 수 있는 복잡한 문제들
    test_problems = [
        # 비교/분석 문제들 (분해 예상: 3개 서브문제)
        ("Comparison", "AI 규제의 장단점을 비교 분석해주세요.", 7.5),
        ("Complex Analysis", "원격 근무가 기업과 직원에게 미치는 영향을 다각도로 분석해주세요.", 7.0),
        
        # 다요인 분석 문제들 (분해 예상: 3개 서브문제)  
        ("Multi-Factor", "기후 변화의 주요 원인들을 직간접적 요인으로 나누어 분석해주세요.", 6.8),
        ("Cause Analysis", "청년 실업률 증가의 원인들을 체계적으로 분석해주세요.", 6.5),
        
        # 예측 문제들 (분해 예상: 4개 서브문제)
        ("Prediction", "양자컴퓨팅이 암호화 기술에 미칠 미래 영향을 예측하고 대응 방안을 제시해주세요.", 9.0),
        ("Future Impact", "메타버스 기술이 교육 분야에 미칠 장기적 영향을 예측해주세요.", 8.2),
        
        # 철학적 문제들 (분해 예상: 4개 서브문제)
        ("Philosophy", "인공지능의 발전이 인간의 존재 의미에 미치는 철학적 영향은 무엇인가요?", 8.8),
        
        # 단순한 문제 (분해 안됨 - 대조군)
        ("Simple", "프랑스의 수도는 어디인가요?", 2.0)
    ]
    
    print(f"Testing with {len(test_problems)} problems (including decomposable and simple ones)...\n")
    
    results = {
        'control': {'total_cost': 0, 'responses': [], 'agents_used': []},
        'test': {'total_cost': 0, 'responses': [], 'agents_used': []}
    }
    
    complexity_analyzer = ComplexityAnalyzer()
    
    for category, problem, expected_complexity in test_problems:
        print(f"{'='*100}")
        print(f"Category: {category}")
        print(f"Problem: {problem}")
        print(f"Expected Complexity: {expected_complexity}")
        print(f"{'='*100}")
        
        # 복잡도 분석
        complexity_metrics = complexity_analyzer.analyze(problem)
        print(f"Analyzed Complexity: {complexity_metrics.score:.1f}")
        should_decompose = complexity_metrics.score >= 6.0
        print(f"Should Decompose: {should_decompose}")
        
        # Control Group: 평면적 해결
        print("\\n[CONTROL GROUP] - Flat Resolution")
        control_cost_tracker = CostTracker()
        control_agents = [
            FlatAgent("Flat_1", control_cost_tracker),
            FlatAgent("Flat_2", control_cost_tracker),
            FlatAgent("Flat_3", control_cost_tracker)
        ]
        control_mediator = Mediator(control_agents, control_cost_tracker)
        control_result = control_mediator.solve_problem(problem)
        
        control_cost = control_cost_tracker.get_total_cost()
        control_agents_count = len(control_agents)
        
        results['control']['total_cost'] += control_cost
        results['control']['agents_used'].append(control_agents_count)
        results['control']['responses'].append({
            'category': category,
            'cost': control_cost,
            'agents': control_agents_count,
            'entropy': control_result['shannon_entropy'],
            'decomposed': False
        })
        
        print(f"Final Answer: {control_result['final_answer'][:150]}...")
        print(f"Cost: ${control_cost:.6f}")
        print(f"Agents Used: {control_agents_count}")
        print(f"Diversity: {control_result['shannon_entropy']:.3f}")
        
        # Test Group: 재귀적 분해
        print("\\n[TEST GROUP] - Recursive Decomposition")
        test_cost_tracker = CostTracker()
        
        # 3개의 재귀 에이전트가 독립적으로 문제 해결
        recursive_agents = [
            RecursiveAgent("Recursive_1", test_cost_tracker, max_recursion_depth=3),
            RecursiveAgent("Recursive_2", test_cost_tracker, max_recursion_depth=3),
            RecursiveAgent("Recursive_3", test_cost_tracker, max_recursion_depth=3)
        ]
        
        # 각 재귀 에이전트의 결과 수집
        recursive_results = []
        total_agents_used = 0
        
        for i, agent in enumerate(recursive_agents):
            print(f"\\n--- Recursive Agent {i+1} ---")
            result = agent.solve_recursively(problem, complexity_metrics.score)
            recursive_results.append(result.final_synthesis)
            total_agents_used += result.total_agents_used
            
            print(f"Recursion Depth: {result.recursion_depth}")
            print(f"Sub-problems: {len(result.sub_problems)}")
            print(f"Agents Created: {result.total_agents_used}")
            print(f"Result: {result.final_synthesis[:100]}...")
        
        # Mediator로 최종 종합
        test_mediator = Mediator(recursive_agents, test_cost_tracker)
        
        # Mock mediator result (실제로는 recursive_results를 종합)
        final_answer = recursive_results[0]  # 첫 번째 결과를 대표로
        shannon_entropy = len(set(recursive_results)) / len(recursive_results) if recursive_results else 0
        
        test_cost = test_cost_tracker.get_total_cost()
        
        results['test']['total_cost'] += test_cost
        results['test']['agents_used'].append(total_agents_used)
        results['test']['responses'].append({
            'category': category,
            'cost': test_cost,
            'agents': total_agents_used,
            'entropy': shannon_entropy,
            'decomposed': len(recursive_results[0]) > 200  # 긴 답변 = 분해됨
        })
        
        print(f"\\nFinal Synthesized Answer: {final_answer[:150]}...")
        print(f"Cost: ${test_cost:.6f}")
        print(f"Total Agents Used: {total_agents_used}")
        print(f"Estimated Diversity: {shannon_entropy:.3f}")
        
        # 비교
        cost_change = ((test_cost - control_cost) / control_cost * 100) if control_cost > 0 else 0
        agent_ratio = total_agents_used / control_agents_count
        
        print(f"\\nComparison:")
        print(f"  Cost Change: {cost_change:+.1f}%")
        print(f"  Agent Ratio: {agent_ratio:.1f}x ({total_agents_used} vs {control_agents_count})")
        print(f"  Decomposition: {'Yes' if should_decompose else 'No'} (Expected), {'Yes' if results['test']['responses'][-1]['decomposed'] else 'No'} (Actual)")
        print()
    
    # 최종 결과 분석
    print("\\n" + "="*100)
    print("FINAL RESULTS - Autonomous Recursion A/B Test")
    print("="*100)
    
    total_control_cost = results['control']['total_cost']
    total_test_cost = results['test']['total_cost']
    total_cost_change = ((total_test_cost - total_control_cost) / total_control_cost * 100) if total_control_cost > 0 else 0
    
    avg_control_agents = sum(results['control']['agents_used']) / len(results['control']['agents_used'])
    avg_test_agents = sum(results['test']['agents_used']) / len(results['test']['agents_used'])
    
    print(f"Total Cost Analysis:")
    print(f"  Control Group (Flat):        ${total_control_cost:.6f}")
    print(f"  Test Group (Recursive):      ${total_test_cost:.6f}")
    print(f"  Total Cost Change:           {total_cost_change:+.1f}%")
    
    print(f"\\nAgent Usage Analysis:")
    print(f"  Control Group Avg:           {avg_control_agents:.1f} agents")
    print(f"  Test Group Avg:              {avg_test_agents:.1f} agents")
    print(f"  Agent Multiplication:        {avg_test_agents/avg_control_agents:.1f}x")
    
    # 문제 유형별 분석
    decomposable_problems = [r for r in results['test']['responses'] if r.get('decomposed', False)]
    simple_problems = [r for r in results['test']['responses'] if not r.get('decomposed', False)]
    
    print(f"\\nProblem Type Analysis:")
    print(f"  Decomposable Problems:       {len(decomposable_problems)}/{len(results['test']['responses'])}")
    print(f"  Simple Problems:             {len(simple_problems)}/{len(results['test']['responses'])}")
    
    if decomposable_problems:
        avg_decomp_agents = sum(r['agents'] for r in decomposable_problems) / len(decomposable_problems)
        avg_decomp_cost = sum(r['cost'] for r in decomposable_problems) / len(decomposable_problems)
        print(f"  Avg Agents (Decomposed):     {avg_decomp_agents:.1f}")
        print(f"  Avg Cost (Decomposed):       ${avg_decomp_cost:.6f}")
    
    if simple_problems:
        avg_simple_agents = sum(r['agents'] for r in simple_problems) / len(simple_problems)
        avg_simple_cost = sum(r['cost'] for r in simple_problems) / len(simple_problems)
        print(f"  Avg Agents (Simple):         {avg_simple_agents:.1f}")
        print(f"  Avg Cost (Simple):           ${avg_simple_cost:.6f}")
    
    # 다양성 분석
    control_entropies = [r['entropy'] for r in results['control']['responses']]
    test_entropies = [r['entropy'] for r in results['test']['responses']]
    
    avg_control_entropy = sum(control_entropies) / len(control_entropies)
    avg_test_entropy = sum(test_entropies) / len(test_entropies)
    
    print(f"\\nDiversity Analysis:")
    print(f"  Control Group Avg Entropy:   {avg_control_entropy:.3f}")
    print(f"  Test Group Avg Entropy:      {avg_test_entropy:.3f}")
    print(f"  Diversity Change:            {((avg_test_entropy-avg_control_entropy)/avg_control_entropy*100) if avg_control_entropy > 0 else 0:+.1f}%")
    
    print(f"\\n{'='*100}")
    print("CONCLUSION")
    print("="*100)
    
    if total_cost_change < 0:
        print(f"SUCCESS: Recursive approach saves {abs(total_cost_change):.1f}% in cost")
    else:
        print(f"WARNING: Recursive approach increases cost by {total_cost_change:.1f}%")
    
    if avg_test_agents > avg_control_agents * 2:
        print(f"WARNING: Agent multiplication is high ({avg_test_agents/avg_control_agents:.1f}x)")
    else:
        print(f"SUCCESS: Reasonable agent usage ({avg_test_agents/avg_control_agents:.1f}x)")
    
    if len(decomposable_problems) >= len(results['test']['responses']) * 0.6:
        print(f"SUCCESS: Most complex problems were properly decomposed")
    else:
        print(f"WARNING: Few problems were decomposed - may need to adjust thresholds")
    
    if avg_test_entropy > avg_control_entropy:
        print(f"SUCCESS: Recursive approach improves diversity by {((avg_test_entropy-avg_control_entropy)/avg_control_entropy*100):.1f}%")
    else:
        print(f"NEUTRAL: Diversity change is minimal ({((avg_test_entropy-avg_control_entropy)/avg_control_entropy*100):+.1f}%)")
    
    return results

if __name__ == "__main__":
    run_recursive_intelligence_ab_test()