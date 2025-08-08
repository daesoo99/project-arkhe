"""
1단계 실험: 경제적 지능 A/B 테스트
Control Group: 고정 모델 vs Test Group: 동적 모델 선택
"""

from src.agents.hierarchy import CostTracker, Mediator
from src.agents.economic_agent import EconomicAgent, FixedModelAgent

def run_economic_intelligence_ab_test():
    """경제적 지능 A/B 테스트 실행"""
    print("=== Economic Intelligence A/B Test ===\n")
    
    # 다양한 복잡도의 테스트 문제들
    test_problems = [
        # 단순한 문제들 (복잡도 1-3)
        ("Simple Math", "2 + 2는 무엇인가요?"),
        ("Basic Fact", "프랑스의 수도는 어디인가요?"),
        ("Yes/No", "지구는 평평한가요? 예/아니오로 답해주세요."),
        
        # 중간 복잡도 문제들 (복잡도 4-6)
        ("Analysis", "AI 규제가 필요한 이유를 설명해주세요."),
        ("Explanation", "기후 변화의 주요 원인들을 나열해주세요."),
        
        # 복잡한 문제들 (복잡도 7-10)
        ("Comparison", "AI 규제의 장단점을 비교 분석해주세요."),
        ("Philosophy", "인공지능의 발전이 인간의 존재 의미에 미치는 철학적 영향은 무엇인가요?"),
        ("Prediction", "양자컴퓨팅이 암호화 기술에 미칠 미래 영향을 예측하고 대응 방안을 제시해주세요."),
    ]
    
    print(f"Testing with {len(test_problems)} problems of varying complexity...\n")
    
    results = {
        'control': {'total_cost': 0, 'responses': []},
        'test': {'total_cost': 0, 'responses': []}
    }
    
    for category, problem in test_problems:
        print(f"{'='*80}")
        print(f"Category: {category}")
        print(f"Problem: {problem}")
        print(f"{'='*80}")
        
        # Control Group: 고정 모델 (GPT-3.5)
        print("\\n[CONTROL GROUP] - Fixed Model (GPT-3.5)")
        control_cost_tracker = CostTracker()
        control_agents = [
            FixedModelAgent("Control_1", control_cost_tracker),
            FixedModelAgent("Control_2", control_cost_tracker),
            FixedModelAgent("Control_3", control_cost_tracker)
        ]
        control_mediator = Mediator(control_agents, control_cost_tracker)
        control_result = control_mediator.solve_problem(problem)
        
        control_cost = control_cost_tracker.get_total_cost()
        results['control']['total_cost'] += control_cost
        results['control']['responses'].append({
            'category': category,
            'cost': control_cost,
            'entropy': control_result['shannon_entropy']
        })
        
        print(f"Final Answer: {control_result['final_answer'][:100]}...")
        print(f"Cost: ${control_cost:.6f}")
        print(f"Diversity: {control_result['shannon_entropy']:.3f}")
        
        # Test Group: 동적 모델 선택 (경제적 지능)
        print("\\n[TEST GROUP] - Dynamic Model Selection (Economic Intelligence)")
        test_cost_tracker = CostTracker()
        test_agents = [
            EconomicAgent("Economic_1", test_cost_tracker),
            EconomicAgent("Economic_2", test_cost_tracker),
            EconomicAgent("Economic_3", test_cost_tracker)
        ]
        test_mediator = Mediator(test_agents, test_cost_tracker)
        test_result = test_mediator.solve_problem(problem)
        
        test_cost = test_cost_tracker.get_total_cost()
        results['test']['total_cost'] += test_cost
        results['test']['responses'].append({
            'category': category,
            'cost': test_cost,
            'entropy': test_result['shannon_entropy']
        })
        
        print(f"Final Answer: {test_result['final_answer'][:100]}...")
        print(f"Cost: ${test_cost:.6f}")
        print(f"Diversity: {test_result['shannon_entropy']:.3f}")
        
        # 개별 에이전트 모델 선택 확인
        print("\\nIndividual Agent Responses (Test Group):")
        for response in test_result['all_responses']:
            print(f"  - {response[:80]}...")
        
        print(f"\\nCost Comparison: Control=${control_cost:.6f} vs Test=${test_cost:.6f} (Savings: {((control_cost-test_cost)/control_cost*100) if control_cost > 0 else 0:.1f}%)")
        print()
    
    # 최종 결과 분석
    print("\\n" + "="*80)
    print("FINAL RESULTS - Economic Intelligence A/B Test")
    print("="*80)
    
    total_control_cost = results['control']['total_cost']
    total_test_cost = results['test']['total_cost']
    total_savings = ((total_control_cost - total_test_cost) / total_control_cost * 100) if total_control_cost > 0 else 0
    
    print(f"Total Cost Analysis:")
    print(f"  Control Group (Fixed Model):     ${total_control_cost:.6f}")
    print(f"  Test Group (Economic Intel):     ${total_test_cost:.6f}")
    print(f"  Total Cost Savings:              {total_savings:.1f}%")
    
    # 복잡도별 분석
    simple_problems = ['Simple Math', 'Basic Fact', 'Yes/No']
    complex_problems = ['Comparison', 'Philosophy', 'Prediction']
    
    simple_control_cost = sum(r['cost'] for r in results['control']['responses'] if r['category'] in simple_problems)
    simple_test_cost = sum(r['cost'] for r in results['test']['responses'] if r['category'] in simple_problems)
    
    complex_control_cost = sum(r['cost'] for r in results['control']['responses'] if r['category'] in complex_problems)
    complex_test_cost = sum(r['cost'] for r in results['test']['responses'] if r['category'] in complex_problems)
    
    print(f"\\nComplexity Analysis:")
    print(f"  Simple Problems:")
    print(f"    Control: ${simple_control_cost:.6f} | Test: ${simple_test_cost:.6f}")
    print(f"    Savings: {((simple_control_cost-simple_test_cost)/simple_control_cost*100) if simple_control_cost > 0 else 0:.1f}%")
    
    print(f"  Complex Problems:")
    print(f"    Control: ${complex_control_cost:.6f} | Test: ${complex_test_cost:.6f}")
    print(f"    Savings: {((complex_control_cost-complex_test_cost)/complex_control_cost*100) if complex_control_cost > 0 else 0:.1f}%")
    
    # 다양성 분석
    avg_control_entropy = sum(r['entropy'] for r in results['control']['responses']) / len(results['control']['responses'])
    avg_test_entropy = sum(r['entropy'] for r in results['test']['responses']) / len(results['test']['responses'])
    
    print(f"\\nDiversity Analysis:")
    print(f"  Control Group Average Entropy: {avg_control_entropy:.3f}")
    print(f"  Test Group Average Entropy:    {avg_test_entropy:.3f}")
    print(f"  Diversity Change:              {((avg_test_entropy-avg_control_entropy)/avg_control_entropy*100) if avg_control_entropy > 0 else 0:+.1f}%")
    
    print(f"\\n{'='*80}")
    print("CONCLUSION")
    print("="*80)
    
    if total_savings > 0:
        print(f"SUCCESS: Economic Intelligence shows {total_savings:.1f}% cost reduction")
        print(f"SUCCESS: Particularly effective on {'simple' if simple_control_cost - simple_test_cost > complex_control_cost - complex_test_cost else 'complex'} problems")
    else:
        print(f"WARNING: Economic Intelligence shows {abs(total_savings):.1f}% cost increase")
    
    if avg_test_entropy > avg_control_entropy:
        print(f"SUCCESS: Diversity improved by {((avg_test_entropy-avg_control_entropy)/avg_control_entropy*100):.1f}%")
    else:
        print(f"WARNING: Diversity decreased by {((avg_control_entropy-avg_test_entropy)/avg_control_entropy*100):.1f}%")
    
    return results

if __name__ == "__main__":
    run_economic_intelligence_ab_test()