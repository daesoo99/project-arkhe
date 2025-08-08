"""
3단계 실험: 정보 비대칭 A/B 테스트
Control Group: 완전한 정보 공유 vs Test Group: 의도적 정보 격리
"""

from typing import List
from src.agents.hierarchy import CostTracker, Mediator
from src.agents.information_asymmetry import (
    IsolatedAgent, TransparentAgent, InformationIsolationEngine, 
    IsolationLevel, CrossValidationEngine
)

def run_information_asymmetry_ab_test():
    """정보 비대칭 A/B 테스트 실행"""
    print("=== Information Asymmetry A/B Test ===\\n")
    
    # 정보 비대칭의 효과를 측정할 수 있는 문제들
    test_problems = [
        # 다양한 관점이 중요한 문제들
        ("Multi-Perspective", "AI 윤리 가이드라인 수립 시 고려해야 할 핵심 요소들을 제시해주세요.", "고등"),
        ("Controversial", "원격근무의 장단점을 다각도로 분석해주세요.", "중등"),
        ("Creative", "미래 교육 시스템의 혁신적 방향을 제안해주세요.", "고등"),
        
        # 군중사고가 발생하기 쉬운 문제들
        ("Popular Opinion", "기후변화 대응을 위한 개인의 역할은 무엇인가요?", "중등"),
        ("Social Issue", "소셜미디어가 청소년에게 미치는 영향을 평가해주세요.", "중등"),
        
        # 창의적 사고가 필요한 문제들
        ("Innovation", "도시 교통 체증을 해결할 혁신적 아이디어를 제안해주세요.", "고등"),
        ("Future Scenario", "2040년 인간과 AI의 협업 모델을 상상해서 설명해주세요.", "고등"),
        
        # 간단한 문제 (대조군)
        ("Simple Fact", "태양계에서 가장 큰 행성은 무엇인가요?", "하등")
    ]
    
    print(f"Testing with {len(test_problems)} problems focusing on perspective diversity...\\n")
    
    results = {
        'control': {'total_cost': 0, 'responses': [], 'diversity_scores': [], 'consistency_scores': []},
        'test': {'total_cost': 0, 'responses': [], 'diversity_scores': [], 'consistency_scores': []}
    }
    
    cross_validator = CrossValidationEngine()
    
    for category, problem, difficulty in test_problems:
        print(f"{'='*100}")
        print(f"Category: {category}")
        print(f"Problem: {problem}")
        print(f"Difficulty: {difficulty}")
        print(f"{'='*100}")
        
        # Control Group: 완전한 정보 공유 (기존 방식)
        print("\\n[CONTROL GROUP] - Complete Information Sharing")
        control_cost_tracker = CostTracker()
        
        # 투명한 에이전트들 (서로의 정보 공유)
        transparent_agents = [
            TransparentAgent("Transparent_1", control_cost_tracker),
            TransparentAgent("Transparent_2", control_cost_tracker), 
            TransparentAgent("Transparent_3", control_cost_tracker)
        ]
        
        # 순차적으로 해결하면서 정보 공유
        control_responses = []
        for i, agent in enumerate(transparent_agents):
            # 이전 에이전트들의 응답을 공유 맥락으로 추가
            for prev_response in control_responses:
                agent.add_shared_context(prev_response[:100])  # 처음 100자만
            
            response = agent.solve_with_shared_info(problem)
            control_responses.append(response)
            print(f"Agent {i+1}: {response[:120]}...")
        
        # Mediator로 최종 종합
        control_mediator = Mediator(transparent_agents, control_cost_tracker)
        control_result = control_mediator.solve_problem(problem)
        
        control_cost = control_cost_tracker.get_total_cost()
        
        # 교차 검증
        control_validation = cross_validator.cross_validate(control_responses)
        
        results['control']['total_cost'] += control_cost
        results['control']['diversity_scores'].append(control_validation['diversity'])
        results['control']['consistency_scores'].append(control_validation['consistency'])
        results['control']['responses'].append({
            'category': category,
            'cost': control_cost,
            'diversity': control_validation['diversity'],
            'consistency': control_validation['consistency'],
            'confidence': control_validation['confidence']
        })
        
        print(f"Final Answer: {control_result['final_answer'][:120]}...")
        print(f"Cost: ${control_cost:.6f}")
        print(f"Diversity: {control_validation['diversity']:.3f}")
        print(f"Consistency: {control_validation['consistency']:.3f}")
        print(f"Confidence: {control_validation['confidence']:.3f}")
        print(f"Validated Points: {', '.join(control_validation['validated_points'][:3])}")
        
        # Test Group: 의도적 정보 격리
        print("\\n[TEST GROUP] - Intentional Information Isolation")
        test_cost_tracker = CostTracker()
        
        # 정보 격리 엔진 초기화
        isolation_engine = InformationIsolationEngine(IsolationLevel.COMPLETE)
        
        # 다양한 사고 스타일의 격리된 에이전트들
        isolated_agents = [
            IsolatedAgent("Isolated_1", test_cost_tracker, "analytical"),
            IsolatedAgent("Isolated_2", test_cost_tracker, "creative"),
            IsolatedAgent("Isolated_3", test_cost_tracker, "skeptical")
        ]
        
        # 각 에이전트에게 격리된 맥락 설정
        test_responses = []
        for i, agent in enumerate(isolated_agents):
            # 완전히 독립적인 맥락 생성 (다른 에이전트 정보 차단)
            context = isolation_engine.create_isolated_context(f"agent_{i+1}", problem)
            agent.set_context(context)
            
            response = agent.solve_isolated(problem)
            test_responses.append(response)
            print(f"Agent {i+1} ({agent.thinking_style}): {response[:120]}...")
            print(f"  Thinking Seed: {context.thinking_seed}")
            print(f"  Hidden Info: {len(context.hidden_info)} items")
        
        test_cost = test_cost_tracker.get_total_cost()
        
        # 독립적 결과들의 교차 검증
        test_validation = cross_validator.cross_validate(test_responses)
        
        results['test']['total_cost'] += test_cost
        results['test']['diversity_scores'].append(test_validation['diversity'])
        results['test']['consistency_scores'].append(test_validation['consistency'])
        results['test']['responses'].append({
            'category': category,
            'cost': test_cost,
            'diversity': test_validation['diversity'],
            'consistency': test_validation['consistency'],
            'confidence': test_validation['confidence']
        })
        
        # 최종 종합 (격리된 결과들만 활용)
        final_synthesis = _synthesize_isolated_results(test_responses, test_validation)
        
        print(f"\\nFinal Synthesized Answer: {final_synthesis[:120]}...")
        print(f"Cost: ${test_cost:.6f}")
        print(f"Diversity: {test_validation['diversity']:.3f}")
        print(f"Consistency: {test_validation['consistency']:.3f}")
        print(f"Confidence: {test_validation['confidence']:.3f}")
        print(f"Divergent Views: {len(test_validation['divergent_views'])} unique perspectives")
        
        # 비교 분석
        diversity_improvement = ((test_validation['diversity'] - control_validation['diversity']) / control_validation['diversity'] * 100) if control_validation['diversity'] > 0 else 0
        cost_change = ((test_cost - control_cost) / control_cost * 100) if control_cost > 0 else 0
        consistency_change = ((test_validation['consistency'] - control_validation['consistency']) / control_validation['consistency'] * 100) if control_validation['consistency'] > 0 else 0
        
        print(f"\\nComparison:")
        print(f"  Diversity Change: {diversity_improvement:+.1f}%")
        print(f"  Consistency Change: {consistency_change:+.1f}%") 
        print(f"  Cost Change: {cost_change:+.1f}%")
        print(f"  Groupthink Risk: {'Reduced' if test_validation['diversity'] > control_validation['diversity'] else 'Similar'}")
        print()
    
    # 최종 결과 분석
    print("\\n" + "="*100)
    print("FINAL RESULTS - Information Asymmetry A/B Test")
    print("="*100)
    
    total_control_cost = results['control']['total_cost']
    total_test_cost = results['test']['total_cost']
    total_cost_change = ((total_test_cost - total_control_cost) / total_control_cost * 100) if total_control_cost > 0 else 0
    
    avg_control_diversity = sum(results['control']['diversity_scores']) / len(results['control']['diversity_scores'])
    avg_test_diversity = sum(results['test']['diversity_scores']) / len(results['test']['diversity_scores'])
    diversity_improvement = ((avg_test_diversity - avg_control_diversity) / avg_control_diversity * 100) if avg_control_diversity > 0 else 0
    
    avg_control_consistency = sum(results['control']['consistency_scores']) / len(results['control']['consistency_scores'])
    avg_test_consistency = sum(results['test']['consistency_scores']) / len(results['test']['consistency_scores'])
    consistency_change = ((avg_test_consistency - avg_control_consistency) / avg_control_consistency * 100) if avg_control_consistency > 0 else 0
    
    print(f"Cost Analysis:")
    print(f"  Control Group (Shared Info):     ${total_control_cost:.6f}")
    print(f"  Test Group (Isolated):           ${total_test_cost:.6f}")
    print(f"  Total Cost Change:               {total_cost_change:+.1f}%")
    
    print(f"\\nDiversity Analysis:")
    print(f"  Control Group Avg Diversity:     {avg_control_diversity:.3f}")
    print(f"  Test Group Avg Diversity:        {avg_test_diversity:.3f}")
    print(f"  Diversity Improvement:           {diversity_improvement:+.1f}%")
    
    print(f"\\nConsistency Analysis:")
    print(f"  Control Group Avg Consistency:   {avg_control_consistency:.3f}")
    print(f"  Test Group Avg Consistency:      {avg_test_consistency:.3f}")
    print(f"  Consistency Change:              {consistency_change:+.1f}%")
    
    # 문제 난이도별 분석
    high_difficulty = [r for r in results['test']['responses'] if any(p[2] == "고등" for p in test_problems if p[0] == r['category'])]
    medium_difficulty = [r for r in results['test']['responses'] if any(p[2] == "중등" for p in test_problems if p[0] == r['category'])]
    
    if high_difficulty:
        avg_high_diversity = sum(r['diversity'] for r in high_difficulty) / len(high_difficulty)
        print(f"\\nDifficulty Analysis:")
        print(f"  High Difficulty Avg Diversity:   {avg_high_diversity:.3f}")
    
    if medium_difficulty:
        avg_med_diversity = sum(r['diversity'] for r in medium_difficulty) / len(medium_difficulty)
        print(f"  Medium Difficulty Avg Diversity: {avg_med_diversity:.3f}")
    
    print(f"\\n{'='*100}")
    print("CONCLUSION")
    print("="*100)
    
    if diversity_improvement > 10:
        print(f"SUCCESS: Information isolation significantly improves diversity by {diversity_improvement:.1f}%")
    elif diversity_improvement > 0:
        print(f"SUCCESS: Information isolation moderately improves diversity by {diversity_improvement:.1f}%")
    else:
        print(f"WARNING: Information isolation reduces diversity by {abs(diversity_improvement):.1f}%")
    
    if abs(total_cost_change) < 20:
        print(f"SUCCESS: Cost impact is minimal ({total_cost_change:+.1f}%)")
    else:
        print(f"WARNING: Significant cost change ({total_cost_change:+.1f}%)")
    
    if consistency_change > -30 and consistency_change < 0:
        print(f"SUCCESS: Healthy reduction in groupthink (consistency {consistency_change:+.1f}%)")
    elif consistency_change >= 0:
        print(f"NEUTRAL: Consistency maintained or increased ({consistency_change:+.1f}%)")
    else:
        print(f"WARNING: Excessive consistency reduction may indicate fragmentation ({consistency_change:+.1f}%)")
    
    # 최종 권고사항
    if diversity_improvement > 5 and abs(total_cost_change) < 30:
        print("\\nRECOMMENDATION: Information asymmetry approach shows promise for complex problems")
    else:
        print("\\nRECOMMENDATION: Information asymmetry needs parameter tuning for optimal results")
    
    return results

def _synthesize_isolated_results(responses: List[str], validation: dict) -> str:
    """격리된 결과들을 종합하여 최종 답변 생성"""
    # 검증된 포인트들과 분기된 관점들을 활용한 종합
    validated_points = validation.get('validated_points', [])
    divergent_views = validation.get('divergent_views', [])
    
    synthesis = "[정보 격리 종합] 독립적 분석 결과:\\n"
    
    if validated_points:
        synthesis += f"공통 검증 포인트: {', '.join(validated_points[:3])}\\n"
    
    if divergent_views:
        synthesis += f"다양한 관점들:\\n"
        for view in divergent_views[:3]:
            synthesis += f"- {view}\\n"
    
    synthesis += "\\n결론: 다양한 독립적 관점들을 통해 더 포괄적이고 균형잡힌 이해를 도출했습니다."
    
    return synthesis

if __name__ == "__main__":
    run_information_asymmetry_ab_test()