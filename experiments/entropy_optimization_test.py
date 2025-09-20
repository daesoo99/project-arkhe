#!/usr/bin/env python3
"""
Project Arkhē - Shannon Entropy Optimization Test
정보 이론 기반 Multi-Agent 파이프라인 최적화 실험
"""

import json
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm.simple_llm import create_llm_auto
from orchestrator.pipeline import run_3stage_with_context
from utils.information_theory import ShannonEntropyAnalyzer, EntropyBasedOptimizer
from utils.scorers import score_task

def run_entropy_tracked_pipeline(prompt: str, llm_factory, analyzer: ShannonEntropyAnalyzer):
    """엔트로피 추적이 포함된 파이프라인 실행"""
    
    start_time = time.time()
    
    # 수동으로 각 단계 실행하며 엔트로피 추적
    llm_draft = llm_factory("qwen2:0.5b")  
    llm_review = llm_factory("gemma:2b")
    llm_judge = llm_factory("llama3:8b")
    
    # Stage 1: Draft
    draft_prompt = f"다음 질문에 대해 초안을 작성하세요: {prompt}"
    draft_response_dict = llm_draft.generate(draft_prompt)
    draft_response = draft_response_dict.get('response', str(draft_response_dict)) if isinstance(draft_response_dict, dict) else str(draft_response_dict)
    
    # 엔트로피 추적: 입력 → Draft
    flow1 = analyzer.track_information_flow("Draft", prompt, draft_response)
    
    # Stage 2: Review  
    review_prompt = f"""
이전 초안을 검토하고 개선하세요:

원래 질문: {prompt}
초안: {draft_response}

개선된 답변을 제시하세요:
"""
    review_response_dict = llm_review.generate(review_prompt)
    review_response = review_response_dict.get('response', str(review_response_dict)) if isinstance(review_response_dict, dict) else str(review_response_dict)
    
    # 엔트로피 추적: Draft → Review
    flow2 = analyzer.track_information_flow("Review", draft_response, review_response)
    
    # Stage 3: Judge
    judge_prompt = f"""
다음 질문에 대한 최종 답변을 제시하세요:

질문: {prompt}
초안: {draft_response}
검토 의견: {review_response}

위 내용을 종합하여 최고 품질의 최종 답변을 작성하세요:
"""
    judge_response_dict = llm_judge.generate(judge_prompt)
    judge_response = judge_response_dict.get('response', str(judge_response_dict)) if isinstance(judge_response_dict, dict) else str(judge_response_dict)
    
    # 엔트로피 추적: Review → Judge
    flow3 = analyzer.track_information_flow("Judge", review_response, judge_response)
    
    end_time = time.time()
    
    return {
        "prompt": prompt,
        "draft": draft_response,
        "review": review_response, 
        "final": judge_response,
        "execution_time": end_time - start_time,
        "entropy_flows": [flow1, flow2, flow3],
        "tokens": len(prompt.split()) + len(draft_response.split()) + len(review_response.split()) + len(judge_response.split())
    }

def run_single_model_baseline(prompt: str, llm_factory, analyzer: ShannonEntropyAnalyzer):
    """Single 모델 기준 실험 (엔트로피 추적 포함)"""
    
    start_time = time.time()
    
    llm = llm_factory("llama3:8b")
    response_dict = llm.generate(prompt)
    response = response_dict.get('response', str(response_dict)) if isinstance(response_dict, dict) else str(response_dict)
    
    end_time = time.time()
    
    # 엔트로피 추적: 입력 → 출력
    flow = analyzer.track_information_flow("Single", prompt, response)
    
    return {
        "prompt": prompt,
        "final": response,
        "execution_time": end_time - start_time,
        "entropy_flows": [flow],
        "tokens": len(prompt.split()) + len(response.split())
    }

def analyze_entropy_performance(multi_result, single_result, analyzer: ShannonEntropyAnalyzer):
    """엔트로피 기반 성능 분석"""
    
    # Multi-Agent 파이프라인 분석
    multi_analysis = analyzer.analyze_pipeline_efficiency()
    analyzer.stage_history.clear()  # 기록 초기화
    
    # Single Model 분석  
    single_analysis = analyzer.analyze_pipeline_efficiency()
    analyzer.stage_history.clear()
    
    # 비교 메트릭
    comparison = {
        "information_preservation": {
            "multi": multi_analysis.get("information_preservation_rate", 0),
            "single": single_analysis.get("information_preservation_rate", 0)
        },
        "average_efficiency": {
            "multi": multi_analysis.get("average_stage_efficiency", 0),
            "single": single_analysis.get("average_stage_efficiency", 0)
        },
        "final_entropy": {
            "multi": multi_analysis.get("final_entropy", 0),
            "single": single_analysis.get("final_entropy", 0)
        },
        "information_gain": {
            "multi": multi_analysis.get("total_information_gain", 0),
            "single": single_analysis.get("total_information_gain", 0)
        }
    }
    
    return {
        "multi_pipeline_analysis": multi_analysis,
        "single_analysis": single_analysis,
        "comparison": comparison
    }

def main():
    """Shannon Entropy 최적화 실험 메인"""
    print("Shannon Entropy Optimization Test: Information Theory Meets Multi-Agent")
    print("=" * 80)
    
    # 테스트 문제들 (다양한 복잡도)
    test_problems = [
        {
            "id": "simple",
            "prompt": "파리는 어느 나라의 수도인가요?",
            "expected_complexity": 2.0,
            "answer": "프랑스"
        },
        {
            "id": "moderate", 
            "prompt": "기후변화 문제를 해결하기 위한 3가지 혁신적 방법을 제시하고 각각의 장단점을 설명하시오.",
            "expected_complexity": 6.0,
            "answer": "재생에너지 전환, 탄소포집기술, 라이프스타일 변화"
        },
        {
            "id": "creative",
            "prompt": "AI와 인간이 협업하는 2050년 미래 직장의 모습을 상상하여 구체적으로 묘사하시오.",
            "expected_complexity": 8.0,
            "answer": "창의적 상상력 기반 미래 직장 시나리오"
        }
    ]
    
    llm_factory = create_llm_auto
    results = []
    
    for i, problem in enumerate(test_problems):
        print(f"\n--- Entropy Test {i+1}: {problem['id'].upper()} ---")
        print(f"Problem: {problem['prompt'][:60]}...")
        print(f"Expected Complexity: {problem['expected_complexity']}")
        
        try:
            # Multi-Agent 엔트로피 추적 실험
            print("\\n  Running Multi-Agent with Entropy Tracking...")
            multi_analyzer = ShannonEntropyAnalyzer()
            multi_result = run_entropy_tracked_pipeline(problem['prompt'], llm_factory, multi_analyzer)
            
            # Single Model 기준 실험
            print("  Running Single Model baseline...")
            single_analyzer = ShannonEntropyAnalyzer()
            single_result = run_single_model_baseline(problem['prompt'], llm_factory, single_analyzer)
            
            # 엔트로피 성능 분석
            entropy_analysis = analyze_entropy_performance(multi_result, single_result, multi_analyzer)
            
            # 정확도 평가
            multi_score = score_task("reason", problem["answer"], multi_result["final"])
            single_score = score_task("reason", problem["answer"], single_result["final"])
            
            result = {
                "problem": problem,
                "multi_result": multi_result,
                "single_result": single_result,
                "entropy_analysis": entropy_analysis,
                "accuracy": {
                    "multi": multi_score["score"],
                    "single": single_score["score"]
                }
            }
            results.append(result)
            
            # 중간 결과 출력
            print(f"\\n  Information Flow Analysis:")
            
            # Multi-Agent 단계별 정보 흐름
            print(f"    Multi-Agent Pipeline:")
            for flow in multi_result["entropy_flows"]:
                gain_symbol = "📈" if flow.information_gain > 0 else "📉" if flow.information_gain < -0.5 else "➡️"
                print(f"      {gain_symbol} {flow.stage_name}: {flow.input_entropy:.2f} → {flow.output_entropy:.2f} (gain: {flow.information_gain:+.2f})")
            
            # Single Model 정보 흐름
            single_flow = single_result["entropy_flows"][0]
            single_symbol = "📈" if single_flow.information_gain > 0 else "📉" if single_flow.information_gain < -0.5 else "➡️"
            print(f"    Single Model:")
            print(f"      {single_symbol} {single_flow.stage_name}: {single_flow.input_entropy:.2f} → {single_flow.output_entropy:.2f} (gain: {single_flow.information_gain:+.2f})")
            
            # 성능 비교
            comparison = entropy_analysis["comparison"]
            print(f"\\n  Performance Comparison:")
            print(f"    Information Preservation: Multi {comparison['information_preservation']['multi']:.2f} vs Single {comparison['information_preservation']['single']:.2f}")
            print(f"    Final Entropy: Multi {comparison['final_entropy']['multi']:.2f} vs Single {comparison['final_entropy']['single']:.2f}")
            print(f"    Accuracy: Multi {multi_score['score']:.2f} vs Single {single_score['score']:.2f}")
            print(f"    Time: Multi {multi_result['execution_time']:.1f}s vs Single {single_result['execution_time']:.1f}s")
            
            # 엔트로피 기반 판정
            if comparison['final_entropy']['multi'] > comparison['final_entropy']['single'] * 1.2:
                print(f"    🎯 Multi-Agent shows superior information richness!")
            elif comparison['information_preservation']['multi'] > comparison['information_preservation']['single']:
                print(f"    📊 Multi-Agent preserves information better")
            else:
                print(f"    ⚖️ Mixed results - Single model more efficient")
                
        except Exception as e:
            print(f"    Error: {e}")
            continue
    
    # 전체 분석
    print("\\n" + "=" * 80)
    print("Overall Shannon Entropy Analysis")
    print("=" * 80)
    
    if results:
        # 평균 메트릭 계산
        avg_multi_entropy = sum(r["entropy_analysis"]["comparison"]["final_entropy"]["multi"] for r in results) / len(results)
        avg_single_entropy = sum(r["entropy_analysis"]["comparison"]["final_entropy"]["single"] for r in results) / len(results)
        avg_multi_accuracy = sum(r["accuracy"]["multi"] for r in results) / len(results)
        avg_single_accuracy = sum(r["accuracy"]["single"] for r in results) / len(results)
        
        print(f"\\nOverall Information Theory Results:")
        print(f"  Average Final Entropy: Multi {avg_multi_entropy:.2f} vs Single {avg_single_entropy:.2f}")
        print(f"  Average Accuracy: Multi {avg_multi_accuracy:.2f} vs Single {avg_single_accuracy:.2f}")
        print(f"  Information Richness Ratio: {avg_multi_entropy / avg_single_entropy:.2f}x")
        
        # 결론
        entropy_advantage = avg_multi_entropy / avg_single_entropy
        accuracy_advantage = avg_multi_accuracy / avg_single_accuracy
        
        print(f"\\nShannon Entropy Conclusion:")
        if entropy_advantage > 1.3 and accuracy_advantage > 0.9:
            print(f"  📊 Multi-Agent excels in information richness!")
            print(f"  🎯 {((entropy_advantage-1)*100):.1f}% richer information content")
            print(f"  💡 Optimal for problems requiring diverse perspectives")
        elif entropy_advantage > 1.1:
            print(f"  📈 Multi-Agent shows information advantages")
            print(f"  🔍 Better for complex, open-ended problems")
        else:
            print(f"  📉 Single model more information-efficient")
            print(f"  ⚡ Better for straightforward, focused tasks")
        
        # 최적화 제안
        print(f"\\nOptimization Suggestions:")
        optimizer = EntropyBasedOptimizer(ShannonEntropyAnalyzer())
        
        for result in results:
            problem_id = result["problem"]["id"]
            complexity = result["problem"]["expected_complexity"]
            optimal_config = optimizer.calculate_optimal_model_allocation(complexity)
            print(f"  {problem_id.upper()}: {optimal_config}")
        
        # 결과 저장
        timestamp = int(time.time())
        output_file = Path(__file__).parent.parent / "results" / f"entropy_optimization_results_{timestamp}.json"
        output_file.parent.mkdir(exist_ok=True)
        
        # JSON 직렬화를 위해 결과 정리
        json_results = []
        for result in results:
            json_result = {
                "problem": result["problem"],
                "multi_accuracy": result["accuracy"]["multi"],
                "single_accuracy": result["accuracy"]["single"],
                "multi_entropy": result["entropy_analysis"]["comparison"]["final_entropy"]["multi"],
                "single_entropy": result["entropy_analysis"]["comparison"]["final_entropy"]["single"],
                "multi_time": result["multi_result"]["execution_time"],
                "single_time": result["single_result"]["execution_time"]
            }
            json_results.append(json_result)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "experiment": "shannon_entropy_optimization",
                "timestamp": timestamp,
                "results": json_results,
                "summary": {
                    "avg_entropy_advantage": entropy_advantage,
                    "avg_accuracy_advantage": accuracy_advantage,
                    "information_richness_improvement": (entropy_advantage - 1) * 100
                }
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\\nResults saved: {output_file}")

if __name__ == "__main__":
    main()