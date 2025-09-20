#!/usr/bin/env python3
"""
Project Arkhē - Creative Collaboration Test
Multi-Agent가 정말 빛날 수 있는 창의적 협업 영역 테스트
"""

import json
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm.simple_llm import create_llm_auto
from orchestrator.pipeline import run_3stage_with_context

def run_creative_brainstorming(topic, llm_factory):
    """창의적 브레인스토밍: 각 Agent가 다른 관점에서 아이디어 생성"""
    
    # 각 Agent에게 다른 페르소나와 관점 부여
    personas = {
        "optimist": "당신은 낙관적이고 가능성을 보는 미래학자입니다. 긍정적이고 혁신적인 관점에서",
        "realist": "당신은 현실적이고 실용적인 엔지니어입니다. 구현 가능성과 제약사항을 고려하여", 
        "contrarian": "당신은 비판적 사고를 하는 철학자입니다. 문제점과 예상치 못한 결과를 분석하여"
    }
    
    prompt = f"""
{topic}에 대한 혁신적인 아이디어를 3가지 제시하시오.

각 단계별 관점:
1. Draft: {personas["optimist"]} 자유롭고 창의적인 아이디어를 제시
2. Review: {personas["realist"]} Draft의 아이디어를 현실성 있게 개선
3. Judge: {personas["contrarian"]} 모든 아이디어의 장단점을 종합 평가

서로 다른 관점이 만나 더 풍부한 결과를 만들어보세요.
"""
    
    start_time = time.time()
    result = run_3stage_with_context(llm_factory, prompt)
    end_time = time.time()
    
    return {
        "topic": topic,
        "approach": "creative_multi_agent", 
        "result": result,
        "time": end_time - start_time,
        "stages": 3
    }

def run_single_creative(topic, llm_factory):
    """단일 모델로 창의적 작업 수행"""
    
    prompt = f"{topic}에 대한 혁신적인 아이디어를 3가지 제시하시오."
    
    start_time = time.time()
    llm = llm_factory("llama3:8b")
    response_dict = llm.generate(prompt)
    response = response_dict.get('response', str(response_dict)) if isinstance(response_dict, dict) else str(response_dict)
    end_time = time.time()
    
    return {
        "topic": topic,
        "approach": "single_model",
        "result": {"final": response},
        "time": end_time - start_time,
        "stages": 1
    }

def analyze_creativity_metrics(multi_result, single_result):
    """창의성 메트릭 분석"""
    
    # 아이디어 다양성 측정 (단어 다양성으로 근사)
    multi_text = str(multi_result["result"])
    single_text = str(single_result["result"])
    
    multi_words = set(multi_text.lower().split())
    single_words = set(single_text.lower().split())
    
    # 메트릭 계산
    metrics = {
        "multi_word_diversity": len(multi_words),
        "single_word_diversity": len(single_words),
        "diversity_ratio": len(multi_words) / len(single_words) if len(single_words) > 0 else 0,
        
        "multi_length": len(multi_text),
        "single_length": len(single_text),
        "elaboration_ratio": len(multi_text) / len(single_text) if len(single_text) > 0 else 0,
        
        "multi_time": multi_result["time"],
        "single_time": single_result["time"],
        "efficiency_cost": multi_result["time"] / single_result["time"] if single_result["time"] > 0 else 0
    }
    
    return metrics

def main():
    """창의적 협업 실험 메인"""
    print("Creative Collaboration Test: Where Multi-Agent Shines")
    print("=" * 60)
    
    # 창의적 문제들 (정답이 없고 다양성이 중요한 영역)
    creative_topics = [
        "미래 도시의 교통 체증 해결 방안",
        "AI와 인간이 공존하는 새로운 직업군",
        "기후변화에 대응하는 혁신적 라이프스타일",
        "메타버스에서의 새로운 교육 방식",
        "우주 시대의 새로운 스포츠 종목"
    ]
    
    llm_factory = create_llm_auto
    results = []
    
    for i, topic in enumerate(creative_topics[:3]):  # 3개만 테스트
        print(f"\n--- Creative Test {i+1}: {topic} ---")
        
        try:
            # Multi-Agent 창의적 협업
            print("  Testing Multi-Agent Creative Collaboration...")
            multi_result = run_creative_brainstorming(topic, llm_factory)
            
            # Single Model 창의적 작업
            print("  Testing Single Model Creative Work...")
            single_result = run_single_creative(topic, llm_factory)
            
            # 창의성 메트릭 분석
            metrics = analyze_creativity_metrics(multi_result, single_result)
            
            result = {
                "topic": topic,
                "multi_agent": multi_result,
                "single_model": single_result,
                "creativity_metrics": metrics
            }
            results.append(result)
            
            # 결과 출력
            print(f"\n  Results:")
            print(f"    Word Diversity: Multi {metrics['multi_word_diversity']} vs Single {metrics['single_word_diversity']} (ratio: {metrics['diversity_ratio']:.2f})")
            print(f"    Elaboration: Multi {metrics['multi_length']} vs Single {metrics['single_length']} chars (ratio: {metrics['elaboration_ratio']:.2f})")
            print(f"    Time Cost: {metrics['efficiency_cost']:.2f}x slower")
            
            # 창의성 판단
            if metrics['diversity_ratio'] > 1.2 and metrics['elaboration_ratio'] > 1.1:
                print(f"    🌟 Multi-Agent shows superior creativity!")
            elif metrics['diversity_ratio'] > 1.0:
                print(f"    ✨ Multi-Agent shows better diversity")
            else:
                print(f"    🤔 Single Model performs comparably")
                
        except Exception as e:
            print(f"    Error: {e}")
            continue
    
    # 전체 분석
    print("\n" + "=" * 60)
    print("Creative Collaboration Analysis")
    print("=" * 60)
    
    if results:
        # 평균 메트릭
        avg_metrics = {
            "diversity_ratio": sum(r["creativity_metrics"]["diversity_ratio"] for r in results) / len(results),
            "elaboration_ratio": sum(r["creativity_metrics"]["elaboration_ratio"] for r in results) / len(results),
            "efficiency_cost": sum(r["creativity_metrics"]["efficiency_cost"] for r in results) / len(results)
        }
        
        print(f"\nOverall Creative Performance:")
        print(f"  Average Diversity Ratio: {avg_metrics['diversity_ratio']:.2f}")
        print(f"  Average Elaboration Ratio: {avg_metrics['elaboration_ratio']:.2f}") 
        print(f"  Average Time Cost: {avg_metrics['efficiency_cost']:.2f}x")
        
        # 결론
        print(f"\nCreative Collaboration Conclusion:")
        if avg_metrics['diversity_ratio'] > 1.3:
            print(f"  🎨 Multi-Agent excels in creative domains!")
            print(f"  📈 {((avg_metrics['diversity_ratio']-1)*100):.1f}% more diverse ideas")
            print(f"  🔍 This is where collaborative intelligence truly shines")
        elif avg_metrics['diversity_ratio'] > 1.1:
            print(f"  🌱 Multi-Agent shows creative advantages")
            print(f"  💡 Worth the extra cost for creative projects")
        else:
            print(f"  📊 Mixed results - need deeper analysis")
            
        # 결과 저장
        timestamp = int(time.time())
        output_file = Path(__file__).parent.parent / "results" / f"creative_collaboration_results_{timestamp}.json"
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "experiment": "creative_collaboration_test",
                "timestamp": timestamp,
                "results": results,
                "summary": avg_metrics
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved: {output_file}")

if __name__ == "__main__":
    main()