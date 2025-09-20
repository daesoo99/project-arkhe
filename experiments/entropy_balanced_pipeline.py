#!/usr/bin/env python3
"""
Project Arkhē - Entropy Balanced Pipeline
Shannon Entropy 이론을 활용한 균형잡힌 Multi-Agent 파이프라인
"""

import sys
from pathlib import Path

# Add src to path  
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm.simple_llm import create_llm_auto
from utils.information_theory import ShannonEntropyAnalyzer

class EntropyBalancedPipeline:
    """Shannon Entropy 균형을 맞춘 최적화된 파이프라인"""
    
    def __init__(self, llm_factory):
        self.llm_factory = llm_factory
        self.analyzer = ShannonEntropyAnalyzer()
        
        # 최적 엔트로피 범위 (실험적으로 도출)
        self.target_entropy_ranges = {
            "draft": (2.0, 4.0),      # 다양하지만 너무 과하지 않게
            "review": (3.0, 5.0),     # 좀 더 풍부하게  
            "judge": (3.5, 6.0)       # 최종 결정을 위한 적정 복잡도
        }
    
    def entropy_controlled_draft(self, prompt: str) -> str:
        """엔트로피 제어된 Draft 단계"""
        
        # 초기 Draft 생성
        llm = self.llm_factory("qwen2:7b")  # 더 큰 모델로 품질 향상
        
        draft_prompt = f"""
다음 질문에 대해 핵심적이면서도 다각도의 초안을 작성하세요:

{prompt}

요구사항:
- 3-4개의 서로 다른 관점을 간결하게 제시
- 각 관점마다 2-3문장으로 설명
- 너무 반복적이지 않도록 다양한 표현 사용
"""
        
        response_dict = llm.generate(draft_prompt)
        draft = response_dict.get('response', str(response_dict)) if isinstance(response_dict, dict) else str(response_dict)
        
        # 엔트로피 체크 및 조정
        current_entropy = self.analyzer.calculate_shannon_entropy(draft)
        target_min, target_max = self.target_entropy_ranges["draft"]
        
        if current_entropy < target_min:
            # 엔트로피 너무 낮음 - 다양성 추가
            enhancement_prompt = f"""
다음 초안을 더 다양하고 창의적으로 확장하세요:

{draft}

추가로 고려할 관점:
- 다른 업계/분야의 유사 사례
- 예상치 못한 부작용이나 기회
- 혁신적이거나 비전통적 접근법
"""
            enhanced_dict = llm.generate(enhancement_prompt)
            draft = enhanced_dict.get('response', str(enhanced_dict)) if isinstance(enhanced_dict, dict) else str(enhanced_dict)
            
        elif current_entropy > target_max:
            # 엔트로피 너무 높음 - 핵심 정리
            refinement_prompt = f"""
다음 초안의 핵심 내용을 정리하여 더 집중된 형태로 재작성하세요:

{draft}

요구사항:
- 가장 중요한 3개 포인트만 선별
- 중복되는 내용 제거  
- 명확하고 간결한 표현 사용
"""
            refined_dict = llm.generate(refinement_prompt)
            draft = refined_dict.get('response', str(refined_dict)) if isinstance(refined_dict, dict) else str(refined_dict)
        
        return draft
    
    def entropy_controlled_review(self, prompt: str, draft: str) -> str:
        """엔트로피 제어된 Review 단계"""
        
        llm = self.llm_factory("qwen2:7b")
        
        review_prompt = f"""
다음 초안을 검토하고 균형잡힌 개선안을 제시하세요:

원래 질문: {prompt}
초안: {draft}

검토 기준:
1. 누락된 중요한 관점이 있는가?
2. 논리적 일관성은 충분한가?  
3. 실용성과 창의성의 균형은 적절한가?

개선된 답변을 작성하되, 지나치게 복잡하지 않도록 주의하세요.
"""
        
        response_dict = llm.generate(review_prompt)
        review = response_dict.get('response', str(response_dict)) if isinstance(response_dict, dict) else str(response_dict)
        
        # 엔트로피 균형 체크
        current_entropy = self.analyzer.calculate_shannon_entropy(review)
        target_min, target_max = self.target_entropy_ranges["review"]
        
        # Review 단계에서는 적당한 복잡도 유지
        if current_entropy < target_min or current_entropy > target_max:
            balance_prompt = f"""
다음 내용을 적절한 복잡도로 조정하세요:

{review}

목표: 너무 단순하지도, 너무 복잡하지도 않은 균형잡힌 분석
- 핵심 아이디어는 유지
- 적당한 세부사항 포함
- 명확한 구조와 흐름
"""
            balanced_dict = llm.generate(balance_prompt)
            review = balanced_dict.get('response', str(balanced_dict)) if isinstance(balanced_dict, dict) else str(balanced_dict)
        
        return review
    
    def entropy_controlled_judge(self, prompt: str, draft: str, review: str) -> str:
        """엔트로피 제어된 Judge 단계"""
        
        llm = self.llm_factory("llama3:8b")  # 최고 성능 모델
        
        judge_prompt = f"""
다음 내용을 종합하여 최적의 최종 답변을 작성하세요:

질문: {prompt}
초안: {draft}
검토의견: {review}

최종 답변 작성 지침:
- 초안과 검토의견의 장점을 모두 활용
- 명확하고 실용적인 결론 도출
- 적절한 깊이와 폭을 갖춘 종합적 답변
- 독자가 이해하고 실행할 수 있는 수준

균형잡힌 최고 품질의 답변을 제시하세요.
"""
        
        response_dict = llm.generate(judge_prompt)
        final = response_dict.get('response', str(response_dict)) if isinstance(response_dict, dict) else str(response_dict)
        
        return final
    
    def execute_balanced_pipeline(self, prompt: str):
        """균형잡힌 엔트로피 파이프라인 실행"""
        
        print(f"Executing Entropy-Balanced Pipeline...")
        print(f"Question: {prompt[:60]}...")
        
        # Stage 1: Entropy-Controlled Draft
        print("  Stage 1: Generating balanced draft...")
        draft = self.entropy_controlled_draft(prompt)
        draft_entropy = self.analyzer.calculate_shannon_entropy(draft)
        print(f"    Draft entropy: {draft_entropy:.2f}")
        
        # Stage 2: Entropy-Controlled Review  
        print("  Stage 2: Balanced review...")
        review = self.entropy_controlled_review(prompt, draft)
        review_entropy = self.analyzer.calculate_shannon_entropy(review)
        print(f"    Review entropy: {review_entropy:.2f}")
        
        # Stage 3: Entropy-Controlled Judge
        print("  Stage 3: Final balanced judgment...")
        final = self.entropy_controlled_judge(prompt, draft, review)
        final_entropy = self.analyzer.calculate_shannon_entropy(final)
        print(f"    Final entropy: {final_entropy:.2f}")
        
        return {
            "prompt": prompt,
            "draft": draft,
            "review": review, 
            "final": final,
            "entropy_progression": [draft_entropy, review_entropy, final_entropy],
            "pipeline_type": "entropy_balanced"
        }

def test_entropy_balanced_pipeline():
    """엔트로피 균형 파이프라인 테스트"""
    
    llm_factory = create_llm_auto
    pipeline = EntropyBalancedPipeline(llm_factory)
    
    # 테스트 문제들
    test_questions = [
        "미래 도시의 교통 체증을 해결할 수 있는 혁신적 방법은?",
        "AI와 인간이 협업하는 이상적인 직장 환경을 어떻게 만들 수 있을까?",
        "기후변화에 대응하기 위한 개인 차원의 실천 방안은?"
    ]
    
    results = []
    
    for question in test_questions:
        print(f"\n{'='*60}")
        result = pipeline.execute_balanced_pipeline(question)
        results.append(result)
        
        print(f"\nEntropy Progression:")
        entropies = result["entropy_progression"]
        print(f"  Draft → Review → Final: {entropies[0]:.2f} → {entropies[1]:.2f} → {entropies[2]:.2f}")
        
        # 엔트로피 안정성 체크
        if 3.0 <= entropies[2] <= 6.0:
            print(f"  ✓ Final entropy in optimal range")
        else:
            print(f"  ⚠ Final entropy outside optimal range")
        
        print(f"\nFinal Answer Preview:")
        print(f"  {result['final'][:200]}...")
    
    print(f"\n{'='*60}")
    print("Entropy-Balanced Pipeline Test Complete")
    
    # 전체 엔트로피 패턴 분석
    all_final_entropies = [r["entropy_progression"][2] for r in results]
    avg_final_entropy = sum(all_final_entropies) / len(all_final_entropies)
    
    print(f"\nOverall Results:")
    print(f"  Average final entropy: {avg_final_entropy:.2f}")
    print(f"  Entropy consistency: {min(all_final_entropies):.2f} - {max(all_final_entropies):.2f}")
    
    if 3.5 <= avg_final_entropy <= 5.5:
        print(f"  🎯 Optimal entropy balance achieved!")
        print(f"  💡 This pipeline should provide rich but focused responses")
    else:
        print(f"  🔧 Entropy balance needs adjustment")

if __name__ == "__main__":
    test_entropy_balanced_pipeline()