# -*- coding: utf-8 -*-
"""
Project Arkhē - 통합 파이프라인 테스트
파이프라인별 AB 비교 및 경제적 지능 측정
"""

import sys
import time
from pathlib import Path
import json
from typing import List, Dict, Any

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.orchestrator.pipeline import Pipeline, PipelineBuilder, PRESET_PIPELINES

class EconomicIntelligenceMetrics:
    """경제적 지능 메트릭 계산기"""
    
    def __init__(self):
        # 비용 가중치 (α, β)
        self.latency_weight = 0.3      # 지연 시간 가중치
        self.compute_weight = 0.7      # 계산 비용 가중치
        
        # 모델별 추정 컴퓨팅 비용 (상대적)
        self.compute_costs = {
            "gemma:2b": 1.0,    # 기준값
            "phi3:mini": 1.2,
            "qwen2:0.5b": 0.8,
            "mistral:7b": 3.5,
            "llama3:8b": 4.0,
            "codellama:7b": 3.5,
            "gpt-4o-mini": 10.0,  # 클라우드 비용 추정
        }
    
    def calculate_cost_score(self, pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
        """비용 점수 계산"""
        total_latency = pipeline_result.get("total_latency_ms", 0) / 1000.0  # 초 단위
        
        # 계산 비용 추정
        total_compute_cost = 0
        for step_result in pipeline_result.get("step_results", []):
            model = step_result.get("model", "unknown")
            base_cost = self.compute_costs.get(model, 2.0)  # 기본값
            step_latency = step_result.get("latency_ms", 0) / 1000.0
            total_compute_cost += base_cost * step_latency
        
        # 통합 비용 점수
        cost_score = (self.latency_weight * total_latency + 
                     self.compute_weight * total_compute_cost)
        
        return {
            "cost_score": cost_score,
            "total_latency_sec": total_latency,
            "estimated_compute_cost": total_compute_cost,
            "latency_weight": self.latency_weight,
            "compute_weight": self.compute_weight
        }

class IntegratedTester:
    """통합 파이프라인 테스터"""
    
    def __init__(self):
        self.metrics = EconomicIntelligenceMetrics()

    def run_pipeline_comparison(self, test_prompts: List[str], 
                              pipelines: Dict[str, Pipeline]) -> Dict[str, Any]:
        """파이프라인 AB 비교 실행"""
        
        results = {}
        
        print(f"파이프라인 비교 테스트 시작")
        print(f"프롬프트 수: {len(test_prompts)}")
        print(f"파이프라인 수: {len(pipelines)}")
        print("=" * 60)
        
        for pipeline_name, pipeline in pipelines.items():
            print(f"\n[{pipeline_name}] 파이프라인 실행 중...")
            
            pipeline_results = []
            total_cost_score = 0
            
            for i, prompt in enumerate(test_prompts, 1):
                print(f"  프롬프트 {i}/{len(test_prompts)}: {prompt[:50]}...")
                
                try:
                    # 파이프라인 실행
                    result = pipeline.execute(prompt)
                    
                    # 비용 점수 계산
                    cost_metrics = self.metrics.calculate_cost_score(result)
                    result.update(cost_metrics)
                    
                    pipeline_results.append(result)
                    total_cost_score += cost_metrics["cost_score"]
                    
                    print(f"    답변: {result['final_answer'][:60]}...")
                    print(f"    시간: {result['total_latency_ms']}ms")
                    print(f"    비용점수: {cost_metrics['cost_score']:.3f}")
                    print(f"    성공률: {result['successful_steps']}/{result['total_steps']}")
                    
                except Exception as e:
                    print(f"    오류: {e}")
                    pipeline_results.append({
                        "pipeline_name": pipeline_name,
                        "final_answer": f"[ERROR] {e}",
                        "total_latency_ms": 0,
                        "successful_steps": 0,
                        "total_steps": len(pipeline.steps),
                        "cost_score": 999.0  # 페널티
                    })
            
            # 파이프라인별 집계
            avg_cost_score = total_cost_score / len(test_prompts) if test_prompts else 999.0
            avg_latency = sum(r.get("total_latency_ms", 0) for r in pipeline_results) / len(pipeline_results)
            success_rate = sum(r.get("successful_steps", 0) for r in pipeline_results) / sum(r.get("total_steps", 1) for r in pipeline_results)
            
            results[pipeline_name] = {
                "results": pipeline_results,
                "metrics": {
                    "avg_cost_score": avg_cost_score,
                    "avg_latency_ms": avg_latency,
                    "success_rate": success_rate,
                    "total_prompts": len(test_prompts)
                }
            }
            
            print(f"  [{pipeline_name}] 완료:")
            print(f"    평균 비용점수: {avg_cost_score:.3f}")
            print(f"    평균 지연시간: {avg_latency:.0f}ms") 
            print(f"    성공률: {success_rate:.1%}")
        
        return results

    def generate_comparison_report(self, results: Dict[str, Any]) -> str:
        """비교 리포트 생성"""
        report = []
        report.append("파이프라인 성능 비교 리포트")
        report.append("=" * 50)
        
        # 메트릭별 순위
        pipelines = list(results.keys())
        
        # 1. 비용 효율성 (낮을수록 좋음)
        cost_ranking = sorted(pipelines, 
                            key=lambda p: results[p]["metrics"]["avg_cost_score"])
        report.append("\n비용 효율성 순위 (낮을수록 좋음):")
        for i, pipeline in enumerate(cost_ranking, 1):
            cost = results[pipeline]["metrics"]["avg_cost_score"]
            report.append(f"  {i}. {pipeline}: {cost:.3f}")
        
        # 2. 속도 (낮을수록 좋음)
        speed_ranking = sorted(pipelines,
                             key=lambda p: results[p]["metrics"]["avg_latency_ms"])
        report.append("\n속도 순위 (낮을수록 좋음):")
        for i, pipeline in enumerate(speed_ranking, 1):
            latency = results[pipeline]["metrics"]["avg_latency_ms"]
            report.append(f"  {i}. {pipeline}: {latency:.0f}ms")
        
        # 3. 안정성 (높을수록 좋음)
        reliability_ranking = sorted(pipelines,
                                   key=lambda p: results[p]["metrics"]["success_rate"],
                                   reverse=True)
        report.append("\n안정성 순위 (높을수록 좋음):")
        for i, pipeline in enumerate(reliability_ranking, 1):
            success = results[pipeline]["metrics"]["success_rate"]
            report.append(f"  {i}. {pipeline}: {success:.1%}")
        
        # 종합 점수 (가중 평균)
        report.append("\n종합 점수 (경제적 지능):")
        composite_scores = {}
        for pipeline in pipelines:
            metrics = results[pipeline]["metrics"]
            # 정규화 (0~1 범위)
            cost_norm = 1 / (1 + metrics["avg_cost_score"])  # 낮을수록 좋음
            speed_norm = 1 / (1 + metrics["avg_latency_ms"] / 1000)  # 낮을수록 좋음  
            reliability_norm = metrics["success_rate"]  # 높을수록 좋음
            
            composite = (0.4 * cost_norm + 0.3 * speed_norm + 0.3 * reliability_norm)
            composite_scores[pipeline] = composite
        
        composite_ranking = sorted(pipelines, key=lambda p: composite_scores[p], reverse=True)
        for i, pipeline in enumerate(composite_ranking, 1):
            score = composite_scores[pipeline]
            report.append(f"  {i}. {pipeline}: {score:.3f}")
        
        return "\n".join(report)

def main():
    """메인 실행 함수"""
    
    # 테스트 프롬프트 (다양한 난이도)
    test_prompts = [
        "프랑스의 수도는?",  # 간단한 사실
        "AI 규제가 필요한 이유를 3가지로 설명하세요.",  # 추론
        "원격근무의 장단점을 비교 분석하세요.",  # 분석
        "2030년 한국의 기술 트렌드를 예측하세요.",  # 예측
    ]
    
    # 사용 가능한 모델만 사용하는 간단한 파이프라인
    simple_pipelines = {
        "A-Single": PipelineBuilder.create_single_agent("gemma:2b", "Single-Gemma"),
        "B-Double": PipelineBuilder.create_multi_independent(["gemma:2b", "gemma:2b"], "Double-Gemma")
    }
    
    # 통합 테스터 실행
    tester = IntegratedTester()
    
    print("Project Arkhe - 통합 파이프라인 테스트")
    print("사용 가능한 모델에 따라 자동으로 파이프라인 조정")
    
    try:
        results = tester.run_pipeline_comparison(test_prompts, simple_pipelines)
        
        # 리포트 생성 및 출력
        report = tester.generate_comparison_report(results)
        print(f"\n{report}")
        
        # 결과를 JSON 파일로 저장
        output_file = "results/integrated_test_results.json"
        Path("results").mkdir(exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n상세 결과 저장: {output_file}")
        
    except Exception as e:
        print(f"테스트 실행 중 오류: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())