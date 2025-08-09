# -*- coding: utf-8 -*-
"""
Project Arkhē - Perfect Economic Intelligence Pipeline V2
모든 High Priority 이슈 해결:
- LLM 클라이언트 재사용
- Ollama 메타데이터 직접 수집  
- 에러 격리/복구
- 완전한 로깅/재현성
- 프롬프트 위생 (JSON 스키마)
"""

import time
import json
import requests
import numpy as np
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import logging
import random

@dataclass
class PerfectStageMetrics:
    """완벽한 단계별 실측 메트릭"""
    stage_name: str
    model: str
    start_time: float
    end_time: float
    prompt_hash: str
    seed: int
    temperature: float
    
    # Ollama 직접 메타데이터
    eval_count: int = 0           # 실제 생성된 토큰 수
    eval_duration: int = 0        # 나노초 단위 평가 시간
    prompt_eval_count: int = 0    # 프롬프트 토큰 수
    prompt_eval_duration: int = 0 # 프롬프트 처리 시간
    
    # 계산된 메트릭
    tokens_per_second: float = 0.0
    first_token_latency: float = 0.0  # ms
    total_latency: float = 0.0        # ms
    cost_factor: float = 1.0
    real_cost: float = 0.0
    success: bool = True
    error: Optional[str] = None
    fallback_used: bool = False

@dataclass 
class StructuredSample:
    """구조화된 샘플 결과"""
    raw_text: str
    parsed_json: Optional[Dict] = None
    answer: str = ""
    rationale: str = ""
    confidence: float = 0.5
    tokens: int = 0
    parsing_success: bool = False

@dataclass
class PerfectEconomicResult:
    """완벽한 경제적 지능 결과"""
    query: str
    query_hash: str
    execution_id: str
    timestamp: str
    
    final_answer: str
    final_confidence: float
    final_rationale: str
    
    total_stages: int
    executed_stages: int
    stage_metrics: List[PerfectStageMetrics]
    entropy_progression: List[float]
    promotion_decisions: List[Tuple[bool, Dict]]
    fallback_count: int
    
    total_cost: float
    total_time: float
    economic_efficiency: float
    cost_saved_ratio: float
    
    # 로깅 정보
    log_file: str
    reproducible: bool = True

class OllamaDirectClient:
    """Ollama REST API 직접 클라이언트 - 메타데이터 수집"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        
    def generate(self, model: str, prompt: str, temperature: float = 0.7, 
                max_tokens: int = 300, seed: int = None) -> Dict[str, Any]:
        """Ollama API 직접 호출 - 완전한 메타데이터 수집"""
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        if seed is not None:
            payload["options"]["seed"] = seed
            
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            data = response.json()
            return {
                "response": data.get("response", ""),
                "eval_count": data.get("eval_count", 0),
                "eval_duration": data.get("eval_duration", 0),
                "prompt_eval_count": data.get("prompt_eval_count", 0),  
                "prompt_eval_duration": data.get("prompt_eval_duration", 0),
                "total_duration": data.get("total_duration", 0),
                "model": data.get("model", model),
                "done": data.get("done", True)
            }
            
        except Exception as e:
            return {
                "response": f"ERROR: {str(e)}",
                "error": str(e),
                "eval_count": 0,
                "eval_duration": 0,
                "prompt_eval_count": 0,
                "prompt_eval_duration": 0,
                "total_duration": 0,
                "model": model,
                "done": False
            }

class PromptHygiene:
    """프롬프트 위생 - JSON 스키마 강제"""
    
    @staticmethod
    def draft_prompt(query: str, seed: int) -> str:
        """초안 프롬프트 - 간결한 답변"""
        return f"""Answer this question concisely and clearly:

Question: {query}

Instructions:
- Keep answer under 50 words
- Be factual and direct
- No unnecessary elaboration

Answer:"""

    @staticmethod  
    def review_prompt(query: str, drafts: List[str], seed: int) -> str:
        """검토 프롬프트 - JSON 응답 강제"""
        drafts_text = "\n".join([f"Draft {i+1}: {d}" for i, d in enumerate(drafts)])
        
        return f"""Review these draft answers and provide an improved response in exact JSON format:

Question: {query}

{drafts_text}

Respond in this exact JSON format:
{{
  "answer": "your improved answer here",
  "rationale": "why this is better than the drafts", 
  "confidence": 0.75,
  "improvements": ["what you improved"]
}}

JSON Response:"""

    @staticmethod
    def judge_prompt(query: str, drafts: List[str], reviews: List[str], seed: int) -> str:
        """판정 프롬프트 - 엄격한 JSON 스키마"""
        drafts_text = "\n".join([f"Draft {i+1}: {d}" for i, d in enumerate(drafts)])
        reviews_text = "\n".join([f"Review {i+1}: {r}" for i, r in enumerate(reviews)])
        
        return f"""Provide the highest quality final answer by analyzing all previous attempts:

Question: {query}

Previous attempts:
{drafts_text}

{reviews_text}

Respond in this EXACT JSON format:
{{
  "answer": "final authoritative answer",
  "rationale": "detailed reasoning for this answer",
  "confidence": 0.95,
  "sources_analyzed": ["draft", "review"],
  "key_improvements": ["specific improvements made"],
  "quality_score": 9
}}

JSON Response:"""

    @staticmethod
    def parse_json_response(text: str) -> StructuredSample:
        """JSON 응답 파싱 - 강력한 오류 처리"""
        sample = StructuredSample(raw_text=text)
        
        try:
            # JSON 추출 시도
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = text[json_start:json_end]
                parsed = json.loads(json_str)
                
                sample.parsed_json = parsed
                sample.answer = parsed.get("answer", "")
                sample.rationale = parsed.get("rationale", "")
                sample.confidence = float(parsed.get("confidence", 0.5))
                sample.parsing_success = True
                
        except Exception as e:
            # JSON 파싱 실패 - 텍스트 그대로 사용
            sample.answer = text.strip()
            sample.rationale = "JSON parsing failed"
            sample.confidence = 0.3  # 낮은 신뢰도
            sample.parsing_success = False
            
        sample.tokens = len(sample.answer.split())
        return sample

class PerfectLogger:
    """완벽한 로깅 시스템"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"economic_intelligence_{timestamp}.jsonl"
        
        # 실험 설정 로깅
        self.setup_logging()
        
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('EconomicIntelligence')
        
    def log_stage(self, stage_metrics: PerfectStageMetrics, samples: List[StructuredSample]):
        """단계별 로깅"""
        log_entry = {
            "type": "stage",
            "timestamp": datetime.now().isoformat(),
            "stage": asdict(stage_metrics),
            "samples": [
                {
                    "answer": s.answer[:100],  # 처음 100자만
                    "confidence": s.confidence,
                    "tokens": s.tokens,
                    "parsing_success": s.parsing_success
                } for s in samples
            ]
        }
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def log_result(self, result: PerfectEconomicResult):
        """최종 결과 로깅"""
        log_entry = {
            "type": "final_result",
            "timestamp": datetime.now().isoformat(),
            "result": asdict(result)
        }
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

class PerfectEconomicIntelligence:
    """완벽한 경제적 지능 파이프라인"""
    
    def __init__(self, cost_sensitivity: float = 0.3):
        self.cost_sensitivity = cost_sensitivity
        
        # LLM 클라이언트 재사용 (한 번만 생성)
        self.ollama_client = OllamaDirectClient()
        
        # 모델별 비용 계수
        self.cost_factors = {
            "qwen2:0.5b": 0.8,
            "gemma:2b": 1.0,
            "llama3:8b": 4.0
        }
        
        # 프롬프트 위생
        self.prompt_hygiene = PromptHygiene()
        
        # 로깅
        self.logger = PerfectLogger()
        
        # 재현성을 위한 시드
        self.master_seed = random.randint(1000, 9999)
        
    def _generate_with_fallback(self, model: str, prompt: str, temperature: float,
                               max_tokens: int, seed: int, stage_name: str) -> Tuple[StructuredSample, PerfectStageMetrics]:
        """에러 복구가 있는 생성"""
        
        start_time = time.time()
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
        
        metrics = PerfectStageMetrics(
            stage_name=stage_name,
            model=model,
            start_time=start_time,
            end_time=0,
            prompt_hash=prompt_hash,
            seed=seed,
            temperature=temperature,
            cost_factor=self.cost_factors.get(model, 1.0)
        )
        
        try:
            # 메인 시도
            response = self.ollama_client.generate(
                model=model, 
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed
            )
            
            if "error" not in response and response.get("done", False):
                # 성공 - Ollama 메타데이터 직접 수집
                metrics.eval_count = response.get("eval_count", 0)
                metrics.eval_duration = response.get("eval_duration", 0)
                metrics.prompt_eval_count = response.get("prompt_eval_count", 0)
                metrics.prompt_eval_duration = response.get("prompt_eval_duration", 0)
                
                # 실제 계산
                if metrics.eval_duration > 0:
                    metrics.tokens_per_second = metrics.eval_count / (metrics.eval_duration / 1e9)
                
                metrics.end_time = time.time()
                metrics.total_latency = (metrics.end_time - start_time) * 1000
                metrics.real_cost = metrics.eval_count * metrics.cost_factor / 1000
                metrics.success = True
                
                # 구조화된 파싱
                sample = self.prompt_hygiene.parse_json_response(response["response"])
                
                return sample, metrics
                
        except Exception as e:
            metrics.error = str(e)
            
        # 폴백 - 간단한 응답
        metrics.end_time = time.time()
        metrics.total_latency = (metrics.end_time - start_time) * 1000
        metrics.success = False
        metrics.fallback_used = True
        metrics.real_cost = 0.1  # 폴백 비용
        
        fallback_sample = StructuredSample(
            raw_text="FALLBACK: Unable to generate response",
            answer="Error in generation",
            confidence=0.1,
            tokens=5
        )
        
        return fallback_sample, metrics
    
    def _multi_sample_stage(self, model: str, prompt: str, stage_name: str, 
                           n_samples: int, temperature: float) -> Tuple[List[StructuredSample], PerfectStageMetrics]:
        """다중 샘플링 with 에러 복구"""
        
        samples = []
        all_metrics = []
        
        for i in range(n_samples):
            seed = self.master_seed + i  # 재현 가능한 시드
            sample, metrics = self._generate_with_fallback(
                model, prompt, temperature, 300, seed, f"{stage_name}_sample_{i+1}"
            )
            samples.append(sample)
            all_metrics.append(metrics)
        
        # 집계된 메트릭
        total_metrics = PerfectStageMetrics(
            stage_name=stage_name,
            model=model,
            start_time=min(m.start_time for m in all_metrics),
            end_time=max(m.end_time for m in all_metrics),
            prompt_hash=all_metrics[0].prompt_hash,
            seed=self.master_seed,
            temperature=temperature,
            eval_count=sum(m.eval_count for m in all_metrics),
            eval_duration=sum(m.eval_duration for m in all_metrics),
            prompt_eval_count=sum(m.prompt_eval_count for m in all_metrics),
            prompt_eval_duration=sum(m.prompt_eval_duration for m in all_metrics),
            cost_factor=self.cost_factors.get(model, 1.0),
            real_cost=sum(m.real_cost for m in all_metrics),
            total_latency=sum(m.total_latency for m in all_metrics),
            success=any(m.success for m in all_metrics),
            fallback_used=any(m.fallback_used for m in all_metrics)
        )
        
        if total_metrics.eval_duration > 0:
            total_metrics.tokens_per_second = total_metrics.eval_count / (total_metrics.eval_duration / 1e9)
        
        # 로깅
        self.logger.log_stage(total_metrics, samples)
        
        return samples, total_metrics
    
    def execute(self, query: str) -> PerfectEconomicResult:
        """완벽한 경제적 지능 실행"""
        
        execution_id = f"ei_{int(time.time())}_{random.randint(1000,9999)}"
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        
        print(f"\n🧠 Perfect Economic Intelligence: {query}")
        print(f"Execution ID: {execution_id}")
        print(f"Master seed: {self.master_seed}")
        
        stage_metrics = []
        entropy_progression = []
        promotion_decisions = []
        fallback_count = 0
        
        # Stage 1: Draft (qwen2:0.5b)
        print("\n📝 Stage 1 (Draft): qwen2:0.5b - Multi-sampling...")
        draft_prompt = self.prompt_hygiene.draft_prompt(query, self.master_seed)
        draft_samples, draft_metrics = self._multi_sample_stage(
            "qwen2:0.5b", draft_prompt, "draft", n_samples=5, temperature=0.8
        )
        stage_metrics.append(draft_metrics)
        
        if draft_metrics.fallback_used:
            fallback_count += 1
            
        # 엔트로피 계산
        draft_texts = [s.answer for s in draft_samples if s.answer]
        draft_entropy = self._calculate_entropy(draft_texts) if draft_texts else 0
        entropy_progression.append(draft_entropy)
        
        print(f"  Samples: {len(draft_samples)}, Entropy: {draft_entropy:.3f}")
        print(f"  Tokens/s: {draft_metrics.tokens_per_second:.1f}, Cost: ${draft_metrics.real_cost:.4f}")
        
        # 승급 결정 1
        promote_1, promo_1_info = self._promotion_decision(draft_samples, 0.8, 1.0, draft_entropy)
        promotion_decisions.append((promote_1, promo_1_info))
        
        if not promote_1:
            print(f"  🛑 Early termination: {promo_1_info}")
            best_draft = max(draft_samples, key=lambda x: x.confidence)
            return self._create_result(query, execution_id, query_hash, best_draft.answer,
                                     best_draft.confidence, best_draft.rationale, 1, 1,
                                     stage_metrics, entropy_progression, promotion_decisions, fallback_count)
        
        # Stage 2: Review (gemma:2b) 
        print("\n🔍 Stage 2 (Review): gemma:2b - JSON structured...")
        top_drafts = sorted(draft_samples, key=lambda x: x.confidence, reverse=True)[:3]
        review_prompt = self.prompt_hygiene.review_prompt(
            query, [d.answer for d in top_drafts], self.master_seed + 100
        )
        review_samples, review_metrics = self._multi_sample_stage(
            "gemma:2b", review_prompt, "review", n_samples=3, temperature=0.5
        )
        stage_metrics.append(review_metrics)
        
        if review_metrics.fallback_used:
            fallback_count += 1
        
        review_texts = [s.answer for s in review_samples if s.answer]
        review_entropy = self._calculate_entropy(review_texts) if review_texts else draft_entropy
        entropy_progression.append(review_entropy)
        
        print(f"  Samples: {len(review_samples)}, Entropy: {review_entropy:.3f}")
        print(f"  JSON parsing: {sum(1 for s in review_samples if s.parsing_success)}/{len(review_samples)}")
        
        # 승급 결정 2
        promote_2, promo_2_info = self._promotion_decision(review_samples, 1.0, 4.0, review_entropy)
        promotion_decisions.append((promote_2, promo_2_info))
        
        if not promote_2:
            print(f"  🛑 Early termination: {promo_2_info}")
            best_review = max(review_samples, key=lambda x: x.confidence)
            return self._create_result(query, execution_id, query_hash, best_review.answer,
                                     best_review.confidence, best_review.rationale, 2, 2,
                                     stage_metrics, entropy_progression, promotion_decisions, fallback_count)
        
        # Stage 3: Judge (llama3:8b)
        print("\n⚖️  Stage 3 (Judge): llama3:8b - Final judgment...")
        top_reviews = sorted(review_samples, key=lambda x: x.confidence, reverse=True)[:2]
        judge_prompt = self.prompt_hygiene.judge_prompt(
            query, [d.answer for d in top_drafts[:2]], [r.answer for r in top_reviews], self.master_seed + 200
        )
        judge_samples, judge_metrics = self._multi_sample_stage(
            "llama3:8b", judge_prompt, "judge", n_samples=2, temperature=0.2
        )
        stage_metrics.append(judge_metrics)
        
        if judge_metrics.fallback_used:
            fallback_count += 1
            
        judge_texts = [s.answer for s in judge_samples if s.answer]
        judge_entropy = self._calculate_entropy(judge_texts) if judge_texts else review_entropy
        entropy_progression.append(judge_entropy)
        
        print(f"  Samples: {len(judge_samples)}, Entropy: {judge_entropy:.3f}")
        print(f"  JSON parsing: {sum(1 for s in judge_samples if s.parsing_success)}/{len(judge_samples)}")
        
        # 최종 결과
        if judge_samples:
            best_judge = max(judge_samples, key=lambda x: x.confidence)
            final_answer = best_judge.answer
            final_confidence = best_judge.confidence
            final_rationale = best_judge.rationale
        else:
            # Judge 완전 실패 - Review 폴백
            best_review = max(review_samples, key=lambda x: x.confidence)
            final_answer = best_review.answer
            final_confidence = best_review.confidence * 0.8  # 페널티
            final_rationale = "Judge failed, using review result"
            fallback_count += 1
            
        return self._create_result(query, execution_id, query_hash, final_answer,
                                 final_confidence, final_rationale, 3, 3,
                                 stage_metrics, entropy_progression, promotion_decisions, fallback_count)
    
    def _calculate_entropy(self, texts: List[str]) -> float:
        """Shannon Entropy 계산"""
        if not texts:
            return 0.0
            
        word_counts = []
        for text in texts:
            words = text.lower().split()
            word_counts.extend(words)
        
        if not word_counts:
            return 0.0
            
        from collections import Counter
        counter = Counter(word_counts)
        total = sum(counter.values())
        
        entropy = 0.0
        for count in counter.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
                
        return entropy
    
    def _promotion_decision(self, samples: List[StructuredSample], current_cost: float, 
                           next_cost: float, entropy: float) -> Tuple[bool, Dict]:
        """승급 결정"""
        if not samples:
            return True, {"reason": "no_samples"}
            
        avg_confidence = np.mean([s.confidence for s in samples])
        cost_ratio = next_cost / current_cost
        
        # 승급 조건
        high_entropy = entropy > 2.0
        low_confidence = avg_confidence < 0.7
        cost_effective = (entropy * (1 - avg_confidence)) > (self.cost_sensitivity * cost_ratio)
        
        should_promote = high_entropy or low_confidence or cost_effective
        
        return should_promote, {
            "entropy": entropy,
            "confidence": avg_confidence,
            "cost_ratio": cost_ratio,
            "high_entropy": high_entropy,
            "low_confidence": low_confidence,
            "cost_effective": cost_effective
        }
    
    def _create_result(self, query: str, execution_id: str, query_hash: str,
                      final_answer: str, final_confidence: float, final_rationale: str,
                      total_stages: int, executed_stages: int,
                      stage_metrics: List[PerfectStageMetrics], 
                      entropy_progression: List[float],
                      promotion_decisions: List[Tuple[bool, Dict]],
                      fallback_count: int) -> PerfectEconomicResult:
        """완벽한 결과 생성"""
        
        total_cost = sum(m.real_cost for m in stage_metrics)
        total_time = sum(m.total_latency for m in stage_metrics)
        
        # 경제적 효율성
        utility = final_confidence * len(final_answer) / 100
        economic_efficiency = (utility - self.cost_sensitivity * total_cost) / max(total_time/1000, 0.001)
        
        # 비용 절약
        max_cost = sum(self.cost_factors.values()) * 3  # 최대 가능 비용
        cost_saved_ratio = 1 - (total_cost / max_cost) if max_cost > 0 else 0
        
        result = PerfectEconomicResult(
            query=query,
            query_hash=query_hash,
            execution_id=execution_id,
            timestamp=datetime.now().isoformat(),
            final_answer=final_answer,
            final_confidence=final_confidence,
            final_rationale=final_rationale,
            total_stages=total_stages,
            executed_stages=executed_stages,
            stage_metrics=stage_metrics,
            entropy_progression=entropy_progression,
            promotion_decisions=promotion_decisions,
            fallback_count=fallback_count,
            total_cost=total_cost,
            total_time=total_time,
            economic_efficiency=economic_efficiency,
            cost_saved_ratio=cost_saved_ratio,
            log_file=str(self.logger.log_file),
            reproducible=True
        )
        
        # 최종 로깅
        self.logger.log_result(result)
        
        return result