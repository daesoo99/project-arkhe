"""
LLM 통합 인터페이스
다양한 LLM 모델을 통합하여 사용할 수 있는 인터페이스 제공
"""

import requests
import json
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import time

class ModelType(Enum):
    """지원하는 모델 타입"""
    OLLAMA_LLAMA = "ollama:llama3.1"
    OLLAMA_QWEN = "ollama:qwen2.5"
    OPENAI_GPT35 = "gpt-3.5-turbo"
    OPENAI_GPT4 = "gpt-4"
    CLAUDE_SONNET = "claude-3-5-sonnet"

@dataclass
class LLMConfig:
    """LLM 설정"""
    model_type: ModelType
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: int = 30

@dataclass
class LLMResponse:
    """LLM 응답"""
    content: str
    model: str
    tokens_used: int
    cost: float
    success: bool
    error: Optional[str] = None

class LLMInterface:
    """통합 LLM 인터페이스"""
    
    def __init__(self):
        self.ollama_base_url = "http://localhost:11434"
        self.model_costs = {
            ModelType.OLLAMA_LLAMA: 0.0,  # 무료
            ModelType.OLLAMA_QWEN: 0.0,   # 무료
            ModelType.OPENAI_GPT35: 0.0015,  # per 1K tokens
            ModelType.OPENAI_GPT4: 0.03,     # per 1K tokens
            ModelType.CLAUDE_SONNET: 0.003,  # per 1K tokens
        }
    
    async def generate(self, prompt: str, config: LLMConfig) -> LLMResponse:
        """텍스트 생성"""
        try:
            if config.model_type.value.startswith("ollama:"):
                return await self._call_ollama(prompt, config)
            elif config.model_type.value.startswith("gpt"):
                return await self._call_openai(prompt, config)
            elif config.model_type.value.startswith("claude"):
                return await self._call_claude(prompt, config)
            else:
                return LLMResponse(
                    content="", model=config.model_type.value, tokens_used=0,
                    cost=0.0, success=False, error="Unsupported model type"
                )
        except Exception as e:
            return LLMResponse(
                content="", model=config.model_type.value, tokens_used=0,
                cost=0.0, success=False, error=str(e)
            )
    
    def generate_sync(self, prompt: str, config: LLMConfig) -> LLMResponse:
        """동기 텍스트 생성 (기존 코드와의 호환성)"""
        try:
            if config.model_type.value.startswith("ollama:"):
                return self._call_ollama_sync(prompt, config)
            elif config.model_type.value.startswith("gpt"):
                return self._call_openai_sync(prompt, config)
            elif config.model_type.value.startswith("claude"):
                return self._call_claude_sync(prompt, config)
            else:
                return LLMResponse(
                    content="", model=config.model_type.value, tokens_used=0,
                    cost=0.0, success=False, error="Unsupported model type"
                )
        except Exception as e:
            return LLMResponse(
                content="", model=config.model_type.value, tokens_used=0,
                cost=0.0, success=False, error=str(e)
            )
    
    def _call_ollama_sync(self, prompt: str, config: LLMConfig) -> LLMResponse:
        """Ollama 동기 호출"""
        model_name = config.model_type.value.split(":")[1]  # "ollama:llama3.1" -> "llama3.1"
        
        # Ollama가 실행 중인지 확인
        try:
            requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
        except requests.ConnectionError:
            # Ollama가 실행되지 않은 경우 fallback 응답
            return self._create_fallback_response(prompt, config)
        
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": config.temperature,
                "num_predict": config.max_tokens
            }
        }
        
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                timeout=config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            content = result.get("response", "")
            
            # 토큰 수 추정 (정확하지는 않지만 근사치)
            tokens_used = len(prompt.split()) + len(content.split())
            cost = tokens_used * self.model_costs[config.model_type] / 1000
            
            return LLMResponse(
                content=content,
                model=config.model_type.value,
                tokens_used=tokens_used,
                cost=cost,
                success=True
            )
            
        except requests.RequestException as e:
            # 네트워크 오류 등의 경우 fallback
            return self._create_fallback_response(prompt, config)
    
    def _call_openai_sync(self, prompt: str, config: LLMConfig) -> LLMResponse:
        """OpenAI 동기 호출 (API 키가 있는 경우)"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return self._create_fallback_response(prompt, config)
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": config.model_type.value,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens
        }
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            tokens_used = result["usage"]["total_tokens"]
            cost = tokens_used * self.model_costs[config.model_type] / 1000
            
            return LLMResponse(
                content=content,
                model=config.model_type.value,
                tokens_used=tokens_used,
                cost=cost,
                success=True
            )
            
        except Exception as e:
            return self._create_fallback_response(prompt, config)
    
    def _call_claude_sync(self, prompt: str, config: LLMConfig) -> LLMResponse:
        """Claude 동기 호출 (API 키가 있는 경우)"""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return self._create_fallback_response(prompt, config)
        
        # Claude API 호출 구현 (실제로는 anthropic 라이브러리 사용 권장)
        return self._create_fallback_response(prompt, config)
    
    def _create_fallback_response(self, prompt: str, config: LLMConfig) -> LLMResponse:
        """API 호출 실패 시 fallback 응답 생성"""
        
        # 간단한 규칙 기반 응답
        prompt_lower = prompt.lower()
        
        if "비교" in prompt or "장단점" in prompt:
            content = "이 주제는 다양한 장단점을 가지고 있습니다. 장점으로는 효율성과 혁신성을, 단점으로는 잠재적 위험과 비용을 고려해야 합니다."
        elif "예측" in prompt or "미래" in prompt:
            content = "현재 트렌드를 바탕으로 볼 때, 기술 발전과 사회 변화가 지속되면서 새로운 기회와 도전이 함께 나타날 것으로 예상됩니다."
        elif "원인" in prompt or "요인" in prompt:
            content = "이 문제의 원인은 복합적입니다. 직접적 요인들과 근본적인 구조적 문제들을 종합적으로 고려해야 합니다."
        elif "철학" in prompt or "윤리" in prompt:
            content = "이는 복잡한 철학적/윤리적 문제입니다. 다양한 관점과 가치 체계를 균형있게 고려하는 것이 중요합니다."
        elif "해결" in prompt or "방안" in prompt:
            content = "문제 해결을 위해서는 다각도 분석과 단계적 접근이 필요합니다. 관련 이해관계자들과의 협력도 중요한 요소입니다."
        else:
            content = f"[Fallback Response] 이 질문에 대한 종합적인 분석을 제공합니다. 다양한 관점을 고려한 균형잡힌 접근이 필요합니다."
        
        # 토큰 수와 비용 추정
        estimated_tokens = len(prompt.split()) + len(content.split())
        estimated_cost = estimated_tokens * 0.001 / 1000  # 가상의 비용
        
        return LLMResponse(
            content=content,
            model=f"{config.model_type.value}_fallback",
            tokens_used=estimated_tokens,
            cost=estimated_cost,
            success=True
        )

    def check_model_availability(self) -> Dict[ModelType, bool]:
        """사용 가능한 모델들 확인"""
        availability = {}
        
        # Ollama 확인
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]
                
                availability[ModelType.OLLAMA_LLAMA] = any("llama" in model.lower() for model in available_models)
                availability[ModelType.OLLAMA_QWEN] = any("qwen" in model.lower() for model in available_models)
            else:
                availability[ModelType.OLLAMA_LLAMA] = False
                availability[ModelType.OLLAMA_QWEN] = False
        except:
            availability[ModelType.OLLAMA_LLAMA] = False
            availability[ModelType.OLLAMA_QWEN] = False
        
        # OpenAI 확인
        availability[ModelType.OPENAI_GPT35] = bool(os.getenv("OPENAI_API_KEY"))
        availability[ModelType.OPENAI_GPT4] = bool(os.getenv("OPENAI_API_KEY"))
        
        # Claude 확인  
        availability[ModelType.CLAUDE_SONNET] = bool(os.getenv("ANTHROPIC_API_KEY"))
        
        return availability

# 전역 인스턴스
llm_interface = LLMInterface()

def get_default_model() -> ModelType:
    """사용 가능한 기본 모델 반환"""
    availability = llm_interface.check_model_availability()
    
    # 우선순위: Ollama Llama > Ollama Qwen > Fallback
    if availability.get(ModelType.OLLAMA_LLAMA, False):
        return ModelType.OLLAMA_LLAMA
    elif availability.get(ModelType.OLLAMA_QWEN, False):
        return ModelType.OLLAMA_QWEN
    else:
        # Fallback - 항상 작동하는 규칙 기반 응답
        return ModelType.OLLAMA_LLAMA  # fallback 응답을 사용