# -*- coding: utf-8 -*-
"""
Project Arkhē - 간단한 LLM 클래스들
파이프라인과 호환되는 통일된 인터페이스 제공
"""

import requests
import json
import time
import os
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

class LLM(ABC):
    """통일된 LLM 인터페이스"""
    
    @abstractmethod
    def generate(self, prompt: str, **opts) -> Dict[str, Any]:
        """텍스트 생성 메서드"""
        pass

class OllamaLLM(LLM):
    """Ollama LLM 클라이언트"""
    
    def __init__(self, model_id: str, base_url: str = "http://127.0.0.1:11434"):
        self.model_id = model_id
        self.base_url = base_url
        
    def generate(self, prompt: str, **opts) -> Dict[str, Any]:
        """Ollama API 호출"""
        temperature = opts.get("temperature", 0.2)
        max_tokens = opts.get("max_tokens", 512)
        timeout = opts.get("timeout", 120)
        
        start_time = time.time()
        
        try:
            payload = {
                "model": self.model_id,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            
            result = response.json()
            latency_ms = int((time.time() - start_time) * 1000)
            
            return {
                "response": result.get("response", ""),
                "latency_ms": latency_ms,
                "model": self.model_id,
                "success": True,
                "eval_count": result.get("eval_count", 0),
                "eval_duration": result.get("eval_duration", 0),
                "total_duration": result.get("total_duration", latency_ms * 1e6)
            }
            
        except Exception as e:
            return {
                "response": f"[ERROR] {str(e)}",
                "latency_ms": int((time.time() - start_time) * 1000),
                "model": self.model_id,
                "success": False,
                "error": str(e)
            }

class OpenAILLM(LLM):
    """OpenAI LLM 클라이언트"""
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.api_key = os.getenv("OPENAI_API_KEY")
        
    def generate(self, prompt: str, **opts) -> Dict[str, Any]:
        """OpenAI API 호출"""
        if not self.api_key:
            return {
                "response": "[ERROR] OPENAI_API_KEY not found",
                "latency_ms": 0,
                "model": self.model_id,
                "success": False,
                "error": "Missing API key"
            }
        
        temperature = opts.get("temperature", 0.2)
        max_tokens = opts.get("max_tokens", 512)
        timeout = opts.get("timeout", 120)
        
        start_time = time.time()
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            
            result = response.json()
            latency_ms = int((time.time() - start_time) * 1000)
            
            return {
                "response": result["choices"][0]["message"]["content"],
                "latency_ms": latency_ms,
                "model": self.model_id,
                "success": True,
                "tokens_used": result["usage"]["total_tokens"],
                "cost_estimate": result["usage"]["total_tokens"] * 0.0015 / 1000  # GPT-4o-mini 기준
            }
            
        except Exception as e:
            return {
                "response": f"[ERROR] {str(e)}",
                "latency_ms": int((time.time() - start_time) * 1000),
                "model": self.model_id,
                "success": False,
                "error": str(e)
            }

class AnthropicLLM(LLM):
    """Anthropic Claude LLM 클라이언트"""
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        
    def generate(self, prompt: str, **opts) -> Dict[str, Any]:
        """Anthropic API 호출"""
        if not self.api_key:
            return {
                "response": "[ERROR] ANTHROPIC_API_KEY not found", 
                "latency_ms": 0,
                "model": self.model_id,
                "success": False,
                "error": "Missing API key"
            }
        
        # Claude SDK 사용 권장하지만 간단한 구현으로 대체
        return {
            "response": "[ERROR] Anthropic client not implemented yet",
            "latency_ms": 0,
            "model": self.model_id,
            "success": False,
            "error": "Not implemented"
        }

def create_llm(provider: str, model_id: str) -> LLM:
    """LLM 인스턴스 팩토리"""
    if provider.lower() == "ollama":
        return OllamaLLM(model_id)
    elif provider.lower() == "openai":
        return OpenAILLM(model_id)
    elif provider.lower() == "anthropic":
        return AnthropicLLM(model_id)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def detect_provider(model_id: str) -> str:
    """모델 ID로부터 프로바이더 자동 감지"""
    if ":" in model_id and not model_id.startswith("gpt"):
        return "ollama"  # gemma:2b, llama3:8b 등
    elif model_id.startswith("gpt"):
        return "openai"  # gpt-4, gpt-3.5-turbo 등
    elif model_id.startswith("claude"):
        return "anthropic"  # claude-3-sonnet 등
    else:
        return "ollama"  # 기본값

def create_llm_auto(model_id: str) -> LLM:
    """모델 ID로부터 자동으로 LLM 인스턴스 생성"""
    provider = detect_provider(model_id)
    return create_llm(provider, model_id)