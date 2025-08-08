"""
GPT-OSS 오픈소스 모델 에이전트 클래스
Ollama를 통해 GPT-OSS 모델을 실행하는 기능 제공
"""

import subprocess
import json
import os
from typing import Dict, Any

class GPTOSSAgent:
    """GPT-OSS 오픈소스 모델을 사용하는 에이전트"""
    
    def __init__(self, name: str, cost_tracker, model: str = "gpt-oss-20b"):
        self.name = name
        self.model = model
        self.cost_tracker = cost_tracker
        
    def _run_ollama_command(self, prompt: str) -> Dict[str, Any]:
        """Ollama 명령어를 실행하여 모델 응답을 얻음"""
        try:
            # Ollama가 PATH에 있는지 확인
            result = subprocess.run(['where', 'ollama'], 
                                  capture_output=True, text=True, shell=True)
            if result.returncode != 0:
                return {"error": "Ollama가 설치되지 않았거나 PATH에 없습니다."}
            
            # 모델이 다운로드되어 있는지 확인
            list_result = subprocess.run(['ollama', 'list'], 
                                       capture_output=True, text=True)
            if self.model not in list_result.stdout:
                # 모델이 없으면 자동으로 다운로드 시도
                print(f"{self.model} 모델을 다운로드 중...")
                pull_result = subprocess.run(['ollama', 'pull', self.model], 
                                          capture_output=True, text=True)
                if pull_result.returncode != 0:
                    return {"error": f"모델 {self.model} 다운로드 실패: {pull_result.stderr}"}
            
            # 모델 실행
            cmd = ['ollama', 'run', self.model, prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                return {
                    "response": result.stdout.strip(),
                    "error": None,
                    "tokens_used": len(prompt.split()) + len(result.stdout.split())  # 간단한 토큰 추정
                }
            else:
                return {"error": f"Ollama 실행 오류: {result.stderr}"}
                
        except subprocess.TimeoutExpired:
            return {"error": "모델 응답 시간 초과 (120초)"}
        except Exception as e:
            return {"error": f"예상치 못한 오류: {str(e)}"}
    
    def solve(self, problem: str) -> str:
        """문제를 해결하여 답변 반환"""
        prompt = f"""다음 문제를 독립적으로 분석하고 명확한 답변을 제공하세요:

문제: {problem}

답변:"""
        
        result = self._run_ollama_command(prompt)
        
        if result.get("error"):
            return f"[{self.name}] 오류 발생: {result['error']}"
        
        # 비용 추적 (GPT-OSS는 무료이므로 0으로 설정)
        tokens_used = result.get("tokens_used", 0)
        self.cost_tracker.add_cost("gpt-oss-20b", tokens_used // 2, tokens_used // 2)
        
        return result.get("response", "응답을 받을 수 없습니다.")

class MockGPTOSSAgent:
    """GPT-OSS 모델의 Mock 버전 (테스트용)"""
    
    def __init__(self, name: str, cost_tracker, model: str = "gpt-oss-20b-mock"):
        self.name = name
        self.model = model
        self.cost_tracker = cost_tracker
        
    def solve(self, problem: str) -> str:
        """문제를 해결하여 Mock 답변 반환"""
        
        # 간단한 규칙 기반 응답 생성
        mock_responses = {
            "capital": "파리는 프랑스의 수도입니다.",
            "earth": "아니오, 지구는 평평하지 않습니다. 지구는 구형입니다.",
            "math": "2 + 2 = 4입니다.",
            "regulation": "예, AI는 적절한 규제가 필요하다고 생각합니다.",
            "climate": "기후 변화의 주요 원인은 온실가스 배출입니다."
        }
        
        problem_lower = problem.lower()
        response = "질문에 대한 구체적인 답변을 제공할 수 없습니다."
        
        for keyword, answer in mock_responses.items():
            if keyword in problem_lower or any(word in problem_lower for word in ["수도", "capital", "프랑스"]) and keyword == "capital":
                response = f"[{self.name} Mock GPT-OSS] {answer}"
                break
            elif any(word in problem_lower for word in ["flat", "평평", "지구"]) and keyword == "earth":
                response = f"[{self.name} Mock GPT-OSS] {answer}"
                break
            elif any(word in problem_lower for word in ["2+2", "2 + 2", "더하기"]) and keyword == "math":
                response = f"[{self.name} Mock GPT-OSS] {answer}"
                break
            elif any(word in problem_lower for word in ["regulation", "규제", "ai"]) and keyword == "regulation":
                response = f"[{self.name} Mock GPT-OSS] {answer}"
                break
            elif any(word in problem_lower for word in ["climate", "기후", "변화"]) and keyword == "climate":
                response = f"[{self.name} Mock GPT-OSS] {answer}"
                break
        
        # Mock 토큰 사용량 (무료 모델이므로 비용은 0)
        estimated_tokens = len(problem.split()) + len(response.split())
        self.cost_tracker.add_cost("gpt-oss-20b", estimated_tokens // 2, estimated_tokens // 2)
        
        return response