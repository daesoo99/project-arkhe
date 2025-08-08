import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

class ClaudeAgent:
    """Claude API를 사용하는 독립적 사고 에이전트"""
    
    def __init__(self, name, cost_tracker):
        self.name = name
        self.model = "claude-3-5-sonnet-20241022"
        self.cost_tracker = cost_tracker
        self.client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
    
    def solve(self, problem):
        """Claude를 사용하여 문제 해결"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[{
                    "role": "user", 
                    "content": f"문제를 독립적으로 분석하고 답변하세요: {problem}"
                }]
            )
            
            content = response.content[0].text
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            
            # Claude 가격: $3/$15 per million tokens
            self.cost_tracker.add_cost("claude-3-5-sonnet", input_tokens, output_tokens)
            
            return content
            
        except Exception as e:
            return f"Claude API 오류: {e}"