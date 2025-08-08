import ollama
import math
from collections import Counter

class CostTracker:
    """Tracks the cost of API calls based on token usage."""
    def __init__(self):
        self._total_cost = 0.0
        # Prices per 1 million tokens
        self._model_prices = {
            "gemma:2b": {"input": 0.15, "output": 0.15},
            "claude-3-5-sonnet": {"input": 3.0, "output": 15.0},
            "gpt-4": {"input": 30.0, "output": 60.0},
            "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
            "gpt-oss-20b": {"input": 0.0, "output": 0.0},  # 오픈소스 무료
            "gpt-oss-120b": {"input": 0.0, "output": 0.0}  # 오픈소스 무료
        }

    def add_cost(self, model, input_tokens, output_tokens):
        """Adds the cost of a single API call."""
        prices = self._model_prices.get(model, {"input": 0, "output": 0})
        cost = ((input_tokens / 1_000_000) * prices["input"]) + \
               ((output_tokens / 1_000_000) * prices["output"])
        self._total_cost += cost

    def get_total_cost(self):
        """Returns the total accumulated cost."""
        return self._total_cost

class BiasDetector:
    """A simple module to detect response bias."""
    def __init__(self):
        self.contradiction_pairs = [
            ("yes", "no"),
            ("true", "false"),
            ("agree", "disagree"),
            ("correct", "incorrect")
        ]

    def calculate_shannon_entropy(self, responses):
        """Calculates the diversity of responses using Shannon entropy."""
        if not responses:
            return 0.0
        
        counts = Counter(responses)
        total_responses = len(responses)
        entropy = 0.0
        
        for count in counts.values():
            probability = count / total_responses
            entropy -= probability * math.log2(probability)
            
        return entropy

    def detect_simple_contradictions(self, responses):
        """Detects simple, direct contradictions in a list of responses."""
        cleaned_responses = [r.lower().strip(" .") for r in responses]
        found_contradictions = []
        
        for r1, r2 in self.contradiction_pairs:
            if r1 in cleaned_responses and r2 in cleaned_responses:
                found_contradictions.append(f"Found '{r1}' and '{r2}'")
        
        if not found_contradictions:
            return "No simple contradictions detected."
        return "; ".join(found_contradictions)

class IndependentThinker:
    """An agent that solves a problem independently."""
    def __init__(self, name, cost_tracker, model='gemma:2b'):
        self.name = name
        self.model = model
        self.cost_tracker = cost_tracker

    def solve(self, problem):
        """Solves a given problem using the specified LLM."""
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': problem}]
            )
            
            content = response['message']['content']
            input_tokens = response.get('prompt_eval_count', 0)
            output_tokens = response.get('eval_count', 0)
            
            self.cost_tracker.add_cost(self.model, input_tokens, output_tokens)
            
            return content
        except Exception as e:
            return f"Error during model call: {e}"

class Mediator:
    """Aggregates results from multiple thinkers and provides a final answer."""
    def __init__(self, thinkers, cost_tracker):
        self.thinkers = thinkers
        self.cost_tracker = cost_tracker
        self.bias_detector = BiasDetector()
        # Initial strategy: Rule-based (take the first valid response)
        self.aggregation_strategy = "rule_based" 

    def solve_problem(self, problem):
        """Orchestrates solving a problem across all thinkers."""
        print(f"\n--- Solving Problem: '{problem}' ---")
        
        responses = [thinker.solve(problem) for thinker in self.thinkers]
        
        # Simple rule-based aggregation
        final_answer = next((r for r in responses if r), "No valid answer could be determined.")
        
        entropy = self.bias_detector.calculate_shannon_entropy(responses)
        contradiction_report = self.bias_detector.detect_simple_contradictions(responses)
        
        return {
            "problem": problem,
            "final_answer": final_answer,
            "all_responses": responses,
            "shannon_entropy": entropy,
            "contradiction_report": contradiction_report
        }
