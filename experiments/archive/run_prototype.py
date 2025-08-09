from src.agents.hierarchy import CostTracker, IndependentThinker, Mediator

def run_prototype():
    """Runs a small-scale prototype of the agent hierarchy."""
    # 1. Initialize components
    cost_tracker = CostTracker()
    
    # Create three independent thinkers
    thinker1 = IndependentThinker(name="Agent_Alpha", cost_tracker=cost_tracker)
    thinker2 = IndependentThinker(name="Agent_Beta", cost_tracker=cost_tracker)
    thinker3 = IndependentThinker(name="Agent_Gamma", cost_tracker=cost_tracker)
    
    # The mediator oversees the thinkers
    mediator = Mediator(thinkers=[thinker1, thinker2, thinker3], cost_tracker=cost_tracker)

    # 2. Define a small set of problems
    problems = [
        "What is the capital of France?",
        "Is the Earth flat? Answer with a simple 'yes' or 'no'.",
        "What is 2 + 2?",
        "Should AI be regulated? Answer with a simple 'yes' or 'no'.",
        "What is the main cause of climate change?"
    ]

    # 3. Solve problems and display results
    for problem in problems:
        result = mediator.solve_problem(problem)
        
        print(f"  Final Answer: {result['final_answer']}")
        print(f"  Diversity (Shannon Entropy): {result['shannon_entropy']:.2f}")
        print(f"  Contradiction Report: {result['contradiction_report']}")
        print("  Individual Responses:")
        for i, resp in enumerate(result['all_responses']):
            print(f"    - Agent {i+1}: {resp}")

    # 4. Report total cost
    total_cost = cost_tracker.get_total_cost()
    print(f"\n--- Run Complete ---")
    print(f"Total Estimated Cost: ${total_cost:.6f}")

if __name__ == "__main__":
    run_prototype()
