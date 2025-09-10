# -*- coding: utf-8 -*-
"""
Simplified hierarchical comparison - avoid Unicode issues
"""

import sys
import time
import json
sys.path.append('.')

from src.llm.simple_llm import create_llm_auto

def simple_hierarchical_test():
    """Simplified hierarchical test"""
    
    # Initialize models
    draft_llm = create_llm_auto("qwen2:0.5b")
    review_llm = create_llm_auto("qwen2:7b") 
    judge_llm = create_llm_auto("llama3:8b")
    
    # Test case
    question = "What is the capital of South Korea?"
    expected = "Seoul"
    
    print("=== Hierarchical Multi-Agent Comparison ===")
    print("Option 1: Draft(0.5B) -> Review(7B) -> Judge(8B)")
    print("Option 2: Draft(0.5B) -> Judge(8B)")
    print("Option 3: Single 8B Model")
    print()
    
    results = []
    
    # Option 1: Draft -> Review -> Judge
    print("Running Option 1...")
    start_time = time.time()
    total_tokens = 0
    
    # Draft stage
    draft_responses = []
    for i in range(3):
        prompt = f"Question: {question}\n\nAnswer with detailed reasoning:"
        response = draft_llm.generate(prompt, temperature=0.3 + i*0.1, max_tokens=200)
        if isinstance(response, dict):
            draft_text = response.get("response", "").strip()
        else:
            draft_text = str(response).strip()
        draft_responses.append(draft_text)
        total_tokens += len(prompt.split()) + len(draft_text.split())
    
    # Review stage
    review_responses = []
    for reviewer_id in range(2):
        review_prompt = f"""Question: {question}

Draft answers to review:
""" + "\n".join(f"Draft {i+1}: {resp}" for i, resp in enumerate(draft_responses)) + f"""

As reviewer {reviewer_id + 1}:
1. Identify common core content
2. Verify accuracy of reasoning
3. Identify errors or issues
4. Provide improved integrated answer

Analysis:"""

        response = review_llm.generate(review_prompt, temperature=0.4, max_tokens=250)
        if isinstance(response, dict):
            review_text = response.get("response", "").strip()
        else:
            review_text = str(response).strip()
        review_responses.append(review_text)
        total_tokens += len(review_prompt.split()) + len(review_text.split())
    
    # Judge stage
    judge_prompt = f"""Question: {question}

Draft originals:
""" + "\n".join(f"Draft {i+1}: {draft}" for i, draft in enumerate(draft_responses)) + f"""

Review analyses:
Review 1: {review_responses[0]}
Review 2: {review_responses[1]}

As advanced judge:
1. Verify draft reasoning logically
2. Evaluate review analyses validity
3. Correct errors with broader knowledge
4. Provide most accurate final answer

Final judgment:"""

    judge_response = judge_llm.generate(judge_prompt, temperature=0.2, max_tokens=200)
    if isinstance(judge_response, dict):
        judge_text = judge_response.get("response", "").strip()
    else:
        judge_text = str(judge_response).strip()
    total_tokens += len(judge_prompt.split()) + len(judge_text.split())
    
    option1_time = int((time.time() - start_time) * 1000)
    option1_accuracy = 1.0 if expected.lower() in judge_text.lower() else 0.0
    
    results.append({
        "approach": "Option 1 (Draft->Review->Judge)",
        "accuracy": option1_accuracy,
        "tokens": total_tokens,
        "time_ms": option1_time,
        "efficiency": option1_accuracy / total_tokens if total_tokens > 0 else 0,
        "final_answer": judge_text[:100] + "..."
    })
    
    print(f"Option 1 - Accuracy: {option1_accuracy:.2f}, Tokens: {total_tokens}, Time: {option1_time}ms, Efficiency: {option1_accuracy/total_tokens:.6f}")
    
    # Option 2: Draft -> Judge (skip Review)
    print("Running Option 2...")
    start_time = time.time()
    total_tokens = 0
    
    # Draft stage (same)
    draft_responses = []
    for i in range(3):
        prompt = f"Question: {question}\n\nAnswer with detailed reasoning:"
        response = draft_llm.generate(prompt, temperature=0.3 + i*0.1, max_tokens=200)
        if isinstance(response, dict):
            draft_text = response.get("response", "").strip()
        else:
            draft_text = str(response).strip()
        draft_responses.append(draft_text)
        total_tokens += len(prompt.split()) + len(draft_text.split())
    
    # Judge stage (direct)
    judge_prompt = f"""Question: {question}

3 Draft answers with reasoning:
""" + "\n".join(f"Draft {i+1}: {draft}" for i, draft in enumerate(draft_responses)) + f"""

As advanced judge:
1. Analyze each draft's answer and reasoning thoroughly
2. Identify logical errors, factual errors, reasoning issues
3. Identify agreements and differences between drafts
4. Verify with your broad knowledge as larger model
5. Synthesize most accurate answer from all information

Special instruction: If all drafts are wrong, provide correct answer from your knowledge.
Use good parts from draft reasoning but correct errors.

Final judgment:"""

    judge_response = judge_llm.generate(judge_prompt, temperature=0.2, max_tokens=200)
    if isinstance(judge_response, dict):
        judge_text = judge_response.get("response", "").strip()
    else:
        judge_text = str(judge_response).strip()
    total_tokens += len(judge_prompt.split()) + len(judge_text.split())
    
    option2_time = int((time.time() - start_time) * 1000)
    option2_accuracy = 1.0 if expected.lower() in judge_text.lower() else 0.0
    
    results.append({
        "approach": "Option 2 (Draft->Judge)",
        "accuracy": option2_accuracy,
        "tokens": total_tokens,
        "time_ms": option2_time,
        "efficiency": option2_accuracy / total_tokens if total_tokens > 0 else 0,
        "final_answer": judge_text[:100] + "..."
    })
    
    print(f"Option 2 - Accuracy: {option2_accuracy:.2f}, Tokens: {total_tokens}, Time: {option2_time}ms, Efficiency: {option2_accuracy/total_tokens:.6f}")
    
    # Option 3: Single 8B Model baseline
    print("Running Single 8B baseline...")
    start_time = time.time()
    
    prompt = f"Question: {question}\n\nAnswer:"
    response = judge_llm.generate(prompt, temperature=0.3, max_tokens=150)
    if isinstance(response, dict):
        answer_text = response.get("response", "").strip()
    else:
        answer_text = str(response).strip()
    
    single_tokens = len(prompt.split()) + len(answer_text.split())
    single_time = int((time.time() - start_time) * 1000)
    single_accuracy = 1.0 if expected.lower() in answer_text.lower() else 0.0
    
    results.append({
        "approach": "Single 8B Model",
        "accuracy": single_accuracy,
        "tokens": single_tokens,
        "time_ms": single_time,
        "efficiency": single_accuracy / single_tokens if single_tokens > 0 else 0,
        "final_answer": answer_text[:100] + "..."
    })
    
    print(f"Single 8B - Accuracy: {single_accuracy:.2f}, Tokens: {single_tokens}, Time: {single_time}ms, Efficiency: {single_accuracy/single_tokens:.6f}")
    
    # Analysis
    print("\n=== Analysis ===")
    option1 = results[0]
    option2 = results[1]
    single = results[2]
    
    print(f"Accuracy: Option1({option1['accuracy']:.2f}) vs Option2({option2['accuracy']:.2f}) vs Single({single['accuracy']:.2f})")
    print(f"Tokens: Option1({option1['tokens']}) vs Option2({option2['tokens']}) vs Single({single['tokens']})")
    print(f"Time: Option1({option1['time_ms']}ms) vs Option2({option2['time_ms']}ms) vs Single({single['time_ms']}ms)")
    print(f"Efficiency: Option1({option1['efficiency']:.6f}) vs Option2({option2['efficiency']:.6f}) vs Single({single['efficiency']:.6f})")
    
    # Review stage value analysis
    accuracy_diff = option1['accuracy'] - option2['accuracy']
    token_diff = option1['tokens'] - option2['tokens']
    time_diff = option1['time_ms'] - option2['time_ms']
    
    print(f"\nReview Stage Value:")
    print(f"Accuracy difference: {accuracy_diff:+.2f} ({'Option 1 better' if accuracy_diff > 0 else 'Option 2 better' if accuracy_diff < 0 else 'Same'})")
    print(f"Token cost: {token_diff:+} ({'Review adds cost' if token_diff > 0 else 'Review saves cost'})")
    print(f"Time cost: {time_diff:+}ms ({'Review adds time' if time_diff > 0 else 'Review saves time'})")
    
    if accuracy_diff > 0.1:
        print("Conclusion: Review stage provides significant accuracy improvement")
    elif accuracy_diff < -0.1:
        print("Conclusion: Review stage actually hurts performance")
    else:
        print("Conclusion: Review stage has minimal impact, consider efficiency")
    
    # Save results
    try:
        with open("results/simple_hierarchical_results.json", "w", encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("\nResults saved to results/simple_hierarchical_results.json")
    except Exception as e:
        print(f"Save failed: {e}")
    
    return results

if __name__ == "__main__":
    results = simple_hierarchical_test()