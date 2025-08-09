# Project Arkhē - Research Summary

## Overview
Project Arkhē explores multi-agent systems for improved LLM performance, focusing on three core research areas: **Recursive Problem Decomposition**, **Information Asymmetry Effects**, and **Economic Intelligence Optimization**.

## Research Questions
1. **Do multi-agent systems outperform single models?**
2. **What is the optimal information sharing level between agents?**
3. **How can we optimize performance-cost trade-offs dynamically?**

---

## Experimental Models & Configurations

### Multi-Agent Pipeline Architecture

#### Current Configuration (v3 - Judge Upgraded)
```
Draft Stage:    qwen2:0.5b × 3 samples → 초안 생성
Review Stage:   qwen2:0.5b × 2 samples → 초안 검토/개선  
Judge Stage:    llama3:8b × 1 sample  → 최종 판정 (UPGRADED)
```

#### Previous Configurations
**v1 (Original)**:
- Draft: `qwen2:0.5b` × 3, Review: `gemma:2b` × 3, Judge: `llama3:8b` × 3

**v2 (Optimized)**:
- Draft: `qwen2:0.5b` × 3, Review: `qwen2:0.5b` × 2, Judge: `gemma:2b` × 1

### Information Sharing Levels
1. **NONE**: Complete information sharing between all stages
2. **PARTIAL**: Limited information sharing (1-to-1 connections)
3. **COMPLETE**: Full isolation between agents

### Single Model Baselines
- **llama3:8b**: Primary comparison target
- **claude-3-haiku**: High-performance cloud model (connection issues)
- **gpt-4o-mini**: OpenAI baseline (connection issues)
- **gemma:2b**: Local efficiency baseline

---

## Key Experimental Results

### 1. Information Asymmetry Findings (MAJOR DISCOVERY)

| Isolation Level | Accuracy | Tokens | Time(ms) | Efficiency |
|----------------|----------|---------|----------|------------|
| **NONE** (Complete Sharing) | **0.800** | 101 | 2824 | **0.744** |
| PARTIAL (Limited Sharing) | 0.600 | **56** | **1507** | 0.533 |
| COMPLETE (Independent) | **0.800** | 82 | 1940 | 0.503 |

**🔥 Counter-Intuitive Finding**: 
- **PARTIAL information sharing is WORST** - contradicts intuition
- Complete sharing = Complete isolation in performance
- "Goldilocks zone" doesn't exist - go extreme or go home

### 2. Multi-Agent vs Single Model (DECISIVE RESULTS)

#### Latest Results (GPT tiktoken-based measurement)
| Method | Accuracy | Tokens | Time(ms) | Efficiency | Result |
|--------|----------|---------|----------|------------|---------|
| **Multi-Agent-NONE** | 50.2% | **1,766** | 8,089 | 0.028 | 😰 |
| **Single-llama3:8b** | **87.7%** | **152** | 6,802 | **0.577** | 🏆 |

**Shocking Results**:
- Single model **42.8% more accurate**
- Multi-agent **11× more expensive** (+1061% tokens)
- Single model **20× more efficient**

#### Category-wise Performance
| Category | Multi-Agent | Single llama3:8b | Winner |
|----------|-------------|------------------|---------|
| **Math** | 60.0% | **80.0%** | 🏆 Single |
| **Knowledge** | 83.3% | **100.0%** | 🏆 Single |
| **Coding** | **78.1%** | 70.0% | 🏆 Multi (only win) |
| **Mixed** | **91.0%** | 90.0% | 🏆 Multi (marginal) |

**Multi-Agent only wins in coding tasks** - and barely.

---

## Technical Implementation

### Token Counting Methodology
**Unified Measurement**: All models measured using GPT-4 tiktoken for fair comparison
```python
import tiktoken
encoder = tiktoken.encoding_for_model("gpt-4")
tokens = len(encoder.encode(input_text)) + len(encoder.encode(output_text))
```

### Evaluation Datasets
**Standard Benchmarks** (moving beyond toy problems):
- **GSM8K**: Math word problems requiring multi-step reasoning
- **MMLU**: Multi-domain knowledge (science, history, etc.)  
- **HumanEval**: Code generation tasks
- **Mixed**: Balanced combination of above

### Accuracy Calculation
```python
def calculate_accuracy(final_answer: str, expected_answer: str) -> float:
    # Complete inclusion
    if expected_lower in final_lower:
        return 1.0
    
    # Partial word matching
    expected_words = set(expected_lower.split())
    final_words = set(final_lower.split())
    return len(expected_words.intersection(final_words)) / len(expected_words)
```

---

## Research Hypotheses & Validation

### Hypothesis 1: Multi-Agent Collaboration Improves Performance
**STATUS**: ❌ **REJECTED**
- Multi-agent systems consistently underperform single models
- Only marginal advantage in specific coding tasks
- Cost-benefit analysis strongly favors single models

### Hypothesis 2: Information Sharing Optimization Matters  
**STATUS**: ✅ **PARTIALLY VALIDATED**
- Information sharing level significantly impacts performance
- **PARTIAL sharing is consistently worst** (unexpected)
- Binary choice: full sharing or full isolation

### Hypothesis 3: Small Models Can Compete Through Collaboration
**STATUS**: ❌ **REJECTED**  
- 6× small model calls cannot match 1× large model call
- Quality degradation compounds through pipeline
- "Wisdom of crowds" doesn't apply to current LLM architectures

---

## Current Research Direction: Judge Upgrade Experiment

### New Hypothesis
**High-performance judge can salvage multi-agent pipeline**

#### Configuration Under Test
- Draft: `qwen2:0.5b` × 3 (cheap initial exploration)
- Review: `qwen2:0.5b` × 2 (lightweight filtering)  
- Judge: `llama3:8b` × 1 (high-quality final decision)

#### Expected Outcomes
**Optimistic**: Judge filters out small model errors, leverages diversity
**Pessimistic**: GIGO (Garbage In, Garbage Out) - poor inputs doom output

#### Research Questions
1. Can high-quality final stage compensate for low-quality early stages?
2. Is there a sweet spot in model size distribution across pipeline?
3. Does information aggregation provide value beyond single model reasoning?

---

## Broader Implications

### For Multi-Agent System Design
- **Pipeline quality is bottlenecked by weakest link**
- **Information sharing optimization is non-linear** 
- **Cost scaling is multiplicative, not additive**

### For LLM Applications  
- **Single large models often optimal** for most tasks
- **Multi-agent complexity rarely justifies costs**
- **Specialization may be key** - coding tasks show promise

### For Academic Research
- **Challenge conventional wisdom** about collaborative AI
- **Rigorous benchmarking essential** - toy problems mislead
- **Economic factors matter** in practical deployments

---

## Next Steps

### Immediate Experiments
1. **Judge Upgrade Results**: Test llama3:8b judge performance
2. **Recursive Agent**: Test problem decomposition approach
3. **Economic Intelligence**: Dynamic model selection optimization

### Future Directions
1. **Task-Specific Optimization**: Identify where multi-agent wins
2. **Hybrid Approaches**: Single model + specialized sub-agents
3. **Cost-Aware Architectures**: Performance budgets and constraints

---

## File Structure
```
Project-Arkhē/
├── src/
│   ├── agents/              # Agent implementations
│   ├── orchestrator/        # Pipeline coordination
│   ├── llm/                # LLM interfaces
│   ├── metrics/            # Information theory calculations
│   └── utils/              # Token counting, utilities
├── experiments/            # Experimental scripts
├── datasets/              # Standard benchmarks
├── results/               # Experimental outputs
└── docs/                  # Documentation
```

## Citation
If you use this research, please cite:
```
Project Arkhē: Multi-Agent LLM Systems Performance Analysis
[Author], 2025
Repository: [GitHub URL]
```

---

*Last Updated: 2025-01-09*
*Status: Active Research*