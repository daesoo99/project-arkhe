# Models and Assumptions - Project Arkhē

## Core Assumptions

### 1. Multi-Agent Architecture Assumptions
- **Pipeline Sequential Processing**: Each stage builds upon previous stage outputs
- **Information Flow Control**: Different isolation levels affect performance
- **Collaborative Intelligence**: Multiple weaker models can potentially outperform single strong model
- **Stage Specialization**: Different roles (Draft/Review/Judge) optimize for different aspects

### 2. Evaluation Assumptions  
- **GPT-4 Tokenizer as Universal Standard**: All models compared using same token counting method
- **Standard Benchmarks Represent Real Tasks**: GSM8K, MMLU, HumanEval reflect practical use cases
- **Accuracy Through String Matching**: Simple inclusion/word-overlap captures semantic correctness
- **Equal Weighting Across Categories**: Math, knowledge, coding tasks equally important

### 3. Economic Assumptions
- **Token Cost Linearity**: More tokens = proportionally higher cost
- **Time Cost Ignored**: Focus on computational cost, not latency
- **Efficiency = Accuracy/Tokens**: Simple ratio captures value proposition
- **No Infrastructure Costs**: Only direct model inference costs considered

---

## Model Configurations

### Multi-Agent Pipeline Models

#### Stage 1: Draft Generation
**Model**: `qwen2:0.5b`  
**Samples**: 3  
**Purpose**: Generate initial diverse responses  
**Assumptions**:
- Diversity more important than individual quality
- Small model sufficient for initial exploration
- Multiple samples capture different perspectives

#### Stage 2: Review & Refinement  
**Model**: `qwen2:0.5b` (current) / `gemma:2b` (previous)
**Samples**: 2  
**Purpose**: Filter and improve draft responses
**Assumptions**:
- Same-size model can improve upon peer outputs through different prompting
- Fewer samples needed as quality improves
- Review stage adds value through error correction

#### Stage 3: Final Judgment
**Model**: `llama3:8b` (current) / `gemma:2b` (previous)
**Samples**: 1  
**Purpose**: Make authoritative final decision
**Assumptions**:
- Larger model produces highest quality output
- Single high-quality sample better than multiple lower-quality
- Judge can effectively synthesize previous stage information

### Single Model Baselines

#### Primary Comparison Target
**Model**: `llama3:8b`  
**Samples**: 1  
**Assumptions**:
- Represents current "reasonable" model size for practical use
- Single inference provides baseline efficiency comparison
- Ollama local deployment reflects real-world usage

#### High-Performance Cloud Models  
**Models**: `claude-3-haiku`, `gpt-4o-mini`
**Samples**: 1 each
**Assumptions**:
- Cloud models represent performance upper bound
- API connectivity issues acceptable for research (noted in results)
- Cost comparison with local models valid despite deployment differences

---

## Information Sharing Models

### NONE (Complete Information Sharing)
**Draft Input**: Query only  
**Review Input**: Query + All draft responses  
**Judge Input**: Query + All draft responses + All review responses  

**Assumptions**:
- More information always helps decision making
- Agents can effectively process and synthesize multiple inputs
- No information overflow or confusion effects

### PARTIAL (Limited Information Sharing)  
**Draft Input**: Query only  
**Review Input**: Query + One draft response (round-robin)  
**Judge Input**: Query + One review response  

**Assumptions**:
- Focused information reduces noise
- One-to-one connections prevent information overload  
- Sequential refinement maintains quality

### COMPLETE (Information Isolation)
**Draft Input**: Query only  
**Review Input**: Query only (no draft information)  
**Judge Input**: Query only (no previous stage information)  

**Assumptions**:
- Independent reasoning reduces bias
- Parallel processing maintains diversity
- Final aggregation captures best of all approaches

---

## Experimental Design Assumptions

### Dataset Selection
**GSM8K (Math)**: Tests logical reasoning chains
**MMLU (Knowledge)**: Tests factual knowledge recall  
**HumanEval (Coding)**: Tests structured problem solving
**Mixed**: Tests general capability

**Assumptions**:
- These domains represent core LLM capabilities
- Performance across domains indicates general intelligence
- Equal weighting across domains reflects practical importance

### Sample Size  
**Questions per category**: 5-8  
**Total experimental runs**: ~60 per configuration

**Assumptions**:
- Small sample sufficient for directional insights
- Patterns emerge quickly with standard benchmarks
- Statistical significance secondary to magnitude of differences

### Measurement Methodology
**Token Counting**: GPT-4 tiktoken for all models  
**Accuracy**: String inclusion + word overlap  
**Efficiency**: Accuracy ÷ (Tokens ÷ 100)

**Assumptions**:
- GPT tokenizer generalizes reasonably to all models
- Simple string matching captures semantic correctness
- Linear relationship between accuracy and efficiency value

---

## Performance Assumptions

### Multi-Agent Hypotheses
1. **Diversity Benefit**: Multiple perspectives improve final output
2. **Error Correction**: Later stages can fix earlier stage mistakes  
3. **Specialization**: Different models excel at different subtasks
4. **Information Synthesis**: Combining partial solutions beats individual reasoning

### Single Model Hypotheses  
1. **Coherence Advantage**: Single reasoning chain maintains consistency
2. **Efficiency Advantage**: One inference cheaper than many
3. **Quality Scaling**: Larger models generally outperform smaller ones
4. **Simplicity Advantage**: Fewer failure modes in simple architecture

---

## Validation Assumptions

### What Constitutes "Success"
- **Multi-Agent Success**: ≥20% accuracy improvement justifies cost increase
- **Information Sharing Success**: Clear ranking between NONE/PARTIAL/COMPLETE
- **Economic Success**: Higher efficiency score indicates better approach

### What Constitutes "Failure"  
- **Multi-Agent Failure**: Higher cost with equal/lower accuracy
- **Information Sharing Failure**: PARTIAL performs worse than extremes
- **Economic Failure**: Efficiency score <50% of baseline

### Generalizability Assumptions
- **Task Generalization**: Results on these benchmarks predict performance on similar tasks
- **Scale Generalization**: Patterns hold for different model sizes  
- **Domain Generalization**: Findings apply beyond specific test categories
- **Time Generalization**: Results remain valid as models evolve

---

## Limitations and Caveats

### Known Model Limitations
- **Local Models Only**: No comparison with latest commercial models
- **Size Constraints**: Limited to models runnable on consumer hardware
- **Version Dependence**: Results specific to current model versions

### Experimental Limitations  
- **Small Sample Size**: Directional findings, not definitive conclusions
- **Simple Tasks**: May not reflect complex real-world problems  
- **Static Evaluation**: No interactive or iterative task performance

### Methodological Limitations
- **Token Counting Approximation**: GPT tokenizer imperfect proxy for all models
- **Accuracy Measurement**: String matching misses semantic equivalence
- **Cost Model Simplification**: Ignores infrastructure, latency, development costs

---

## Future Research Directions

### Model Architecture Extensions
1. **Heterogeneous Pipelines**: Different model families at each stage
2. **Dynamic Routing**: Adaptive model selection based on task type
3. **Feedback Loops**: Iterative refinement between stages

### Evaluation Extensions  
1. **Larger Scale Studies**: 100s-1000s of test cases per category  
2. **Human Evaluation**: Semantic quality assessment
3. **Task Complexity Analysis**: Performance vs problem difficulty curves

### Economic Model Extensions
1. **Full Cost Analysis**: Infrastructure, development, maintenance costs
2. **Value-Based Evaluation**: Task importance weighting  
3. **Real-Time Performance**: Latency-sensitive use cases

---

*Document Version: 1.0*  
*Last Updated: 2025-01-09*  
*Status: Living Document - Updated with new experimental findings*