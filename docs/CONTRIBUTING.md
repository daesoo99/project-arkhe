# Contributing to Project Arkhē

Thank you for your interest in contributing to Project Arkhē! This project implements an economic intelligence system for multi-agent AI with a working MVP.

## Quick Start

1. Fork the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment: `.\scripts\setup.ps1` (Windows) or equivalent setup
4. Run quick test: `python experiments/bench_simple.py --limit 3`
5. Submit a pull request

## Current System Status

**✅ MVP Complete**: 3-stage pipeline system with proven 4x efficiency gains
- **Pipeline Orchestrator**: `src/orchestrator/pipeline.py`
- **LLM Integration**: `src/llm/simple_llm.py` (Ollama, OpenAI, Anthropic)
- **Advanced Scoring**: `src/utils/scorers.py` (6 task-specific evaluators)
- **Economic Intelligence**: Real-time cost-performance tracking

## Core Concepts

- **Economic Intelligence**: Cost-effective model allocation (`qwen2:0.5b → gemma:2b → llama3:8b`)
- **Smart Resource Allocation**: Expensive models only for critical decisions
- **Pipeline Orchestration**: Flexible multi-agent workflow engine

## Development Areas

### 🎯 Priority Contributions
1. **Model Integration**: Add support for more LLM providers
2. **Pipeline Patterns**: Novel multi-agent orchestration strategies
3. **Evaluation Methods**: New task-specific scoring functions
4. **Information Theory**: Shannon entropy tracking and promotion policies

### 🧪 Experiment Extensions
- Test with different model combinations
- Implement promotion policies (route high-entropy cases to expensive models)
- Validate on specialized domains (coding, creative tasks, etc.)

## Code Style

- Follow existing patterns in `src/orchestrator/` and `src/llm/`
- Use the unified LLM interface from `src/llm/simple_llm.py`
- Add comprehensive tests for new pipeline components
- Include cost tracking for any new LLM calls

## Testing

```bash
# Quick validation (3 tasks)
python experiments/bench_simple.py --limit 3

# Pipeline comparison
python experiments/integrated_test.py

# Full benchmark matrix
.\scripts\run_matrix.ps1
```

## Reporting Issues

Please use GitHub Issues for:
- Bug reports with pipeline orchestration
- New LLM provider integration requests
- Performance optimization suggestions
- Economic intelligence metric improvements
- Documentation updates

## Research Collaboration

We welcome academic collaborators interested in:
- Information theory applications to multi-agent systems
- Cost-performance frontier optimization
- Large-scale validation studies
- Novel evaluation methodologies

## License

By contributing, you agree that your contributions will be licensed under the MIT License.