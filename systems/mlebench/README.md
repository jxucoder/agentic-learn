# agentic-learn MLE-bench Agent

An MLE-bench competition agent built on the agentic-learn framework.

## Architecture

The agent runs inside the MLE-bench Docker container and:

1. **Parses** the competition description and data files
2. **Generates** ML solution code via LLM
3. **Executes** the code and captures results
4. **Reflects** on errors/scores and iteratively improves
5. **Searches** over solution variants using MCTS (Phase 2+)

## Quick Start

```bash
# Build the agent image
docker build --platform=linux/amd64 -t agentic-learn \
  systems/mlebench/ \
  --build-arg SUBMISSION_DIR=/home/submission \
  --build-arg LOGS_DIR=/home/logs \
  --build-arg CODE_DIR=/home/code \
  --build-arg AGENT_DIR=/home/agent

# Run on a single competition
python run_agent.py \
  --agent-id agentic-learn \
  --competition-set experiments/splits/spaceship-titanic.txt

# Run on Lite split (22 competitions)
python run_agent.py \
  --agent-id agentic-learn \
  --competition-set experiments/splits/low.txt \
  --n-seeds 3
```

## Configuration

Set environment variables for your LLM provider:

```bash
export ANTHROPIC_API_KEY=...   # For Claude
export OPENAI_API_KEY=...      # For OpenAI/DeepSeek
```

## Tool Tiers

The agent uses agentic-learn's tiered tool system:

- **Phase 1**: Direct LLM code generation loop (no tools, just code gen + exec)
- **Phase 2**: MCTS search over solution variants (adapted from Silver)
- **Phase 3**: Hierarchical memory + structured reflection (adapted from Minsky)

## Phases

| Phase | Target Score | Key Feature |
|-------|-------------|-------------|
| 1 | ~20-25% | Greedy iteration loop |
| 2 | ~35-45% | MCTS search |
| 3 | ~45-55% | Memory + reflection |
| 4 | ~55%+ | Multi-model + ensembling |
