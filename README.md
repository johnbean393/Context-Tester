# LLM Context Window Tester

Efficiently measure the maximum context window size of any LLM using binary search and OpenAI-compatible APIs.

## Quick Start

```bash
# Install dependencies
pip install openai

# Test a model
python context_tester.py --endpoint <API_ENDPOINT> --api-key <API_KEY> --model <MODEL_NAME>
```

## Examples

```bash
# OpenAI GPT-4
python context_tester.py --endpoint "https://api.openai.com/v1" --api-key "sk-..." --model "gpt-4"

# Zhipu GLM-4.6
python context_tester.py --endpoint "https://open.bigmodel.cn/api/paas/v4/" --api-key "sk-..." --model "glm-4.6"

# Custom bounds
python context_tester.py --endpoint "..." --api-key "..." --model "..." --lower-bound 1000 --upper-bound 16000
```

## Options

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--endpoint` | Yes | - | API endpoint URL |
| `--api-key` | No* | - | API key (or use `OPENAI_API_KEY` env var) |
| `--model` | Yes | - | Model name to test |
| `--lower-bound` | No | 128,000 | Lower bound for testing (tokens) |
| `--upper-bound` | No | 2,000,000 | Upper bound for testing (tokens) |
| `--chars-per-token` | No | 4 | Character-to-token ratio estimate |

*Required unless `OPENAI_API_KEY` environment variable is set.

## How It Works

Uses binary search to efficiently find the maximum context length by sending test text of varying sizes to the model. Provides real-time visual progress and reports the maximum successful context size in tokens.

## Compatible APIs

Works with any OpenAI-compatible API:
- OpenAI (GPT-3.5, GPT-4, etc.)
- Anthropic Claude (via compatible wrappers)  
- Local models (Ollama, vLLM, FastChat)
- Cloud providers (Azure OpenAI, AWS Bedrock)
- Custom APIs implementing OpenAI chat format