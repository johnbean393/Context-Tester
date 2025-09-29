# LLM Context Window Tester

A Python tool for measuring the maximum context window size of Large Language Models (LLMs) accessible through OpenAI-compatible APIs. This tool is particularly useful for testing unannounced or proprietary models where the context window size is unknown.

## Overview

This tool uses binary search to efficiently determine the maximum context length that a model can handle. It generates test text of varying lengths and attempts to get the model to process it, narrowing down the maximum successful context size through systematic testing.

### Key Features

- **Binary Search Algorithm**: Efficiently finds the maximum context length in logarithmic time
- **Visual Progress Tracking**: Real-time visualization of the search progress with ASCII art
- **OpenAI-Compatible APIs**: Works with OpenAI, Anthropic, local models (Ollama, vLLM), and other compatible endpoints
- **Configurable Parameters**: Customizable token bounds, character-to-token ratios, and API settings
- **Error Handling**: Graceful handling of API errors and edge cases
- **Detailed Output**: Clear reporting of results with token and character counts

## Installation

### Prerequisites

- Python 3.7 or higher
- `openai` Python package

### Setup

1. Clone or download this repository:
```bash
git clone <repository-url>
cd context-tester
```

2. Install required dependencies:
```bash
pip install openai
```

## Usage

### Basic Usage

```bash
python context_tester.py --endpoint <API_ENDPOINT> --api-key <API_KEY> --model <MODEL_NAME>
```

### Examples

#### Testing GLM-4.6
```bash
python context_tester.py \
  --endpoint "https://open.bigmodel.cn/api/paas/v4/" \
  --api-key "sk-xxxxx" \
  --model "glm-4.6"
```

#### Testing with Custom Token Bounds
```bash
python context_tester.py \
  --endpoint "https://open.bigmodel.cn/api/paas/v4/" \
  --api-key "sk-xxxxx" \
  --model "glm-4.6" \
  --lower-bound 1000 \
  --upper-bound 16000
```

#### Using Environment Variable for API Key
```bash
export ZHIPU_API_KEY="sk-xxxxx"
python context_tester.py \
  --endpoint "https://open.bigmodel.cn/api/paas/v4/" \
  --api-key $ZHIPU_API_KEY \
  --model "glm-4.6"
```

## Command Line Options

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--endpoint` | Yes | - | API endpoint URL (e.g., `https://api.openai.com/v1`) |
| `--api-key` | No* | - | API key for authentication. Can use `OPENAI_API_KEY` env var |
| `--model` | Yes | - | Model name to test (e.g., `gpt-4`, `llama2`) |
| `--lower-bound` | No | 128,000 | Lower bound for context length test (in tokens) |
| `--upper-bound` | No | 2,000,000 | Upper bound for context length test (in tokens) |
| `--chars-per-token` | No | 4 | Estimated characters per token for estimation |

*Required unless `OPENAI_API_KEY` environment variable is set.

## How It Works

### Algorithm

1. **Binary Search**: The tool uses binary search to efficiently find the maximum context length
2. **Test Text Generation**: Creates repetitive test text of specific character lengths
3. **API Testing**: Sends the text to the model with a request to summarize it
4. **Result Analysis**: Determines success/failure based on whether the API call completes
5. **Boundary Adjustment**: Adjusts search boundaries based on results until convergence

### Token Estimation

Since exact tokenization varies by model and isn't always available via API, the tool estimates tokens based on character count:

- **Default Ratio**: 4 characters per token (suitable for English text)
- **Customizable**: Use `--chars-per-token` to adjust for different languages or models
- **Accuracy**: Provides reasonable approximations for most use cases

### Visual Output

The tool provides real-time visual feedback during testing:

```
Testing gpt-4: 128,000 - 2,000,000 tokens
Selected range: 128,000 - 2,000,000 tokens

 1:  1,064,000 [··················━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━▼━━━━━━━━··················] ✓
 2:  1,532,000 [··················━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━▼━━━━━━━━━━━━━━··················] ✗
 3:  1,298,000 [··················━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━▼━━━━━━━━━━━━━━━━━━━━━··················] ✓
...
Maximum successful context: 1,456,789 tokens
```

### Stopping Criteria

The search stops when:
- The search space narrows to less than 1,000 tokens
- The binary search algorithm converges
- An unrecoverable error occurs

## Compatible APIs

This tool works with any OpenAI-compatible API, including:

- **OpenAI**: GPT-3.5, GPT-4, etc.
- **Anthropic**: Claude models (via compatible wrappers)
- **Local Models**: Ollama, vLLM, FastChat
- **Cloud Providers**: Azure OpenAI, AWS Bedrock (with compatible endpoints)
- **Custom APIs**: Any service implementing OpenAI's chat completions format

## Troubleshooting

### Common Issues

#### Authentication Errors
```
Error: API key must be provided via --api-key argument or OPENAI_API_KEY environment variable
```
**Solution**: Ensure you provide a valid API key either via command line or environment variable.

#### Connection Errors
```
Error: Connection failed
```
**Solution**: 
- Verify the endpoint URL is correct
- Check if the service is running (for local models)
- Ensure network connectivity

#### Model Not Found
```
Error: Model 'model-name' not found
```
**Solution**: 
- Verify the model name is correct
- Ensure the model is available on your API endpoint
- Check if you have access to the requested model

### Performance Considerations

- **API Rate Limits**: The tool includes small delays between requests to respect rate limits
- **Large Context Windows**: Testing very large contexts (>1M tokens) may take several minutes
- **Cost**: Be aware that testing large contexts may consume significant API credits

### Accuracy Considerations

- **Token Estimation**: Character-based token estimation is approximate; actual token counts may vary
- **Model Variations**: Different model versions may have different context limits
- **API Limitations**: Some APIs may have stricter limits than the model's actual capacity

## Output Interpretation

### Successful Run
```
Maximum successful context: 1,456,789 tokens
```
This indicates the largest context size that the model successfully processed.

### Incomplete Search
```
Search space narrowed to 500 tokens - stopping search
Maximum successful context: 1,450,000 tokens
```
The search stopped before finding the exact limit but narrowed it down to within 500 tokens.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Potential Improvements

- Support for other tokenization methods
- Integration with actual tokenizer libraries
- More sophisticated test text generation
- Parallel testing of multiple models
- Export results to various formats
- Web interface

## License

This project is open source. Please check the license file for details.

## Changelog

### Current Version
- Binary search algorithm for efficient context testing
- Visual progress indicators
- Support for OpenAI-compatible APIs
- Configurable token bounds and estimation
- Comprehensive error handling

---

## Quick Start

For the impatient, here's a minimal example:

```bash
# Install dependencies
pip install openai
```

```bash
# Test Zhipu GLM-4.6 (replace with your API key)
python context_tester.py \
  --endpoint "https://open.bigmodel.cn/api/paas/v4/" \
  --api-key $ZHIPU_API_KEY \
  --model "glm-4.6"
```

That's it! The tool will automatically find the maximum context window for your model.
