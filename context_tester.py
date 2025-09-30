import os
import time
import sys
import argparse
from openai import OpenAI

# Global client variable - will be initialized based on CLI args

def generate_test_text(length_chars):
    """Generate test text of approximately specified character length."""
    base_text = "This is a test sentence used to fill up the context window for testing purposes. "
    repeat_count = max(1, length_chars // len(base_text))
    text = base_text * repeat_count
    return text[:length_chars]

def count_tokens_estimate(text, chars_per_token=4):
    """Rough estimate of token count.
    
    Args:
        text (str): Text to estimate tokens for
        chars_per_token (int): Estimated characters per token (default: 4)
                              Common values: 3-4 for English, 2-4 for other languages
    """
    return len(text) // chars_per_token

def tokens_to_chars_estimate(tokens, chars_per_token=4):
    """Convert estimated token count to character count.
    
    Args:
        tokens (int): Number of tokens
        chars_per_token (int): Estimated characters per token (default: 4)
    """
    return tokens * chars_per_token

def clear_lines(num_lines=1):
    """Clear multiple lines and return cursor to beginning."""
    for _ in range(num_lines):
        sys.stdout.write('\033[1A\033[K')  # Move up one line and clear it
    sys.stdout.write('\r')  # Return to beginning of line
    sys.stdout.flush()

def clear_line():
    """Clear the current line and return cursor to beginning."""
    sys.stdout.write('\r\033[K')
    sys.stdout.flush()

# Global variable to track visualization state
_search_line_shown = False

def update_search_visualization(iteration, current_tokens, current_chars, low_tokens, high_tokens, original_low_tokens, original_high_tokens, last_result=None):
    """Display a visual representation of the binary search space."""
    global _search_line_shown
    
    # Clear the entire previous content and return to our search line
    if _search_line_shown:
        # Move cursor up and clear all lines from our search position
        sys.stdout.write('\033[1A\033[K')  # Move up one line and clear it
        sys.stdout.flush()
    
    # Create visual representation of search space
    bar_width = 60
    total_range = original_high_tokens - original_low_tokens
    
    # Calculate positions within the visual bar
    if total_range > 0:
        current_pos = int(((current_tokens - original_low_tokens) / total_range) * bar_width)
        low_pos = int(((low_tokens - original_low_tokens) / total_range) * bar_width)
        high_pos = int(((high_tokens - original_low_tokens) / total_range) * bar_width)
    else:
        current_pos = low_pos = high_pos = 0
    
    # Ensure positions are within bounds
    current_pos = max(0, min(bar_width - 1, current_pos))
    low_pos = max(0, min(bar_width - 1, low_pos))
    high_pos = max(0, min(bar_width - 1, high_pos))
    
    # Build the visual bar
    bar = list('·' * bar_width)  # Inactive space
    
    # Mark the search boundaries
    for i in range(low_pos, high_pos + 1):
        if i < bar_width:
            bar[i] = '━'  # Active search space
    
    # Mark current test position with result indicator
    if current_pos < bar_width:
        if last_result is None:
            bar[current_pos] = '▼'  # Testing in progress
        elif last_result:
            bar[current_pos] = '✓'  # Success
        else:
            bar[current_pos] = '✗'  # Failed
    
    bar_str = ''.join(bar)
    
    # Show only the visual bar with minimal info
    result_indicator = ""
    if last_result is not None:
        result_indicator = " ✓" if last_result else " ✗"
    
    # Single line output: iteration, tokens, and visual bar
    line = f"{iteration:2d}: {current_tokens:>8,} [{bar_str}]{result_indicator}\n"
    sys.stdout.write(line)
    sys.stdout.flush()
    
    _search_line_shown = True

def display_final_visualization(model_name, max_tokens, max_chars, total_iterations, search_history):
    """Display final results with a simple visualization of the search path."""
    print(f"\nMaximum successful context: {max_tokens:,} tokens")

def test_context_length_binary_search(model_name, lower_bound_tokens=250, upper_bound_tokens=250000, chars_per_token=4):
    """Test the maximum context length using binary search.
    
    Args:
        model_name (str): The model to test
        lower_bound_tokens (int): Minimum context length to test in tokens
        upper_bound_tokens (int): Maximum context length to test in tokens
        chars_per_token (int): Estimated characters per token for token estimation
    """
    # Convert token bounds to character bounds for internal processing
    lower_bound_chars = tokens_to_chars_estimate(lower_bound_tokens, chars_per_token)
    upper_bound_chars = tokens_to_chars_estimate(upper_bound_tokens, chars_per_token)
    
    # Reset visualization state for new search
    global _search_line_shown
    _search_line_shown = False
    
    print(f"Testing {model_name}: {lower_bound_tokens:,} - {upper_bound_tokens:,} tokens")
    
    # Binary search variables (work in characters internally)
    low = lower_bound_chars
    high = upper_bound_chars
    max_successful_length_chars = 0
    max_successful_tokens = 0
    iteration = 0
    search_history = []
    total_range_tokens = upper_bound_tokens - lower_bound_tokens
    
    while low <= high:
        # Calculate current search boundaries
        current_low_tokens = count_tokens_estimate(generate_test_text(low), chars_per_token)
        current_high_tokens = count_tokens_estimate(generate_test_text(high), chars_per_token)
        
        # Exit if search space is under 1K tokens
        search_space = current_high_tokens - current_low_tokens
        if search_space < 1000:
            # Clear the current search line before showing final message
            if _search_line_shown:
                sys.stdout.write('\033[1A\033[K')
                sys.stdout.flush()
            print(f"Search space narrowed to {search_space:,} tokens - stopping search")
            break
            
        iteration += 1
        mid = (low + high) // 2
        mid_tokens = count_tokens_estimate(generate_test_text(mid), chars_per_token)
        
        # Show visual progress (testing in progress)
        update_search_visualization(iteration, mid_tokens, mid, current_low_tokens, current_high_tokens, lower_bound_tokens, upper_bound_tokens)
        
        success, error_msg, usage_data, truncated = test_single_context_length(model_name, mid, chars_per_token)
        
        # Use actual token count if available, otherwise fall back to estimate
        actual_tokens = mid_tokens
        if usage_data and 'prompt_tokens' in usage_data:
            actual_tokens = usage_data['prompt_tokens']
        
        # Update visualization with result
        update_search_visualization(iteration, actual_tokens, mid, current_low_tokens, current_high_tokens, lower_bound_tokens, upper_bound_tokens, success)
        
        # Show usage confirmation if available
        if usage_data and success:
            print(f"Actual: {usage_data['prompt_tokens']:,} tokens (est: {usage_data['estimated_tokens']:,})")
        
        # If truncation detected, stop the search and report the truncated token count
        if truncated:
            # Clear the current search line before showing final message
            if _search_line_shown:
                sys.stdout.write('\033[1A\033[K')
                sys.stdout.flush()
            print(f"Token truncation detected - context window limit: {actual_tokens:,} tokens")
            max_successful_tokens = actual_tokens
            max_successful_length_chars = mid
            break
        
        # Note: Errors are silently handled - no persistent error messages
        
        # Track search history with actual tokens
        search_history.append((actual_tokens, success, False))
        
        if success:
            max_successful_length_chars = mid
            max_successful_tokens = actual_tokens
            low = mid + 1
        else:
            high = mid - 1
        
        # Small delay to show result before next iteration
        time.sleep(0.8)
    
    # Mark the final successful result in history
    if search_history:
        search_history[-1] = (search_history[-1][0], search_history[-1][1], True)
    
    # Clear the final search line before showing results
    if _search_line_shown:
        sys.stdout.write('\033[1A\033[K')
        sys.stdout.flush()
    
    # Display final results with visualization
    display_final_visualization(model_name, max_successful_tokens, max_successful_length_chars, iteration, search_history)

def test_single_context_length(model_name, char_length, chars_per_token=4):
    """Test a single context length and return success/failure.
    
    Args:
        model_name (str): The model to test
        char_length (int): Context length in characters to test
        chars_per_token (int): Estimated characters per token for token estimation
        
    Returns:
        tuple: (bool, str or None, dict or None, bool) - Success status, error message, usage data, and truncation flag
    """
    test_text = generate_test_text(char_length)
    estimated_tokens = count_tokens_estimate(test_text, chars_per_token)
    
    try:
        # Create a simple message that includes our test text
        messages = [
            {
                "role": "user", 
                "content": f"Please summarize this text in one sentence: {test_text}"
            }
        ]
        
        start_time = time.time()
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=100,  # Keep response short to focus on input context
            temperature=0.1
        )
        end_time = time.time()
        
        # Extract usage data if available
        usage_data = None
        truncated = False
        if hasattr(response, 'usage') and response.usage:
            usage_data = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens,
                'estimated_tokens': estimated_tokens
            }
            
            # Check for truncation: if actual tokens are significantly less than estimated
            # (more than 10% difference), the API likely truncated the input
            if usage_data['prompt_tokens'] < estimated_tokens * 0.9:
                truncated = True
                return False, "Token truncation detected", usage_data, truncated
        
        return True, None, usage_data, False
        
    except Exception as e:
        return False, str(e), None, False
    

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test the maximum context length of OpenAI API compatible models using binary search.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test OpenAI GPT-4
  python context_tester.py --endpoint https://api.openai.com/v1 --api-key sk-... --model gpt-4

  # Test local model with Ollama
  python context_tester.py --endpoint http://localhost:11434/v1 --api-key ollama --model llama2

  # Test with custom token bounds
  python context_tester.py --endpoint https://api.openai.com/v1 --api-key sk-... --model gpt-3.5-turbo --lower-bound 1000 --upper-bound 16000

  # Use API key from environment variable
  export OPENAI_API_KEY=sk-...
  python context_tester.py --endpoint https://api.openai.com/v1 --model gpt-4
        """)
    
    parser.add_argument("--endpoint", "--base-url", 
                       required=True,
                       help="API endpoint URL (e.g., https://api.openai.com/v1)")
    
    parser.add_argument("--api-key", 
                       help="API key for authentication. If not provided, will try OPENAI_API_KEY environment variable")
    
    parser.add_argument("--model", 
                       required=True,
                       help="Model name to test (e.g., gpt-4, gpt-3.5-turbo, llama2)")
    
    parser.add_argument("--lower-bound", 
                       type=int, 
                       default=128000,
                       help="Lower bound for context length test in tokens (default: 128,000)")
    
    parser.add_argument("--upper-bound", 
                       type=int, 
                       default=2000000,
                       help="Upper bound for context length test in tokens (default: 2,000,000)")
    
    parser.add_argument("--chars-per-token", 
                       type=int, 
                       default=4,
                       help="Estimated characters per token for token estimation (default: 4)")
    
    args = parser.parse_args()
    
    # Get API key from environment if not provided
    if not args.api_key:
        args.api_key = os.getenv("OPENAI_API_KEY")
        if not args.api_key:
            parser.error("API key must be provided via --api-key argument or OPENAI_API_KEY environment variable")
    
    # Validate bounds
    if args.lower_bound <= 0:
        parser.error("Lower bound must be positive")
    
    if args.upper_bound <= args.lower_bound:
        parser.error("Upper bound must be greater than lower bound")
    
    if args.chars_per_token <= 0:
        parser.error("Characters per token must be positive")
    
    return args

if __name__ == "__main__":
    global client
    
    args = parse_args()
    
    # Initialize the global client with the provided arguments
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.endpoint
    )
    
    print(f"Selected range: {args.lower_bound:,} - {args.upper_bound:,} tokens")
    print(f"Testing model: {args.model}")
    print(f"API endpoint: {args.endpoint}")
    print()
    
    test_context_length_binary_search(args.model, args.lower_bound, args.upper_bound, args.chars_per_token)
