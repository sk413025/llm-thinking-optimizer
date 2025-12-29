#!/usr/bin/env python3
"""Debug script to see raw model output."""

import torch
torch._dynamo.config.disable = True

from fd_thinking_optimizer import (
    FunctionGemmaRunner,
    ThinkingPrompt,
    TEST_QUERIES,
)


def main():
    print("Loading model...")
    runner = FunctionGemmaRunner()

    # Test with simple prompt
    prompt = ThinkingPrompt(
        system_prompt="""You are a helpful assistant. Before using any tool, you should think step by step about:
1. What the user is asking for
2. Which tool to use
3. What parameters to provide

Format your response as:
<think>
[Your reasoning here]
</think>
[Then make the tool call]""",
        few_shot_examples=[]
    )

    print("\n" + "="*60)
    print("Testing with explicit format instructions")
    print("="*60)

    for i, query in enumerate(TEST_QUERIES[:2]):  # Just first 2 for brevity
        print(f"\n--- Query {i+1}: {query.user_content} ---")
        result = runner.generate(prompt, query)
        print(f"\n[Full Output]:\n{result.output}")
        print(f"\n[Parsed]:")
        print(f"  Has thinking: {result.has_thinking}")
        print(f"  Has tool call: {result.has_tool_call}")


if __name__ == "__main__":
    main()
