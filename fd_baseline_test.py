#!/usr/bin/env python3
"""
Baseline test for FunctionGemma thinking capability.
This script tests the model without any FD optimization to establish a baseline.
"""

import torch
torch._dynamo.config.disable = True

from fd_thinking_optimizer import (
    FunctionGemmaRunner,
    ThinkingPrompt,
    FewShotExample,
    TEST_QUERIES,
    INITIAL_PROMPT,
    THINK_TAG_OPEN,
    THINK_TAG_CLOSE,
)


def create_baseline_prompt() -> ThinkingPrompt:
    """Create a minimal baseline prompt without any thinking instructions."""
    return ThinkingPrompt(
        system_prompt="You are a helpful assistant with access to tools.",
        few_shot_examples=[]
    )


def create_simple_thinking_prompt() -> ThinkingPrompt:
    """Create a simple prompt with thinking instructions but no few-shot."""
    return ThinkingPrompt(
        system_prompt="""You are a helpful assistant with access to tools.
Before calling any tool, first think about the problem inside <think>...</think> tags.
Then make the tool call.""",
        few_shot_examples=[]
    )


def main():
    print("="*60)
    print("FunctionGemma Baseline Test")
    print("="*60)

    # Load model
    print("\n[1/4] Loading FunctionGemma model...")
    runner = FunctionGemmaRunner()

    # Define prompts to test
    prompts = {
        "Baseline (no instructions)": create_baseline_prompt(),
        "Simple thinking instruction": create_simple_thinking_prompt(),
        "Full prompt with few-shot": INITIAL_PROMPT,
    }

    results = {}

    for prompt_name, prompt in prompts.items():
        print(f"\n[Testing] {prompt_name}")
        print("-" * 40)

        thinking_count = 0
        tool_call_count = 0

        for i, query in enumerate(TEST_QUERIES):
            print(f"\n  Query {i+1}: {query.user_content[:50]}...")

            result = runner.generate(prompt, query)

            # Print summary
            print(f"    Has thinking: {result.has_thinking}")
            print(f"    Has tool call: {result.has_tool_call}")
            if result.thinking_content:
                print(f"    Thinking: {result.thinking_content[:100]}...")
            if result.tool_call_name:
                print(f"    Tool called: {result.tool_call_name}")

            if result.has_thinking:
                thinking_count += 1
            if result.has_tool_call:
                tool_call_count += 1

        thinking_rate = thinking_count / len(TEST_QUERIES)
        tool_call_rate = tool_call_count / len(TEST_QUERIES)

        results[prompt_name] = {
            "thinking_rate": thinking_rate,
            "tool_call_rate": tool_call_rate,
        }

        print(f"\n  Summary for '{prompt_name}':")
        print(f"    Thinking rate: {thinking_rate:.0%}")
        print(f"    Tool call rate: {tool_call_rate:.0%}")

    # Final summary
    print("\n" + "="*60)
    print("Baseline Results Summary")
    print("="*60)

    for prompt_name, metrics in results.items():
        print(f"\n{prompt_name}:")
        print(f"  Thinking rate: {metrics['thinking_rate']:.0%}")
        print(f"  Tool call rate: {metrics['tool_call_rate']:.0%}")


if __name__ == "__main__":
    main()
