#!/usr/bin/env python3
"""
FD 可行性深度分析實驗

驗證假設:
- H1: 模型完全無法產生 <think> 標籤
- H2: 模型對 prompt 變化沒有反應
- H3: Few-shot 對模型無效
- H4: 模型無法遵循任何格式指令
- H5: 模型的輸出分佈是完全固定的
"""

import torch
torch._dynamo.config.disable = True

import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Callable, Tuple
from datetime import datetime
from collections import defaultdict
import re

# =============================================================================
# Configuration
# =============================================================================

THINK_TAG_OPEN = "<think>"
THINK_TAG_CLOSE = "</think>"
NUM_SAMPLES_PER_CONDITION = 3
MAX_NEW_TOKENS = 256

# Test queries (simplified for efficiency)
TEST_QUERIES = [
    {
        "content": "What's the weather in Tokyo?",
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }
        }]
    },
    {
        "content": "Calculate 15% of 200",
        "tools": [{
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Do math calculations",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"]
                }
            }
        }]
    },
    {
        "content": "Search for Python tutorials",
        "tools": [{
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }
        }]
    },
]


# =============================================================================
# Model Runner
# =============================================================================

class ModelRunner:
    def __init__(self):
        print("Loading FunctionGemma-270M...")
        from unsloth import FastLanguageModel

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/functiongemma-270m-it",
            max_seq_length=2048,
            load_in_4bit=False,
            load_in_8bit=False,
            load_in_16bit=True,
        )
        FastLanguageModel.for_inference(self.model)
        print("Model loaded!")

    def generate(
        self,
        system_prompt: str,
        user_content: str,
        tools: List[Dict],
        few_shot: List[Dict] = None,
        temperature: float = 0.7,
        num_samples: int = 1
    ) -> List[str]:
        """Generate outputs with given configuration."""

        messages = [{"role": "system", "content": system_prompt}]

        # Add few-shot examples
        if few_shot:
            for ex in few_shot:
                messages.append({"role": "user", "content": ex["user"]})
                messages.append({"role": "assistant", "content": ex["assistant"]})

        messages.append({"role": "user", "content": user_content})

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )
        if text.startswith("<bos>"):
            text = text[5:]

        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        outputs = []

        for _ in range(num_samples):
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True if temperature > 0 else False,
                    top_p=0.95,
                    top_k=64,
                    temperature=temperature if temperature > 0 else 1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            output_tokens = generated[0][inputs["input_ids"].shape[1]:]
            output_text = self.tokenizer.decode(output_tokens, skip_special_tokens=False)
            outputs.append(output_text)

        return outputs


# =============================================================================
# Experiment 1: Prompt Variants
# =============================================================================

PROMPT_VARIANTS = {
    "baseline": "You are a helpful assistant.",

    "explicit_think": """You are a helpful assistant.
Before calling any tool, you MUST think step by step in <think>...</think> tags.
Format: <think>your reasoning</think> then tool call.""",

    "chain_of_thought": """You are a helpful assistant.
Let's think step by step before acting.
First explain your reasoning, then make the tool call.""",

    "expert_role": """You are an expert assistant who always explains reasoning first.
Before any action, describe what you're thinking about the problem.""",

    "teacher_role": """You are a teacher who shows your work.
Always explain your thought process before answering or using tools.""",

    "json_thinking": """You are a helpful assistant.
Before any tool call, output your reasoning in this format:
REASONING: <your thought process here>
Then make the tool call.""",

    "no_direct_action": """You are a helpful assistant.
IMPORTANT: Do NOT call tools directly!
First, write a short explanation of what you plan to do and why.
Only then, make the tool call.""",

    "thinking_required": """You are a helpful assistant with tools.
REQUIRED FORMAT:
1. First output <think>your reasoning about what to do</think>
2. Then make the appropriate tool call
This format is MANDATORY.""",

    "step_by_step": """You are a helpful assistant.
Always follow these steps:
Step 1: Think about what the user needs
Step 2: Explain which tool to use and why
Step 3: Call the tool with correct parameters
Show all steps in your response.""",

    "reasoning_prefix": """You are a helpful assistant.
Your response must start with "Let me think:" followed by your reasoning.
Then call the appropriate tool.""",
}


def run_experiment_1(runner: ModelRunner) -> Dict:
    """Experiment 1: Test different prompt variants."""
    print("\n" + "="*60)
    print("EXPERIMENT 1: Prompt Variants")
    print("="*60)

    results = {}

    for prompt_name, system_prompt in PROMPT_VARIANTS.items():
        print(f"\n[Testing] {prompt_name}")
        variant_results = []

        for query in TEST_QUERIES:
            outputs = runner.generate(
                system_prompt=system_prompt,
                user_content=query["content"],
                tools=query["tools"],
                temperature=0.7,
                num_samples=NUM_SAMPLES_PER_CONDITION,
            )

            for output in outputs:
                has_think = THINK_TAG_OPEN in output and THINK_TAG_CLOSE in output
                has_reasoning = any(marker in output.lower() for marker in
                    ["reasoning:", "let me think", "i think", "step 1", "first,"])
                has_tool_call = "call:" in output or "<start_function_call>" in output

                variant_results.append({
                    "query": query["content"],
                    "output": output[:500],  # Truncate for storage
                    "has_think_tag": has_think,
                    "has_any_reasoning": has_reasoning or has_think,
                    "has_tool_call": has_tool_call,
                    "output_length": len(output),
                })

        # Calculate metrics
        think_rate = sum(1 for r in variant_results if r["has_think_tag"]) / len(variant_results)
        reasoning_rate = sum(1 for r in variant_results if r["has_any_reasoning"]) / len(variant_results)
        tool_rate = sum(1 for r in variant_results if r["has_tool_call"]) / len(variant_results)
        avg_length = sum(r["output_length"] for r in variant_results) / len(variant_results)

        results[prompt_name] = {
            "think_rate": think_rate,
            "reasoning_rate": reasoning_rate,
            "tool_call_rate": tool_rate,
            "avg_output_length": avg_length,
            "samples": variant_results,
        }

        print(f"  Think tag rate: {think_rate:.0%}")
        print(f"  Any reasoning rate: {reasoning_rate:.0%}")
        print(f"  Tool call rate: {tool_rate:.0%}")

    return results


# =============================================================================
# Experiment 2: Few-Shot Effect
# =============================================================================

FEW_SHOT_EXAMPLES = [
    {
        "user": "What time is it in London?",
        "assistant": "<think>\nThe user wants to know the current time in London.\nI should use the get_time tool with timezone Europe/London.\n</think>"
    },
    {
        "user": "Convert 100 USD to EUR",
        "assistant": "<think>\nUser needs currency conversion from USD to EUR.\nI'll use the convert_currency tool with amount=100, from=USD, to=EUR.\n</think>"
    },
    {
        "user": "Find nearby restaurants",
        "assistant": "<think>\nUser wants restaurant recommendations nearby.\nI should use search_places with query='restaurants' and get location.\n</think>"
    },
]

SYSTEM_PROMPT_FOR_FEWSHOT = """You are a helpful assistant with tools.
Before calling any tool, think through the problem in <think>...</think> tags."""


def run_experiment_2(runner: ModelRunner) -> Dict:
    """Experiment 2: Test few-shot effect (0, 1, 2, 3 shot)."""
    print("\n" + "="*60)
    print("EXPERIMENT 2: Few-Shot Effect")
    print("="*60)

    results = {}

    for n_shot in [0, 1, 2, 3]:
        print(f"\n[Testing] {n_shot}-shot")
        few_shot = FEW_SHOT_EXAMPLES[:n_shot] if n_shot > 0 else None
        shot_results = []

        for query in TEST_QUERIES:
            outputs = runner.generate(
                system_prompt=SYSTEM_PROMPT_FOR_FEWSHOT,
                user_content=query["content"],
                tools=query["tools"],
                few_shot=few_shot,
                temperature=0.7,
                num_samples=NUM_SAMPLES_PER_CONDITION,
            )

            for output in outputs:
                has_think = THINK_TAG_OPEN in output and THINK_TAG_CLOSE in output
                has_tool_call = "call:" in output or "<start_function_call>" in output

                shot_results.append({
                    "query": query["content"],
                    "output": output[:500],
                    "has_think_tag": has_think,
                    "has_tool_call": has_tool_call,
                })

        think_rate = sum(1 for r in shot_results if r["has_think_tag"]) / len(shot_results)
        tool_rate = sum(1 for r in shot_results if r["has_tool_call"]) / len(shot_results)

        results[f"{n_shot}_shot"] = {
            "think_rate": think_rate,
            "tool_call_rate": tool_rate,
            "samples": shot_results,
        }

        print(f"  Think rate: {think_rate:.0%}")
        print(f"  Tool call rate: {tool_rate:.0%}")

    return results


# =============================================================================
# Experiment 3: Format Compliance
# =============================================================================

FORMAT_TESTS = [
    {
        "name": "chinese_response",
        "instruction": "You must respond in Chinese (中文). Use Chinese characters in your response.",
        "check": lambda x: any(ord(c) > 0x4E00 and ord(c) < 0x9FFF for c in x),
    },
    {
        "name": "prefix_ok",
        "instruction": "Your response MUST start with 'OK:' (including the colon).",
        "check": lambda x: x.strip().startswith("OK:"),
    },
    {
        "name": "suffix_end",
        "instruction": "Your response MUST end with '[DONE]' marker.",
        "check": lambda x: "[DONE]" in x,
    },
    {
        "name": "numbered_steps",
        "instruction": "List your response as numbered steps: 1. ... 2. ... etc.",
        "check": lambda x: "1." in x and "2." in x,
    },
]


def run_experiment_3(runner: ModelRunner) -> Dict:
    """Experiment 3: Test format compliance ability."""
    print("\n" + "="*60)
    print("EXPERIMENT 3: Format Compliance")
    print("="*60)

    results = {}

    for format_test in FORMAT_TESTS:
        print(f"\n[Testing] {format_test['name']}")
        test_results = []

        system_prompt = f"You are a helpful assistant.\nIMPORTANT: {format_test['instruction']}"

        for query in TEST_QUERIES:
            outputs = runner.generate(
                system_prompt=system_prompt,
                user_content=query["content"],
                tools=query["tools"],
                temperature=0.7,
                num_samples=1,  # Just 1 sample per format test
            )

            for output in outputs:
                complies = format_test["check"](output)
                test_results.append({
                    "query": query["content"],
                    "output": output[:300],
                    "complies": complies,
                })

        compliance_rate = sum(1 for r in test_results if r["complies"]) / len(test_results)

        results[format_test["name"]] = {
            "compliance_rate": compliance_rate,
            "samples": test_results,
        }

        print(f"  Compliance rate: {compliance_rate:.0%}")

    return results


# =============================================================================
# Experiment 4: Output Variance
# =============================================================================

def run_experiment_4(runner: ModelRunner) -> Dict:
    """Experiment 4: Test output variance across temperatures."""
    print("\n" + "="*60)
    print("EXPERIMENT 4: Output Variance")
    print("="*60)

    results = {}
    temperatures = [0.3, 0.5, 0.7, 1.0]
    samples_per_temp = 10

    query = TEST_QUERIES[0]  # Use first query for variance test
    system_prompt = PROMPT_VARIANTS["explicit_think"]

    for temp in temperatures:
        print(f"\n[Testing] temperature={temp}")

        outputs = runner.generate(
            system_prompt=system_prompt,
            user_content=query["content"],
            tools=query["tools"],
            temperature=temp,
            num_samples=samples_per_temp,
        )

        # Calculate variance metrics
        unique_outputs = len(set(outputs))
        has_think_count = sum(1 for o in outputs if THINK_TAG_OPEN in o)

        # Calculate average pairwise similarity (simplified: just check if identical)
        identical_pairs = 0
        total_pairs = 0
        for i in range(len(outputs)):
            for j in range(i+1, len(outputs)):
                total_pairs += 1
                if outputs[i] == outputs[j]:
                    identical_pairs += 1

        diversity_score = 1 - (identical_pairs / total_pairs) if total_pairs > 0 else 0

        results[f"temp_{temp}"] = {
            "unique_outputs": unique_outputs,
            "total_outputs": samples_per_temp,
            "diversity_score": diversity_score,
            "think_rate": has_think_count / samples_per_temp,
            "sample_outputs": [o[:200] for o in outputs[:3]],  # Store first 3 samples
        }

        print(f"  Unique outputs: {unique_outputs}/{samples_per_temp}")
        print(f"  Diversity score: {diversity_score:.2f}")
        print(f"  Think rate: {has_think_count/samples_per_temp:.0%}")

    return results


# =============================================================================
# Analysis and Conclusion
# =============================================================================

def analyze_results(exp1: Dict, exp2: Dict, exp3: Dict, exp4: Dict) -> Dict:
    """Analyze all experimental results and draw conclusions."""

    analysis = {
        "hypothesis_results": {},
        "key_metrics": {},
        "conclusion": "",
        "fd_viability": "",
    }

    # H1: Can model produce <think> tags?
    max_think_rate = max(r["think_rate"] for r in exp1.values())
    any_think_in_fewshot = max(r["think_rate"] for r in exp2.values())
    h1_result = max_think_rate > 0 or any_think_in_fewshot > 0

    analysis["hypothesis_results"]["H1_can_produce_think"] = {
        "result": h1_result,
        "max_think_rate_prompts": max_think_rate,
        "max_think_rate_fewshot": any_think_in_fewshot,
        "conclusion": "Model CAN produce <think> tags" if h1_result else "Model CANNOT produce <think> tags"
    }

    # H2: Is model sensitive to prompts?
    lengths = [r["avg_output_length"] for r in exp1.values()]
    length_variance = max(lengths) - min(lengths)
    reasoning_rates = [r["reasoning_rate"] for r in exp1.values()]
    reasoning_variance = max(reasoning_rates) - min(reasoning_rates)

    h2_result = length_variance > 50 or reasoning_variance > 0.1

    analysis["hypothesis_results"]["H2_prompt_sensitivity"] = {
        "result": h2_result,
        "output_length_variance": length_variance,
        "reasoning_rate_variance": reasoning_variance,
        "conclusion": "Model IS sensitive to prompts" if h2_result else "Model is NOT sensitive to prompts"
    }

    # H3: Does few-shot help?
    shot_0_think = exp2["0_shot"]["think_rate"]
    shot_3_think = exp2["3_shot"]["think_rate"]
    fewshot_improvement = shot_3_think - shot_0_think

    h3_result = fewshot_improvement > 0.1

    analysis["hypothesis_results"]["H3_fewshot_effect"] = {
        "result": h3_result,
        "0_shot_think_rate": shot_0_think,
        "3_shot_think_rate": shot_3_think,
        "improvement": fewshot_improvement,
        "conclusion": "Few-shot HELPS" if h3_result else "Few-shot does NOT help"
    }

    # H4: Can model follow format instructions?
    format_rates = [r["compliance_rate"] for r in exp3.values()]
    avg_format_compliance = sum(format_rates) / len(format_rates)

    h4_result = avg_format_compliance > 0.3

    analysis["hypothesis_results"]["H4_format_compliance"] = {
        "result": h4_result,
        "avg_compliance_rate": avg_format_compliance,
        "per_format": {name: r["compliance_rate"] for name, r in exp3.items()},
        "conclusion": "Model CAN follow some format instructions" if h4_result else "Model CANNOT follow format instructions"
    }

    # H5: Does output vary?
    diversities = [r["diversity_score"] for r in exp4.values()]
    avg_diversity = sum(diversities) / len(diversities)

    h5_result = avg_diversity > 0.3

    analysis["hypothesis_results"]["H5_output_variance"] = {
        "result": h5_result,
        "avg_diversity_score": avg_diversity,
        "per_temperature": {name: r["diversity_score"] for name, r in exp4.items()},
        "conclusion": "Output VARIES across samples" if h5_result else "Output is FIXED"
    }

    # Overall conclusion
    analysis["key_metrics"] = {
        "max_think_rate_achieved": max(max_think_rate, any_think_in_fewshot),
        "prompt_sensitivity": h2_result,
        "fewshot_effective": h3_result,
        "format_compliance": avg_format_compliance,
        "output_diversity": avg_diversity,
    }

    # Determine FD viability based on logic tree
    if h1_result:
        analysis["fd_viability"] = "⭐⭐⭐+ (VIABLE - Model can produce thinking)"
        analysis["conclusion"] = """
FD 可能有效！模型在某些條件下能產生 <think> 標籤。
建議：進一步優化能產生 thinking 的 prompt 策略。
"""
    elif h2_result:
        analysis["fd_viability"] = "⭐⭐ (PARTIALLY VIABLE - Model responds to prompts)"
        analysis["conclusion"] = """
FD 部分可行。模型對 prompt 敏感，但尚未找到能產生 thinking 的策略。
建議：嘗試更多 prompt 變體或不同的推理標記格式。
"""
    elif h4_result:
        analysis["fd_viability"] = "⭐⭐ (PARTIALLY VIABLE - Try different format)"
        analysis["conclusion"] = """
FD 部分可行。模型能遵循某些格式指令，但不是 <think> 標籤。
建議：使用其他推理標記（如 REASONING:）替代 <think>。
"""
    elif h5_result:
        analysis["fd_viability"] = "⭐ (LOW VIABILITY - Output varies but uncontrollable)"
        analysis["conclusion"] = """
FD 可行性低。模型輸出有變異但不受 prompt 控制。
建議：考慮使用 fine-tuning 方案。
"""
    else:
        analysis["fd_viability"] = "⭐ (NOT VIABLE - Output is fixed)"
        analysis["conclusion"] = """
FD 不可行。模型輸出固定，不受 prompt 影響。
建議：必須使用 fine-tuning 來改變模型行為。
"""

    return analysis


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*60)
    print("FD Deep Analysis Experiment")
    print("="*60)
    print(f"Started at: {datetime.now()}")

    # Initialize
    runner = ModelRunner()

    # Run experiments
    exp1_results = run_experiment_1(runner)
    exp2_results = run_experiment_2(runner)
    exp3_results = run_experiment_3(runner)
    exp4_results = run_experiment_4(runner)

    # Analyze
    print("\n" + "="*60)
    print("ANALYSIS AND CONCLUSIONS")
    print("="*60)

    analysis = analyze_results(exp1_results, exp2_results, exp3_results, exp4_results)

    # Print hypothesis results
    print("\n--- Hypothesis Testing Results ---")
    for h_name, h_result in analysis["hypothesis_results"].items():
        status = "✓" if h_result["result"] else "✗"
        print(f"\n{status} {h_name}: {h_result['conclusion']}")

    print(f"\n--- FD Viability Assessment ---")
    print(f"Rating: {analysis['fd_viability']}")
    print(f"\nConclusion:{analysis['conclusion']}")

    # Save full results
    full_results = {
        "timestamp": datetime.now().isoformat(),
        "experiment_1_prompt_variants": exp1_results,
        "experiment_2_fewshot": exp2_results,
        "experiment_3_format": exp3_results,
        "experiment_4_variance": exp4_results,
        "analysis": analysis,
    }

    output_file = "fd_deep_analysis_results.json"
    with open(output_file, "w") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nFull results saved to: {output_file}")
    print(f"\nCompleted at: {datetime.now()}")


if __name__ == "__main__":
    main()
