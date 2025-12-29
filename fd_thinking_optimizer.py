#!/usr/bin/env python3
"""
Feedback Descent (FD) Optimizer for FunctionGemma Thinking Capability

This script implements the Feedback Descent algorithm to optimize prompts
for inducing thinking behavior in FunctionGemma-270M without fine-tuning.

Paper: https://arxiv.org/abs/2511.07919
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import torch

# =============================================================================
# Configuration
# =============================================================================

# Disable torch.compile for RTX 5090 compatibility
torch._dynamo.config.disable = True

THINK_TAG_OPEN = "<think>"
THINK_TAG_CLOSE = "</think>"

MAX_ITERATIONS = 15
EARLY_STOP_K = 5
MAX_NEW_TOKENS = 512

# OpenAI API settings (for GPT-4o evaluator)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class FewShotExample:
    """A single few-shot example demonstrating thinking + tool call."""
    user_content: str
    assistant_thinking: str
    tool_call: Dict  # {"name": str, "arguments": Dict}


@dataclass
class ThinkingPrompt:
    """The prompt artifact to be optimized by Feedback Descent."""
    system_prompt: str
    few_shot_examples: List[FewShotExample] = field(default_factory=list)

    def to_messages(self, user_query: str) -> List[Dict]:
        """Assemble the full messages list for the model."""
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add few-shot examples
        for ex in self.few_shot_examples:
            messages.append({"role": "user", "content": ex.user_content})
            messages.append({
                "role": "assistant",
                "content": f"{THINK_TAG_OPEN}\n{ex.assistant_thinking}\n{THINK_TAG_CLOSE}",
                "tool_calls": [{
                    "id": "example_call",
                    "type": "function",
                    "function": ex.tool_call
                }]
            })

        # Add the actual user query
        messages.append({"role": "user", "content": user_query})
        return messages

    def to_json(self) -> str:
        """Serialize to JSON for storage and prompt generation."""
        return json.dumps({
            "system_prompt": self.system_prompt,
            "few_shot_examples": [
                {
                    "user_content": ex.user_content,
                    "assistant_thinking": ex.assistant_thinking,
                    "tool_call": ex.tool_call
                }
                for ex in self.few_shot_examples
            ]
        }, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "ThinkingPrompt":
        """Deserialize from JSON."""
        data = json.loads(json_str)
        return cls(
            system_prompt=data["system_prompt"],
            few_shot_examples=[
                FewShotExample(
                    user_content=ex["user_content"],
                    assistant_thinking=ex["assistant_thinking"],
                    tool_call=ex["tool_call"]
                )
                for ex in data.get("few_shot_examples", [])
            ]
        )


@dataclass
class TestQuery:
    """A test query with tools for evaluation."""
    user_content: str
    tools: List[Dict]
    expected_tool_name: str  # The expected tool to be called


@dataclass
class Feedback:
    """Feedback from pairwise comparison."""
    preference: str  # "A" or "B"
    rationale: str
    improvement_suggestions: str


@dataclass
class EvaluationResult:
    """Result of evaluating model output."""
    output: str
    has_thinking: bool
    thinking_content: Optional[str]
    has_tool_call: bool
    tool_call_name: Optional[str]


# =============================================================================
# Test Queries (5 diverse queries for validation)
# =============================================================================

TEST_QUERIES = [
    TestQuery(
        user_content="What's the current weather in Tokyo?",
        tools=[{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "The city name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["city"]
                }
            }
        }],
        expected_tool_name="get_weather"
    ),
    TestQuery(
        user_content="Search for recent news about AI regulations in Europe",
        tools=[{
            "type": "function",
            "function": {
                "name": "search_news",
                "description": "Search for news articles on a topic",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "region": {"type": "string", "description": "Geographic region"}
                    },
                    "required": ["query"]
                }
            }
        }],
        expected_tool_name="search_news"
    ),
    TestQuery(
        user_content="Calculate 15% tip on a $85.50 restaurant bill",
        tools=[{
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Math expression to evaluate"}
                    },
                    "required": ["expression"]
                }
            }
        }],
        expected_tool_name="calculate"
    ),
    TestQuery(
        user_content="Find restaurants near Times Square that are open now",
        tools=[{
            "type": "function",
            "function": {
                "name": "search_places",
                "description": "Search for places near a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to search for"},
                        "location": {"type": "string", "description": "Location to search near"},
                        "open_now": {"type": "boolean", "description": "Only show open places"}
                    },
                    "required": ["query", "location"]
                }
            }
        }],
        expected_tool_name="search_places"
    ),
    TestQuery(
        user_content="Translate 'Hello, how are you?' to Japanese",
        tools=[{
            "type": "function",
            "function": {
                "name": "translate",
                "description": "Translate text between languages",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to translate"},
                        "target_language": {"type": "string", "description": "Target language code"}
                    },
                    "required": ["text", "target_language"]
                }
            }
        }],
        expected_tool_name="translate"
    ),
]


# =============================================================================
# Initial Prompt Design
# =============================================================================

INITIAL_PROMPT = ThinkingPrompt(
    system_prompt="""You are a helpful assistant with access to tools.
Before calling any tool, you MUST first think through the problem inside <think>...</think> tags.
In your thinking, consider:
- What information does the user need?
- Which tool can best provide this information?
- What parameters should be used?
After thinking, make the appropriate tool call.""",
    few_shot_examples=[
        FewShotExample(
            user_content="What time is it in London right now?",
            assistant_thinking="""The user wants to know the current time in London.
I have access to the get_time tool which can retrieve the current time for a specific timezone.
London is in the Europe/London timezone (GMT/BST).
I should call get_time with timezone="Europe/London".""",
            tool_call={
                "name": "get_time",
                "arguments": {"timezone": "Europe/London"}
            }
        )
    ]
)


# =============================================================================
# Model Interface
# =============================================================================

class FunctionGemmaRunner:
    """Interface for running FunctionGemma inference."""

    def __init__(self, model_name: str = "unsloth/functiongemma-270m-it"):
        print(f"Loading FunctionGemma from {model_name}...")
        from unsloth import FastLanguageModel

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            load_in_4bit=False,
            load_in_8bit=False,
            load_in_16bit=True,
        )
        FastLanguageModel.for_inference(self.model)
        print("Model loaded successfully!")

    def generate(self, prompt: ThinkingPrompt, query: TestQuery) -> EvaluationResult:
        """Generate output for a query using the given prompt."""
        messages = prompt.to_messages(query.user_content)

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tools=query.tools,
            tokenize=False,
            add_generation_prompt=True,
        )
        if text.startswith("<bos>"):
            text = text[5:]

        # Generate
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                top_p=0.95,
                top_k=64,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode output
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        output_text = self.tokenizer.decode(generated, skip_special_tokens=False)

        # Parse output
        return self._parse_output(output_text)

    def _parse_output(self, output: str) -> EvaluationResult:
        """Parse the model output to extract thinking and tool calls."""
        has_thinking = THINK_TAG_OPEN in output and THINK_TAG_CLOSE in output
        thinking_content = None

        if has_thinking:
            start = output.find(THINK_TAG_OPEN) + len(THINK_TAG_OPEN)
            end = output.find(THINK_TAG_CLOSE)
            thinking_content = output[start:end].strip()

        # Check for tool call markers
        has_tool_call = "<start_function_call>" in output or "call:" in output
        tool_call_name = None

        if has_tool_call:
            # Try to extract tool name
            if "call:" in output:
                try:
                    start = output.find("call:") + 5
                    end = output.find("{", start)
                    if end > start:
                        tool_call_name = output[start:end].strip()
                except:
                    pass

        return EvaluationResult(
            output=output,
            has_thinking=has_thinking,
            thinking_content=thinking_content,
            has_tool_call=has_tool_call,
            tool_call_name=tool_call_name
        )


# =============================================================================
# GPT-4o Evaluator
# =============================================================================

class GPT4oEvaluator:
    """Evaluator using GPT-4o for pairwise comparison."""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")

        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

    def compare(
        self,
        outputs_a: List[EvaluationResult],
        outputs_b: List[EvaluationResult],
        queries: List[TestQuery]
    ) -> Tuple[str, Feedback]:
        """Compare two sets of outputs and return preference and feedback."""

        # Build comparison prompt
        comparison_text = self._build_comparison_prompt(outputs_a, outputs_b, queries)

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": """You are an expert evaluator for language model outputs.
You will compare two sets of outputs (A and B) from a small language model (FunctionGemma-270M).
The goal is to determine which set better demonstrates "thinking before tool calling" behavior.

Evaluation criteria:
1. Thinking presence: Does the output contain <think>...</think> tags BEFORE the tool call?
2. Thinking quality: Is the reasoning relevant, logical, and helpful?
3. Tool call correctness: Is the right tool called with appropriate parameters?

You MUST respond in JSON format:
{
    "preference": "A" or "B",
    "rationale": "Detailed explanation of why the preferred set is better",
    "improvement_suggestions": "Specific suggestions for improving the prompt to get better outputs"
}"""},
                {"role": "user", "content": comparison_text}
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )

        result = json.loads(response.choices[0].message.content)

        return result["preference"], Feedback(
            preference=result["preference"],
            rationale=result["rationale"],
            improvement_suggestions=result["improvement_suggestions"]
        )

    def _build_comparison_prompt(
        self,
        outputs_a: List[EvaluationResult],
        outputs_b: List[EvaluationResult],
        queries: List[TestQuery]
    ) -> str:
        """Build the comparison prompt text."""
        lines = ["Compare the following two sets of outputs:\n"]

        for i, (query, out_a, out_b) in enumerate(zip(queries, outputs_a, outputs_b)):
            lines.append(f"=== Query {i+1}: {query.user_content} ===")
            lines.append(f"Expected tool: {query.expected_tool_name}")
            lines.append(f"\n--- Output A ---")
            lines.append(f"Has thinking: {out_a.has_thinking}")
            if out_a.thinking_content:
                lines.append(f"Thinking: {out_a.thinking_content[:200]}...")
            lines.append(f"Has tool call: {out_a.has_tool_call}")
            lines.append(f"Tool called: {out_a.tool_call_name}")

            lines.append(f"\n--- Output B ---")
            lines.append(f"Has thinking: {out_b.has_thinking}")
            if out_b.thinking_content:
                lines.append(f"Thinking: {out_b.thinking_content[:200]}...")
            lines.append(f"Has tool call: {out_b.has_tool_call}")
            lines.append(f"Tool called: {out_b.tool_call_name}")
            lines.append("")

        # Add summary statistics
        a_thinking_rate = sum(1 for o in outputs_a if o.has_thinking) / len(outputs_a)
        b_thinking_rate = sum(1 for o in outputs_b if o.has_thinking) / len(outputs_b)

        lines.append(f"\n=== Summary ===")
        lines.append(f"Set A thinking rate: {a_thinking_rate:.0%}")
        lines.append(f"Set B thinking rate: {b_thinking_rate:.0%}")

        return "\n".join(lines)

    def generate_improved_prompt(
        self,
        current_prompt: ThinkingPrompt,
        feedback_history: List[Feedback]
    ) -> ThinkingPrompt:
        """Use GPT-4o to generate an improved prompt based on feedback."""

        history_text = "\n\n".join([
            f"Feedback {i+1}:\n- Rationale: {f.rationale}\n- Suggestions: {f.improvement_suggestions}"
            for i, f in enumerate(feedback_history)
        ])

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": """You are an expert prompt engineer.
Your task is to improve a prompt template that guides a small language model (FunctionGemma-270M)
to think before calling tools.

The model should:
1. Output <think>...</think> tags with reasoning BEFORE making tool calls
2. Have relevant, logical thinking content
3. Make correct tool calls after thinking

Important constraints:
- The model is only 270M parameters, so keep instructions VERY CLEAR and SIMPLE
- Few-shot examples are crucial for small models
- Be EXPLICIT about the expected format

You MUST respond with a valid JSON object containing the improved prompt."""},
                {"role": "user", "content": f"""Current prompt:
{current_prompt.to_json()}

Accumulated feedback from previous iterations:
{history_text if history_text else "No feedback yet - this is the first improvement iteration."}

Based on this feedback, generate an improved prompt. The response must be valid JSON in this format:
{{
    "system_prompt": "Your improved system prompt here",
    "few_shot_examples": [
        {{
            "user_content": "Example user query",
            "assistant_thinking": "Example thinking content",
            "tool_call": {{"name": "tool_name", "arguments": {{"key": "value"}}}}
        }}
    ]
}}

You may add 1-2 few-shot examples if helpful. Keep the system prompt concise but explicit."""}
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )

        try:
            result = json.loads(response.choices[0].message.content)
            return ThinkingPrompt.from_json(json.dumps(result))
        except Exception as e:
            print(f"Error parsing improved prompt: {e}")
            return current_prompt


# =============================================================================
# Feedback Descent Main Loop
# =============================================================================

class FeedbackDescentOptimizer:
    """Main Feedback Descent optimization loop."""

    def __init__(
        self,
        model_runner: FunctionGemmaRunner,
        evaluator: GPT4oEvaluator,
        test_queries: List[TestQuery],
        max_iterations: int = MAX_ITERATIONS,
        early_stop_k: int = EARLY_STOP_K,
    ):
        self.model = model_runner
        self.evaluator = evaluator
        self.queries = test_queries
        self.max_iterations = max_iterations
        self.early_stop_k = early_stop_k

        # Logging
        self.log_dir = "fd_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(
            self.log_dir,
            f"fd_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        self.history = []

    def run_evaluation(self, prompt: ThinkingPrompt) -> List[EvaluationResult]:
        """Run the model on all test queries with the given prompt."""
        results = []
        for query in self.queries:
            result = self.model.generate(prompt, query)
            results.append(result)
        return results

    def compute_metrics(self, results: List[EvaluationResult]) -> Dict:
        """Compute evaluation metrics from results."""
        thinking_rate = sum(1 for r in results if r.has_thinking) / len(results)
        tool_call_rate = sum(1 for r in results if r.has_tool_call) / len(results)
        correct_tool_rate = sum(
            1 for r, q in zip(results, self.queries)
            if r.tool_call_name == q.expected_tool_name
        ) / len(results)

        return {
            "thinking_rate": thinking_rate,
            "tool_call_rate": tool_call_rate,
            "correct_tool_rate": correct_tool_rate,
        }

    def optimize(self, initial_prompt: ThinkingPrompt) -> ThinkingPrompt:
        """Run the Feedback Descent optimization loop."""

        print("\n" + "="*60)
        print("Starting Feedback Descent Optimization")
        print("="*60)

        best_prompt = initial_prompt
        feedback_history: List[Feedback] = []
        no_improvement_count = 0

        # Baseline evaluation
        print("\n[Iteration 0] Baseline evaluation...")
        best_outputs = self.run_evaluation(best_prompt)
        best_metrics = self.compute_metrics(best_outputs)

        self._log_iteration(0, best_prompt, best_outputs, best_metrics, None)
        print(f"  Thinking rate: {best_metrics['thinking_rate']:.0%}")
        print(f"  Tool call rate: {best_metrics['tool_call_rate']:.0%}")
        print(f"  Correct tool rate: {best_metrics['correct_tool_rate']:.0%}")

        for iteration in range(1, self.max_iterations + 1):
            print(f"\n[Iteration {iteration}/{self.max_iterations}]")

            # 1. Propose: Generate improved prompt
            print("  Generating improved prompt...")
            candidate_prompt = self.evaluator.generate_improved_prompt(
                best_prompt, feedback_history
            )

            # 2. Evaluate: Run on test queries
            print("  Evaluating candidate prompt...")
            candidate_outputs = self.run_evaluation(candidate_prompt)
            candidate_metrics = self.compute_metrics(candidate_outputs)

            print(f"  Candidate thinking rate: {candidate_metrics['thinking_rate']:.0%}")

            # 3. Compare: Get preference and feedback
            print("  Comparing with current best...")
            preference, feedback = self.evaluator.compare(
                best_outputs, candidate_outputs, self.queries
            )

            print(f"  Preference: {'Candidate (B)' if preference == 'B' else 'Current best (A)'}")
            print(f"  Rationale: {feedback.rationale[:100]}...")

            # 4. Update
            if preference == "B":
                print("  -> Accepting candidate as new best!")
                best_prompt = candidate_prompt
                best_outputs = candidate_outputs
                best_metrics = candidate_metrics
                feedback_history = []  # Reset
                no_improvement_count = 0
            else:
                print("  -> Keeping current best, accumulating feedback.")
                feedback_history.append(feedback)
                no_improvement_count += 1

            self._log_iteration(iteration, candidate_prompt, candidate_outputs,
                              candidate_metrics, feedback)

            # Early stopping
            if no_improvement_count >= self.early_stop_k:
                print(f"\n[Early Stop] No improvement for {self.early_stop_k} iterations.")
                break

        # Final summary
        print("\n" + "="*60)
        print("Optimization Complete")
        print("="*60)
        print(f"Final thinking rate: {best_metrics['thinking_rate']:.0%}")
        print(f"Final tool call rate: {best_metrics['tool_call_rate']:.0%}")
        print(f"Final correct tool rate: {best_metrics['correct_tool_rate']:.0%}")
        print(f"\nOptimized prompt saved to: {self.log_file}")

        return best_prompt

    def _log_iteration(
        self,
        iteration: int,
        prompt: ThinkingPrompt,
        outputs: List[EvaluationResult],
        metrics: Dict,
        feedback: Optional[Feedback]
    ):
        """Log iteration results."""
        entry = {
            "iteration": iteration,
            "prompt": json.loads(prompt.to_json()),
            "metrics": metrics,
            "outputs": [
                {
                    "query": self.queries[i].user_content,
                    "has_thinking": o.has_thinking,
                    "thinking_content": o.thinking_content,
                    "has_tool_call": o.has_tool_call,
                    "tool_call_name": o.tool_call_name,
                }
                for i, o in enumerate(outputs)
            ],
            "feedback": asdict(feedback) if feedback else None,
        }
        self.history.append(entry)

        # Save to file
        with open(self.log_file, "w") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for the FD optimization experiment."""

    print("="*60)
    print("Feedback Descent for FunctionGemma Thinking Capability")
    print("="*60)
    print(f"Max iterations: {MAX_ITERATIONS}")
    print(f"Early stop after: {EARLY_STOP_K} iterations without improvement")
    print(f"Test queries: {len(TEST_QUERIES)}")

    # Check API key
    if not OPENAI_API_KEY:
        print("\nError: OPENAI_API_KEY environment variable not set!")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        return

    # Initialize components
    print("\n[1/3] Loading FunctionGemma model...")
    model_runner = FunctionGemmaRunner()

    print("\n[2/3] Initializing GPT-4o evaluator...")
    evaluator = GPT4oEvaluator(OPENAI_API_KEY)

    print("\n[3/3] Setting up optimizer...")
    optimizer = FeedbackDescentOptimizer(
        model_runner=model_runner,
        evaluator=evaluator,
        test_queries=TEST_QUERIES,
    )

    # Run optimization
    optimized_prompt = optimizer.optimize(INITIAL_PROMPT)

    # Save final prompt
    final_prompt_file = "fd_optimized_prompt.json"
    with open(final_prompt_file, "w") as f:
        f.write(optimized_prompt.to_json())
    print(f"\nFinal optimized prompt saved to: {final_prompt_file}")


if __name__ == "__main__":
    main()
