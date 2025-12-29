#!/usr/bin/env python3
"""
Ollama Feedback Descent (FD) Optimizer

使用 Feedback Descent 算法優化 prompt，讓 mistral:7b 在 tool-calling 前
產生 <think>...</think> 推理標籤。

評估器使用 gpt-oss:20b（本地模型），完全不需要外部 API。

Paper: https://arxiv.org/abs/2511.07919
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import requests

# =============================================================================
# 配置
# =============================================================================

# 模型配置
TARGET_MODEL = "mistral:7b"        # 被優化的模型
EVALUATOR_MODEL = "gpt-oss:20b"    # 評估器模型
OLLAMA_BASE_URL = "http://localhost:11434"

# FD 超參數
MAX_ITERATIONS = 15      # 最大優化輪數
EARLY_STOP_K = 5         # 連續 K 輪無改進則停止
SAMPLES_PER_PROMPT = 3   # 每個 prompt 採樣次數
TEMPERATURE = 0.7        # 生成溫度
MIN_TOOL_ACCURACY = 0.4  # 最低 tool accuracy 門檻（約束條件）

# 標籤
THINK_TAG_OPEN = "<think>"
THINK_TAG_CLOSE = "</think>"

# 輸出目錄
OUTPUT_DIR = "ollama_fd_results"


# =============================================================================
# 數據結構
# =============================================================================

@dataclass
class FewShotExample:
    """Few-shot 範例"""
    user_content: str
    assistant_thinking: str
    tool_name: str
    tool_arguments: Dict


@dataclass
class ThinkingPrompt:
    """要優化的 prompt artifact"""
    system_prompt: str
    few_shot_examples: List[FewShotExample] = field(default_factory=list)

    def to_messages(self, user_query: str, tools: List[Dict]) -> List[Dict]:
        """組裝完整的 messages 列表"""
        messages = [{"role": "system", "content": self.system_prompt}]

        # 添加 few-shot 範例
        for ex in self.few_shot_examples:
            messages.append({"role": "user", "content": ex.user_content})

            # 組裝 assistant 回應（thinking + tool call 說明）
            thinking_block = f"{THINK_TAG_OPEN}\n{ex.assistant_thinking}\n{THINK_TAG_CLOSE}"
            tool_info = f"\n\nI'll call {ex.tool_name} with: {json.dumps(ex.tool_arguments)}"
            messages.append({
                "role": "assistant",
                "content": thinking_block + tool_info
            })

        # 添加實際用戶查詢
        messages.append({"role": "user", "content": user_query})
        return messages

    def to_json(self) -> str:
        """序列化為 JSON"""
        return json.dumps({
            "system_prompt": self.system_prompt,
            "few_shot_examples": [
                {
                    "user_content": ex.user_content,
                    "assistant_thinking": ex.assistant_thinking,
                    "tool_name": ex.tool_name,
                    "tool_arguments": ex.tool_arguments
                }
                for ex in self.few_shot_examples
            ]
        }, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "ThinkingPrompt":
        """從 JSON 反序列化"""
        data = json.loads(json_str)
        return cls(
            system_prompt=data["system_prompt"],
            few_shot_examples=[
                FewShotExample(
                    user_content=ex["user_content"],
                    assistant_thinking=ex["assistant_thinking"],
                    tool_name=ex["tool_name"],
                    tool_arguments=ex["tool_arguments"]
                )
                for ex in data.get("few_shot_examples", [])
            ]
        )


@dataclass
class TestQuery:
    """測試查詢"""
    user_content: str
    tools: List[Dict]
    expected_tool_name: str


@dataclass
class EvaluationResult:
    """模型輸出評估結果"""
    output: str
    has_thinking: bool
    thinking_content: Optional[str]
    has_tool_call: bool
    tool_call_name: Optional[str]
    tool_call_correct: bool


@dataclass
class Feedback:
    """Pairwise comparison 的反饋"""
    preference: str  # "A" or "B"
    rationale: str
    improvement_suggestions: str


@dataclass
class OptimizationStep:
    """優化步驟記錄"""
    iteration: int
    prompt: ThinkingPrompt
    thinking_rate: float
    tool_accuracy: float
    score: float
    feedback: Optional[Feedback] = None


# =============================================================================
# 測試查詢集
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
                        "expression": {"type": "string", "description": "Math expression"}
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
                        "location": {"type": "string", "description": "Location"},
                        "open_now": {"type": "boolean", "description": "Only open places"}
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
                        "target_language": {"type": "string", "description": "Target language"}
                    },
                    "required": ["text", "target_language"]
                }
            }
        }],
        expected_tool_name="translate"
    ),
]


# =============================================================================
# 初始 Prompt 設計
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
            tool_name="get_time",
            tool_arguments={"timezone": "Europe/London"}
        )
    ]
)


# =============================================================================
# Ollama Runner
# =============================================================================

class OllamaRunner:
    """Ollama 模型推理介面"""

    def __init__(self, model: str = TARGET_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url

    def generate(
        self,
        prompt: ThinkingPrompt,
        query: TestQuery,
        temperature: float = TEMPERATURE
    ) -> EvaluationResult:
        """使用給定 prompt 生成輸出並評估"""
        messages = prompt.to_messages(query.user_content, query.tools)

        payload = {
            "model": self.model,
            "messages": messages,
            "tools": query.tools,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120
            )

            if response.status_code == 200:
                data = response.json()
                message = data.get("message", {})

                # 組合輸出
                content = message.get("content", "")
                tool_calls = message.get("tool_calls", [])

                output_parts = []
                if content:
                    output_parts.append(content)
                if tool_calls:
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        output_parts.append(
                            f"[TOOL_CALL: {func.get('name', 'unknown')}({func.get('arguments', {})})]"
                        )

                output = "\n".join(output_parts) if output_parts else "[NO OUTPUT]"
                return self._parse_output(output, query.expected_tool_name)
            else:
                return EvaluationResult(
                    output=f"[ERROR: {response.status_code}]",
                    has_thinking=False,
                    thinking_content=None,
                    has_tool_call=False,
                    tool_call_name=None,
                    tool_call_correct=False
                )

        except Exception as e:
            return EvaluationResult(
                output=f"[ERROR: {str(e)}]",
                has_thinking=False,
                thinking_content=None,
                has_tool_call=False,
                tool_call_name=None,
                tool_call_correct=False
            )

    def _parse_output(self, output: str, expected_tool: str) -> EvaluationResult:
        """解析模型輸出"""
        # 檢查 thinking 標籤
        has_thinking = THINK_TAG_OPEN.lower() in output.lower() and THINK_TAG_CLOSE.lower() in output.lower()
        thinking_content = None

        if has_thinking:
            try:
                start = output.lower().find(THINK_TAG_OPEN.lower()) + len(THINK_TAG_OPEN)
                end = output.lower().find(THINK_TAG_CLOSE.lower())
                thinking_content = output[start:end].strip()
            except:
                pass

        # 檢查工具調用（多種格式）
        has_tool_call = False
        tool_call_name = None

        # 格式 1: 結構化 API 回應 [TOOL_CALL: name(...)]
        if "[TOOL_CALL:" in output:
            has_tool_call = True
            try:
                start = output.find("[TOOL_CALL:") + len("[TOOL_CALL:")
                end = output.find("(", start)
                if end > start:
                    tool_call_name = output[start:end].strip()
            except:
                pass

        # 格式 2: JSON 文本格式 [{"name":"tool_name",...}] 或 {"name":"tool_name",...}
        if not tool_call_name:
            import re
            # 匹配 JSON 中的 "name": "xxx" 模式
            json_pattern = r'"name"\s*:\s*"([^"]+)"'
            match = re.search(json_pattern, output)
            if match:
                has_tool_call = True
                tool_call_name = match.group(1)

        # 格式 3: 函數調用文本格式 call:function_name{...}
        if not tool_call_name and "call:" in output:
            has_tool_call = True
            try:
                start = output.find("call:") + 5
                end = output.find("{", start)
                if end > start:
                    tool_call_name = output[start:end].strip()
            except:
                pass

        tool_call_correct = tool_call_name == expected_tool if tool_call_name else False

        return EvaluationResult(
            output=output,
            has_thinking=has_thinking,
            thinking_content=thinking_content,
            has_tool_call=has_tool_call,
            tool_call_name=tool_call_name,
            tool_call_correct=tool_call_correct
        )


# =============================================================================
# 本地 LLM 評估器
# =============================================================================

class LocalLLMEvaluator:
    """使用本地 Ollama 模型進行 pairwise comparison"""

    def __init__(self, model: str = EVALUATOR_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url

    def compare(
        self,
        outputs_a: List[EvaluationResult],
        outputs_b: List[EvaluationResult],
        queries: List[TestQuery]
    ) -> Tuple[str, Feedback]:
        """比較兩組輸出並返回偏好和反饋"""

        comparison_prompt = self._build_comparison_prompt(outputs_a, outputs_b, queries)

        messages = [
            {"role": "system", "content": """You are an expert evaluator for language model outputs.
Compare two sets of outputs (A and B) and determine which better demonstrates "thinking before tool calling".

Evaluation criteria:
1. Thinking presence: Does the output contain <think>...</think> tags BEFORE tool calls?
2. Thinking quality: Is the reasoning relevant, logical, and helpful?
3. Tool call correctness: Is the right tool called?

You MUST respond in this exact JSON format:
{
    "preference": "A" or "B",
    "rationale": "Why the preferred set is better",
    "improvement_suggestions": "How to improve the prompt for better outputs"
}"""},
            {"role": "user", "content": comparison_prompt}
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.3,
            }
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=180
            )

            if response.status_code == 200:
                data = response.json()
                content = data.get("message", {}).get("content", "{}")

                try:
                    result = json.loads(content)
                    preference = result.get("preference", "A")
                    return preference, Feedback(
                        preference=preference,
                        rationale=result.get("rationale", ""),
                        improvement_suggestions=result.get("improvement_suggestions", "")
                    )
                except json.JSONDecodeError:
                    # 如果 JSON 解析失敗，嘗試從文本中提取
                    preference = "A" if "prefer A" in content.lower() or "set A" in content.lower() else "B"
                    return preference, Feedback(
                        preference=preference,
                        rationale=content[:500],
                        improvement_suggestions=""
                    )
            else:
                return "A", Feedback(preference="A", rationale="Evaluator error", improvement_suggestions="")

        except Exception as e:
            print(f"  [Evaluator Error] {e}")
            return "A", Feedback(preference="A", rationale=str(e), improvement_suggestions="")

    def _build_comparison_prompt(
        self,
        outputs_a: List[EvaluationResult],
        outputs_b: List[EvaluationResult],
        queries: List[TestQuery]
    ) -> str:
        """構建比較 prompt"""
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
            lines.append(f"Correct tool: {out_a.tool_call_correct}")

            lines.append(f"\n--- Output B ---")
            lines.append(f"Has thinking: {out_b.has_thinking}")
            if out_b.thinking_content:
                lines.append(f"Thinking: {out_b.thinking_content[:200]}...")
            lines.append(f"Has tool call: {out_b.has_tool_call}")
            lines.append(f"Tool called: {out_b.tool_call_name}")
            lines.append(f"Correct tool: {out_b.tool_call_correct}")
            lines.append("")

        # 添加統計摘要
        a_thinking_rate = sum(1 for o in outputs_a if o.has_thinking) / len(outputs_a)
        b_thinking_rate = sum(1 for o in outputs_b if o.has_thinking) / len(outputs_b)
        a_tool_rate = sum(1 for o in outputs_a if o.tool_call_correct) / len(outputs_a)
        b_tool_rate = sum(1 for o in outputs_b if o.tool_call_correct) / len(outputs_b)

        lines.append(f"\n=== Summary ===")
        lines.append(f"Set A: thinking={a_thinking_rate:.0%}, tool_accuracy={a_tool_rate:.0%}")
        lines.append(f"Set B: thinking={b_thinking_rate:.0%}, tool_accuracy={b_tool_rate:.0%}")

        return "\n".join(lines)

    def generate_improved_prompt(
        self,
        current_prompt: ThinkingPrompt,
        feedback: Feedback
    ) -> ThinkingPrompt:
        """根據反饋生成改進的 prompt"""

        improvement_request = f"""Current system prompt:
{current_prompt.system_prompt}

Feedback from evaluation:
- Rationale: {feedback.rationale}
- Suggestions: {feedback.improvement_suggestions}

Generate an improved system prompt that addresses the feedback.
The goal is to make the model ALWAYS output <think>...</think> tags BEFORE making tool calls.

Respond with ONLY the new system prompt text, no JSON or formatting."""

        messages = [
            {"role": "system", "content": "You are an expert prompt engineer. Generate improved prompts based on feedback."},
            {"role": "user", "content": improvement_request}
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.7,
            }
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120
            )

            if response.status_code == 200:
                data = response.json()
                new_system_prompt = data.get("message", {}).get("content", "")

                if new_system_prompt and len(new_system_prompt) > 50:
                    return ThinkingPrompt(
                        system_prompt=new_system_prompt.strip(),
                        few_shot_examples=current_prompt.few_shot_examples
                    )

            return current_prompt

        except Exception as e:
            print(f"  [Prompt Generation Error] {e}")
            return current_prompt


# =============================================================================
# FD 優化器
# =============================================================================

class FDOptimizer:
    """Feedback Descent 優化器"""

    def __init__(
        self,
        runner: OllamaRunner,
        evaluator: LocalLLMEvaluator,
        queries: List[TestQuery]
    ):
        self.runner = runner
        self.evaluator = evaluator
        self.queries = queries
        self.history: List[OptimizationStep] = []
        self.best_prompt: Optional[ThinkingPrompt] = None
        self.best_score: float = 0.0
        self.no_improvement_count: int = 0

    def evaluate_prompt(self, prompt: ThinkingPrompt) -> Tuple[List[EvaluationResult], float, float]:
        """評估一個 prompt 的效果"""
        all_results = []

        for query in self.queries:
            for _ in range(SAMPLES_PER_PROMPT):
                result = self.runner.generate(prompt, query)
                all_results.append(result)

        thinking_rate = sum(1 for r in all_results if r.has_thinking) / len(all_results)
        tool_accuracy = sum(1 for r in all_results if r.tool_call_correct) / len(all_results)

        return all_results, thinking_rate, tool_accuracy

    def calculate_score(self, thinking_rate: float, tool_accuracy: float) -> float:
        """計算綜合分數"""
        # 權重：thinking 更重要，但也要保持 tool 準確度
        return 0.6 * thinking_rate + 0.4 * tool_accuracy

    def optimize(
        self,
        initial_prompt: ThinkingPrompt,
        max_iterations: int = MAX_ITERATIONS
    ) -> ThinkingPrompt:
        """執行 FD 優化循環"""

        print("\n" + "="*60)
        print("Feedback Descent Optimization")
        print("="*60)
        print(f"Target Model: {TARGET_MODEL}")
        print(f"Evaluator Model: {EVALUATOR_MODEL}")
        print(f"Max Iterations: {max_iterations}")
        print(f"Samples per Prompt: {SAMPLES_PER_PROMPT}")
        print(f"Test Queries: {len(self.queries)}")
        print("="*60)

        current_prompt = initial_prompt

        # 初始評估
        print("\n[Iteration 0] Initial evaluation...")
        results, thinking_rate, tool_accuracy = self.evaluate_prompt(current_prompt)
        score = self.calculate_score(thinking_rate, tool_accuracy)

        print(f"  Thinking Rate: {thinking_rate:.1%}")
        print(f"  Tool Accuracy: {tool_accuracy:.1%}")
        print(f"  Score: {score:.3f}")

        self.best_prompt = current_prompt
        self.best_score = score

        self.history.append(OptimizationStep(
            iteration=0,
            prompt=current_prompt,
            thinking_rate=thinking_rate,
            tool_accuracy=tool_accuracy,
            score=score
        ))

        for iteration in range(1, max_iterations + 1):
            print(f"\n[Iteration {iteration}] Generating improved prompt...")

            # 生成新的 prompt 變體
            if self.history and self.history[-1].feedback:
                new_prompt = self.evaluator.generate_improved_prompt(
                    current_prompt,
                    self.history[-1].feedback
                )
            else:
                # 如果沒有 feedback，使用隨機變異
                new_prompt = self._mutate_prompt(current_prompt)

            # 評估新 prompt
            print("  Evaluating new prompt...")
            new_results, new_thinking_rate, new_tool_accuracy = self.evaluate_prompt(new_prompt)
            new_score = self.calculate_score(new_thinking_rate, new_tool_accuracy)

            print(f"  Thinking Rate: {new_thinking_rate:.1%}")
            print(f"  Tool Accuracy: {new_tool_accuracy:.1%}")
            print(f"  Score: {new_score:.3f}")

            # 檢查約束條件：tool accuracy 必須達到最低門檻
            if new_tool_accuracy < MIN_TOOL_ACCURACY:
                print(f"  ✗ Rejecting: tool accuracy {new_tool_accuracy:.0%} < {MIN_TOOL_ACCURACY:.0%} threshold")
                self.no_improvement_count += 1

                self.history.append(OptimizationStep(
                    iteration=iteration,
                    prompt=new_prompt,
                    thinking_rate=new_thinking_rate,
                    tool_accuracy=new_tool_accuracy,
                    score=new_score,
                    feedback=Feedback(
                        preference="A",
                        rationale=f"Tool accuracy {new_tool_accuracy:.0%} below minimum threshold {MIN_TOOL_ACCURACY:.0%}",
                        improvement_suggestions="Need to maintain tool calling ability while adding thinking"
                    )
                ))
                continue

            # Pairwise comparison
            print("  Running pairwise comparison...")
            preference, feedback = self.evaluator.compare(
                results,  # 當前最佳
                new_results,  # 新候選
                self.queries
            )
            print(f"  Evaluator prefers: Set {preference}")

            # 更新（只有同時滿足約束條件才接受）
            if preference == "B" or new_score > score:
                print("  ✓ Accepting new prompt (improvement)")
                current_prompt = new_prompt
                results = new_results
                thinking_rate = new_thinking_rate
                tool_accuracy = new_tool_accuracy
                score = new_score

                if score > self.best_score:
                    self.best_prompt = new_prompt
                    self.best_score = score
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1
            else:
                print("  ✗ Rejecting new prompt (no improvement)")
                self.no_improvement_count += 1

            self.history.append(OptimizationStep(
                iteration=iteration,
                prompt=new_prompt,
                thinking_rate=new_thinking_rate,
                tool_accuracy=new_tool_accuracy,
                score=new_score,
                feedback=feedback
            ))

            # Early stopping
            if thinking_rate >= 0.95:
                print(f"\n[Early Stop] Target thinking rate achieved!")
                break

            if self.no_improvement_count >= EARLY_STOP_K:
                print(f"\n[Early Stop] No improvement for {EARLY_STOP_K} iterations")
                break

        print("\n" + "="*60)
        print("Optimization Complete!")
        print("="*60)
        print(f"Best Score: {self.best_score:.3f}")
        print(f"Total Iterations: {len(self.history)}")

        return self.best_prompt

    def _mutate_prompt(self, prompt: ThinkingPrompt) -> ThinkingPrompt:
        """平衡 thinking 和 tool calling 的 prompt 變異策略"""
        mutations = [
            # 強調先思考後調用
            "FORMAT: First write <think>your reasoning</think>, then call the appropriate tool.",
            # 分步驟說明
            "STEPS: 1) Think in <think> tags 2) Call the tool with correct parameters.",
            # 強調兩者都必須
            "REQUIRED: Your response MUST contain BOTH <think> reasoning AND a tool call.",
            # 範例格式
            "OUTPUT FORMAT:\n<think>\n[Analyze the request and decide which tool to use]\n</think>\n[Make the tool call]",
            # 強調完整性
            "COMPLETE RESPONSE: Always include reasoning (<think> tags) followed by the tool invocation.",
        ]

        import random
        mutation = random.choice(mutations)

        # 替換而不是堆疊，避免 prompt 過長
        base_prompt = """You are a helpful assistant with access to tools.
Before calling any tool, think through the problem inside <think>...</think> tags.
Consider what the user needs, which tool to use, and what parameters are appropriate.
After thinking, make the tool call."""

        new_system = mutation + "\n\n" + base_prompt

        return ThinkingPrompt(
            system_prompt=new_system,
            few_shot_examples=prompt.few_shot_examples
        )

    def save_results(self, output_dir: str = OUTPUT_DIR):
        """保存優化結果"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存優化歷史
        history_data = []
        for step in self.history:
            history_data.append({
                "iteration": step.iteration,
                "thinking_rate": step.thinking_rate,
                "tool_accuracy": step.tool_accuracy,
                "score": step.score,
                "system_prompt": step.prompt.system_prompt[:500] + "...",
                "feedback": asdict(step.feedback) if step.feedback else None
            })

        with open(f"{output_dir}/optimization_log_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)

        # 保存最佳 prompt
        if self.best_prompt:
            with open(f"{output_dir}/best_prompt_{timestamp}.json", "w", encoding="utf-8") as f:
                f.write(self.best_prompt.to_json())

        # 保存摘要
        summary = {
            "timestamp": timestamp,
            "target_model": TARGET_MODEL,
            "evaluator_model": EVALUATOR_MODEL,
            "total_iterations": len(self.history),
            "best_score": self.best_score,
            "initial_thinking_rate": self.history[0].thinking_rate if self.history else 0,
            "final_thinking_rate": self.history[-1].thinking_rate if self.history else 0,
            "initial_tool_accuracy": self.history[0].tool_accuracy if self.history else 0,
            "final_tool_accuracy": self.history[-1].tool_accuracy if self.history else 0,
        }

        with open(f"{output_dir}/summary_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to {output_dir}/")
        return summary


# =============================================================================
# 主函數
# =============================================================================

def check_ollama_running() -> bool:
    """檢查 Ollama 服務是否運行"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def check_models_available() -> Tuple[bool, bool]:
    """檢查所需模型是否可用"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            available = [m["name"] for m in data.get("models", [])]
            target_ok = TARGET_MODEL in available
            evaluator_ok = EVALUATOR_MODEL in available
            return target_ok, evaluator_ok
    except:
        pass
    return False, False


def main():
    print("="*60)
    print("Ollama Feedback Descent Optimizer")
    print("="*60)

    # 檢查 Ollama 服務
    if not check_ollama_running():
        print("\n[ERROR] Ollama is not running!")
        print("Please start Ollama with: ollama serve")
        return

    print("\n[OK] Ollama is running")

    # 檢查模型
    target_ok, evaluator_ok = check_models_available()

    if not target_ok:
        print(f"\n[ERROR] Target model '{TARGET_MODEL}' not found!")
        print(f"Please run: ollama pull {TARGET_MODEL}")
        return

    if not evaluator_ok:
        print(f"\n[ERROR] Evaluator model '{EVALUATOR_MODEL}' not found!")
        print(f"Please run: ollama pull {EVALUATOR_MODEL}")
        return

    print(f"[OK] Target model: {TARGET_MODEL}")
    print(f"[OK] Evaluator model: {EVALUATOR_MODEL}")

    # 創建組件
    runner = OllamaRunner(model=TARGET_MODEL)
    evaluator = LocalLLMEvaluator(model=EVALUATOR_MODEL)

    # 創建優化器
    optimizer = FDOptimizer(
        runner=runner,
        evaluator=evaluator,
        queries=TEST_QUERIES
    )

    # 運行優化
    start_time = time.time()
    best_prompt = optimizer.optimize(INITIAL_PROMPT, max_iterations=MAX_ITERATIONS)
    elapsed_time = time.time() - start_time

    # 保存結果
    summary = optimizer.save_results()

    # 打印最終結果
    print("\n" + "="*60)
    print("Final Results")
    print("="*60)
    print(f"Time Elapsed: {elapsed_time/60:.1f} minutes")
    print(f"Initial Thinking Rate: {summary['initial_thinking_rate']:.1%}")
    print(f"Final Thinking Rate: {summary['final_thinking_rate']:.1%}")
    print(f"Initial Tool Accuracy: {summary['initial_tool_accuracy']:.1%}")
    print(f"Final Tool Accuracy: {summary['final_tool_accuracy']:.1%}")
    print(f"Best Score: {summary['best_score']:.3f}")

    print("\n" + "-"*60)
    print("Best System Prompt:")
    print("-"*60)
    print(best_prompt.system_prompt)


if __name__ == "__main__":
    main()
