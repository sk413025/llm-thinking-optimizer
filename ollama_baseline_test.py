#!/usr/bin/env python3
"""
Ollama 基線測試腳本 - 測試模型是否滿足 FD 前提條件

測試項目:
1. Prompt Sensitivity: 不同 prompt 是否產生不同輸出
2. Format Compliance: 是否能遵循格式指令
3. Output Variance: 同一輸入多次採樣是否有變異
4. Thinking Capability: 是否能產生 <think> 標籤

成功標準:
- Thinking Rate > 10% (任一 prompt 變體)
- Output Diversity > 0.1 (temperature=0.7)
- Format Compliance > 30%
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
import requests
from collections import Counter

# 導入共用資料模組
from data_utils import (
    THINK_TAG_OPEN,
    THINK_TAG_CLOSE,
    TestCase,
    load_test_cases,
    generate_few_shot_examples,
)


# ============== 配置 ==============

OLLAMA_BASE_URL = "http://localhost:11434"

# 測試的模型列表
MODELS_TO_TEST = [
    "llama3.1:8b",
    "qwen3:8b",
    "mistral:7b",
    "gpt-oss:20b",
]

# 資料集配置 (對齊 train.py 和 ollama_fd_optimizer.py)
N_EVALUATION_SAMPLES = 20    # 評估樣本數
RANDOM_SEED = 42             # 隨機種子
NUM_SAMPLES_PER_CONDITION = 3  # 每個條件的採樣數

# Prompt 變體 (測試 H1, H2)
PROMPT_VARIANTS = {
    "baseline": "You are a helpful assistant with access to tools.",

    "explicit_think": """You are a helpful assistant with access to tools.
Before calling any tool, you MUST first think inside <think>...</think> tags.
Then make the tool call.""",

    "chain_of_thought": """You are a helpful assistant with access to tools.
Let's think step by step before taking any action.
First explain your reasoning, then use the appropriate tool.""",

    "expert_role": """You are an expert assistant who always explains your reasoning first.
Before using any tool, analyze what the user needs and why you're choosing that tool.
Format: First explain in <think>...</think>, then call the tool.""",

    "teacher_role": """You are a teacher who shows your work.
When helping users, first demonstrate your thinking process in <think>...</think> tags.
Then proceed to use the appropriate tool.""",

    "json_thinking": """You are a helpful assistant.
Before each tool call, output a JSON block with your reasoning:
{"thinking": "your reasoning here"}
Then make the tool call.""",

    "no_direct_action": """You are a helpful assistant with access to tools.
IMPORTANT: Do NOT call tools directly.
First, explain what you will do and why in <think>...</think> tags.
Only after thinking, call the appropriate tool.""",

    "thinking_required": """You are a helpful assistant.
REQUIRED FORMAT:
<think>
[Your analysis of the user's request]
[Which tool to use and why]
</think>
[Tool call]

This format is MANDATORY. Always include the <think> section.""",

    "step_by_step": """You are a helpful assistant.
For every request:
Step 1: Think about what the user needs (write in <think> tags)
Step 2: Decide which tool to use
Step 3: Call the tool

Never skip Step 1.""",

    "reasoning_prefix": """You are a helpful assistant with tools.
Your response must ALWAYS start with reasoning:
<think>I need to...</think>
Then call the tool."""
}

# 格式遵循測試 (測試 H4)
FORMAT_TESTS = [
    {
        "instruction": "使用中文回答",
        "check": lambda x: any(ord(c) > 0x4e00 and ord(c) < 0x9fff for c in x),
        "description": "Response contains Chinese characters"
    },
    {
        "instruction": "回答以 'OK:' 開頭",
        "check": lambda x: x.strip().startswith("OK:"),
        "description": "Response starts with 'OK:'"
    },
    {
        "instruction": "在回答末尾加上 [DONE]",
        "check": lambda x: "[DONE]" in x,
        "description": "Response contains [DONE]"
    },
    {
        "instruction": "用編號步驟 (1. 2. 3.) 來組織你的回答",
        "check": lambda x: "1." in x and "2." in x,
        "description": "Response uses numbered steps"
    }
]


# ============== 數據結構 ==============

@dataclass
class TestQuery:
    """測試查詢（從 TestCase 轉換）"""
    user_content: str
    tools: List[Dict]
    expected_tool_name: str


@dataclass
class TestResult:
    model: str
    test_type: str
    condition: str
    query: str
    expected_tool: str           # 預期的工具名稱
    output: str
    has_thinking: bool = False
    has_tool_call: bool = False
    tool_call_correct: bool = False  # 工具呼叫是否正確
    format_compliant: bool = False
    metadata: Dict = field(default_factory=dict)


@dataclass
class ModelMetrics:
    model: str
    thinking_rate: float = 0.0
    tool_call_rate: float = 0.0
    tool_call_accuracy: float = 0.0  # 工具呼叫準確率
    format_compliance_rate: float = 0.0
    output_diversity: float = 0.0
    prompt_sensitivity: float = 0.0
    best_prompt_variant: str = ""
    best_thinking_rate: float = 0.0
    fd_viable: bool = False
    details: Dict = field(default_factory=dict)


# ============== 資料載入 ==============

def load_test_queries(n_samples: int = N_EVALUATION_SAMPLES) -> List[TestQuery]:
    """
    從 TxT360-3efforts 資料集載入測試查詢。

    Args:
        n_samples: 要載入的樣本數

    Returns:
        TestQuery 列表
    """
    print(f"  Loading {n_samples} samples from TxT360-3efforts...")

    test_cases = load_test_cases(
        n_samples=n_samples,
        require_thinking=True,
        require_tool_call=True,
        seed=RANDOM_SEED,
    )

    # 轉換為 TestQuery 格式
    queries = [
        TestQuery(
            user_content=tc.user_content,
            tools=tc.tools,
            expected_tool_name=tc.expected_tool_name,
        )
        for tc in test_cases
    ]

    print(f"  Loaded {len(queries)} valid test queries")
    return queries


# ============== Ollama 交互 ==============

def check_ollama_running() -> bool:
    """檢查 Ollama 服務是否運行"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_available_models() -> List[str]:
    """獲取已安裝的模型列表"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
    except Exception as e:
        print(f"Error getting models: {e}")
    return []


def generate_with_ollama(
    model: str,
    system_prompt: str,
    user_message: str,
    temperature: float = 0.7,
    tools: Optional[List[Dict]] = None
) -> str:
    """使用 Ollama 生成回應"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
        }
    }

    # 添加工具定義（如果模型支持）
    if tools:
        payload["tools"] = tools

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=120
        )

        if response.status_code == 200:
            data = response.json()
            message = data.get("message", {})

            # 處理工具調用
            tool_calls = message.get("tool_calls", [])
            content = message.get("content", "")

            # 組合輸出
            output_parts = []
            if content:
                output_parts.append(content)
            if tool_calls:
                for tc in tool_calls:
                    func = tc.get("function", {})
                    output_parts.append(
                        f"[TOOL_CALL: {func.get('name', 'unknown')}({func.get('arguments', {})})]"
                    )

            return "\n".join(output_parts) if output_parts else "[NO OUTPUT]"
        else:
            return f"[ERROR: {response.status_code}]"

    except Exception as e:
        return f"[ERROR: {str(e)}]"


# ============== 分析函數 ==============

def has_thinking_tag(output: str) -> bool:
    """檢查輸出是否包含 <think> 標籤"""
    output_lower = output.lower()
    return THINK_TAG_OPEN.lower() in output_lower or THINK_TAG_CLOSE.lower() in output_lower


def extract_tool_call_name(output: str) -> Optional[str]:
    """從輸出中提取工具呼叫名稱"""
    import re

    # 格式 1: [TOOL_CALL: name(...)]
    if "[TOOL_CALL:" in output:
        try:
            start = output.find("[TOOL_CALL:") + len("[TOOL_CALL:")
            end = output.find("(", start)
            if end > start:
                return output[start:end].strip()
        except:
            pass

    # 格式 2: JSON 中的 "name": "xxx"
    json_pattern = r'"name"\s*:\s*"([^"]+)"'
    match = re.search(json_pattern, output)
    if match:
        return match.group(1)

    return None


def has_tool_call(output: str) -> bool:
    """檢查輸出是否包含工具調用"""
    return "[TOOL_CALL:" in output or "function" in output.lower()


def calculate_diversity(outputs: List[str]) -> float:
    """計算輸出多樣性（0-1）"""
    if len(outputs) <= 1:
        return 0.0

    # 統計唯一輸出
    unique_outputs = set(outputs)

    # 多樣性 = (唯一輸出數 - 1) / (總數 - 1)
    diversity = (len(unique_outputs) - 1) / (len(outputs) - 1)
    return diversity


def calculate_edit_distance(s1: str, s2: str) -> int:
    """計算編輯距離（Levenshtein distance）"""
    if len(s1) < len(s2):
        return calculate_edit_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


# ============== 測試函數 ==============

def test_prompt_variants(model: str, test_queries: List[TestQuery], num_samples: int = NUM_SAMPLES_PER_CONDITION) -> List[TestResult]:
    """測試 1: Prompt 變體測試（使用 TxT360 資料）"""
    print(f"\n  [Test 1] Prompt Variants ({len(PROMPT_VARIANTS)} variants × {len(test_queries)} queries × {num_samples} samples)")

    results = []

    for variant_name, system_prompt in PROMPT_VARIANTS.items():
        print(f"    Testing: {variant_name}...", end=" ", flush=True)
        variant_results = []

        for query in test_queries:
            for _ in range(num_samples):
                output = generate_with_ollama(
                    model=model,
                    system_prompt=system_prompt,
                    user_message=query.user_content,
                    temperature=0.7,
                    tools=query.tools  # 使用每個 query 自帶的工具
                )

                # 檢查工具呼叫是否正確
                called_tool = extract_tool_call_name(output)
                tool_correct = called_tool == query.expected_tool_name if called_tool else False

                result = TestResult(
                    model=model,
                    test_type="prompt_variant",
                    condition=variant_name,
                    query=query.user_content[:200],
                    expected_tool=query.expected_tool_name,
                    output=output,
                    has_thinking=has_thinking_tag(output),
                    has_tool_call=has_tool_call(output),
                    tool_call_correct=tool_correct,
                )
                results.append(result)
                variant_results.append(result)

        # 計算此變體的 thinking rate 和 tool accuracy
        thinking_rate = sum(1 for r in variant_results if r.has_thinking) / len(variant_results)
        tool_rate = sum(1 for r in variant_results if r.has_tool_call) / len(variant_results)
        tool_accuracy = sum(1 for r in variant_results if r.tool_call_correct) / len(variant_results)
        print(f"Think: {thinking_rate:.0%}, Tool: {tool_rate:.0%}, Acc: {tool_accuracy:.0%}")

    return results


def test_few_shot(model: str, test_queries: List[TestQuery], fewshot_queries: List[TestQuery], num_samples: int = NUM_SAMPLES_PER_CONDITION) -> List[TestResult]:
    """測試 2: Few-Shot 效果測試（使用 TxT360 資料生成 few-shot 範例）"""
    print(f"\n  [Test 2] Few-Shot Learning (0-3 shots × {len(test_queries)} queries × {num_samples} samples)")

    # 從 fewshot_queries 生成 few-shot 範例
    few_shot_examples = []
    for q in fewshot_queries[:3]:  # 最多 3 個範例
        few_shot_examples.append({
            "user": q.user_content,
            "assistant": f"""{THINK_TAG_OPEN}
The user is asking about something that requires the {q.expected_tool_name} tool.
I should analyze the request and use the appropriate tool.
{THINK_TAG_CLOSE}
[TOOL_CALL: {q.expected_tool_name}(...)]"""
        })

    results = []

    for n_shots in range(4):  # 0, 1, 2, 3 shots
        print(f"    Testing: {n_shots}-shot...", end=" ", flush=True)

        # 構建 system prompt with few-shot
        system_prompt = f"""You are a helpful assistant with access to tools.
Before calling any tool, first think inside {THINK_TAG_OPEN}...{THINK_TAG_CLOSE} tags.
Then make the tool call."""

        if n_shots > 0 and few_shot_examples:
            system_prompt += "\n\nExamples:\n"
            for i in range(min(n_shots, len(few_shot_examples))):
                ex = few_shot_examples[i]
                system_prompt += f"\nUser: {ex['user']}\nAssistant: {ex['assistant']}\n"

        shot_results = []
        for query in test_queries:
            for _ in range(num_samples):
                output = generate_with_ollama(
                    model=model,
                    system_prompt=system_prompt,
                    user_message=query.user_content,
                    temperature=0.7,
                    tools=query.tools
                )

                # 檢查工具呼叫是否正確
                called_tool = extract_tool_call_name(output)
                tool_correct = called_tool == query.expected_tool_name if called_tool else False

                result = TestResult(
                    model=model,
                    test_type="few_shot",
                    condition=f"{n_shots}-shot",
                    query=query.user_content[:200],
                    expected_tool=query.expected_tool_name,
                    output=output,
                    has_thinking=has_thinking_tag(output),
                    has_tool_call=has_tool_call(output),
                    tool_call_correct=tool_correct,
                )
                results.append(result)
                shot_results.append(result)

        thinking_rate = sum(1 for r in shot_results if r.has_thinking) / len(shot_results)
        tool_rate = sum(1 for r in shot_results if r.has_tool_call) / len(shot_results)
        tool_accuracy = sum(1 for r in shot_results if r.tool_call_correct) / len(shot_results)
        print(f"Think: {thinking_rate:.0%}, Tool: {tool_rate:.0%}, Acc: {tool_accuracy:.0%}")

    return results


def test_format_compliance(model: str, test_queries: List[TestQuery], num_samples: int = NUM_SAMPLES_PER_CONDITION) -> List[TestResult]:
    """測試 3: 格式遵循能力測試（使用 TxT360 資料）"""
    print(f"\n  [Test 3] Format Compliance ({len(FORMAT_TESTS)} formats × {num_samples} samples)")

    results = []

    for fmt_test in FORMAT_TESTS:
        print(f"    Testing: {fmt_test['description']}...", end=" ", flush=True)

        system_prompt = f"""You are a helpful assistant.
IMPORTANT: {fmt_test['instruction']}
Please answer the user's question."""

        fmt_results = []
        # 使用第一個測試查詢
        query = test_queries[0] if test_queries else None
        if not query:
            print("No queries available!")
            continue

        for _ in range(num_samples):
            output = generate_with_ollama(
                model=model,
                system_prompt=system_prompt,
                user_message=query.user_content,
                temperature=0.7
            )

            compliant = fmt_test["check"](output)

            result = TestResult(
                model=model,
                test_type="format_compliance",
                condition=fmt_test["instruction"],
                query=query.user_content[:200],
                expected_tool=query.expected_tool_name,
                output=output,
                format_compliant=compliant,
                metadata={"format_type": fmt_test["description"]}
            )
            results.append(result)
            fmt_results.append(result)

        compliance_rate = sum(1 for r in fmt_results if r.format_compliant) / len(fmt_results)
        print(f"Compliance: {compliance_rate:.0%}")

    return results


def test_output_variance(model: str, test_queries: List[TestQuery], num_samples: int = 10) -> List[TestResult]:
    """測試 4: 輸出變異度測試（使用 TxT360 資料）"""
    print(f"\n  [Test 4] Output Variance (4 temps × {num_samples} samples)")

    results = []
    temperatures = [0.3, 0.5, 0.7, 1.0]

    system_prompt = "You are a helpful assistant with access to tools."

    # 使用第一個測試查詢
    query = test_queries[0] if test_queries else None
    if not query:
        print("No queries available!")
        return results

    for temp in temperatures:
        print(f"    Testing: temp={temp}...", end=" ", flush=True)

        temp_outputs = []
        for _ in range(num_samples):
            output = generate_with_ollama(
                model=model,
                system_prompt=system_prompt,
                user_message=query.user_content,
                temperature=temp,
                tools=query.tools
            )

            # 檢查工具呼叫是否正確
            called_tool = extract_tool_call_name(output)
            tool_correct = called_tool == query.expected_tool_name if called_tool else False

            result = TestResult(
                model=model,
                test_type="output_variance",
                condition=f"temp={temp}",
                query=query.user_content[:200],
                expected_tool=query.expected_tool_name,
                output=output,
                has_thinking=has_thinking_tag(output),
                has_tool_call=has_tool_call(output),
                tool_call_correct=tool_correct,
                metadata={"temperature": temp}
            )
            results.append(result)
            temp_outputs.append(output)

        diversity = calculate_diversity(temp_outputs)
        unique_count = len(set(temp_outputs))
        print(f"Diversity: {diversity:.2f} ({unique_count}/{num_samples} unique)")

    return results


# ============== 主函數 ==============

def analyze_results(results: List[TestResult], model: str) -> ModelMetrics:
    """分析測試結果並計算指標"""

    metrics = ModelMetrics(model=model)

    # 1. 計算 Prompt Variant 指標
    variant_results = [r for r in results if r.test_type == "prompt_variant"]
    if variant_results:
        # 總體 thinking rate 和 tool accuracy
        metrics.thinking_rate = sum(1 for r in variant_results if r.has_thinking) / len(variant_results)
        metrics.tool_call_rate = sum(1 for r in variant_results if r.has_tool_call) / len(variant_results)
        metrics.tool_call_accuracy = sum(1 for r in variant_results if r.tool_call_correct) / len(variant_results)

        # 找出最佳 prompt variant
        by_variant = {}
        for r in variant_results:
            if r.condition not in by_variant:
                by_variant[r.condition] = []
            by_variant[r.condition].append(r)

        best_variant = ""
        best_rate = 0.0
        variant_rates = {}
        variant_tool_accuracy = {}
        for variant, vresults in by_variant.items():
            rate = sum(1 for r in vresults if r.has_thinking) / len(vresults)
            tool_acc = sum(1 for r in vresults if r.tool_call_correct) / len(vresults)
            variant_rates[variant] = rate
            variant_tool_accuracy[variant] = tool_acc
            if rate > best_rate:
                best_rate = rate
                best_variant = variant

        metrics.best_prompt_variant = best_variant
        metrics.best_thinking_rate = best_rate
        metrics.details["variant_thinking_rates"] = variant_rates
        metrics.details["variant_tool_accuracy"] = variant_tool_accuracy

        # 計算 prompt sensitivity (輸出間的差異度)
        outputs_by_variant = {v: [r.output for r in rs] for v, rs in by_variant.items()}
        all_outputs = [o for outputs in outputs_by_variant.values() for o in outputs]
        if len(all_outputs) > 1:
            # 抽樣計算編輯距離
            sample_distances = []
            for i in range(min(20, len(all_outputs))):
                for j in range(i+1, min(20, len(all_outputs))):
                    dist = calculate_edit_distance(all_outputs[i][:200], all_outputs[j][:200])
                    avg_len = (len(all_outputs[i][:200]) + len(all_outputs[j][:200])) / 2
                    if avg_len > 0:
                        sample_distances.append(dist / avg_len)
            if sample_distances:
                metrics.prompt_sensitivity = sum(sample_distances) / len(sample_distances)

    # 2. 計算 Format Compliance
    format_results = [r for r in results if r.test_type == "format_compliance"]
    if format_results:
        metrics.format_compliance_rate = sum(1 for r in format_results if r.format_compliant) / len(format_results)

    # 3. 計算 Output Diversity
    variance_results = [r for r in results if r.test_type == "output_variance"]
    if variance_results:
        by_temp = {}
        for r in variance_results:
            temp = r.metadata.get("temperature", 0.7)
            if temp not in by_temp:
                by_temp[temp] = []
            by_temp[temp].append(r.output)

        diversities = []
        for temp, outputs in by_temp.items():
            div = calculate_diversity(outputs)
            diversities.append(div)

        metrics.output_diversity = sum(diversities) / len(diversities) if diversities else 0.0
        metrics.details["diversity_by_temp"] = {t: calculate_diversity(o) for t, o in by_temp.items()}

    # 4. 判斷 FD 可行性
    # 標準: Thinking Rate > 10% OR Output Diversity > 0.1 OR Format Compliance > 30%
    metrics.fd_viable = (
        metrics.best_thinking_rate > 0.1 or
        metrics.output_diversity > 0.1 or
        metrics.format_compliance_rate > 0.3
    )

    return metrics


def run_baseline_test(model: str, test_queries: List[TestQuery], fewshot_queries: List[TestQuery]) -> ModelMetrics:
    """對單個模型運行完整基線測試（使用 TxT360 資料）"""
    print(f"\n{'='*60}")
    print(f"Testing Model: {model}")
    print(f"{'='*60}")

    all_results = []

    # 運行所有測試（傳入 test_queries）
    all_results.extend(test_prompt_variants(model, test_queries, num_samples=NUM_SAMPLES_PER_CONDITION))
    all_results.extend(test_few_shot(model, test_queries, fewshot_queries, num_samples=NUM_SAMPLES_PER_CONDITION))
    all_results.extend(test_format_compliance(model, test_queries, num_samples=NUM_SAMPLES_PER_CONDITION))
    all_results.extend(test_output_variance(model, test_queries, num_samples=10))

    # 分析結果
    metrics = analyze_results(all_results, model)

    # 保存原始結果
    results_file = f"ollama_baseline_{model.replace(':', '_')}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2, ensure_ascii=False)
    print(f"\n  Raw results saved to: {results_file}")

    return metrics


def main():
    print("="*60)
    print("Ollama Baseline Test - FD Feasibility Analysis")
    print("="*60)
    print(f"Dataset: LLM360/TxT360-3efforts")
    print(f"Evaluation samples: {N_EVALUATION_SAMPLES}")

    # 檢查 Ollama 服務
    if not check_ollama_running():
        print("\n[ERROR] Ollama is not running!")
        print("Please start Ollama with: ollama serve")
        return

    print("\n[OK] Ollama is running")

    # 獲取可用模型
    available_models = get_available_models()
    print(f"\nAvailable models: {available_models}")

    # 篩選要測試的模型
    models_to_test = [m for m in MODELS_TO_TEST if m in available_models]

    if not models_to_test:
        print("\n[ERROR] No models from the test list are available!")
        print(f"Expected: {MODELS_TO_TEST}")
        print(f"Available: {available_models}")
        return

    print(f"\nModels to test: {models_to_test}")

    # 從 TxT360 載入測試查詢
    print(f"\n[Loading] Loading {N_EVALUATION_SAMPLES} samples from TxT360-3efforts...")
    all_queries = load_test_queries(N_EVALUATION_SAMPLES)

    if len(all_queries) < 5:
        print("[ERROR] Not enough valid samples loaded! Need at least 5.")
        return

    # 分割：前 5 個用於 few-shot，其餘用於測試
    fewshot_queries = all_queries[:5]
    test_queries = all_queries[5:]

    print(f"[OK] Split: {len(fewshot_queries)} for few-shot, {len(test_queries)} for testing")

    # 運行測試
    all_metrics = []
    for model in models_to_test:
        metrics = run_baseline_test(model, test_queries, fewshot_queries)
        all_metrics.append(metrics)

    # 打印摘要（含 tool accuracy）
    print("\n" + "="*60)
    print("SUMMARY: FD Feasibility Assessment")
    print("="*60)

    print("\n| Model | Think | Best Think | Tool Rate | Tool Acc | Format | Diversity | FD Viable |")
    print("|-------|-------|------------|-----------|----------|--------|-----------|-----------|")

    for m in all_metrics:
        viable_str = "✅ YES" if m.fd_viable else "❌ NO"
        print(f"| {m.model:15} | {m.thinking_rate:>5.0%} | {m.best_thinking_rate:>8.0%} | {m.tool_call_rate:>7.0%} | {m.tool_call_accuracy:>6.0%} | {m.format_compliance_rate:>5.0%} | {m.output_diversity:>8.2f} | {viable_str:>9} |")

    print("\n" + "-"*60)
    print("FD Viability Criteria:")
    print("  - Best Thinking Rate > 10%")
    print("  - OR Output Diversity > 0.1")
    print("  - OR Format Compliance > 30%")

    # 保存摘要
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "LLM360/TxT360-3efforts",
        "n_evaluation_samples": N_EVALUATION_SAMPLES,
        "n_fewshot": len(fewshot_queries),
        "n_test": len(test_queries),
        "models_tested": models_to_test,
        "metrics": [asdict(m) for m in all_metrics],
        "fd_viable_models": [m.model for m in all_metrics if m.fd_viable]
    }

    with open("ollama_baseline_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nSummary saved to: ollama_baseline_summary.json")

    # 推薦
    viable_models = [m for m in all_metrics if m.fd_viable]
    if viable_models:
        print(f"\n🎉 FD-viable models found: {[m.model for m in viable_models]}")
        print("Recommend proceeding with FD experiment on these models.")
    else:
        print("\n⚠️ No FD-viable models found.")
        print("FD may not be effective for these models on this task.")


if __name__ == "__main__":
    main()
