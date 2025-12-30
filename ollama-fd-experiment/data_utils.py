"""
共用資料處理模組

提供 train.py 和 fd_deep_analysis.py 共用的資料載入和處理功能。
"""

import json
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datasets import load_dataset, Dataset

# =============================================================================
# 常數
# =============================================================================

THINK_TAG_OPEN = "<think>"
THINK_TAG_CLOSE = "</think>"
THINK_KEYS = ["think", "think_fast", "think_faster"]

# 資料集設定
DATASET_NAME = "LLM360/TxT360-3efforts"
DATASET_CONFIG = "agent"
DATASET_SPLIT = "medium"


# =============================================================================
# 資料結構
# =============================================================================

@dataclass
class TestCase:
    """從 TxT360 資料集提取的測試案例"""
    # 核心資料
    messages: List[Dict]              # 完整對話 (system, user, assistant, tool, ...)
    tools: List[Dict]                 # 工具定義 (OpenAI 格式)

    # 提取的元資料
    user_content: str                 # 第一個 user 訊息內容
    expected_tool_name: str           # 預期呼叫的工具名稱
    expected_thinking: str            # 資料集中的 thinking 內容
    expected_tool_args: Dict = field(default_factory=dict)  # 預期的工具參數

    # 多輪對話支援
    tool_response: Optional[Dict] = None  # 工具回應
    final_answer: Optional[str] = None    # 最終回答
    is_multi_turn: bool = False           # 是否為多輪對話
    turn_count: int = 2                   # 對話輪數


# =============================================================================
# 資料處理函數
# =============================================================================

def prepare_messages_and_tools(example: Dict) -> Tuple[Optional[List[Dict]], Optional[List[Dict]]]:
    """
    準備訊息和工具定義，從原始資料集範例中提取。

    Args:
        example: 原始資料集範例，包含 "messages" JSON 字串

    Returns:
        (messages, tools) 元組，若資料無效則返回 (None, None)
    """
    raw = json.loads(example["messages"])
    msgs = [dict(m) for m in raw]

    # 提取工具定義
    tools_raw = []
    if msgs and isinstance(msgs[0], dict):
        tlist = msgs[0].get("tools")
        if isinstance(tlist, list) and tlist:
            tools_raw = tlist
            msgs[0].pop("tools", None)

    # 合併 assistant["think"] 到 ["content"]
    has_valid_thought = False

    for m in msgs:
        if m.get("role") == "assistant":
            found_key = next((k for k in THINK_KEYS if m.get(k)), None)

            if found_key:
                think_text = m[found_key]
                content = m.get("content")
                think_block = f"{THINK_TAG_OPEN}{think_text}{THINK_TAG_CLOSE}"

                if isinstance(content, str) and content:
                    m["content"] = think_block + "\n" + content
                else:
                    m["content"] = think_block

                has_valid_thought = True

                for k in THINK_KEYS:
                    m.pop(k, None)
            else:
                return None, None

    if not has_valid_thought:
        return None, None

    # 正規化 tool_calls
    for m in msgs:
        if "tool_calls" not in m or not m["tool_calls"]:
            continue

        new_tool_calls = []
        for tc in m["tool_calls"]:
            if not isinstance(tc, dict):
                continue

            if "function" in tc and isinstance(tc["function"], dict):
                new_tool_calls.append(tc)
                continue

            fn_name = tc.get("name", "")
            args = tc.get("arguments", {})

            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass

            new_tool_calls.append({
                "id": tc.get("id") or tc.get("tool_call_id"),
                "type": tc.get("type", "function"),
                "function": {
                    "name": fn_name,
                    "arguments": args,
                },
            })

        m["tool_calls"] = new_tool_calls

    # 建立 tool_call_id -> function name 映射
    id_to_name = {}
    for m in msgs:
        for tc in m.get("tool_calls", []) or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            name = fn.get("name") or tc.get("name")
            tc_id = tc.get("id") or tc.get("tool_call_id")
            if tc_id and name:
                id_to_name[tc_id] = name

    # 確保 tool response 訊息有 'name' 欄位
    for m in msgs:
        if m.get("role") == "tool":
            if not m.get("name"):
                tc_id = m.get("tool_call_id")
                inferred = id_to_name.get(tc_id) if tc_id else None
                m["name"] = inferred or "unknown_tool"

    # 正規化工具 schema
    adapted_tools = []
    for t in tools_raw:
        if not isinstance(t, dict):
            continue

        if "function" in t and isinstance(t["function"], dict):
            adapted_tools.append(t)
            continue

        name = t.get("name", "")
        description = t.get("description", "")
        parameters = t.get("parameters") or {"type": "object", "properties": {}}

        adapted_tools.append({
            "type": t.get("type", "function"),
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        })

    # 刪除空的 system 訊息
    first_message = msgs[0]
    if first_message["role"] == "system" and "content" not in first_message:
        msgs.pop(0)

    return msgs, adapted_tools


def extract_test_case(example: Dict) -> Optional[TestCase]:
    """
    從資料集範例中提取 TestCase。

    Args:
        example: 原始資料集範例

    Returns:
        TestCase 物件，若資料無效則返回 None
    """
    messages, tools = prepare_messages_and_tools(example)

    if messages is None or tools is None or len(messages) == 0:
        return None

    # 提取 user content
    user_content = ""
    for m in messages:
        if m.get("role") == "user":
            user_content = m.get("content", "")
            break

    if not user_content:
        return None

    # 提取 expected thinking 和 tool call
    expected_thinking = ""
    expected_tool_name = ""
    expected_tool_args = {}

    for m in messages:
        if m.get("role") == "assistant":
            content = m.get("content", "")
            # 提取 thinking
            if THINK_TAG_OPEN in content and THINK_TAG_CLOSE in content:
                start = content.find(THINK_TAG_OPEN) + len(THINK_TAG_OPEN)
                end = content.find(THINK_TAG_CLOSE)
                expected_thinking = content[start:end].strip()

            # 提取 tool call
            tool_calls = m.get("tool_calls", [])
            if tool_calls and len(tool_calls) > 0:
                tc = tool_calls[0]
                fn = tc.get("function", {})
                expected_tool_name = fn.get("name", "")
                expected_tool_args = fn.get("arguments", {})
            break

    if not expected_tool_name:
        return None

    # 檢查是否為多輪對話 (有 tool response)
    tool_response = None
    final_answer = None
    is_multi_turn = False

    for i, m in enumerate(messages):
        if m.get("role") == "tool":
            tool_response = {
                "role": "tool",
                "name": m.get("name", ""),
                "tool_call_id": m.get("tool_call_id", ""),
                "content": m.get("content", "")
            }
            is_multi_turn = True

            # 檢查是否有後續的 assistant 回應
            if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant":
                final_answer = messages[i + 1].get("content", "")
            break

    # 計算對話輪數
    turn_count = sum(1 for m in messages if m.get("role") in ["user", "assistant"])

    return TestCase(
        messages=messages,
        tools=tools,
        user_content=user_content,
        expected_tool_name=expected_tool_name,
        expected_thinking=expected_thinking,
        expected_tool_args=expected_tool_args,
        tool_response=tool_response,
        final_answer=final_answer,
        is_multi_turn=is_multi_turn,
        turn_count=turn_count,
    )


def load_test_cases(
    n_samples: int = 50,
    require_thinking: bool = True,
    require_tool_call: bool = True,
    require_multi_turn: bool = False,
    seed: int = 42,
    max_scan: int = 5000,
) -> List[TestCase]:
    """
    從 TxT360-3efforts 資料集載入測試案例。

    Args:
        n_samples: 要載入的測試案例數量
        require_thinking: 是否要求有 thinking 內容
        require_tool_call: 是否要求有 tool call
        require_multi_turn: 是否只載入多輪對話
        seed: 隨機種子
        max_scan: 最多掃描的資料集範例數

    Returns:
        TestCase 物件列表
    """
    print(f"Loading test cases from {DATASET_NAME}...")

    # 載入資料集 (streaming 模式)
    dataset = load_dataset(
        DATASET_NAME,
        name=DATASET_CONFIG,
        split=DATASET_SPLIT,
        streaming=True
    )

    # 收集有效案例
    valid_cases = []
    scanned = 0

    for example in dataset:
        if scanned >= max_scan:
            break
        scanned += 1

        test_case = extract_test_case(example)
        if test_case is None:
            continue

        # 套用篩選條件
        if require_thinking and not test_case.expected_thinking:
            continue
        if require_tool_call and not test_case.expected_tool_name:
            continue
        if require_multi_turn and not test_case.is_multi_turn:
            continue

        valid_cases.append(test_case)

    print(f"Scanned {scanned} examples, found {len(valid_cases)} valid cases")

    # 隨機取樣
    random.seed(seed)
    if len(valid_cases) > n_samples:
        valid_cases = random.sample(valid_cases, n_samples)

    print(f"Loaded {len(valid_cases)} test cases")
    return valid_cases


def generate_few_shot_examples(test_cases: List[TestCase], n_examples: int = 3) -> List[Dict]:
    """
    從測試案例中生成 few-shot 範例。

    Args:
        test_cases: TestCase 列表
        n_examples: 要生成的範例數量

    Returns:
        few-shot 範例列表
    """
    examples = []
    for tc in test_cases[:n_examples]:
        if tc.expected_thinking:
            examples.append({
                "user": tc.user_content,
                "assistant": f"{THINK_TAG_OPEN}\n{tc.expected_thinking}\n{THINK_TAG_CLOSE}",
                "tool_name": tc.expected_tool_name,
            })
    return examples
