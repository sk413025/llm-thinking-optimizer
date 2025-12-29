# FD 優化器調試記錄

## 問題發現與解決過程

本文檔記錄了在實現 Ollama Feedback Descent 優化器時遇到的關鍵問題及其解決方案。

---

## 問題 1：Thinking Rate 與 Tool Accuracy 的矛盾

### 現象描述

第一次運行 FD 優化器時，出現了看似矛盾的結果：

```
[Iteration 0] Initial evaluation...
  Thinking Rate: 20.0%
  Tool Accuracy: 80.0%
  Score: 0.440

[Iteration 1] Generating improved prompt...
  Thinking Rate: 100.0%
  Tool Accuracy: 0.0%      ← 問題！
  Score: 0.600
  ✓ Accepting new prompt (improvement)

[Early Stop] Target thinking rate achieved!
```

**問題**：當 thinking rate 達到 100% 時，tool accuracy 卻變成 0%！

### 初步分析

評分函數是 `Score = 0.6 × ThinkingRate + 0.4 × ToolAccuracy`

- 舊 prompt: 0.6 × 0.20 + 0.4 × 0.80 = 0.44
- 新 prompt: 0.6 × 1.00 + 0.4 × 0.00 = 0.60

由於 0.60 > 0.44，優化器接受了新 prompt，但這個 prompt **完全丟失了工具調用能力**！

### 根本原因假設

當時有兩個可能的原因：
1. 優化器缺乏約束條件，允許「只有 thinking 沒有 tool call」的解
2. 解析器可能有問題，無法正確識別某些格式的 tool call

---

## 解決方案 1：添加 Tool Accuracy 約束條件

### 修改內容

在 `ollama_fd_optimizer.py` 中添加約束：

```python
# 新增配置
MIN_TOOL_ACCURACY = 0.4  # 最低 tool accuracy 門檻（約束條件）

# 在優化循環中添加檢查
if new_tool_accuracy < MIN_TOOL_ACCURACY:
    print(f"  ✗ Rejecting: tool accuracy {new_tool_accuracy:.0%} < {MIN_TOOL_ACCURACY:.0%} threshold")
    self.no_improvement_count += 1
    continue  # 跳過這個 prompt，不接受
```

### 結果

添加約束後再次運行：

```
[Iteration 1] Generating improved prompt...
  Thinking Rate: 100.0%
  Tool Accuracy: 0.0%
  ✗ Rejecting: tool accuracy 0% < 40% threshold

[Iteration 2] Generating improved prompt...
  Thinking Rate: 100.0%
  Tool Accuracy: 0.0%
  ✗ Rejecting: tool accuracy 0% < 40% threshold

... (連續 15 輪都被拒絕)
```

**所有迭代都被拒絕**，因為 tool accuracy 始終為 0%。

這表明問題不是缺少約束，而是**解析器本身有問題**。

---

## 問題 2：解析器無法識別 JSON 格式的 Tool Call

### 診斷過程

為了找出問題，我們手動測試不同的 prompt 並觀察實際輸出：

```python
# 測試代碼
prompt = ThinkingPrompt(
    system_prompt='FORMAT: First write <think>your reasoning</think>, then call the appropriate tool.',
    few_shot_examples=[]
)
result = runner.generate(prompt, query)
print(f'Output: {result.output}')
print(f'Has thinking: {result.has_thinking}')
print(f'Tool call: {result.tool_call_name}')
```

### 實際輸出

```
Output:  <think>I need to find the current weather for a specific location.
To do this, I will call the 'get_weather' function with 'Tokyo' as the city name.</think>

[{"name":"get_weather","arguments":{"city":"Tokyo"}}]

Has thinking: True
Tool call: None      ← 問題！模型確實輸出了 tool call，但解析器沒識別到
```

### 關鍵發現

模型**確實同時產生了 thinking 和 tool call**！

但 tool call 是以 **JSON 文本格式** 輸出的：
```json
[{"name":"get_weather","arguments":{"city":"Tokyo"}}]
```

而我們的解析器只識別這種格式：
```
[TOOL_CALL: get_weather({'city': 'Tokyo'})]
```

---

## 解決方案 2：多格式解析器

### 原始解析器（有問題）

```python
def _parse_output(self, output: str, expected_tool: str) -> EvaluationResult:
    # 只識別結構化 API 回應
    has_tool_call = "[TOOL_CALL:" in output
    tool_call_name = None

    if "[TOOL_CALL:" in output:
        try:
            start = output.find("[TOOL_CALL:") + len("[TOOL_CALL:")
            end = output.find("(", start)
            if end > start:
                tool_call_name = output[start:end].strip()
        except:
            pass
```

### 修復後的解析器

```python
def _parse_output(self, output: str, expected_tool: str) -> EvaluationResult:
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

    # 格式 2: JSON 文本格式 [{"name":"tool_name",...}]
    if not tool_call_name:
        import re
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
    return EvaluationResult(...)
```

---

## 輸出格式對比

### 格式 1：結構化 API 回應（Ollama tools API 原生）

當模型不產生 thinking 時，Ollama 返回結構化的 tool_calls：

```python
# API 回應
{
    "message": {
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "get_weather",
                    "arguments": {"city": "Tokyo"}
                }
            }
        ]
    }
}

# 我們的 OllamaRunner 轉換為：
"[TOOL_CALL: get_weather({'city': 'Tokyo'})]"
```

### 格式 2：JSON 文本格式（模型產生 thinking 時）

當模型被要求產生 thinking 時，它會在 content 中輸出 JSON：

```
<think>
I need to find the current weather for a specific location.
To do this, I will call the 'get_weather' function with 'Tokyo' as the city name.
</think>

[{"name":"get_weather","arguments":{"city":"Tokyo"}}]
```

**關鍵區別**：這時 tool call 不是通過 Ollama 的 `tool_calls` 字段返回，而是作為普通文本內容！

### 格式 3：函數調用文本格式（FunctionGemma 風格）

某些模型可能輸出類似 FunctionGemma 的格式：

```
<think>...</think>
call:get_weather{"city":"Tokyo"}
```

---

## 修復後的結果

### 修復前

```
Query: What's the current weather in Tokyo?
Output: <think>...</think>[{"name":"get_weather","arguments":{...}}]
Has thinking: True
Tool call: None        ← 錯誤！
Correct: False
```

### 修復後

```
Query: What's the current weather in Tokyo?
Output: <think>...</think>[{"name":"get_weather","arguments":{...}}]
Has thinking: True
Tool call: get_weather  ← 正確識別！
Correct: True
```

### 最終 FD 優化結果

```
[Iteration 0] Initial evaluation...
  Thinking Rate: 26.7%
  Tool Accuracy: 66.7%
  Score: 0.427

[Iteration 3] Generating improved prompt...
  Thinking Rate: 100.0%
  Tool Accuracy: 60.0%    ← 現在正確識別了！
  Score: 0.840
  ✓ Accepting new prompt (improvement)

[Early Stop] Target thinking rate achieved!
```

---

## 經驗總結

### 1. 調試流程

```
觀察異常 → 提出假設 → 添加約束驗證 → 發現真正問題 → 手動測試確認 → 修復根本原因
```

### 2. 關鍵教訓

| 教訓 | 說明 |
|------|------|
| 不要只看數字 | 0% tool accuracy 不代表模型沒輸出 tool call |
| 查看原始輸出 | 手動檢查實際輸出內容，不要只依賴解析結果 |
| 多格式支持 | LLM 輸出格式可能多樣，解析器需要靈活 |
| 約束 ≠ 解決方案 | 添加約束只能避免接受壞結果，不能修復根本問題 |

### 3. Ollama 工具調用的特殊行為

當 system prompt 要求模型先思考再行動時：
- 模型會在 `content` 字段輸出 thinking + JSON 文本格式的 tool call
- 而不是使用 Ollama 的結構化 `tool_calls` 字段

這是因為模型學到的「工具調用」行為模式來自訓練數據，而訓練數據中工具調用可能是以 JSON 文本形式出現的。

---

## 相關檔案

| 檔案 | 說明 |
|------|------|
| `ollama_fd_optimizer.py:361-420` | 多格式解析器實現 |
| `ollama_fd_optimizer.py:682-699` | Tool accuracy 約束條件 |
| `ollama_fd_results/optimization_log_*.json` | 優化歷史記錄 |

---

## 時間線

| 時間 | 事件 |
|------|------|
| 18:41 | 第一次運行，發現 100% thinking + 0% tool accuracy |
| 18:42 | 添加 MIN_TOOL_ACCURACY 約束 |
| 18:44 | 所有迭代被拒絕，懷疑解析器問題 |
| 18:45 | 手動測試確認模型輸出 JSON 格式 tool call |
| 18:47 | 實現多格式解析器 |
| 18:48 | 最終運行成功：100% thinking + 60% tool accuracy |
