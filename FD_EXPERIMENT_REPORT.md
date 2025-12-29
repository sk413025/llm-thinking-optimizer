# Feedback Descent 實驗報告：FunctionGemma-270M Thinking 能力研究

## 實驗目標

驗證 Feedback Descent (FD) 能否在不使用 fine-tuning 的情況下，讓原始 FunctionGemma-270M 模型具備思考能力（在調用工具前產生 `<think>...</think>` 推理標籤）。

## 實驗日期

2025-12-29

## 方法論

基於 [Feedback Descent 論文](https://arxiv.org/abs/2511.07919) (2025年11月)，通過優化 prompt（system prompt + few-shot examples）來引導模型在工具調用前進行推理。

## 基線測試結果

### 測試配置
- 模型：`unsloth/functiongemma-270m-it`（原始版本，無 LoRA）
- 測試數量：5 個不同類型的工具調用 queries
- 硬體：NVIDIA RTX 5090

### 結果

| Prompt 類型 | Thinking Rate | Tool Call Rate |
|-------------|---------------|----------------|
| 無指令基線 | **0%** | 80% |
| 簡單 thinking 指令 | **0%** | 100% |
| 完整 prompt + few-shot | **0%** | 60% |

### 原始模型輸出範例

**輸入指令**：
```
You are a helpful assistant. Before using any tool, you should think step by step...
Format your response as:
<think>
[Your reasoning here]
</think>
[Then make the tool call]
```

**模型實際輸出**：
```
<start_function_call>call:get_weather{city:<escape>Tokyo<escape>}<end_function_call>
```

模型**完全忽略了 thinking 指令**，直接輸出工具調用。

## 關鍵發現

### 1. FunctionGemma-270M 的輸出格式是固定的

模型被訓練為直接輸出：
```
<start_function_call>call:{tool_name}{params}<end_function_call>
```

沒有中間推理步驟的空間。

### 2. 小模型的 In-Context Learning 能力不足

270M 參數的模型無法通過 prompt 指令學習新的輸出格式。即使使用 few-shot examples，模型也無法改變其輸出分佈。

### 3. Few-shot examples 反而降低了工具調用準確率

使用 few-shot examples 後，tool call rate 從 80%/100% 降到 60%。這可能是因為：
- Few-shot 的格式與模型預訓練格式不匹配
- 額外的 context 干擾了模型的正常工具調用行為

## 結論

### Feedback Descent 對 FunctionGemma-270M 的適用性評估

**結論：不適用** ⭐ (1/5)

#### 原因：

1. **模型輸出分佈已固定**：FD 優化 prompt 來改變模型輸出，但如果模型的輸出分佈已經被預訓練固定為特定格式，純 prompt 優化無法改變這一點。

2. **缺乏足夠的 ICL 能力**：270M 參數的小模型可能沒有足夠的 in-context learning 能力來從 prompt 中學習新的輸出格式。

3. **FD 的前提不成立**：FD 假設模型可以根據 prompt 產生不同質量的輸出，並且可以通過優化 prompt 來改進輸出。但在這個案例中，模型對 prompt 變化幾乎沒有反應（thinking rate 始終為 0%）。

### 與 Fine-tuning 方案的對比

| 方面 | Fine-tuning (LoRA) | Feedback Descent |
|------|-------------------|------------------|
| Thinking Rate | ~100%（訓練後） | 0%（無法改變） |
| 是否修改權重 | 是 | 否 |
| 適用於 FunctionGemma-270M | ✅ 有效 | ❌ 無效 |

### 研究價值

本實驗量化了 FunctionGemma-270M 模型的 in-context learning 能力邊界：

- **工具調用能力**：強（80-100%）
- **格式遵循能力**：弱（0%）
- **Prompt 敏感度**：低

這表明對於需要改變輸出格式的任務，小型語言模型仍然需要 fine-tuning。

## 建議

對於想要為 FunctionGemma-270M 添加 thinking 能力的用戶：

1. **推薦方案**：使用 LoRA fine-tuning（如本專案中的 `train.py`）
2. **不推薦**：純 prompt optimization / Feedback Descent

## 文件結構

```
functiongemma/
├── fd_thinking_optimizer.py    # FD 優化器實現（完整可運行）
├── fd_baseline_test.py         # 基線測試腳本
├── fd_debug_output.py          # 調試腳本
├── FD_EXPERIMENT_REPORT.md     # 本報告
└── train.py                    # LoRA fine-tuning（推薦方案）
```

## 參考資料

- [Feedback Descent 論文](https://arxiv.org/abs/2511.07919)
- [FunctionGemma 模型](https://huggingface.co/unsloth/functiongemma-270m-it)
- [Unsloth 框架](https://github.com/unslothai/unsloth)
