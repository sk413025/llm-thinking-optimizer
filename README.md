# FunctionGemma-270M Fine-tuning

使用 Unsloth + LoRA 對 FunctionGemma-270M 進行微調，訓練具有思考能力的 tool-calling 模型。

## 硬體需求

- **GPU**: NVIDIA GPU，建議 16GB+ VRAM
- **測試環境**: NVIDIA RTX 5090 (32GB VRAM)
- **磁碟空間**: ~10GB（模型 + 資料集快取）

## 環境安裝

### 1. 建立 Conda 環境

Unsloth 需要 Python 3.10-3.12，不支援 Python 3.13+。

```bash
conda create -n functiongemma python=3.11 -y
conda activate functiongemma
```

### 2. 安裝 PyTorch (CUDA 12.8)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 3. 安裝 Unsloth

```bash
pip install unsloth
```

### 4. 安裝其他依賴

```bash
pip install transformers==4.57.3
pip install --no-deps trl==0.24.0
pip install datasets sentencepiece protobuf huggingface_hub hf_transfer
```

### 5. 安裝 GGUF 轉換所需的系統套件

```bash
sudo apt-get install build-essential cmake libcurl4-openssl-dev -y
```

### 6. 驗證安裝

```bash
python -c "from unsloth import FastLanguageModel; print('Unsloth OK')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 測試環境版本參考

| 套件 | 版本 |
|------|------|
| Python | 3.11.14 |
| PyTorch | 2.9.1+cu128 |
| CUDA | 12.8 |
| Unsloth | 2025.12.9 |
| Transformers | 4.57.3 |
| TRL | 0.24.0 |
| Datasets | 4.3.0 |

## 訓練配置

| 參數 | 值 |
|------|-----|
| 基礎模型 | `unsloth/functiongemma-270m-it` |
| 精度 | 16-bit |
| LoRA Rank | 128 |
| LoRA Alpha | 256 |
| 序列長度 | 2048 |
| Batch Size | 2 |
| Gradient Accumulation | 4 |
| 有效 Batch Size | 8 |
| 學習率 | 2e-4 |
| 訓練步數 | 500 |
| 資料集 | `LLM360/TxT360-3efforts` (agent split, 50K samples) |

## 執行訓練

```bash
cd /home/sbplab/jiawei/functiongemma
conda activate functiongemma
python -u train.py 2>&1 | tee training.log
```

訓練完成後會自動：
1. 儲存 LoRA 適配器到 `./functiongemma-lora/`
2. 轉換為 GGUF 格式（Q8_0 量化）

## 單獨執行 GGUF 轉換

如果只需要轉換已訓練的模型：

```bash
python -u convert_to_gguf.py
```

## 輸出檔案

| 路徑 | 說明 |
|------|------|
| `./functiongemma-lora/` | LoRA 適配器（可用於繼續訓練或合併） |
| `./functiongemma-gguf/` | 合併後的完整模型 |
| `./functiongemma-270m-it.Q8_0.gguf` | Q8_0 量化的 GGUF 檔案（用於 LM Studio） |
| `./training.log` | 訓練日誌 |

## 在 LM Studio 使用

1. 開啟 LM Studio
2. 點擊 "My Models" → "Import"
3. 選擇 `functiongemma-270m-it.Q8_0.gguf` 檔案
4. 匯入後即可使用

## 已知問題與解決方案

### RTX 5090 (Blackwell 架構) 相容性

RTX 5090 使用 Blackwell 架構，需要停用 torch.compile：

```python
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
```

此設定已包含在 `train.py` 中。

### CUDA 記憶體不足

如果遇到 OOM 錯誤，可調整以下參數：
- 降低 `MAX_SEQ_LENGTH`（例如 2048 → 1024）
- 降低 `BATCH_SIZE`（例如 2 → 1）
- 增加 `GRADIENT_ACCUMULATION_STEPS` 以維持有效 batch size

## Feedback Descent 優化器

使用 Feedback Descent (FD) 算法優化 prompt，讓本地 LLM 在 tool-calling 前產生 `<think>` 推理標籤。

論文：[Feedback Descent (arXiv:2511.07919)](https://arxiv.org/abs/2511.07919)

### 環境要求

- Ollama 服務運行中
- 目標模型：`ollama pull mistral:7b`
- 評估模型：`ollama pull gpt-oss:20b`

### 執行

```bash
python ollama_fd_optimizer.py
```

### 配置參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| N_EVALUATION_SAMPLES | 5 | 評估樣本數 |
| MAX_ITERATIONS | 15 | 最大迭代輪數 |
| SAMPLES_PER_PROMPT | 3 | 每 prompt 採樣次數 |
| MIN_TOOL_ACCURACY | 0.4 | 最低 tool accuracy 門檻 |

### 輸出文件

| 路徑 | 說明 |
|------|------|
| `ollama_fd_results/best_prompt_*.json` | 優化後的最佳 prompt |
| `ollama_fd_results/optimization_log_*.json` | 完整優化歷史 |
| `ollama_fd_results/summary_*.json` | 實驗摘要 |

### 實驗結果

| 指標 | 初始值 | 最終值 | 變化 |
|------|--------|--------|------|
| Thinking Rate | 100% | 100% | 維持 |
| Tool Accuracy | 20% | 80% | +60pp (4x) |

## 檔案結構

```
functiongemma/
├── train.py                 # 主訓練腳本
├── convert_to_gguf.py       # GGUF 轉換腳本
├── ollama_fd_optimizer.py   # Feedback Descent 優化器
├── ollama_baseline_test.py  # Ollama 模型基線測試
├── data_utils.py            # 共用資料處理模組
├── README.md                # 本文件
├── training.log             # 訓練日誌
├── functiongemma-lora/      # 訓練輸出（LoRA 適配器）
├── functiongemma-gguf/      # 合併後模型
├── functiongemma-270m-it.Q8_0.gguf  # GGUF 量化模型
└── ollama_fd_results/       # FD 優化結果
```

## 參考資料

- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [FunctionGemma 模型](https://huggingface.co/unsloth/functiongemma-270m-it)
- [TxT360 資料集](https://huggingface.co/datasets/LLM360/TxT360-3efforts)
