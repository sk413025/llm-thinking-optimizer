"""
Convert trained FunctionGemma LoRA model to GGUF format.
"""

import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from unsloth import FastLanguageModel

LORA_PATH = "functiongemma-lora"
MAX_SEQ_LENGTH = 2048

print(f"Loading trained model from {LORA_PATH}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=LORA_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,
    load_in_8bit=False,
    load_in_16bit=True,
)

print("\nConverting to GGUF format (Q8_0)...")
model.save_pretrained_gguf(
    "functiongemma-gguf",
    tokenizer,
    quantization_method="Q8_0",
)
print("GGUF model saved to ./functiongemma-gguf")

print("\nDone! You can now use the GGUF file in LM Studio.")
