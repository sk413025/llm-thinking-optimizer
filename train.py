"""
FunctionGemma-270M Fine-tuning Script
Fine-tunes FunctionGemma for tool-calling with thinking capability using Unsloth + LoRA.
"""

import os
# Disable torch.compile for RTX 5090 (Blackwell) compatibility
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import json
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig

# =============================================================================
# Configuration
# =============================================================================
MODEL_NAME = "unsloth/functiongemma-270m-it"
MAX_SEQ_LENGTH = 2048  # Reduced from 4096 to save memory
LORA_RANK = 128
LORA_ALPHA = 256
BATCH_SIZE = 2  # Reduced from 4 to save memory
GRADIENT_ACCUMULATION_STEPS = 4  # Increased to maintain effective batch size
MAX_STEPS = 500
LEARNING_RATE = 2e-4
OUTPUT_DIR = "outputs"
DATASET_SIZE = 50000

# =============================================================================
# Load Model
# =============================================================================
print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,
    load_in_8bit=False,
    load_in_16bit=True,
    full_finetuning=False,
)

# =============================================================================
# Add LoRA Adapters
# =============================================================================
print("Adding LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# =============================================================================
# Data Preparation Helpers
# =============================================================================
THINK_TAG_OPEN = "<think>"
THINK_TAG_CLOSE = "</think>"

def prepare_messages_and_tools(example):
    """Prepare messages and tools from raw dataset example."""
    raw = json.loads(example["messages"])
    msgs = [dict(m) for m in raw]

    # Extract tools
    tools_raw = []
    if msgs and isinstance(msgs[0], dict):
        tlist = msgs[0].get("tools")
        if isinstance(tlist, list) and tlist:
            tools_raw = tlist
            msgs[0].pop("tools", None)

    # Merge assistant["think"] into ["content"]
    THINK_KEYS = ["think", "think_fast", "think_faster"]
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

    # Normalize tool_calls
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

    # Build map from tool_call_id -> function name
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

    # Ensure tool response messages have a 'name'
    for m in msgs:
        if m.get("role") == "tool":
            if not m.get("name"):
                tc_id = m.get("tool_call_id")
                inferred = id_to_name.get(tc_id) if tc_id else None
                m["name"] = inferred or "unknown_tool"

    # Normalize tool schemas
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

    # Delete empty system message
    first_message = msgs[0]
    if first_message["role"] == "system" and "content" not in first_message:
        msgs.pop(0)

    return msgs, adapted_tools


def format_example(example):
    """Format a single example for training."""
    messages, tools = prepare_messages_and_tools(example)

    if messages is None or len(messages) == 0:
        return {"text": None}

    chat_str = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=False,
        tokenize=False,
    ).removeprefix("<bos>")

    return {"text": chat_str}


# =============================================================================
# Load and Process Dataset
# =============================================================================
print(f"Loading dataset (first {DATASET_SIZE} examples)...")
dataset = load_dataset("LLM360/TxT360-3efforts", name="agent", split="medium", streaming=True)
dataset = Dataset.from_list(list(dataset.take(DATASET_SIZE)))

print("Processing dataset...")
train_dataset = dataset.map(format_example)
train_dataset = train_dataset.filter(lambda x: x["text"] is not None)
print(f"Dataset size after filtering: {len(train_dataset)}")

# =============================================================================
# Setup Trainer
# =============================================================================
print("Setting up trainer...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=None,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=10,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_DIR,
        report_to="none",
    ),
)

# Train only on model responses
trainer = train_on_responses_only(
    trainer,
    instruction_part="<start_of_turn>user\n",
    response_part="<start_of_turn>model\n",
)

# =============================================================================
# Show GPU Stats
# =============================================================================
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU: {gpu_stats.name}, Max memory: {max_memory} GB")
print(f"Reserved memory before training: {start_gpu_memory} GB")

# =============================================================================
# Train
# =============================================================================
print("\nStarting training...")
trainer_stats = trainer.train()

# =============================================================================
# Show Training Stats
# =============================================================================
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"\nTraining complete!")
print(f"Peak reserved memory: {used_memory} GB ({used_percentage}% of max)")
print(f"Memory used for LoRA training: {used_memory_for_lora} GB ({lora_percentage}% of max)")

# =============================================================================
# Save Model
# =============================================================================
print("\nSaving LoRA adapters...")
model.save_pretrained("functiongemma-lora")
tokenizer.save_pretrained("functiongemma-lora")
print("Model saved to ./functiongemma-lora")

# =============================================================================
# Convert to GGUF for LM Studio
# =============================================================================
print("\nConverting to GGUF format (Q8_0)...")
model.save_pretrained_gguf(
    "functiongemma-gguf",
    tokenizer,
    quantization_method="Q8_0",  # Options: Q8_0, BF16, F16
)
print("GGUF model saved to ./functiongemma-gguf")

print("\nDone! You can now use the GGUF file in LM Studio.")
