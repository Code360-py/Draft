!pip install unsloth
!pip install --no-deps xformers trl peft accelerate bitsandbytes

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments


max_seq_length = 2048
dtype = None # None for auto detection
load_in_4bit = True # Use 4bit quantization to fit in Colab memory

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit", # You can swap this for a 'Dolphin' version
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Add LoRA adapters to make it trainable on a single GPU
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# 1. Load a specific clean file to avoid the "CastError" column mismatch
dataset = load_dataset(
    "cognitivecomputations/dolphin-2.9",
    data_files="openhermes200k_unfiltered.jsonl",
    split="train[:5000]"
)

# 2. Alpaca-style formatting
prompt_format = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

# 3. Updated formatting function
def formatting_prompts_func(examples):
    texts = []
    # OpenHermes uses 'conversations' but we need to extract the 'value' fields
    for conv in examples["conversations"]:
        if len(conv) >= 2:
            # Assumes index 0 is Human and index 1 is AI
            user_text = conv[0]['value']
            assistant_text = conv[1]['value']
            text = prompt_format.format(user_text, assistant_text) + tokenizer.eos_token
            texts.append(text)
    return { "text" : texts, }

# 4. Apply the mapping
dataset = dataset.map(formatting_prompts_func, batched = True)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 10,
        learning_rate = 2e-4,
        # Corrected the function name below:
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        output_dir = "outputs",
        optim = "adamw_8bit", # Optimization for Colab memory
    ),
)

trainer.train()

model.save_pretrained("uncensored_model_lora")
tokenizer.save_pretrained("uncensored_model_lora")
