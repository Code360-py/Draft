import logging
import warnings
import os
from transformers.utils import logging as hf_logging
from unsloth import FastLanguageModel
from transformers import TextStreamer

# -------------------------------
# 🔇 Suppress warnings
# -------------------------------
warnings.filterwarnings("ignore")

# Reduce Transformers + HF verbosity
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

hf_logging.set_verbosity_error()

# -------------------------------
# 📋 Configure logging
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------------
# 🚀 Imports after logging setup
# -------------------------------

logger.info("Loading model and tokenizer...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./finetuned_tinyllama",
    max_seq_length=256,
    dtype=None,
    load_in_4bit=True,
)

logger.info("Applying LoRA adapters...")

unsloth_model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=["q_proj", "v_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing=True,
)

# -------------------------------
# 🧠 Inference setup
# -------------------------------
instruction = "You are a helpful ai assistant"
input_text = "What is Python?"
logger.info("Formatting prompt...")

formatted_prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}"}],
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(
    [formatted_prompt],
    return_tensors="pt"
).to("cuda")

text_streamer = TextStreamer(tokenizer)

logger.info("Starting inference...")

_ = unsloth_model.generate(
    **inputs,
    streamer=text_streamer,
    max_new_tokens=64,
    use_cache=True
)

logger.info("Inference complete!")
