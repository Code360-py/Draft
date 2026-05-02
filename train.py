!pip install -q unsloth
import os
import shutil
import torch, json
from unsloth import FastLanguageModel
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset

"""
print("Attempting to remove all directories in the current working directory. THIS IS DESTRUCTIVE!")
print("This will remove directories like .config, finetuned_tinyllama, model, etc.")

for item in os.listdir('.'):
  item_path = os.path.join('.', item)
  if os.path.isdir(item_path):
    try:
      print(f"Removing directory: {item_path}")
      shutil.rmtree(item_path)
      print(f"Successfully removed: {item_path}")
    except Exception as e:
      print(f"Failed to remove {item_path}: {e}")
  else:
    print(f"Skipping file: {item_path}")

"""
print("CUDA available: ", torch.cuda.is_available())

dataset = [
  {
    'instruction': 'Translate English to French',
    'input': 'Good morning!',
    'output': 'Bonjour!'
  },
  {
    'instruction': 'summarize',
    'input': 'Deep learning enables computers to learn complex patterns',
    'output': 'Deep learning lets computers learn patterns'
  }
]


os.makedirs('data', exist_ok=True)
with open('data/sample.jsonl', 'w') as f:
  for ex in dataset:
    f.write(json.dumps(ex) + '\n')
print('Sample data saved.')


model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
model, tokenizer = FastLanguageModel.from_pretrained(model_name=model_name, load_in_4bit=True)


model = FastLanguageModel.get_peft_model(
  model,
  r=8,
  lora_alpha=16,
  target_modules=['q_proj', 'v_proj'],
  lora_dropout=0.05,
  bias='none',
  use_gradient_checkpointing=True
)
dataset = load_dataset('json', data_files='data/sample.jsonl', split='train')


def tokenize(example):
  prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
  tokenized = tokenizer(
    prompt,
    truncation=True,
    padding="max_length",
    max_length=256
  )
  tokenized['labels'] = tokenized['input_ids'].copy()
  return tokenized
tokenized_dataset = dataset.map(tokenize)


training_args = TrainingArguments(
  output_dir='./tinyllama',
  per_device_train_batch_size=2,
  gradient_accumulation_steps=2,
  learning_rate=2e-4,
  logging_steps=5,
  num_train_epochs=20,
  bf16=torch.cuda.is_bf16_supported(),
  fp16=not torch.cuda.is_bf16_supported(),
  optim='adamw_torch',
  lr_scheduler_type='cosine',
  report_to='none'
)

data_collator = DataCollatorForLanguageModeling(
  tokenizer=tokenizer,
  mlm=False
)

trainer = Trainer(
  model=model,
  train_dataset=tokenized_dataset,
  args=training_args,
  data_collator=data_collator
)

trainer.train()
print('Training complete!')


model.save_pretrained("./finetuned_tinyllama")
tokenizer.save_pretrained("./finetuned_tinyllama")

# model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
# model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
