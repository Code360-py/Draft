!pip install -q diffusers transformers accelerate bitsandbytes ftfy
!pip install -q torchvision  # For image processing
import os
import torch

# Set your paths
IMAGE_DIR = "/content/train_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# Tip: Upload your images to this folder now using the Colab file sidebar.

from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

model_id = "runwayml/stable-diffusion-v1-5"

# Load tokenizer and text encoder (The "Brain" of the prompt)
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float16).to("cuda")

# Load the UNet (The "Artist" that draws the pixels)
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float16).to("cuda")

# Force download the script again to be sure
!wget -q https://raw.githubusercontent.com/huggingface/diffusers/v0.27.2/examples/lora/train_text_to_image_lora.py -O train_text_to_image_lora.py

!python train_text_to_image_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="/content/train_images" \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=500 \
  --learning_rate=1e-4 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir="lora_model" \
  --seed=42

from diffusers import StableDiffusionPipeline
import torch
import os

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to("cuda")

# Look inside the output folder for the weight file
# It's usually named 'pytorch_lora_weights.safetensors'
lora_path = "/content/lora_model/pytorch_lora_weights.safetensors"

if os.path.exists(lora_path):
    pipe.load_lora_weights(lora_path)
    print("LoRA loaded successfully!")
else:
    print("Weight file not found. Did the training finish?")

# CRITICAL: Bypassing the filter
pipe.safety_checker = None
pipe.requires_safety_checker = False

prompt = "nude indian aunty"
image = pipe(prompt, num_inference_steps=30).images[0]
image.save("result.png")
image
