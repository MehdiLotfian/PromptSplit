import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, DPMSolverMultistepScheduler
from transformers import AutoImageProcessor, AutoModel
import time
import logging
import os
from pathlib import Path

# --- ⚙️ Configuration for High-Quality SDXL ---

# 1. SDXL requires a base and a refiner model for the best results.
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
refiner_model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"  # Loaded but not used in this script; can be integrated if needed

# 2. More descriptive prompts produce much better images.
# Note: Original prompts dictionary is overridden below with MS-COCO
prompts = {
    "Carpenter": "Photorealistic portrait of a carpenter in a dusty, sun-drenched workshop, sawdust in the air, focused on carving a detailed piece of oak wood. High detail, sharp focus, award-winning photography.",
    "Teacher": "Warm, candid photo of an enthusiastic teacher in a vibrant, modern classroom, surrounded by curious high school students. Natural light from a large window, cinematic style, shallow depth of field.",
    "Driver": "Intense close-up of a driver in a car, focused and determined. motion blur, dramatic lighting.",
    "Dancer": "Dynamic action shot of a dancer on a grand stage, dramatic cinematic lighting, masterpiece, 8k resolution, intricate details.",
    "Chef": "Intense close-up of a chef meticulously garnishing a gourmet dish in a busy, high-end restaurant kitchen. Steam rising, focused expression, stainless steel reflections, Michelin-star quality, dramatic lighting.",
    "Firefighter": "Heroic portrait of a firefighter, against a blazing inferno. Embers flying, intense heat visible, sense of courage and duty, cinematic, photorealistic.",
    "Police Officer": "Candid street portrait of a police officer on patrol, observing the surroundings with a vigilant expression, shallow depth of field, moody atmosphere.",
    "Judge": "Authoritative portrait of a senior judge sitting in a grand, wood-paneled courtroom, gavel in hand. Stern but fair expression, soft light from a large window, detailed robes, sense of wisdom and justice.",
    "Nurse": "Photorealistic portrait of a nurse in a bright, clean hospital room. Warm and reassuring atmosphere, soft morning light, highly detailed, realistic."
}

prompts_path = Path(__file__).parent.joinpath("ms_coco_experiments/GigaGAN/GigaGAN_t2i_coco256_rep/captions.txt")
if not prompts_path.exists():
    raise FileNotFoundError(f"Prompts file not found: {prompts_path} — create the captions file at this path or update the script to point to your captions list")
with prompts_path.open('r', encoding='utf-8') as f:
    engineer_prompts = [line.strip() for line in f if line.strip()]
prompts = {f"MS_COCO_{idx+1}": txt for idx, txt in enumerate(engineer_prompts)}

# 3. An effective negative prompt for SDXL.
negative_prompt = "cartoon, sketch, painting, cgi, 3d, render, poor quality, distorted, blurry, deformed, ugly, bad anatomy"

# 4. Set the number of images to generate per prompt.
num_images_per_prompt = 1  # Increase this to batch-generate (e.g., 4) for more efficiency if VRAM allows

# --- Model Loading ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("sdxl_generation_mountain.log"),
        logging.StreamHandler()
    ]
)

# --- Load Base Model ---
logging.info(f"Loading SDXL base model: {base_model_id}")
base_pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)

# Use the fp16-fixed VAE to avoid dtype mismatches in decoding
logging.info("Loading fp16-fixed VAE...")
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16,
    use_safetensors=True
)
base_pipe.vae = vae

# NEW: Enable VAEG slicing for faster/more memory-efficient decoding (especially useful with offload)
base_pipe.vae.enable_slicing()
base_pipe.vae.enable_tiling()

# NEW: Switch to a faster scheduler (DPM++ 2M Karras is great for speed/quality balance)
base_pipe.scheduler = DPMSolverMultistepScheduler.from_config(base_pipe.scheduler.config)

# Move pipeline to CUDA (no offload needed for RTX 3090 with 24GB VRAM)
base_pipe.to("cuda")

base_pipe.enable_xformers_memory_efficient_attention()

# OPTIONAL: If PyTorch >=2.0, compile for speedup (uncomment if available; test first as it may increase initial load time)
# base_pipe.unet = torch.compile(base_pipe.unet, mode="reduce-overhead", fullgraph=True)

logging.info("Base model loaded successfully.")

# --- Load DINOv2 Model for Embeddings ---
logging.info("Loading DINOv2 model for embeddings...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
dino_model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
logging.info("DINOv2 model loaded successfully.")

total_time = 0

# --- 🚀 Image Generation ---
j = -1
name = "mountain_photo_realistic"
output_dir = f"sdxl_images/MS_COCO_images_4"
os.makedirs(output_dir, exist_ok=True)
prompt_embeddings = []  # List to collect embeddings for the current prompt

for _, prompt_text in prompts.items():
    j += 1
    for i in range(num_images_per_prompt):
        generator = torch.Generator(device=device).manual_seed(j + 3)

        logging.info(f"Generating image {(j + 1)}/{len(prompts) * num_images_per_prompt} for prompt: '{prompt_text}' ---")
        start_time = time.time()

        output_filename = f"{output_dir}/{j}.png"

        # --- High-Quality SDXL Workflow ---
        # NEW: Reduced steps for speedup (adjust to 20-30; quality vs speed tradeoff)
        base_out = base_pipe(
            prompt=prompt_text,
            negative_prompt=negative_prompt,
            num_inference_steps=50,  # ← Reduced from 50
            guidance_scale=7.5,
            generator=generator,
            output_type="pil"  # Directly output PIL image
        )
        image = base_out.images[0]

        end_time = time.time()
        logging.info(f"Image generated in {end_time - start_time:.2f} seconds.")

        # --- Save the Image and Collect Embedding ---
        try:
            image.save(output_filename)
            logging.info(f"Image successfully saved as '{output_filename}'")

            # --- Generate and Collect DINOv2 Embedding ---
            inputs = dino_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = dino_model(**inputs)
            image_embedding = outputs.last_hidden_state[:, 0, :].cpu()
            prompt_embeddings.append(image_embedding)

        except Exception as e:
            logging.error(f"Error saving image or generating embedding: {e}")

        total_time += end_time - start_time

# --- Save all embeddings ---
if prompt_embeddings:
    embedding_filename = f"{output_dir}/embeddings.pt"
    torch.save(torch.cat(prompt_embeddings, dim=0), embedding_filename)
    logging.info(f"All embeddings for prompt '{name}' saved to '{embedding_filename}'")

logging.info(f"Total time taken for image generation: {total_time:.2f} seconds.")