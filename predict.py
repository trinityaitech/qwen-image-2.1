import os

# Enable ultra-fast Rust-based parallel downloads from Hugging Face
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from typing import Optional
from cog import BasePredictor, Input, Path
from diffusers import QwenImage21Pipeline
from PIL import Image
import torch

MODEL_ID = "Qwen/Qwen-Image-2.1"


class Predictor(BasePredictor):

  def setup(self):
    """Loads Qwen-Image-2.1 natively into L40S GPU VRAM."""
    print("Loading native Qwen-Image-2.1 Diffusers pipeline...")
    self.pipe = QwenImage21Pipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16
    ).to("cuda")
    print("Pipeline ready.")

  def predict(
      self,
      prompt: str = Input(
          description=(
              "Prompt or editing instruction. Refer to target as <image1> and"
              " reference as <image2>."
          ),
          default=(
              "Keep the character and pose in <image1> unchanged, put this"
              " light blue denim shirt from <image2> on the character."
          ),
      ),
      target_image: Optional[Path] = Input(
          description=(
              "Primary target image (<image1>). Leave blank for pure"
              " text-to-image."
          ),
          default=None,
      ),
      reference_image: Optional[Path] = Input(
          description=(
              "Optional secondary reference image (<image2>) for multi-image"
              " edits."
          ),
          default=None,
      ),
      num_inference_steps: int = Input(
          description="Inference steps (official default is 40)",
          ge=15,
          le=60,
          default=40,
      ),
      true_cfg_scale: float = Input(
          description="CFG scale (1.0 default for Qwen-Image-2.1)",
          ge=1.0,
          le=5.0,
          default=1.0,
      ),
      seed: int = Input(
          description="Random seed for reproducibility (-1 for random)",
          default=-1,
      ),
  ) -> Path:
    """Executes native PyTorch inference."""
    if seed < 0:
      seed = torch.randint(0, 2**32 - 1, (1,)).item()
    print(f"Running generation with seed: {seed}")

    generator = torch.Generator("cuda").manual_seed(seed)

    # Build image conditioning list
    images = []
    if target_image:
      images.append(Image.open(str(target_image)).convert("RGB"))
    if reference_image:
      images.append(Image.open(str(reference_image)).convert("RGB"))

    call_kwargs = {
        "prompt": prompt,
        "num_inference_steps": num_inference_steps,
        "true_cfg_scale": true_cfg_scale,
        "generator": generator,
    }

    # Pass conditioning images to pipeline if provided
    if images:
      call_kwargs["image"] = images if len(images) > 1 else images[0]

    # Run inference directly in GPU memory
    output_image = self.pipe(**call_kwargs).images[0]

    out_path = Path("/tmp/output.png")
    output_image.save(str(out_path))
    return out_path