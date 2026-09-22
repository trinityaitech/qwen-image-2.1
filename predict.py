import json
import os
import random
import shutil
import subprocess
import time
import urllib.request
import uuid
from typing import Optional
from cog import BaseModel, BasePredictor, Input, Path
import requests
import websocket

COMFY_HOST = "127.0.0.1:8188"
COMFY_PYTHON = "/root/comfy_env/bin/python"

# Asset Paths & URLs
GGUF_URL = "https://huggingface.co/Abiray/Qwen-Image-2.1-GGUF/resolve/main/qwen_image_2.1_Q6_K.gguf"
GGUF_PATH = (
    "/root/ComfyUI/models/diffusion_models/qwen_image_2.1_Q6_K.gguf"
)

TEXT_ENC_URL = "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/text_encoders/qwen3vl_8b_int8_convrot.safetensors"
TEXT_ENC_PATH = (
    "/root/ComfyUI/models/text_encoders/qwen3vl_8b_int8_convrot.safetensors"
)

VAE_URL = "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors"
VAE_PATH = "/root/ComfyUI/models/vae/qwen_image_2.1_vae_bf16.safetensors"

DEFAULT_PROMPT = (
    "Keep the character and pose in <image1> unchanged, put this light blue"
    " denim shirt from <image2> on the character, preserve the original facial"
    " features, hair, body shape and pose, the denim shirt fits naturally on"
    " body, realistic denim fabric texture, natural clothing folds, keep the"
    " original background and lighting, high fashion editorial photography,"
    " sharp details."
)


class Output(BaseModel):
  image: Path


class Predictor(BasePredictor):

  def setup(self):
    """Downloads Qwen-Image-2.1 checkpoints dynamically and boots isolated ComfyUI server."""
    os.makedirs(os.path.dirname(GGUF_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(TEXT_ENC_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(VAE_PATH), exist_ok=True)
    os.makedirs("/root/ComfyUI/input", exist_ok=True)
    os.makedirs("/root/ComfyUI/output", exist_ok=True)

    # 1. Download GGUF Diffusion Model (~5.88 GB)
    if not os.path.exists(GGUF_PATH) or os.path.getsize(GGUF_PATH) < 5_000_000_000:
      print("Downloading Qwen-Image-2.1 GGUF Model (~5.88 GB)...")
      subprocess.run(
          ["wget", "-c", "--progress=dot:giga", GGUF_URL, "-O", GGUF_PATH],
          check=True,
      )

    # 2. Download Text Encoder (~8.71 GB)
    if (
        not os.path.exists(TEXT_ENC_PATH)
        or os.path.getsize(TEXT_ENC_PATH) < 7_000_000_000
    ):
      print("Downloading Qwen-Image-2.1 Text Encoder (~8.71 GB)...")
      subprocess.run(
          [
              "wget",
              "-c",
              "--progress=dot:giga",
              TEXT_ENC_URL,
              "-O",
              TEXT_ENC_PATH,
          ],
          check=True,
      )

    # 3. Download VAE (~644 MB)
    if not os.path.exists(VAE_PATH) or os.path.getsize(VAE_PATH) < 500_000_000:
      print("Downloading Qwen-Image-2.1 VAE (~644 MB)...")
      subprocess.run(
          ["wget", "-c", "--progress=dot:giga", VAE_URL, "-O", VAE_PATH],
          check=True,
      )

    # 4. Start ComfyUI Studio server in isolated virtualenv
    print("Starting ComfyUI server in isolated virtualenv...")
    cmd = [
        COMFY_PYTHON,
        "/root/ComfyUI/main.py",
        "--listen",
        "127.0.0.1",
        "--port",
        "8188",
        "--fast",
        "--verbose",
        "INFO",
    ]
    self.comfy_process = subprocess.Popen(cmd)

    ready = False
    for _ in range(60):
      if self.comfy_process.poll() is not None:
        raise RuntimeError(
            f"ComfyUI process exited prematurely with code"
            f" {self.comfy_process.returncode}"
        )
      try:
        res = requests.get(f"http://{COMFY_HOST}/system_stats", timeout=1)
        if res.status_code == 200:
          ready = True
          break
      except Exception:
        time.sleep(1)

    if not ready:
      raise RuntimeError("ComfyUI failed to start within 60 seconds.")
    print("ComfyUI Studio server is online.")

  def predict(
      self,
      target_image: Path = Input(
          description=(
              "Primary target image to be edited (referenced as <image1> in"
              " prompt)"
          )
      ),
      prompt_text: str = Input(
          description=(
              "Editing instruction. Refer to target as <image1> and optional"
              " reference as <image2>."
          ),
          default=DEFAULT_PROMPT,
      ),
      reference_image: Optional[Path] = Input(
          description=(
              "Optional secondary reference image (e.g. clothing, object, style"
              " - referenced as <image2>)"
          ),
          default=None,
      ),
      steps: int = Input(
          description="Number of inference steps (20-30 recommended)",
          ge=10,
          le=50,
          default=25,
      ),
      cfg: float = Input(
          description=(
              "Guidance scale (1.0 recommended by Qwen official pipeline)"
          ),
          ge=1.0,
          le=4.0,
          default=1.0,
      ),
      resolution: int = Input(
          description=(
              "Pixel budget resolution (1024 standard; 0 to preserve native"
              " image resolution)"
          ),
          default=1024,
      ),
      seed: int = Input(
          description="Random seed for reproducibility (-1 for random)",
          default=-1,
      ),
  ) -> Output:
    """Executes Qwen-Image-2.1 image editing workflow."""
    if seed < 0:
      seed = random.randint(0, 2**32 - 1)
    print(f"Executing Qwen-Image-2.1 run with seed: {seed}")

    with open("workflow_api.json", "r", encoding="utf-8") as f:
      prompt = json.load(f)

    # 1. Process Target Image (<image1>)
    target_filename = f"target_{uuid.uuid4().hex[:8]}.png"
    target_dest = os.path.join("/root/ComfyUI/input", target_filename)
    shutil.copyfile(str(target_image), target_dest)
    prompt["4"]["inputs"]["image"] = target_filename

    # 2. Process Reference Image (<image2>)
    if reference_image and os.path.exists(str(reference_image)):
      ref_filename = f"ref_{uuid.uuid4().hex[:8]}.png"
      ref_dest = os.path.join("/root/ComfyUI/input", ref_filename)
      shutil.copyfile(str(reference_image), ref_dest)
      prompt["5"]["inputs"]["image"] = ref_filename
    else:
      # If no reference image provided, disconnect image_2 and drop node 5
      if "image_2" in prompt["6"]["inputs"]:
        del prompt["6"]["inputs"]["image_2"]
      if "5" in prompt:
        del prompt["5"]

    # 3. Inject User Parameters into Native Nodes
    prompt["6"]["inputs"]["prompt"] = prompt_text
    prompt["6"]["inputs"]["resolution"] = resolution

    prompt["8"]["inputs"]["seed"] = seed
    prompt["8"]["inputs"]["steps"] = steps
    prompt["8"]["inputs"]["cfg"] = cfg

    # 4. Submit Job to ComfyUI via WebSocket
    client_id = str(uuid.uuid4())
    ws = websocket.WebSocket()
    ws.connect(f"ws://{COMFY_HOST}/ws?clientId={client_id}")

    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode("utf-8")
    req = urllib.request.Request(f"http://{COMFY_HOST}/prompt", data=data)
    response = json.loads(urllib.request.urlopen(req).read())
    prompt_id = response["prompt_id"]

    while True:
      out = ws.recv()
      if isinstance(out, str):
        message = json.loads(out)
        if message["type"] == "executing":
          data = message["data"]
          if data["node"] is None and data["prompt_id"] == prompt_id:
            break
      else:
        continue
    ws.close()

    # 5. Retrieve Generated Image Output
    output_root = "/root/ComfyUI/output"
    found_images = []

    for root, _, files in os.walk(output_root):
      for f in files:
        if f.endswith((".png", ".jpg", ".jpeg", ".webp")):
          found_images.append(os.path.join(root, f))

    if not found_images:
      raise RuntimeError("No output image was produced by ComfyUI.")

    latest_image = max(found_images, key=os.path.getmtime)
    return Output(image=Path(latest_image))