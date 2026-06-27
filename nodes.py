from __future__ import annotations

import gc
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

import folder_paths
import comfy.utils


MODEL_REPO = "nvidia/LocateAnything-3B"
MODEL_ROOT = Path(folder_paths.models_dir) / "LocateAnything"
MODEL_ROOT.mkdir(parents=True, exist_ok=True)
folder_paths.add_model_folder_path("locateanything", str(MODEL_ROOT))

BOX_PATTERN = re.compile(
    r"(?:<ref>([^<]*)</ref>\s*)?<box><(\d+)><(\d+)><(\d+)><(\d+)></box>",
)
POINT_PATTERN = re.compile(
    r"(?:<ref>([^<]*)</ref>\s*)?<box><(\d+)><(\d+)></box>",
)

TASKS = [
    "ground_multi",
    "ground_single",
    "detect",
    "ground_text",
    "detect_text",
    "gui_box",
    "gui_point",
    "point",
    "custom",
]


def _safe_repo_folder(repo_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "--", repo_id).strip("-") or "model"


def _resolve_model_source(model_source: str, download_model: bool) -> str:
    source = model_source.strip()
    if not source:
        raise ValueError("model_source cannot be empty")

    local_path = Path(source).expanduser()
    if local_path.exists():
        return str(local_path.resolve())

    comfy_path = MODEL_ROOT / _safe_repo_folder(source)
    if (comfy_path / "config.json").exists():
        return str(comfy_path)

    if not download_model:
        raise FileNotFoundError(
            f"LocateAnything model not found at {comfy_path}. Enable download_model "
            f"or place the Hugging Face snapshot there."
        )

    from huggingface_hub import snapshot_download

    print(f"[LocateAnything] Downloading {source} to {comfy_path}")
    snapshot_download(
        repo_id=source,
        local_dir=str(comfy_path),
        ignore_patterns=[
            "assets/*",
            "all_results.json",
            "trainer_state.json",
            "training_args.bin",
        ],
    )
    return str(comfy_path)


def _resolve_device(device: str) -> torch.device:
    if device != "auto":
        resolved = torch.device(device)
    else:
        try:
            import comfy.model_management as model_management

            resolved = model_management.get_torch_device()
        except Exception:
            if torch.cuda.is_available():
                resolved = torch.device("cuda")
            elif torch.backends.mps.is_available():
                resolved = torch.device("mps")
            else:
                resolved = torch.device("cpu")

    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected, but torch.cuda.is_available() is False")
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was selected, but torch.backends.mps.is_available() is False")
    return resolved


def _resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def _load_batch_runtime(
    model_path: str,
    attn: str,
    vision_attn: str,
    scheduler: str,
    group_size: int,
    strict_attn: bool,
):
    os.environ["LA_FLASH_MODEL"] = model_path
    os.environ["LA_FLASH_ATTN"] = attn
    os.environ["LA_FLASH_VISION_ATTN"] = vision_attn
    os.environ["LA_FLASH_HYBRID_SCHEDULER"] = scheduler
    os.environ["LA_FLASH_HYBRID_GROUP_SIZE"] = str(group_size)
    if strict_attn:
        os.environ["LA_FLASH_STRICT_ATTN"] = "1"
    else:
        os.environ.pop("LA_FLASH_STRICT_ATTN", None)

    if model_path not in sys.path:
        sys.path.insert(0, model_path)

    try:
        from batch_utils import generate_batch_hybrid, get_last_hybrid_stats, load
    except ImportError as exc:
        raise ImportError(
            "Batch inference requires the Hugging Face release files `batch_utils/` and "
            "`kernel_utils/` from nvidia/LocateAnything-3B. Update the local snapshot or "
            "disable use_batch_runtime."
        ) from exc

    tokenizer, processor, model = load()
    return tokenizer, processor, model, generate_batch_hybrid, get_last_hybrid_stats


def _replace_once(content: str, old: str, new: str, path: Path) -> tuple[str, bool]:
    if old not in content:
        return content, False
    return content.replace(old, new, 1), True


def _apply_model_compat_patches(model_path: str) -> None:
    root = Path(model_path)
    if not root.is_dir():
        return

    replacements = {
        "configuration_locateanything.py": [
            (
                """        if text_config['architectures'][0] == 'Qwen2ForCausalLM':\n            self.text_config = Qwen2Config(**text_config)\n        elif text_config['architectures'][0] == 'Qwen3ForCausalLM':\n            self.text_config = Qwen3Config(**text_config)\n        else:\n            raise ValueError('Unsupported architecture: {}. Only Qwen2ForCausalLM and Qwen3ForCausalLM are supported.'.format(text_config['architectures'][0]))\n        self.use_backbone_lora = use_backbone_lora\n""",
                """        if text_config['architectures'][0] == 'Qwen2ForCausalLM':\n            self.text_config = Qwen2Config(**text_config)\n        elif text_config['architectures'][0] == 'Qwen3ForCausalLM':\n            self.text_config = Qwen3Config(**text_config)\n        else:\n            raise ValueError('Unsupported architecture: {}. Only Qwen2ForCausalLM and Qwen3ForCausalLM are supported.'.format(text_config['architectures'][0]))\n        if 'rope_theta' in text_config and not hasattr(self.text_config, 'rope_theta'):\n            self.text_config.rope_theta = text_config['rope_theta']\n        self.use_backbone_lora = use_backbone_lora\n""",
            ),
        ],
        "modeling_qwen2.py": [
            (
                """    def _check_and_adjust_attn_implementation(self, attn_implementation, is_init_check=False):\n        if attn_implementation == "magi":\n            return "magi"\n        return super()._check_and_adjust_attn_implementation(attn_implementation, is_init_check)\n""",
                """    def _check_and_adjust_attn_implementation(\n        self,\n        attn_implementation,\n        is_init_check=False,\n        allow_all_kernels=False,\n    ):\n        if attn_implementation == "magi":\n            return "magi"\n        return super()._check_and_adjust_attn_implementation(\n            attn_implementation,\n            is_init_check,\n            allow_all_kernels=allow_all_kernels,\n        )\n""",
            ),
            (
                """        if use_cache:\n            use_legacy_cache = not isinstance(past_key_values, Cache)\n            if use_legacy_cache:\n                if past_key_values is None:\n                    past_key_values = DynamicCache()\n                else:\n                    past_key_values = DynamicCache.from_legacy_cache(past_key_values)\n            past_key_values_length = past_key_values.get_seq_length()\n""",
                """        if use_cache:\n            use_legacy_cache = not isinstance(past_key_values, Cache)\n            if use_legacy_cache:\n                if past_key_values is None:\n                    past_key_values = DynamicCache()\n                else:\n                    if hasattr(DynamicCache, "from_legacy_cache"):\n                        past_key_values = DynamicCache.from_legacy_cache(past_key_values)\n                    else:\n                        past_key_values = DynamicCache(past_key_values)\n            past_key_values_length = past_key_values.get_seq_length()\n""",
            ),
            (
                """        if use_cache:\n            use_legacy_cache = not isinstance(past_key_values, Cache)\n            if use_legacy_cache:\n                if past_key_values is None:\n                    past_key_values = DynamicCache()\n                else:\n                    if hasattr(DynamicCache, "from_legacy_cache"):\n                        past_key_values = DynamicCache.from_legacy_cache(past_key_values)\n                    else:\n                        past_key_values = DynamicCache(past_key_values)\n                        past_key_values = DynamicCache(past_key_values)\n                    else:\n                        past_key_values = DynamicCache(past_key_values)\n                    else:\n                        past_key_values = DynamicCache(past_key_values)\n            past_key_values_length = past_key_values.get_seq_length()\n""",
                """        if use_cache:\n            use_legacy_cache = not isinstance(past_key_values, Cache)\n            if use_legacy_cache:\n                if past_key_values is None:\n                    past_key_values = DynamicCache()\n                else:\n                    if hasattr(DynamicCache, "from_legacy_cache"):\n                        past_key_values = DynamicCache.from_legacy_cache(past_key_values)\n                    else:\n                        past_key_values = DynamicCache(past_key_values)\n            past_key_values_length = past_key_values.get_seq_length()\n""",
            ),
            (
                """            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache\n""",
                """            if use_legacy_cache:\n                if hasattr(next_decoder_cache, "to_legacy_cache"):\n                    next_cache = next_decoder_cache.to_legacy_cache()\n                else:\n                    next_cache = tuple((layer.keys, layer.values) for layer in next_decoder_cache.layers)\n            else:\n                next_cache = next_decoder_cache\n""",
            ),
            (
                """class Qwen2ForCausalLM(Qwen2PreTrainedModel):\n    _tied_weights_keys = ["lm_head.weight"]\n""",
                """class Qwen2ForCausalLM(Qwen2PreTrainedModel):\n    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}\n""",
            ),
        ],
        "modeling_locateanything.py": [
            (
                """    def _check_and_adjust_attn_implementation(self, attn_implementation, is_init_check=False):\n        if attn_implementation == "magi":\n            return "magi"\n        return super()._check_and_adjust_attn_implementation(attn_implementation, is_init_check)\n""",
                """    def _check_and_adjust_attn_implementation(\n        self,\n        attn_implementation,\n        is_init_check=False,\n        allow_all_kernels=False,\n    ):\n        if attn_implementation == "magi":\n            return "magi"\n        return super()._check_and_adjust_attn_implementation(\n            attn_implementation,\n            is_init_check,\n            allow_all_kernels=allow_all_kernels,\n        )\n""",
            ),
            (
                """        if 'Qwen3' in arch:\n            self._no_split_modules = ["Qwen3DecoderLayer"]\n        else:\n            self._no_split_modules = ["Qwen2DecoderLayer"]\n\n        \n""",
                """        if 'Qwen3' in arch:\n            self._no_split_modules = ["Qwen3DecoderLayer"]\n        else:\n            self._no_split_modules = ["Qwen2DecoderLayer"]\n\n        self.post_init()\n\n""",
            ),
        ],
    }

    patched_files = []
    for filename, file_replacements in replacements.items():
        path = root / filename
        if not path.exists():
            continue
        content = path.read_text()
        original = content
        for old, new in file_replacements:
            content, _ = _replace_once(content, old, new, path)
        if content != original:
            path.write_text(content)
            patched_files.append(filename)

    if patched_files:
        print(f"[LocateAnything] Applied transformers compatibility patches to {', '.join(patched_files)}")


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = image.detach().cpu().clamp(0, 1).numpy()
    return Image.fromarray(np.rint(array * 255.0).astype(np.uint8), mode="RGB")


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array)


def _clamp_coordinate(value: int) -> int:
    return max(0, min(1000, value))


def parse_locations(answer: str, width: int, height: int) -> dict[str, Any]:
    boxes = []
    for match in BOX_PATTERN.finditer(answer):
        label, x1, y1, x2, y2 = match.groups()
        normalized = [_clamp_coordinate(int(value)) for value in (x1, y1, x2, y2)]
        px = [
            normalized[0] / 1000 * width,
            normalized[1] / 1000 * height,
            normalized[2] / 1000 * width,
            normalized[3] / 1000 * height,
        ]
        boxes.append(
            {
                "label": label.strip() if label else "",
                "normalized": normalized,
                "pixel": px,
            }
        )

    points = []
    for match in POINT_PATTERN.finditer(answer):
        label, x, y = match.groups()
        normalized = [_clamp_coordinate(int(value)) for value in (x, y)]
        px = [
            normalized[0] / 1000 * width,
            normalized[1] / 1000 * height,
        ]
        points.append(
            {
                "label": label.strip() if label else "",
                "normalized": normalized,
                "pixel": px,
            }
        )

    return {
        "boxes": boxes,
        "points": points,
        "none": "<box>none</box>" in answer,
        "image_size": {"width": width, "height": height},
    }


def _is_degenerate_answer(answer: str) -> bool:
    stripped = answer.strip()
    if len(stripped) < 32:
        return False
    if re.fullmatch(r"!+", stripped):
        return True
    if len(set(stripped)) == 1 and not stripped[0].isalnum():
        return True
    return False


def _build_prompt(task: str, query: str) -> str:
    phrase = query.strip()
    prompts = {
        "ground_multi": f"Locate all the instances that match the following description: {phrase}.",
        "ground_single": f"Locate a single instance that matches the following description: {phrase}.",
        "detect": f"Locate all the instances that matches the following description: {phrase}.",
        "ground_text": f"Please locate the text referred as {phrase}.",
        "detect_text": "Detect all the text in box format.",
        "gui_box": f"Locate the region that matches the following description: {phrase}.",
        "gui_point": f"Point to: {phrase}.",
        "point": f"Point to: {phrase}.",
        "custom": phrase,
    }
    if task not in prompts:
        raise ValueError(f"Unsupported LocateAnything task: {task}")
    if task != "detect_text" and not phrase:
        raise ValueError(f"query cannot be empty for task {task}")
    return prompts[task]


def _draw_locations(
    image: Image.Image,
    locations: dict[str, Any],
    point_radius: int,
) -> tuple[Image.Image, torch.Tensor]:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size
    mask = np.zeros((height, width), dtype=np.float32)
    colors = ["#00ff66", "#ffcc00", "#00ccff", "#ff6699", "#cc88ff"]

    for index, box in enumerate(locations["boxes"]):
        color = colors[index % len(colors)]
        x1, y1, x2, y2 = box["pixel"]
        left, right = sorted(min(width - 1, max(0, round(x))) for x in (x1, x2))
        top, bottom = sorted(min(height - 1, max(0, round(y))) for y in (y1, y2))
        draw.rectangle((left, top, right, bottom), outline=color, width=3)
        label = box["label"] or f"box {index + 1}"
        draw.text((left + 3, top + 3), label, fill=color, stroke_width=2, stroke_fill="black")
        mask[top : bottom + 1, left : right + 1] = 1.0

    for index, point in enumerate(locations["points"]):
        color = colors[(len(locations["boxes"]) + index) % len(colors)]
        x, y = (round(value) for value in point["pixel"])
        x = min(width - 1, max(0, x))
        y = min(height - 1, max(0, y))
        left, right = max(0, x - point_radius), min(width - 1, x + point_radius)
        top, bottom = max(0, y - point_radius), min(height - 1, y + point_radius)
        draw.ellipse((left, top, right, bottom), outline=color, fill=color, width=2)
        label = point["label"] or f"point {index + 1}"
        draw.text((left + 3, top + 3), label, fill=color, stroke_width=2, stroke_fill="black")
        yy, xx = np.ogrid[:height, :width]
        mask[(xx - x) ** 2 + (yy - y) ** 2 <= point_radius**2] = 1.0

    return annotated, torch.from_numpy(mask)


def _postprocess_mask(mask: torch.Tensor, grow: int, blur_radius: float) -> torch.Tensor:
    output = mask.float().clamp(0, 1).unsqueeze(0).unsqueeze(0)
    if grow != 0:
        kernel_size = abs(grow) * 2 + 1
        if grow > 0:
            output = F.max_pool2d(output, kernel_size=kernel_size, stride=1, padding=abs(grow))
        else:
            output = 1.0 - F.max_pool2d(1.0 - output, kernel_size=kernel_size, stride=1, padding=abs(grow))

    if blur_radius > 0:
        radius = max(1, int(round(blur_radius * 3)))
        coords = torch.arange(-radius, radius + 1, device=output.device, dtype=output.dtype)
        kernel = torch.exp(-(coords**2) / (2 * blur_radius**2))
        kernel = kernel / kernel.sum()
        output = F.pad(output, (radius, radius, 0, 0), mode="replicate")
        output = F.conv2d(output, kernel.view(1, 1, 1, -1))
        output = F.pad(output, (0, 0, radius, radius), mode="replicate")
        output = F.conv2d(output, kernel.view(1, 1, -1, 1))

    return output.squeeze(0).squeeze(0).clamp(0, 1)


def _parse_overlay_color(value: str) -> torch.Tensor:
    color = value.strip().lstrip("#")
    if len(color) == 3:
        color = "".join(character * 2 for character in color)
    if len(color) != 6 or not re.fullmatch(r"[0-9a-fA-F]{6}", color):
        raise ValueError("overlay_color must be a hex RGB color such as #00ff66")
    return torch.tensor([int(color[index : index + 2], 16) / 255.0 for index in (0, 2, 4)])


def _make_mask_overlay(
    image: torch.Tensor,
    mask: torch.Tensor,
    color: torch.Tensor,
    opacity: float,
) -> torch.Tensor:
    alpha = mask.unsqueeze(-1) * opacity
    return (image.detach().cpu().float().clamp(0, 1) * (1.0 - alpha) + color.view(1, 1, 3) * alpha).clamp(0, 1)


def _json_default(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype})"
    return repr(value)


@dataclass
class LocateAnythingRuntime:
    model: Any
    tokenizer: Any
    processor: Any
    device: torch.device
    dtype: torch.dtype
    model_path: str
    use_batch_runtime: bool = False
    scheduler: str = "pipeline"
    group_size: int = 0
    batch_generate: Any = None
    batch_stats: Any = None

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        prompt: str,
        generation_mode: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        seed: int,
        verbose: bool,
    ) -> dict[str, Any]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.py_apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=images,
            videos=videos,
            return_tensors="pt",
        ).to(self.device)

        effective_temperature = temperature
        effective_top_p = top_p
        effective_top_k = top_k
        if self.device.type == "cuda" and temperature > 0:
            print(
                "[LocateAnything] temperature > 0 is unstable on this CUDA path and can trigger "
                "device-side asserts in multinomial sampling. Forcing safe sampling with "
                "temperature=0.0, top_p=1.0, top_k=0."
            )
            effective_temperature = 0.0
            effective_top_p = 1.0
            effective_top_k = 0

        def _generate_once(run_temperature: float, run_top_p: float, run_top_k: int):
            started = time.perf_counter()
            cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(seed)
                top_k_for_generate = None if run_top_k <= 0 else run_top_k
                response = self.model.generate(
                    pixel_values=inputs["pixel_values"].to(self.dtype),
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    image_grid_hws=inputs.get("image_grid_hws"),
                    tokenizer=self.tokenizer,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    generation_mode=generation_mode,
                    temperature=run_temperature,
                    do_sample=run_temperature > 0,
                    top_p=run_top_p,
                    top_k=top_k_for_generate,
                    repetition_penalty=repetition_penalty,
                    verbose=verbose,
                )
            return response, time.perf_counter() - started

        try:
            response, elapsed = _generate_once(effective_temperature, effective_top_p, effective_top_k)
        except Exception as exc:
            if effective_temperature <= 0:
                raise
            print(
                "[LocateAnything] Sampling path failed; retrying with temperature=0.0, top_p=1.0, top_k=0. "
                f"Original error: {exc}"
            )
            response, elapsed = _generate_once(0.0, 1.0, 0)

        payload = response[0] if isinstance(response, tuple) else response
        if isinstance(payload, str):
            answer = payload
        elif isinstance(payload, torch.Tensor):
            generated = payload[0] if payload.ndim > 1 else payload
            input_length = inputs["input_ids"].shape[-1]
            answer = self.tokenizer.decode(generated[input_length:], skip_special_tokens=False)
        else:
            answer = str(payload)

        result = {"answer": answer, "elapsed_seconds": elapsed}
        if isinstance(response, tuple) and len(response) >= 3:
            result["history"] = response[1]
            result["stats"] = response[2]
        return result

    @torch.no_grad()
    def predict_batch(
        self,
        images: list[Image.Image],
        prompt: str,
        generation_mode: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        verbose: bool,
    ) -> list[dict[str, Any]]:
        if not self.use_batch_runtime or self.batch_generate is None:
            return [
                self.predict(
                    image=image,
                    prompt=prompt,
                    generation_mode=generation_mode,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    seed=index,
                    verbose=verbose,
                )
                for index, image in enumerate(images)
            ]

        if generation_mode != "hybrid":
            raise ValueError("Batch runtime currently supports generation_mode='hybrid' only")

        effective_temperature = temperature
        effective_top_p = top_p
        effective_top_k = top_k
        if self.device.type == "cuda" and temperature > 0:
            print(
                "[LocateAnything] temperature > 0 is unstable on this CUDA batch path and can trigger "
                "device-side asserts in multinomial sampling. Forcing safe sampling with "
                "temperature=0.0, top_p=1.0, top_k=0."
            )
            effective_temperature = 0.0
            effective_top_p = 1.0
            effective_top_k = 0

        top_k_for_generate = None if effective_top_k <= 0 else effective_top_k
        top_p_for_generate = None if effective_top_p < 0 else effective_top_p
        try:
            answers = self.batch_generate(
                [(image, prompt) for image in images],
                temperature=effective_temperature,
                top_p=top_p_for_generate,
                top_k=top_k_for_generate,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
                scheduler=self.scheduler,
                group_size=self.group_size,
            )
        except Exception as exc:
            if effective_temperature <= 0:
                raise
            print(
                "[LocateAnything] Batch sampling path failed; retrying serially with temperature=0.0, "
                f"top_p=1.0, top_k=0. Original error: {exc}"
            )
            return [
                self.predict(
                    image=image,
                    prompt=prompt,
                    generation_mode=generation_mode,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    top_k=0,
                    repetition_penalty=repetition_penalty,
                    seed=index,
                    verbose=verbose,
                )
                for index, image in enumerate(images)
            ]
        stats = self.batch_stats() if verbose and self.batch_stats is not None else None
        results = []
        for answer in answers:
            row = {"answer": answer, "elapsed_seconds": 0.0}
            if stats is not None:
                row["stats"] = stats
            results.append(row)
        return results

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.batch_generate = None
        self.batch_stats = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class LocateAnythingModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_source": (
                    "STRING",
                    {
                        "default": MODEL_REPO,
                        "tooltip": "Hugging Face repo ID or local snapshot directory.",
                    },
                ),
                "download_model": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Download missing model files into models/LocateAnything.",
                    },
                ),
                "device": (["auto", "cuda", "cpu", "mps"], {"default": "auto", "tooltip": "Execution device. auto uses the device selected by ComfyUI; cuda is recommended when available."}),
                "dtype": (
                    ["auto", "bfloat16", "float16", "float32"],
                    {"default": "auto", "tooltip": "Model precision. auto selects bfloat16 or float16 on CUDA and float32 on CPU."},
                ),
                "attention": (
                    ["eager", "sdpa", "auto"],
                    {
                        "default": "eager",
                        "tooltip": "Eager is the conservative compatibility path. SDPA can be faster but may be less stable on some setups.",
                    },
                ),
                "use_batch_runtime": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Use the official hybrid batch runtime from the model snapshot when available.",
                    },
                ),
                "runtime_attention": (
                    ["la_flash", "sdpa", "eager", "auto"],
                    {
                        "default": "la_flash",
                        "tooltip": "Attention backend for the optional official batch runtime.",
                    },
                ),
                "vision_attention": (
                    ["auto", "sdpa", "eager", "flash_attention_2"],
                    {
                        "default": "auto",
                        "tooltip": "Vision attention backend for the optional official batch runtime.",
                    },
                ),
                "scheduler": (
                    ["pipeline", "round_robin"],
                    {
                        "default": "pipeline",
                        "tooltip": "Hybrid batch scheduler used by the official batch runtime.",
                    },
                ),
                "group_size": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1024,
                        "tooltip": "Batch runtime hybrid grouping. 0 keeps the upstream default.",
                    },
                ),
                "strict_attn": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Require the configured batch attention backend instead of allowing runtime fallback.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("LOCATEANYTHING_MODEL",)
    RETURN_NAMES = ("model",)
    OUTPUT_TOOLTIPS = ("Loaded LocateAnything runtime shared with grounding nodes.",)
    FUNCTION = "load"
    CATEGORY = "LocateAnything"
    DESCRIPTION = """Loads NVIDIA LocateAnything-3B from a local snapshot or Hugging Face. Missing files can be downloaded automatically into models/LocateAnything. The official checkpoint requires trust_remote_code=True."""

    def load(
        self,
        model_source: str,
        download_model: bool,
        device: str,
        dtype: str,
        attention: str,
        use_batch_runtime: bool,
        runtime_attention: str,
        vision_attention: str,
        scheduler: str,
        group_size: int,
        strict_attn: bool,
    ):
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        model_path = _resolve_model_source(model_source, download_model)
        torch_device = _resolve_device(device)
        torch_dtype = _resolve_dtype(dtype, torch_device)
        kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch_dtype,
        }
        if attention != "auto":
            kwargs["attn_implementation"] = attention

        print(
            f"[LocateAnything] Loading {model_path} on {torch_device} "
            f"with {torch_dtype}, attention={attention}, use_batch_runtime={use_batch_runtime}"
        )
        _apply_model_compat_patches(model_path)
        batch_generate = None
        batch_stats = None
        if use_batch_runtime:
            tokenizer, processor, model, batch_generate, batch_stats = _load_batch_runtime(
                model_path=model_path,
                attn=runtime_attention,
                vision_attn=vision_attention,
                scheduler=scheduler,
                group_size=group_size,
                strict_attn=strict_attn,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_path, **kwargs).to(torch_device).eval()
        return (
            LocateAnythingRuntime(
                model=model,
                tokenizer=tokenizer,
                processor=processor,
                device=torch_device,
                dtype=torch_dtype,
                model_path=model_path,
                use_batch_runtime=use_batch_runtime,
                scheduler=scheduler,
                group_size=group_size,
                batch_generate=batch_generate,
                batch_stats=batch_stats,
            ),
        )


class LocateAnythingGrounding:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("LOCATEANYTHING_MODEL",),
                "image": ("IMAGE",),
                "task": (TASKS, {"default": "ground_multi", "tooltip": "Operation mode. Hover the node help (?) for the complete list. Use custom to send query as the full model prompt."}),
                "query": (
                    "STRING",
                    {
                        "default": "person",
                        "multiline": True,
                        "tooltip": "Description, text, GUI target, or full prompt for custom mode.",
                    },
                ),
                "generation_mode": (
                    ["hybrid", "fast", "slow"],
                    {
                        "default": "hybrid",
                        "tooltip": "Decoding strategy: hybrid uses fast decoding with stable fallback; fast prioritizes speed; slow prioritizes the stable path.",
                    },
                ),
                "max_new_tokens": ("INT", {"default": 2048, "min": 1, "max": 8192, "tooltip": "Maximum generated tokens. Reduce this when shorter answers are sufficient."}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Sampling temperature. 0.0 is the safest path; the node retries with safe settings if higher temperatures fail."},
                ),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Nucleus sampling cutoff. Relevant when temperature is above 0."}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 1024, "tooltip": "Top-k sampling cutoff. 0 disables top-k, matching the official worker."}),
                "repetition_penalty": (
                    "FLOAT",
                    {"default": 1.1, "min": 0.0, "max": 3.0, "step": 0.05, "tooltip": "Penalty for repeated tokens. The official worker uses 1.1."},
                ),
                "point_radius": ("INT", {"default": 12, "min": 1, "max": 512, "tooltip": "Radius in pixels used to draw point results into the output mask."}),
                "mask_grow": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1, "tooltip": "Grow the output mask by this many pixels. Negative values shrink it."}),
                "mask_blur": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "Gaussian blur radius applied after mask grow. Use 0 for hard edges."}),
                "overlay_color": ("STRING", {"default": "#00ff66", "tooltip": "Hex RGB color used by the mask overlay preview, for example #00ff66 or #ff0000."}),
                "overlay_opacity": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Opacity of the colored mask overlay preview."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "Sampling seed. Relevant when temperature is above 0. For IMAGE batches, frame N uses seed + N."}),
                "verbose": ("BOOLEAN", {"default": True, "tooltip": "Print the official generation step log in the terminal. Disable for quieter runs."}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("answer", "locations_json", "annotated_image", "mask", "mask_overlay")
    OUTPUT_TOOLTIPS = (
        "Raw model response. For batches, this is a JSON array of responses.",
        "Structured JSON with prompt, timing, normalized coordinates, pixel coordinates, and batch index.",
        "Input image batch annotated with returned boxes and points.",
        "Post-processed mask batch after grow and blur: filled boxes and circles centered on returned points.",
        "Original image batch with the post-processed mask blended using overlay_color and overlay_opacity.",
    )
    FUNCTION = "run"
    CATEGORY = "LocateAnything"
    DESCRIPTION = """Runs visual grounding frame by frame for an IMAGE batch.

Task modes:
- ground_multi: locate every matching instance.
- ground_single: locate one matching instance.
- detect: detect matching categories or descriptions.
- ground_text: locate a requested text phrase.
- detect_text: detect scene text; query is ignored.
- gui_box: locate a GUI region.
- gui_point: return a point for a GUI target.
- point: return a point for a described target.
- custom: use query as the complete prompt.

Coordinates are parsed from the model's normalized [0, 1000] format. A batch of video frames is processed independently, with native ComfyUI progress updates."""

    def run(
        self,
        model: LocateAnythingRuntime,
        image: torch.Tensor,
        task: str,
        query: str,
        generation_mode: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        point_radius: int,
        mask_grow: int,
        mask_blur: float,
        overlay_color: str,
        overlay_opacity: float,
        seed: int,
        verbose: bool,
    ):
        prompt = _build_prompt(task, query)
        answers = []
        records = []
        annotated_images = []
        masks = []
        mask_overlays = []
        color = _parse_overlay_color(overlay_color)
        frames = list(image)
        pil_images = [_tensor_to_pil(frame) for frame in frames]
        progress_bar = comfy.utils.ProgressBar(len(frames))
        frame_seeds = [((seed + index) & 0xffffffffffffffff) for index in range(len(frames))]

        if model.use_batch_runtime and len(pil_images) > 1:
            results = model.predict_batch(
                images=pil_images,
                prompt=prompt,
                generation_mode=generation_mode,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                verbose=verbose,
            )
        else:
            results = [
                model.predict(
                    image=pil_image,
                    prompt=prompt,
                    generation_mode=generation_mode,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    seed=frame_seed,
                    verbose=verbose,
                )
                for pil_image, frame_seed in zip(pil_images, frame_seeds, strict=True)
            ]

        for index, (frame, pil_image, frame_seed, result) in enumerate(
            zip(frames, pil_images, frame_seeds, results, strict=True)
        ):
            if _is_degenerate_answer(result["answer"]):
                print(
                    "[LocateAnything] Degenerate repeated-token output detected; retrying once with generation_mode=slow."
                )
                result = model.predict(
                    image=pil_image,
                    prompt=prompt,
                    generation_mode="slow",
                    max_new_tokens=min(max_new_tokens, 1024),
                    temperature=0.0,
                    top_p=1.0,
                    top_k=0,
                    repetition_penalty=repetition_penalty,
                    seed=frame_seed,
                    verbose=verbose,
                )
            locations = parse_locations(result["answer"], *pil_image.size)
            if not locations["boxes"] and not locations["points"] and not locations["none"]:
                if task != "custom" and generation_mode != "slow":
                    print(
                        "[LocateAnything] No coordinates parsed in current mode; retrying once with generation_mode=slow."
                    )
                    result = model.predict(
                        image=pil_image,
                        prompt=prompt,
                        generation_mode="slow",
                        max_new_tokens=max_new_tokens,
                        temperature=0.0,
                        top_p=1.0,
                        top_k=0,
                        repetition_penalty=repetition_penalty,
                        seed=frame_seed,
                        verbose=verbose,
                    )
                    locations = parse_locations(result["answer"], *pil_image.size)
                if not locations["boxes"] and not locations["points"] and not locations["none"]:
                    print(
                        "[LocateAnything] No coordinates parsed from model response. "
                        "Try reloading the model with attention=eager, then use generation_mode=slow. "
                        f"Raw answer: {result['answer']!r}"
                    )
            annotated, mask = _draw_locations(pil_image, locations, point_radius)
            mask = _postprocess_mask(mask, mask_grow, mask_blur)
            answers.append(result["answer"])
            records.append(
                {
                    "batch_index": index,
                    "prompt": prompt,
                    "answer": result["answer"],
                    "elapsed_seconds": result["elapsed_seconds"],
                    "seed": frame_seed,
                    "locations": locations,
                    "stats": result.get("stats"),
                }
            )
            annotated_images.append(_pil_to_tensor(annotated))
            masks.append(mask)
            mask_overlays.append(_make_mask_overlay(frame, mask, color, overlay_opacity))
            progress_bar.update_absolute(index + 1)

        answer = answers[0] if len(answers) == 1 else json.dumps(answers, ensure_ascii=False)
        locations_json = json.dumps(records, ensure_ascii=False, indent=2, default=_json_default)
        return (
            answer,
            locations_json,
            torch.stack(annotated_images),
            torch.stack(masks),
            torch.stack(mask_overlays),
        )


class LocateAnythingUnloadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("LOCATEANYTHING_MODEL",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    OUTPUT_TOOLTIPS = ("Confirmation that model references and the CUDA cache were released.",)
    FUNCTION = "unload"
    OUTPUT_NODE = True
    CATEGORY = "LocateAnything"
    DESCRIPTION = "Releases the loaded LocateAnything model references and clears the CUDA cache."

    def unload(self, model: LocateAnythingRuntime):
        model.unload()
        return ("LocateAnything model unloaded",)


NODE_CLASS_MAPPINGS = {
    "LocateAnythingModelLoader": LocateAnythingModelLoader,
    "LocateAnythingGrounding": LocateAnythingGrounding,
    "LocateAnythingUnloadModel": LocateAnythingUnloadModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LocateAnythingModelLoader": "LocateAnything Model Loader",
    "LocateAnythingGrounding": "LocateAnything Grounding",
    "LocateAnythingUnloadModel": "LocateAnything Unload Model",
}
