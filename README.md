# ComfyUI-LocateAnything

ComfyUI custom nodes for NVIDIA's [LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) visual grounding model.

LocateAnything supports object detection, phrase grounding, text localization, document layout analysis, GUI grounding, and point-based localization. This integration processes ComfyUI `IMAGE` batches frame by frame and returns structured coordinates, annotated previews, and masks.

## Important Model License Notice

This repository contains integration code only. Model weights are downloaded separately from Hugging Face.

The NVIDIA LocateAnything-3B checkpoint is released under the [NVIDIA License](https://huggingface.co/nvidia/LocateAnything-3B/blob/main/LICENSE) for non-commercial use. NVIDIA states that use, reproduction, and modification are permitted for academic and non-profit research purposes only. Review the official model license before downloading or using the checkpoint.

## Nodes

### LocateAnything Model Loader

Downloads and loads the checkpoint. The first run downloads approximately 7.8 GB into:

```text
ComfyUI/models/LocateAnything/nvidia--LocateAnything-3B
```

Inputs:

| Parameter | Description |
| --- | --- |
| `model_source` | Hugging Face repository ID or local snapshot directory. Defaults to `nvidia/LocateAnything-3B`. |
| `download_model` | Downloads missing model files automatically when enabled. |
| `device` | `auto`, `cuda`, `cpu`, or `mps`. NVIDIA CUDA is the recommended path. |
| `dtype` | `auto`, `bfloat16`, `float16`, or `float32`. `auto` selects an appropriate dtype for the device. |
| `attention` | `sdpa`, `auto`, or `eager`. SDPA is the broadly compatible default. |

The official checkpoint requires `trust_remote_code=True`. Loading it executes Python modeling files shipped in the Hugging Face model repository.

### LocateAnything Grounding

Runs inference for every image in the connected batch. A video loaded as a ComfyUI `IMAGE` batch is processed frame by frame, with a native ComfyUI progress bar. The node includes a visible `?` icon in its title bar that opens an in-app parameter reference popup.

Outputs:

| Output | Description |
| --- | --- |
| `answer` | Raw model response. For batches, a JSON array of responses. |
| `locations_json` | Structured batch results with prompt, timing, normalized `[0, 1000]` coordinates, pixel coordinates, and `batch_index`. |
| `annotated_image` | Input batch with returned boxes and points drawn on each frame. |
| `mask` | Post-processed mask batch containing rectangular box masks and circular point masks after grow and blur. |
| `mask_overlay` | Original image or video frames with the post-processed mask blended using the selected color and opacity. |

Task modes:

| Mode | Description |
| --- | --- |
| `ground_multi` | Locates every instance matching a free-form description. |
| `ground_single` | Locates one instance matching a description. |
| `detect` | Detects objects matching categories or a description. Multiple categories may be separated with `</c>`. |
| `ground_text` | Locates a requested text phrase. |
| `detect_text` | Detects scene text. The `query` input is ignored. |
| `gui_box` | Locates a GUI region as a bounding box. |
| `gui_point` | Returns a point for a GUI target. |
| `point` | Returns a point for a described target. |
| `custom` | Sends `query` as the complete model prompt. General visual questions may return plain text, but VQA is experimental because the checkpoint is optimized for localization. |

In `custom` mode, a question such as `What is the color of the t-shirt?` may return a plain-text answer. When the response contains no coordinates, `annotated_image` remains unchanged and `mask` is empty. General VQA quality is not guaranteed by this grounding-focused checkpoint.

Generation parameters:

| Parameter | Description |
| --- | --- |
| `generation_mode` | `hybrid`, `fast`, or `slow`. See the table below. |
| `max_new_tokens` | Maximum generated tokens. Increase for dense detections; reduce for short grounding responses. |
| `temperature` | Sampling randomness. Use `0` for deterministic grounding. |
| `top_p` | Nucleus sampling cutoff. Relevant when `temperature` is above `0`. |
| `repetition_penalty` | Penalty for repeated tokens. The official worker uses `1.1`. |
| `point_radius` | Radius in pixels used when points are drawn into the output mask. |
| `mask_grow` | Grow rectangular or circular masks by this many pixels. Negative values shrink them. |
| `mask_blur` | Gaussian blur radius applied after grow. Use `0` for hard edges. |
| `overlay_color` | Hex RGB color used for the overlay preview, such as `#00ff66` or `#ff0000`. |
| `overlay_opacity` | Opacity of the colored overlay preview. |
| `seed` | Sampling seed. Relevant when `temperature` is above `0`. For batches, frame N uses `seed + N`. |
| `verbose` | Prints the official step-by-step generation log in the terminal. Disable it for quieter runs. |

Mask geometry note: LocateAnything is a grounding model, not a segmentation model. Boxes generate rectangular masks and points generate circular masks. `mask_grow` and `mask_blur` modify these shapes but do not extract object silhouettes. Connect a segmentation node such as SAM downstream when a contour mask is required.

Generation modes:

| Mode | Behavior | Tradeoff |
| --- | --- | --- |
| `hybrid` | Uses fast parallel decoding and falls back to autoregressive decoding when needed. | Recommended balance. |
| `fast` | Uses MTP parallel decoding only. | Fastest; best for simpler scenes. |
| `slow` | Uses autoregressive decoding. | Slowest; most robust. |

### LocateAnything Unload Model

Releases model references and clears the CUDA cache.

## Installation

### ComfyUI Manager

Search for `ComfyUI-LocateAnything` in ComfyUI Manager after the package has been published to the Comfy Registry.

### Manual Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/alisson-anjos/ComfyUI-LocateAnything.git
cd ComfyUI-LocateAnything
pip install -r requirements.txt
```

Restart ComfyUI after installation.

## Basic Workflow

```text
Load Image or Load Video
  -> LocateAnything Model Loader
  -> LocateAnything Grounding
  -> Preview Image
```

Connect the `mask` output to a mask preview or downstream crop/inpaint workflow.

For an initial test:

```text
task: ground_multi
query: person
generation_mode: hybrid
```

## Performance Notes

- The official model card recommends `generation_mode="hybrid"` and up to `max_new_tokens=8192` to avoid truncating dense responses.
- Without optional [MagiAttention](https://sandai-org.github.io/MagiAttention/docs/main/user_guide/install.html), the checkpoint falls back to PyTorch SDPA. This remains functional but may be slower for MTP decoding.
- The model card documents optimized NVIDIA GPU execution. CPU execution is exposed for compatibility but will be substantially slower.

## Official References

- [NVIDIA LocateAnything-3B model card](https://huggingface.co/nvidia/LocateAnything-3B)
- [NVlabs Eagle Embodied source](https://github.com/NVlabs/Eagle/tree/main/Embodied)
- [LocateAnything project page](https://research.nvidia.com/labs/lpr/locate-anything/)
- [LocateAnything technical report](https://research.nvidia.com/labs/lpr/locate-anything/LocateAnything.pdf)

## License

The integration code in this repository is licensed under GPL-3.0. The NVIDIA model weights have separate terms described above.
