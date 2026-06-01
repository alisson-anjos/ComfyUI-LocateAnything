import { app } from "../../scripts/app.js";

const TARGET_NODES = new Set([
  "LocateAnythingModelLoader",
  "LocateAnythingGrounding",
  "LocateAnythingUnloadModel",
]);

const HELP_CONTENT = {
  LocateAnythingModelLoader: `
    <h2>LocateAnything Model Loader</h2>
    <p>Downloads and loads NVIDIA LocateAnything-3B. The first load downloads approximately 7.8 GB into <code>ComfyUI/models/LocateAnything/nvidia--LocateAnything-3B</code>.</p>
    <h3>Parameters</h3>
    <table>
      <tr><th>Parameter</th><th>Description</th></tr>
      <tr><td><code>model_source</code></td><td>Hugging Face repository ID or a local snapshot path.</td></tr>
      <tr><td><code>download_model</code></td><td>Downloads missing files automatically when enabled.</td></tr>
      <tr><td><code>device</code></td><td><code>auto</code> follows ComfyUI. CUDA is recommended when available.</td></tr>
      <tr><td><code>dtype</code></td><td><code>auto</code> chooses BF16 or FP16 on CUDA and FP32 on CPU.</td></tr>
      <tr><td><code>attention</code></td><td><code>sdpa</code> is the broadly compatible default. Without optional MagiAttention, the official model falls back to SDPA.</td></tr>
    </table>
    <h3>Security Note</h3>
    <p>The official checkpoint requires <code>trust_remote_code=True</code>. Loading it executes modeling Python files shipped with the Hugging Face repository.</p>
  `,
  LocateAnythingGrounding: `
    <h2>LocateAnything Grounding</h2>
    <p>Runs visual grounding for every image in the connected ComfyUI <code>IMAGE</code> batch. Video frames are processed independently, frame by frame.</p>
    <h3>Task Modes</h3>
    <table>
      <tr><th>Mode</th><th>Description</th></tr>
      <tr><td><code>ground_multi</code></td><td>Locate every instance matching a description.</td></tr>
      <tr><td><code>ground_single</code></td><td>Locate one matching instance.</td></tr>
      <tr><td><code>detect</code></td><td>Detect matching categories or descriptions. Separate multiple categories with <code>&lt;/c&gt;</code>.</td></tr>
      <tr><td><code>ground_text</code></td><td>Locate a requested text phrase.</td></tr>
      <tr><td><code>detect_text</code></td><td>Detect scene text. The query input is ignored.</td></tr>
      <tr><td><code>gui_box</code></td><td>Locate a GUI element as a box.</td></tr>
      <tr><td><code>gui_point</code></td><td>Return a point for a GUI element.</td></tr>
      <tr><td><code>point</code></td><td>Return a point for a described target.</td></tr>
      <tr><td><code>custom</code></td><td>Send the query as the complete model prompt. General visual questions may return text answers, but VQA is experimental because this checkpoint is optimized for localization.</td></tr>
    </table>
    <h3>Custom Prompts and VQA</h3>
    <p>Custom mode can be used for questions such as <code>What is the color of the t-shirt?</code>. If the model returns plain text, it is available in <code>answer</code>. Since no coordinates are present, <code>annotated_image</code> remains unchanged and <code>mask</code> is empty. General VQA quality is not guaranteed by this grounding-focused checkpoint.</p>
    <h3>Generation Modes</h3>
    <table>
      <tr><th>Mode</th><th>Behavior</th><th>Tradeoff</th></tr>
      <tr><td><code>hybrid</code></td><td>Parallel decoding with autoregressive fallback when needed.</td><td>Recommended balance.</td></tr>
      <tr><td><code>fast</code></td><td>MTP parallel decoding only.</td><td>Fastest; suited to simpler scenes.</td></tr>
      <tr><td><code>slow</code></td><td>Autoregressive decoding.</td><td>Slowest; most robust.</td></tr>
    </table>
    <h3>Parameters</h3>
    <table>
      <tr><th>Parameter</th><th>Description</th></tr>
      <tr><td><code>query</code></td><td>Description, category list, text phrase, GUI target, or complete prompt in custom mode.</td></tr>
      <tr><td><code>max_new_tokens</code></td><td>Maximum generated tokens. Increase for dense detections. The official model card recommends up to 8192 to avoid truncation.</td></tr>
      <tr><td><code>temperature</code></td><td>Sampling randomness. Use 0 for deterministic grounding.</td></tr>
      <tr><td><code>top_p</code></td><td>Nucleus sampling cutoff. Relevant when temperature is above 0.</td></tr>
      <tr><td><code>repetition_penalty</code></td><td>Penalty for repeated tokens. The official worker uses 1.1.</td></tr>
      <tr><td><code>point_radius</code></td><td>Radius in pixels used to draw point results into the mask.</td></tr>
      <tr><td><code>mask_grow</code></td><td>Grow rectangular or circular masks by this many pixels. Negative values shrink them.</td></tr>
      <tr><td><code>mask_blur</code></td><td>Gaussian blur radius applied after grow. Use 0 for hard edges.</td></tr>
      <tr><td><code>overlay_color</code></td><td>Hex RGB color for the overlay preview, such as <code>#00ff66</code> or <code>#ff0000</code>.</td></tr>
      <tr><td><code>overlay_opacity</code></td><td>Opacity of the colored mask overlay preview.</td></tr>
      <tr><td><code>seed</code></td><td>Sampling seed. Relevant when temperature is above 0. For IMAGE batches, frame N uses <code>seed + N</code> for reproducible frame-by-frame processing.</td></tr>
      <tr><td><code>verbose</code></td><td>Print the official step-by-step generation log in the terminal. Disable for quieter runs.</td></tr>
    </table>
    <h3>Mask Geometry</h3>
    <p>LocateAnything is a grounding model, not a segmentation model. Box results produce rectangular masks and point results produce circular masks. Grow and blur expand or soften those shapes; they do not extract an object silhouette. Use a segmentation node such as SAM downstream when silhouette masks are required.</p>
    <h3>Outputs</h3>
    <table>
      <tr><th>Output</th><th>Description</th></tr>
      <tr><td><code>answer</code></td><td>Raw model response. For batches, a JSON array.</td></tr>
      <tr><td><code>locations_json</code></td><td>Structured results with batch index, timing, normalized [0, 1000] coordinates, and pixel coordinates.</td></tr>
      <tr><td><code>annotated_image</code></td><td>Input frames with boxes and points drawn on them.</td></tr>
      <tr><td><code>mask</code></td><td>Post-processed rectangular box masks and circular point masks after grow and blur.</td></tr>
      <tr><td><code>mask_overlay</code></td><td>Original frames with the post-processed mask blended using the selected color and opacity.</td></tr>
    </table>
  `,
  LocateAnythingUnloadModel: `
    <h2>LocateAnything Unload Model</h2>
    <p>Releases model references and clears the CUDA cache. Use it when you need to reclaim VRAM for another model.</p>
  `,
};

function ensureStyles() {
  if (document.getElementById("locateanything-help-styles")) return;
  const style = document.createElement("style");
  style.id = "locateanything-help-styles";
  style.textContent = `
    .locateanything-help-backdrop {
      position: fixed; inset: 0; z-index: 10000;
      display: flex; align-items: center; justify-content: center;
      padding: 24px; background: rgba(0, 0, 0, 0.62);
    }
    .locateanything-help-modal {
      width: min(920px, 95vw); max-height: 88vh; overflow: auto;
      color: var(--fg-color, #eee); background: var(--comfy-menu-bg, #202020);
      border: 1px solid var(--border-color, #666); border-radius: 10px;
      box-shadow: 0 12px 48px rgba(0, 0, 0, 0.55);
      padding: 20px 24px 24px; font: 14px/1.5 Arial, sans-serif;
    }
    .locateanything-help-header { display: flex; justify-content: space-between; gap: 16px; }
    .locateanything-help-modal h2 { color: #7ec8ff; margin: 0 0 12px; }
    .locateanything-help-modal h3 { color: #ffcf70; margin: 20px 0 8px; }
    .locateanything-help-modal p { margin: 8px 0; }
    .locateanything-help-modal table { width: 100%; border-collapse: collapse; }
    .locateanything-help-modal th, .locateanything-help-modal td {
      border: 1px solid var(--border-color, #555); padding: 7px 9px; text-align: left; vertical-align: top;
    }
    .locateanything-help-modal th { background: rgba(255, 255, 255, 0.08); }
    .locateanything-help-modal code { color: #9ee493; }
    .locateanything-help-close {
      border: 1px solid var(--border-color, #666); border-radius: 6px;
      color: var(--fg-color, #eee); background: rgba(255, 255, 255, 0.08);
      cursor: pointer; font-size: 18px; min-width: 34px; height: 34px;
    }
    .locateanything-help-links { margin-top: 20px; }
    .locateanything-help-links a { color: #7ec8ff; }
  `;
  document.head.appendChild(style);
}

function showHelp(nodeName) {
  ensureStyles();
  const backdrop = document.createElement("div");
  backdrop.className = "locateanything-help-backdrop";
  const modal = document.createElement("div");
  modal.className = "locateanything-help-modal";
  modal.innerHTML = `
    <div class="locateanything-help-header">
      <div>${HELP_CONTENT[nodeName] || "<h2>LocateAnything</h2>"}</div>
      <button class="locateanything-help-close" type="button" title="Close">×</button>
    </div>
    <div class="locateanything-help-links">
      <strong>Official references:</strong>
      <a href="https://huggingface.co/nvidia/LocateAnything-3B" target="_blank" rel="noreferrer">model card</a> ·
      <a href="https://github.com/NVlabs/Eagle/tree/main/Embodied" target="_blank" rel="noreferrer">source</a> ·
      <a href="https://research.nvidia.com/labs/lpr/locate-anything/" target="_blank" rel="noreferrer">project page</a>
    </div>
  `;
  const onKeydown = (event) => {
    if (event.key === "Escape") close();
  };
  const close = () => {
    document.removeEventListener("keydown", onKeydown);
    backdrop.remove();
  };
  modal.querySelector(".locateanything-help-close").addEventListener("click", close);
  backdrop.addEventListener("click", (event) => {
    if (event.target === backdrop) close();
  });
  document.addEventListener("keydown", onKeydown);
  backdrop.appendChild(modal);
  document.body.appendChild(backdrop);
}

function addHelpIcon(nodeType, nodeName) {
  const iconSize = 16;
  const iconMargin = 5;
  const iconY = iconSize - 34;
  const originalDraw = nodeType.prototype.onDrawForeground;
  nodeType.prototype.onDrawForeground = function (ctx) {
    const result = originalDraw?.apply(this, arguments);
    if (this.flags?.collapsed) return result;
    const iconX = this.size[0] - iconSize - iconMargin;
    ctx.save();
    ctx.beginPath();
    ctx.arc(iconX + iconSize / 2, iconY + iconSize / 2, iconSize / 2, 0, Math.PI * 2);
    ctx.fillStyle = "#f0a33a";
    ctx.fill();
    ctx.fillStyle = "#1d1d1d";
    ctx.font = "bold 13px Arial";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("?", iconX + iconSize / 2, iconY + iconSize / 2 + 0.5);
    ctx.restore();
    return result;
  };

  const originalMouseDown = nodeType.prototype.onMouseDown;
  nodeType.prototype.onMouseDown = function (event, localPos) {
    const iconX = this.size[0] - iconSize - iconMargin;
    if (
      !this.flags?.collapsed &&
      localPos[0] >= iconX && localPos[0] <= iconX + iconSize &&
      localPos[1] >= iconY && localPos[1] <= iconY + iconSize
    ) {
      showHelp(nodeName);
      return true;
    }
    return originalMouseDown?.apply(this, arguments);
  };
}

app.registerExtension({
  name: "ComfyUI-LocateAnything.HelpPopup",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!TARGET_NODES.has(nodeData.name)) return;
    addHelpIcon(nodeType, nodeData.name);
  },
});
