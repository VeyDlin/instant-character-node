# Instant Character
> [!WARNING]
> This node is still under development and not yet functional

This node integrates [InstantCharacter](https://github.com/Tencent/InstantCharacter) — a character-consistent image generator by Tencent Hunyuan — into the [InvokeAI](https://github.com/invoke-ai/InvokeAI) workflow.

**Features:**

* Generate consistent character images from a single reference image.
* Control pose, style, and environment with a text prompt.
* Supports LoRA-based style adapters (e.g., Ghibli, Shinkai).
* No finetuning required — ready to use out of the box.
* Built on top of the DiT (Diffusion Transformer) architecture.

**Requirements:**

To run this node, make sure to install the `timm` package using `uv`:

```bash
uv pip install timm
```
