# Hunyuan3D CLI

Linux/NVIDIA command-line inference for Hunyuan3D 2.1. The supported interface is a single JSON-producing CLI, designed for people and agents rather than a web server.

## Clone → build → use

Install [uv](https://docs.astral.sh/uv/) first, then run:

```bash
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git
cd Hunyuan3D-2.1
./bootstrap.sh --profile all
.venv/bin/hunyuan3d doctor
.venv/bin/hunyuan3d models pull --components shape,texture
.venv/bin/hunyuan3d generate --image ./flower.png --output-dir ./output --seed 42
```

`bootstrap.sh` creates a Python 3.11 environment from the locked dependency graph and, for the `texture` and `all` profiles, compiles the two required CUDA/native extensions. It never downloads model weights. `models pull` does that explicitly and stores them under `.cache/hunyuan3d/` in the clone (or `--cache-dir` / `HUNYUAN3D_CACHE`).

The `shape` profile omits the PBR texturing dependencies and native extensions:

```bash
./bootstrap.sh --profile shape
.venv/bin/hunyuan3d models pull --components shape
.venv/bin/hunyuan3d generate --image ./flower.png --output-dir ./output --shape-only
```

## CLI contract

Every successful command emits one JSON object on stdout. Progress and failures are written to stderr; a failure also emits an `{"ok": false, ...}` object and exits non-zero.

```bash
# Verify GPU visibility, torch build, CUDA compiler, and cache location.
hunyuan3d doctor

# Inspect or explicitly fetch the pinned model revisions.
hunyuan3d models status
hunyuan3d models pull --components shape
hunyuan3d models pull --components texture

# Keep each recovery boundary as a separate command.
hunyuan3d prepare flower.png --output work/flower.rgba.png
hunyuan3d shape --image work/flower.rgba.png --output work/shape.glb --steps 50 --seed 42
hunyuan3d texture --image work/flower.rgba.png --mesh work/shape.glb --output work/textured

# The convenience path: prepare → shape → texture.
hunyuan3d generate --image flower.png --output-dir output --seed 42
```

`prepare` removes the background and writes RGBA. Supplying an opaque image directly to `shape` is allowed but usually yields a flat background-shaped mesh, so agents should retain the prepared image. On GPUs below 21 GB VRAM, shape and texture automatically use CPU model offloading; this needs ample system RAM and is slower.

## Requirements and support boundary

- Linux, an NVIDIA GPU with working CUDA driver access, and a CUDA 12-compatible toolchain for the texture profile.
- Python 3.11, managed by `uv`.
- Roughly 10 GB VRAM for shape generation; full PBR texturing normally wants 21 GB VRAM. Low-VRAM offload is supported and was validated on a 12 GB RTX 3060.
- Model use remains subject to Tencent's license in [LICENSE](LICENSE).

The project uses `pyproject.toml` plus `uv.lock` as its dependency source of truth. Package versions are bounded to current-compatible releases and the lock file makes installs reproducible. PyTorch wheels come from the CUDA 12.4 PyTorch index. To refresh the lock deliberately, use `uv lock --upgrade`; normal users should use `bootstrap.sh` unchanged.

The previous Gradio, API, notebook, training, and demo entry points are no longer documented or supported. The legacy inference modules remain behind the CLI while the model implementations are progressively consolidated; do not invoke them directly.

## Outputs

`generate` writes:

- `input.rgba.png` — prepared reference image.
- `shape.glb` — untextured shape checkpoint.
- `textured.obj`, `.mtl`, and PBR texture maps — final textured asset.

This structure lets an agent retry only the failed stage without rerunning shape generation.

## Attribution

Hunyuan3D 2.1 is released by Tencent Hunyuan. See the upstream [technical report](https://arxiv.org/abs/2506.15442) and [model card](https://huggingface.co/tencent/Hunyuan3D-2.1).
