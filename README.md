# Hunyuan3D CLI

Linux/NVIDIA command-line inference for Hunyuan3D 2.1. The supported interface is a single JSON-producing CLI, designed for people and agents rather than a web server.

## Clone → build → use

Install [uv](https://docs.astral.sh/uv/) first, then run:

```bash
git clone https://github.com/christophkogler/Hunyuan3D-2.1-low-vram.git
cd Hunyuan3D-2.1-low-vram
./bootstrap.sh --profile all --install-command

# Open a new Bash terminal, then run these from any directory.
command -v hunyuan3d
hunyuan3d --version
hunyuan3d doctor
hunyuan3d models pull --components prepare,shape,texture
hunyuan3d generate --image ./flower.png --output-dir ./output --seed 42
```

`bootstrap.sh` creates a Python 3.11 environment from the locked dependency graph and, for the `texture` and `all` profiles, compiles the two required CUDA/native extensions. Passing `--install-command` installs a user-local `hunyuan3d` command in `~/.local/bin` and adds that directory to `~/.bashrc` when needed. After opening a new Bash terminal, the command is available from any directory without activating the project environment. It never downloads model weights. `models pull` does that explicitly and stores them under `.cache/hunyuan3d/` in the clone (or `--cache-dir` / `HUNYUAN3D_CACHE`); this default is independent of the caller's current directory.

`hunyuan3d --version` prints the installed `hunyuan3d-cli` package version. The user-local command points to the most recently installed clone; running `./bootstrap.sh --profile <profile> --install-command` in another clone deliberately changes that command to use the new clone. Re-run the same command after pulling upgrades to update both that clone's environment and the persistent command.

The `shape` profile omits the PBR texturing dependencies and native extensions:

```bash
./bootstrap.sh --profile shape --install-command
hunyuan3d models pull --components prepare,shape
hunyuan3d generate --image ./flower.png --output-dir ./output --shape-only
```

## CLI contract

Every normal command invocation emits exactly one JSON object on stdout. Lifecycle and progress events are emitted as JSON Lines on stderr; `generate` also persists those events, runtime diagnostics, and any failure traceback to `run.log` in its output directory. Failures exit non-zero. Every result includes `schema_version: 1`. Failure payloads use `error.code`, `error.message`, and, when useful, `error.details`. Stable error codes include `invalid_arguments`, `missing_input`, `invalid_input`, `invalid_output`, `output_conflict`, `missing_model_assets`, `unsupported_runtime`, `dependency_failure`, and `generation_failure`.

```bash
# Run the read-only capability report before downloading models or inferring.
hunyuan3d doctor
# Optionally inspect the filesystem used for generated outputs too.
hunyuan3d doctor --output-dir ./output

# Inspect or explicitly fetch the pinned model revisions.
hunyuan3d models status
hunyuan3d models pull --components prepare,shape
hunyuan3d models pull --components texture

# Keep each recovery boundary as a separate command.
hunyuan3d prepare flower.png --output work/flower.rgba.png
hunyuan3d shape --image work/flower.rgba.png --output work/shape.glb --steps 50 --seed 42
hunyuan3d texture --image work/flower.rgba.png --mesh work/shape.glb --output work/textured

# The convenience path: prepare → shape → texture.
hunyuan3d generate --image flower.png --output-dir output --seed 42

# Human-readable usage is available separately.
hunyuan3d help
```

`prepare` removes the background and writes RGBA. Supplying an opaque image directly to `shape` is allowed but usually yields a flat background-shaped mesh, so agents should retain the prepared image. On GPUs below 21 GB VRAM, shape and texture automatically use CPU model offloading; this needs ample system RAM and is slower.

Before runtime setup or model loading, generating commands validate readable images,
supported mesh formats, positive `--steps`, writable output parents, and all
planned output paths. The default is collision-safe: existing destination files
cause a structured `output_conflict` failure. Pass `--overwrite` to
`prepare`, `shape`, `texture`, or `generate` to explicitly replace them. A
partial `generate` directory is allowed when it contains only unrelated files;
any existing planned file (`input.rgba.png`, `shape.glb`, `run.log`, the
textured OBJ/MTL/maps, or the texture remesh intermediate) blocks the run until
`--overwrite` is supplied.

`doctor` reports GPU/CUDA and VRAM facts, importable dependencies and native
extensions, model component readiness, cache/output disk space, and readiness
for `prepare`, `shape`, `texture`, `generate --shape-only`, and `generate`.
Each blocked workflow includes stable blocker codes, remediation, and suggested
next commands. It does not create cache/output directories or download model
files. Its exit status is 0 when the full `generate` workflow is ready and 4
when one or more required capabilities are blocked.

### Progress event stream

For every normal command except the explicit human-readable `help` path, stderr
is a JSON Lines stream. Each event has this stable envelope:

```json
{
  "schema_version": 1,
  "event": "stage_started",
  "run_id": "6f6c7e25-2c74-4cd0-9a31-5a5c1e7fce9a",
  "stage": "shape",
  "timestamp": "2026-01-01T00:00:00.000000Z",
  "elapsed_seconds": 12.345678
}
```

The current event names are `run_started`, `stage_started`, `progress`,
`stage_completed`, `stage_failed`, `run_completed`, and `run_failed`.
`progress` adds integer `current` and `total` counters. Stage events include
applicable input, output, cache, and component paths; the composite `generate`
stage contains nested `prepare`, `shape`, and, unless `--shape-only` is used,
`texture` stages. Model pulls report one progress counter per selected model
component. Failure events contain only a bounded, redacted exception type,
message, and CLI error code when available; tracebacks are kept in `run.log`,
not in the event payload.

Consumers should parse stderr one line at a time, group records by `run_id`,
ignore unknown event names or fields, and continue to parse stdout as the
single final-result object. The event envelope and existing stdout result
contract are versioned independently through the same `schema_version: 1`
compatibility marker.

## Requirements and support boundary

- Linux, an NVIDIA GPU with working CUDA driver access, and a CUDA 12-compatible toolchain for the texture profile.
- Python 3.11, managed by `uv`.
- Roughly 10 GB VRAM for shape generation; full PBR texturing normally wants 21 GB VRAM. Low-VRAM offload is supported and was validated on a 12 GB RTX 3060.
- Model use remains subject to Tencent's license in [LICENSE](LICENSE).

The project uses `pyproject.toml` plus `uv.lock` as its dependency source of truth. Package versions are bounded to current-compatible releases and the lock file makes installs reproducible. PyTorch wheels come from the CUDA 12.4 PyTorch index. To refresh the lock deliberately, use `uv lock --upgrade`; normal users should use `bootstrap.sh` unchanged.

The previous Gradio, API, notebook, training, Docker-demo, and sample-data surfaces have been removed. The retained model runtime is private implementation detail behind the CLI; do not invoke it directly.

## Outputs

`generate` writes:

- `input.rgba.png` — prepared reference image.
- `shape.glb` — untextured shape checkpoint.
- `textured.obj`, `.mtl`, and PBR texture maps — final textured asset.
- `run.log` — progress, warnings, and any failure traceback from this `generate` invocation.

This structure lets an agent retry only the failed stage without rerunning shape generation.

## Attribution

Hunyuan3D 2.1 is released by Tencent Hunyuan. See the upstream [technical report](https://arxiv.org/abs/2506.15442) and [model card](https://huggingface.co/tencent/Hunyuan3D-2.1).
