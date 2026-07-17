"""Stable JSON CLI around the repository's shape and texture engines."""
from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import os
import shutil
import sys
import traceback
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_REVISIONS = {
    "hunyuan": "0b94677654c57bb9a6b6845cd7b704ccf551d327",
    "dino": "611a9d42f2335e0f921f1e313ad3c1b7178d206d",
}
REAL_ESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
SCHEMA_VERSION = 1


class CliError(Exception):
    """An expected CLI failure that can be represented in the JSON contract."""

    def __init__(self, code: str, message: str, exit_code: int, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.exit_code = exit_code
        self.details = details


class JsonArgumentParser(argparse.ArgumentParser):
    """Raise structured errors instead of writing usage text and exiting."""

    def error(self, message: str) -> None:
        raise CliError("invalid_arguments", message, 2)


def package_version() -> str:
    """Return the version of the installed CLI distribution."""
    return importlib.metadata.version("hunyuan3d-cli")


def emit(payload: dict, code: int = 0) -> int:
    print(json.dumps({"schema_version": SCHEMA_VERSION, **payload}, sort_keys=True), flush=True)
    return code


def emit_error(error: CliError) -> int:
    payload = {"code": error.code, "message": str(error)}
    if error.details is not None:
        payload["details"] = error.details
    return emit({"ok": False, "error": payload}, error.exit_code)


def cache_root(value: str | None) -> Path:
    return Path(value or os.environ.get("HUNYUAN3D_CACHE", ROOT / ".cache/hunyuan3d"))


def legacy_paths() -> None:
    for path in (ROOT, ROOT / "hy3dshape", ROOT / "hy3dpaint", ROOT / "hy3dpaint/custom_rasterizer"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def configure_runtime(cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
    os.environ.setdefault("HY3DGEN_MODELS", str(cache / "models"))
    os.environ.setdefault("U2NET_HOME", str(cache / "rembg"))
    # Importing torch first exposes its CUDA shared objects to the native
    # rasterizer extension, including on distributions without a global CUDA
    # runtime linker path.
    import torch
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    if torch_lib.exists():
        os.environ["LD_LIBRARY_PATH"] = ":".join(filter(None, [str(torch_lib), os.environ.get("LD_LIBRARY_PATH")]))


def require_cuda() -> None:
    import torch
    if not torch.cuda.is_available():
        raise CliError(
            "unsupported_runtime",
            "CUDA is required for generation but is not available.",
            4,
        )


def require_file(path: Path, argument: str) -> None:
    if not path.is_file():
        raise CliError(
            "missing_input",
            f"{argument} does not exist or is not a file: {path}",
            3,
            {"argument": argument, "path": str(path)},
        )


def require_model_assets(cache: Path, components: set[str]) -> None:
    available = model_status(cache)
    missing = sorted(component for component in components if not available[component])
    if missing:
        raise CliError(
            "missing_model_assets",
            "Required model assets are missing. Run `hunyuan3d models pull` first.",
            3,
            {"missing": missing, "cache": str(cache)},
        )


def prepare_image(source: Path, destination: Path) -> Path:
    legacy_paths()
    from PIL import Image
    from hy3dshape.rembg import BackgroundRemover
    with contextlib.redirect_stdout(sys.stderr):
        result = BackgroundRemover()(Image.open(source).convert("RGB")).convert("RGBA")
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.save(destination)
    return destination


def pull_models(cache: Path, components: set[str]) -> list[str]:
    from huggingface_hub import snapshot_download
    pulled = []
    if "shape" in components:
        snapshot_download("tencent/Hunyuan3D-2.1", revision=MODEL_REVISIONS["hunyuan"],
                          allow_patterns=["hunyuan3d-dit-v2-1/*"],
                          local_dir=cache / "models/tencent/Hunyuan3D-2.1")
        pulled.append("shape")
    if "texture" in components:
        snapshot_download("tencent/Hunyuan3D-2.1", revision=MODEL_REVISIONS["hunyuan"],
                          allow_patterns=["hunyuan3d-paintpbr-v2-1/*"],
                          local_dir=cache / "models/tencent/Hunyuan3D-2.1")
        snapshot_download("facebook/dinov2-giant", revision=MODEL_REVISIONS["dino"],
                          local_dir=cache / "models/facebook/dinov2-giant")
        checkpoint = cache / "realesrgan/RealESRGAN_x4plus.pth"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if not checkpoint.exists():
            urllib.request.urlretrieve(REAL_ESRGAN_URL, checkpoint)
        pulled.append("texture")
    return pulled


def shape(image: Path, output: Path, cache: Path, steps: int, seed: int | None) -> Path:
    legacy_paths()
    import torch
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    with contextlib.redirect_stdout(sys.stderr):
        pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            str(cache / "models/tencent/Hunyuan3D-2.1")
        )
        if torch.cuda.get_device_properties(0).total_memory < 21 * 1024**3:
            pipe.enable_model_cpu_offload(device="cuda")
        generator = torch.Generator(device=pipe._execution_device).manual_seed(seed) if seed is not None else None
        mesh = pipe(image=str(image), num_inference_steps=steps, generator=generator)[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output)
    return output


def texture(mesh: Path, image: Path, output: Path, cache: Path) -> Path:
    legacy_paths()
    from torchvision_fix import apply_fix
    apply_fix()
    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline
    config = Hunyuan3DPaintConfig(max_num_view=6, resolution=512, cpu_offload=True)
    config.multiview_pretrained_path = str(cache / "models/tencent/Hunyuan3D-2.1")
    config.dino_ckpt_path = str(cache / "models/facebook/dinov2-giant")
    config.realesrgan_ckpt_path = str(cache / "realesrgan/RealESRGAN_x4plus.pth")
    with contextlib.redirect_stdout(sys.stderr):
        pipeline = Hunyuan3DPaintPipeline(config)
        pipeline(mesh_path=str(mesh), image_path=str(image), output_mesh_path=str(output.with_suffix(".obj")), save_glb=False)
    return output.with_suffix(".obj")


def model_status(cache: Path) -> dict:
    root = cache / "models/tencent/Hunyuan3D-2.1"
    return {
        "shape": (root / "hunyuan3d-dit-v2-1/model.fp16.ckpt").is_file(),
        "texture": (root / "hunyuan3d-paintpbr-v2-1/unet/diffusion_pytorch_model.bin").is_file(),
        "dino": (cache / "models/facebook/dinov2-giant/config.json").is_file(),
        "realesrgan": (cache / "realesrgan/RealESRGAN_x4plus.pth").is_file(),
    }


def build_parser() -> JsonArgumentParser:
    parser = JsonArgumentParser(prog="hunyuan3d", add_help=False)
    parser.add_argument("--version", action="version", version=package_version())
    parser.add_argument("--cache-dir")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", add_help=False)
    models = sub.add_parser("models", add_help=False)
    models.add_argument("action", choices=["pull", "status"])
    models.add_argument("--components", default="shape,texture")
    prep = sub.add_parser("prepare", add_help=False)
    prep.add_argument("image", type=Path)
    prep.add_argument("--output", type=Path, required=True)
    shape_cmd = sub.add_parser("shape", add_help=False)
    shape_cmd.add_argument("--image", type=Path, required=True)
    shape_cmd.add_argument("--output", type=Path, required=True)
    shape_cmd.add_argument("--steps", type=int, default=50)
    shape_cmd.add_argument("--seed", type=int)
    texture_cmd = sub.add_parser("texture", add_help=False)
    texture_cmd.add_argument("--mesh", type=Path, required=True)
    texture_cmd.add_argument("--image", type=Path, required=True)
    texture_cmd.add_argument("--output", type=Path, required=True)
    generate = sub.add_parser("generate", add_help=False)
    generate.add_argument("--image", type=Path, required=True)
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--shape-only", action="store_true")
    generate.add_argument("--steps", type=int, default=50)
    generate.add_argument("--seed", type=int)
    sub.add_parser("help", add_help=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "help":
            parser.print_help()
            return 0
        cache = cache_root(args.cache_dir)
        if args.command == "prepare":
            require_file(args.image, "image")
        elif args.command == "shape":
            require_file(args.image, "--image")
        elif args.command == "texture":
            require_file(args.mesh, "--mesh")
            require_file(args.image, "--image")
        elif args.command == "generate":
            require_file(args.image, "--image")
        if args.command == "models":
            components = set(args.components.split(","))
            invalid = sorted(component for component in components if component not in {"shape", "texture"})
            if invalid:
                raise CliError("invalid_arguments", "Unknown model component.", 2, {"invalid": invalid})
        configure_runtime(cache)
        if args.command == "doctor":
            import torch
            if not torch.cuda.is_available():
                raise CliError("unsupported_runtime", "CUDA is not available.", 4)
            return emit({"ok": True, "cuda": True, "nvcc": shutil.which("nvcc"), "cache": str(cache), "torch": torch.__version__})
        if args.command == "models":
            if args.action == "pull":
                return emit({"ok": True, "pulled": pull_models(cache, components), "cache": str(cache)})
            return emit({"ok": True, "cache": str(cache), "models": model_status(cache)})
        if args.command == "prepare":
            return emit({"ok": True, "image": str(prepare_image(args.image, args.output))})
        if args.command == "shape":
            require_cuda()
            require_model_assets(cache, {"shape"})
            return emit({"ok": True, "shape": str(shape(args.image, args.output, cache, args.steps, args.seed))})
        if args.command == "texture":
            require_cuda()
            require_model_assets(cache, {"texture", "dino", "realesrgan"})
            return emit({"ok": True, "texture": str(texture(args.mesh, args.image, args.output, cache))})
        require_cuda()
        required_models = {"shape"} if args.shape_only else {"shape", "texture", "dino", "realesrgan"}
        require_model_assets(cache, required_models)
        prepared = args.output_dir / "input.rgba.png"
        prepare_image(args.image, prepared)
        glb = shape(prepared, args.output_dir / "shape.glb", cache, args.steps, args.seed)
        result = {"ok": True, "input": str(prepared), "shape": str(glb)}
        if not args.shape_only:
            result["texture"] = str(texture(glb, prepared, args.output_dir / "textured", cache))
        return emit(result)
    except CliError as error:
        return emit_error(error)
    except (ImportError, ModuleNotFoundError) as error:
        return emit_error(CliError("dependency_failure", str(error), 5))
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        return emit_error(CliError("generation_failure", str(error), 1, {"exception": type(error).__name__}))


if __name__ == "__main__":
    raise SystemExit(main())
