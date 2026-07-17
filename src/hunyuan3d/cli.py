"""Stable JSON CLI around the repository's shape and texture engines."""
from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.metadata
import io
import json
import os
import shutil
import subprocess
import sys
import traceback
import urllib.request
from pathlib import Path
from typing import Any, TextIO

ROOT = Path(__file__).resolve().parents[2]
MODEL_REVISIONS = {
    "hunyuan": "0b94677654c57bb9a6b6845cd7b704ccf551d327",
    "dino": "611a9d42f2335e0f921f1e313ad3c1b7178d206d",
}
REAL_ESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
SCHEMA_VERSION = 1
GIB = 1024**3
MIN_WORKFLOW_VRAM = 10 * GIB
VALID_MODEL_COMPONENTS = frozenset({"prepare", "shape", "texture"})
SUPPORTED_MESH_EXTENSIONS = frozenset(
    {".3ds", ".dae", ".fbx", ".glb", ".gltf", ".obj", ".off", ".ply", ".stl", ".x3d"}
)
MIN_STEPS = 1

# Keep these paths as the single source of truth for both `models status` and
# the doctor report. Each entry is a required readiness file for that component.
MODEL_DEFINITIONS = {
    "shape": {
        "files": ("models/tencent/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/model.fp16.ckpt",),
        "pull_component": "shape",
    },
    "texture": {
        "files": (
            "models/tencent/Hunyuan3D-2.1/hunyuan3d-paintpbr-v2-1/unet/diffusion_pytorch_model.bin",
        ),
        "pull_component": "texture",
    },
    "dino": {
        "files": ("models/facebook/dinov2-giant/config.json",),
        "pull_component": "texture",
    },
    "realesrgan": {
        "files": ("realesrgan/RealESRGAN_x4plus.pth",),
        "pull_component": "texture",
    },
    "rembg": {
        "files": ("rembg/u2net.onnx",),
        "pull_component": "prepare",
    },
}

DEPENDENCY_DEFINITIONS = {
    "prepare": {
        "pillow": "PIL",
        "rembg": "rembg",
    },
    "shape": {
        "shape_pipeline": "hy3dshape.pipelines",
    },
    "texture": {
        "texture_pipeline": "textureGenPipeline",
        "basicsr": "basicsr",
        "cupy": "cupy",
        "open3d": "open3d",
        "pymeshlab": "pymeshlab",
        "pytorch_lightning": "pytorch_lightning",
        "realesrgan": "realesrgan",
        "scipy": "scipy",
        "xatlas": "xatlas",
    },
}

NATIVE_EXTENSION_DEFINITIONS = {
    "custom_rasterizer": "custom_rasterizer_kernel",
    "mesh_inpaint": "DifferentiableRenderer.mesh_inpaint_processor",
}

WORKFLOW_DEFINITIONS = {
    "prepare": {
        "command": "prepare",
        "models": ("rembg",),
        "dependencies": ("prepare",),
        "requires_cuda": False,
        "min_vram": None,
    },
    "shape": {
        "command": "shape",
        "models": ("shape",),
        "dependencies": ("shape",),
        "requires_cuda": True,
        "min_vram": MIN_WORKFLOW_VRAM,
    },
    "texture": {
        "command": "texture",
        "models": ("texture", "dino", "realesrgan"),
        "dependencies": ("texture",),
        "requires_cuda": True,
        "min_vram": MIN_WORKFLOW_VRAM,
        "native_extensions": tuple(NATIVE_EXTENSION_DEFINITIONS),
    },
    "generate_shape_only": {
        "command": "generate --shape-only",
        "models": ("rembg", "shape"),
        "dependencies": ("prepare", "shape"),
        "requires_cuda": True,
        "min_vram": MIN_WORKFLOW_VRAM,
    },
    "generate": {
        "command": "generate",
        "models": ("rembg", "shape", "texture", "dino", "realesrgan"),
        "dependencies": ("prepare", "shape", "texture"),
        "requires_cuda": True,
        "min_vram": MIN_WORKFLOW_VRAM,
        "native_extensions": tuple(NATIVE_EXTENSION_DEFINITIONS),
    },
}


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


class StderrTee:
    """Write diagnostic output to the terminal and a generate runtime log."""

    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def __getattr__(self, name: str):
        return getattr(self.streams[0], name)


@contextlib.contextmanager
def generate_runtime_log(args: argparse.Namespace):
    """Mirror generate diagnostics to a per-invocation log once its directory is known."""
    if args.command != "generate":
        yield
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "run.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stderr(StderrTee(sys.stderr, log)):
            yield


def cache_root(value: str | None) -> Path:
    return Path(value or os.environ.get("HUNYUAN3D_CACHE", ROOT / ".cache/hunyuan3d"))


def legacy_paths() -> None:
    for path in (ROOT, ROOT / "hy3dshape", ROOT / "hy3dpaint", ROOT / "hy3dpaint/custom_rasterizer"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def configure_runtime(cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    # The selected CLI cache is the sole model source for this process.  Do
    # not inherit a global Hugging Face or rembg cache when --cache-dir is set.
    os.environ["HF_HOME"] = str(cache / "huggingface")
    os.environ["HY3DGEN_MODELS"] = str(cache / "models")
    os.environ["U2NET_HOME"] = str(cache / "rembg")
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


def require_readable_image(path: Path, argument: str) -> None:
    """Validate an image without constructing any inference pipeline."""
    require_file(path, argument)
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
    except (OSError, SyntaxError, ValueError) as error:
        raise CliError(
            "invalid_input",
            f"{argument} is not a readable image: {path}",
            3,
            {"argument": argument, "path": str(path), "reason": str(error)},
        ) from error


def require_supported_mesh(path: Path, argument: str) -> None:
    """Validate a mesh path before the texture runtime is imported."""
    require_file(path, argument)
    extension = path.suffix.lower()
    if extension not in SUPPORTED_MESH_EXTENSIONS:
        raise CliError(
            "invalid_input",
            f"{argument} has an unsupported mesh format: {path.suffix or '<none>'}",
            3,
            {
                "argument": argument,
                "path": str(path),
                "extension": extension,
                "supported_formats": sorted(SUPPORTED_MESH_EXTENSIONS),
            },
        )


def require_valid_steps(steps: int) -> None:
    if steps < MIN_STEPS:
        raise CliError(
            "invalid_arguments",
            f"--steps must be at least {MIN_STEPS}.",
            2,
            {"argument": "--steps", "value": steps, "minimum": MIN_STEPS},
        )


def parse_model_components(value: str) -> set[str]:
    """Parse and strictly validate the comma-separated model component list."""
    raw_components = [component.strip() for component in value.split(",")]
    if not value.strip() or any(not component for component in raw_components):
        raise CliError(
            "invalid_arguments",
            "--components must contain one or more non-empty component names.",
            2,
            {"argument": "--components", "value": value},
        )

    duplicates = sorted(
        component
        for component in set(raw_components)
        if raw_components.count(component) > 1
    )
    if duplicates:
        raise CliError(
            "invalid_arguments",
            "Duplicate model components are not allowed.",
            2,
            {"argument": "--components", "duplicates": duplicates},
        )

    invalid = sorted(set(raw_components) - VALID_MODEL_COMPONENTS)
    if invalid:
        raise CliError(
            "invalid_arguments",
            "Unknown model component.",
            2,
            {"invalid": invalid},
        )
    return set(raw_components)


def planned_texture_outputs(output: Path) -> list[Path]:
    """Return files written by the checked-in OBJ texture exporter."""
    mesh = output.with_suffix(".obj")
    return [
        mesh,
        mesh.with_suffix(".mtl"),
        mesh.with_suffix(".jpg"),
        mesh.with_name(f"{mesh.stem}_metallic.jpg"),
        mesh.with_name(f"{mesh.stem}_roughness.jpg"),
    ]


def texture_write_plan(mesh: Path, output: Path) -> list[Path]:
    """Include the remesh intermediate as well as all predictable final files."""
    return [mesh.parent / "white_mesh_remesh.obj", *planned_texture_outputs(output)]


def generate_write_plan(output_dir: Path, shape_only: bool) -> list[Path]:
    paths = [
        output_dir / "input.rgba.png",
        output_dir / "shape.glb",
        output_dir / "run.log",
    ]
    if not shape_only:
        paths.extend(texture_write_plan(output_dir / "shape.glb", output_dir / "textured"))
    return paths


def _path_exists_including_broken_symlinks(path: Path) -> bool:
    return os.path.lexists(path)


def validate_output_plan(paths: list[Path], overwrite: bool) -> None:
    """Validate all output parents and collisions before runtime setup."""
    unique_paths = list(dict.fromkeys(paths))
    for path in unique_paths:
        parent = path.parent
        if parent.exists() and not parent.is_dir():
            raise CliError(
                "invalid_output",
                f"Output parent is not a directory: {parent}",
                3,
                {"path": str(path), "parent": str(parent)},
            )

        checked = parent
        while not checked.exists() and checked != checked.parent:
            checked = checked.parent
        if not checked.is_dir() or not os.access(checked, os.W_OK | os.X_OK):
            raise CliError(
                "invalid_output",
                f"Output parent is not writable: {parent}",
                3,
                {"path": str(path), "parent": str(parent), "checked_path": str(checked)},
            )

    colliding_paths = [
        path for path in unique_paths if _path_exists_including_broken_symlinks(path)
    ]
    non_file_collisions = [
        str(path)
        for path in colliding_paths
        if path.is_symlink() or not path.is_file()
    ]
    if non_file_collisions and overwrite:
        raise CliError(
            "invalid_output",
            "Planned output paths must be files when --overwrite is used.",
            3,
            {"paths": non_file_collisions},
        )

    collisions = [str(path) for path in colliding_paths]
    if collisions and not overwrite:
        raise CliError(
            "output_conflict",
            "One or more planned output files already exist. Pass --overwrite to replace them.",
            3,
            {
                "paths": collisions,
                "planned_outputs": [str(path) for path in unique_paths],
                "overwrite_flag": "--overwrite",
            },
        )


def validate_command(args: argparse.Namespace) -> set[str] | None:
    """Perform all deterministic validation before importing runtime dependencies."""
    if args.command == "models":
        return parse_model_components(args.components)
    if args.command == "doctor" or args.command == "help":
        return None

    if args.command == "prepare":
        require_readable_image(args.image, "image")
        validate_output_plan([args.output], args.overwrite)
    elif args.command == "shape":
        require_readable_image(args.image, "--image")
        require_valid_steps(args.steps)
        validate_output_plan([args.output], args.overwrite)
    elif args.command == "texture":
        require_supported_mesh(args.mesh, "--mesh")
        require_readable_image(args.image, "--image")
        validate_output_plan(texture_write_plan(args.mesh, args.output), args.overwrite)
    elif args.command == "generate":
        require_readable_image(args.image, "--image")
        require_valid_steps(args.steps)
        validate_output_plan(
            generate_write_plan(args.output_dir, args.shape_only), args.overwrite
        )
    return None


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
    if "prepare" in components:
        legacy_paths()
        from hy3dshape.rembg import BackgroundRemover
        # Constructing the rembg session downloads U²-Net into U2NET_HOME.
        # configure_runtime() points that environment variable at this cache.
        BackgroundRemover()
        pulled.append("prepare")
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
    config.multiview_cfg_path = str(ROOT / "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml")
    config.realesrgan_ckpt_path = str(cache / "realesrgan/RealESRGAN_x4plus.pth")
    with contextlib.redirect_stdout(sys.stderr):
        pipeline = Hunyuan3DPaintPipeline(config)
        pipeline(mesh_path=str(mesh), image_path=str(image), output_mesh_path=str(output.with_suffix(".obj")), save_glb=False)
    return output.with_suffix(".obj")


def model_readiness(cache: Path) -> dict:
    """Return detailed, read-only readiness for every cached model component."""
    readiness = {}
    for component, definition in MODEL_DEFINITIONS.items():
        paths = [cache / relative_path for relative_path in definition["files"]]
        missing = [str(path) for path in paths if not path.is_file()]
        readiness[component] = {
            "ready": not missing,
            "files": [str(path) for path in paths],
            "missing": missing,
            "pull_component": definition["pull_component"],
        }
    return readiness


def model_status(cache: Path) -> dict:
    """Return the existing compact model status contract."""
    return {
        component: details["ready"]
        for component, details in model_readiness(cache).items()
    }


def probe_import(module_name: str) -> dict:
    """Import a module without allowing its diagnostics to corrupt JSON output."""
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            importlib.import_module(module_name)
    except Exception as error:
        return {
            "ready": False,
            "module": module_name,
            "error": str(error),
            "exception": type(error).__name__,
        }
    return {"ready": True, "module": module_name}


def dependency_status() -> dict:
    """Probe runtime dependencies without constructing pipelines or loading models."""
    legacy_paths()
    status = {}
    for group, definitions in DEPENDENCY_DEFINITIONS.items():
        checks = {
            name: probe_import(module_name) for name, module_name in definitions.items()
        }
        status[group] = {
            "ready": all(check["ready"] for check in checks.values()),
            "checks": checks,
        }
    return status


def native_extension_status() -> dict:
    """Probe the compiled extensions used by the texture renderer."""
    legacy_paths()
    return {
        name: probe_import(module_name)
        for name, module_name in NATIVE_EXTENSION_DEFINITIONS.items()
    }


def driver_version() -> str | None:
    command = shutil.which("nvidia-smi")
    if command is None:
        return None
    try:
        result = subprocess.run(
            [command, "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return next(
        (line.strip() for line in result.stdout.splitlines() if line.strip()), None
    )


def gpu_status() -> dict:
    """Collect GPU facts without changing CUDA state or creating cache files."""
    status: dict[str, Any] = {
        "available": False,
        "visible": os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"},
        "count": 0,
        "name": None,
        "driver_version": driver_version(),
        "torch_version": None,
        "torch_cuda_version": None,
        "nvcc": shutil.which("nvcc"),
        "available_vram_bytes": None,
        "total_vram_bytes": None,
    }
    try:
        import torch
    except Exception as error:
        status["error"] = {"exception": type(error).__name__, "message": str(error)}
        return status

    status["torch_version"] = getattr(torch, "__version__", None)
    status["torch_cuda_version"] = getattr(
        getattr(torch, "version", None), "cuda", None
    )
    try:
        status["available"] = bool(torch.cuda.is_available())
    except Exception as error:
        status["error"] = {"exception": type(error).__name__, "message": str(error)}
        return status
    if not status["available"]:
        return status

    status["visible"] = True
    try:
        status["count"] = int(torch.cuda.device_count())
    except Exception as error:
        status["error"] = {"exception": type(error).__name__, "message": str(error)}
        status["available"] = False
        return status
    if status["count"] == 0:
        status["available"] = False
        return status

    try:
        properties = torch.cuda.get_device_properties(0)
        status["name"] = getattr(
            properties, "name", None
        ) or torch.cuda.get_device_name(0)
        status["total_vram_bytes"] = int(properties.total_memory)
    except Exception as error:
        status["error"] = {"exception": type(error).__name__, "message": str(error)}
        status["available"] = False
        return status
    try:
        available, total = torch.cuda.mem_get_info(0)
        status["available_vram_bytes"] = int(available)
        status["total_vram_bytes"] = int(total)
    except (AttributeError, RuntimeError, TypeError):
        # Older or mocked torch builds may not expose mem_get_info. Total VRAM
        # still gives the agent a useful capability decision.
        pass
    return status


def disk_status(path: Path) -> dict:
    """Inspect the nearest existing parent so a missing cache is not created."""
    requested = path.expanduser()
    checked = requested
    while not checked.exists() and checked != checked.parent:
        checked = checked.parent
    try:
        usage = shutil.disk_usage(checked)
    except OSError as error:
        return {
            "path": str(requested),
            "checked_path": str(checked),
            "available": False,
            "error": {"exception": type(error).__name__, "message": str(error)},
        }
    return {
        "path": str(requested),
        "checked_path": str(checked),
        "available": True,
        "free_bytes": int(usage.free),
        "total_bytes": int(usage.total),
    }


def _blocker(
    code: str,
    message: str,
    remediation: str,
    details: dict | None = None,
    next_command: str | None = None,
) -> dict:
    result = {"code": code, "message": message, "remediation": remediation}
    if details is not None:
        result["details"] = details
    if next_command is not None:
        result["next_command"] = next_command
    return result


def workflow_status(
    gpu: dict, models: dict, dependencies: dict, native_extensions: dict
) -> dict:
    """Build actionable readiness decisions from the independent doctor probes."""
    workflows = {}
    for workflow, definition in WORKFLOW_DEFINITIONS.items():
        blockers = []
        next_commands = []
        model_checks = {
            component: models[component]["ready"] for component in definition["models"]
        }
        dependency_checks = {
            group: dependencies[group]["ready"] for group in definition["dependencies"]
        }
        native_checks = {
            name: native_extensions[name]["ready"]
            for name in definition.get("native_extensions", ())
        }

        for component in definition["models"]:
            if model_checks[component]:
                continue
            pull_component = models[component]["pull_component"]
            blocker = _blocker(
                "missing_model",
                f"The {component} model component is not ready.",
                f"Download the {pull_component} model assets into the selected cache.",
                {"component": component, "missing_files": models[component]["missing"]},
                f"hunyuan3d models pull --components {pull_component}",
            )
            blockers.append(blocker)
            next_commands.append(blocker["next_command"])

        for group in definition["dependencies"]:
            for name, check in dependencies[group]["checks"].items():
                if check["ready"]:
                    continue
                blocker = _blocker(
                    "missing_dependency",
                    f"The {name} dependency is not importable.",
                    "Install the selected CLI profile and its locked dependencies.",
                    {
                        "dependency": name,
                        "module": check["module"],
                        "error": check.get("error"),
                    },
                    "./bootstrap.sh --profile all --install-command"
                    if workflow in {"texture", "generate"}
                    else "./bootstrap.sh --profile shape --install-command",
                )
                blockers.append(blocker)
                next_commands.append(blocker["next_command"])

        for name in definition.get("native_extensions", ()):
            check = native_extensions[name]
            if check["ready"]:
                continue
            blocker = _blocker(
                "missing_native_extension",
                f"The {name} native extension is not importable.",
                "Build the texture profile's native extensions in the active environment.",
                {
                    "extension": name,
                    "module": check["module"],
                    "error": check.get("error"),
                },
                "./bootstrap.sh --profile texture --install-command",
            )
            blockers.append(blocker)
            next_commands.append(blocker["next_command"])

        if definition["requires_cuda"] and not gpu.get("available", False):
            blocker = _blocker(
                "cuda_unavailable",
                "CUDA is not available to the selected PyTorch runtime.",
                "Use a host with a visible NVIDIA GPU and a CUDA-enabled PyTorch environment.",
                {
                    "cuda_visible": gpu.get("visible", False),
                    "torch_cuda_version": gpu.get("torch_cuda_version"),
                },
                "hunyuan3d doctor",
            )
            blockers.append(blocker)
            next_commands.append(blocker["next_command"])
        elif definition["min_vram"] is not None:
            available_vram = gpu.get("available_vram_bytes")
            if available_vram is None:
                available_vram = gpu.get("total_vram_bytes")
            if available_vram is not None and available_vram < definition["min_vram"]:
                blocker = _blocker(
                    "insufficient_vram",
                    f"The {workflow} workflow needs more available VRAM than this GPU provides.",
                    "Free GPU memory or use a GPU with more VRAM; CPU offload is already enabled where supported.",
                    {
                        "required_bytes": definition["min_vram"],
                        "available_bytes": available_vram,
                        "total_bytes": gpu.get("total_vram_bytes"),
                    },
                    "hunyuan3d doctor",
                )
                blockers.append(blocker)
                next_commands.append(blocker["next_command"])

        workflows[workflow] = {
            "command": definition["command"],
            "ready": not blockers,
            "checks": {
                "models": model_checks,
                "dependencies": dependency_checks,
                "native_extensions": native_checks,
                "cuda": gpu.get("available", False)
                if definition["requires_cuda"]
                else True,
                "minimum_vram_bytes": definition["min_vram"],
                "available_vram_bytes": gpu.get("available_vram_bytes"),
            },
            "blockers": blockers,
            "next_commands": list(dict.fromkeys(next_commands)),
        }
    return workflows


def doctor_report(cache: Path, output_dir: Path) -> tuple[dict, int]:
    """Create the complete read-only preflight report and its exit status."""
    gpu = gpu_status()
    models = model_readiness(cache)
    dependencies = dependency_status()
    native_extensions = native_extension_status()
    workflows = workflow_status(gpu, models, dependencies, native_extensions)
    payload = {
        "ok": workflows["generate"]["ready"],
        "cache": str(cache),
        "output_dir": str(output_dir),
        # These fields retain the small doctor contract used by existing agents.
        "cuda": gpu["available"],
        "nvcc": gpu["nvcc"],
        "torch": gpu["torch_version"],
        "torch_cuda": gpu["torch_cuda_version"],
        "gpu": gpu,
        "models": {
            component: details["ready"] for component, details in models.items()
        },
        "model_readiness": models,
        "dependencies": dependencies,
        "native_extensions": native_extensions,
        "disk": {"cache": disk_status(cache), "output": disk_status(output_dir)},
        "workflows": workflows,
        "readiness": {
            workflow: details["ready"] for workflow, details in workflows.items()
        },
    }
    return payload, 0 if payload["ok"] else 4


def build_parser() -> JsonArgumentParser:
    parser = JsonArgumentParser(prog="hunyuan3d", add_help=False)
    parser.add_argument("--version", action="version", version=package_version())
    parser.add_argument("--cache-dir")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", add_help=False)
    doctor.add_argument("--output-dir", type=Path, default=Path.cwd())
    models = sub.add_parser("models", add_help=False)
    models.add_argument("action", choices=["pull", "status"])
    models.add_argument("--components", default="prepare,shape,texture")
    prep = sub.add_parser("prepare", add_help=False)
    prep.add_argument("image", type=Path)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--overwrite", action="store_true")
    shape_cmd = sub.add_parser("shape", add_help=False)
    shape_cmd.add_argument("--image", type=Path, required=True)
    shape_cmd.add_argument("--output", type=Path, required=True)
    shape_cmd.add_argument("--steps", type=int, default=50)
    shape_cmd.add_argument("--seed", type=int)
    shape_cmd.add_argument("--overwrite", action="store_true")
    texture_cmd = sub.add_parser("texture", add_help=False)
    texture_cmd.add_argument("--mesh", type=Path, required=True)
    texture_cmd.add_argument("--image", type=Path, required=True)
    texture_cmd.add_argument("--output", type=Path, required=True)
    texture_cmd.add_argument("--overwrite", action="store_true")
    generate = sub.add_parser("generate", add_help=False)
    generate.add_argument("--image", type=Path, required=True)
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--shape-only", action="store_true")
    generate.add_argument("--steps", type=int, default=50)
    generate.add_argument("--seed", type=int)
    generate.add_argument("--overwrite", action="store_true")
    sub.add_parser("help", add_help=False)
    return parser


def run_command(args: argparse.Namespace, cache: Path, components: set[str] | None) -> int:
    configure_runtime(cache)
    if args.command == "models":
        if args.action == "pull":
            return emit({"ok": True, "pulled": pull_models(cache, components or set()), "cache": str(cache)})
        return emit({"ok": True, "cache": str(cache), "models": model_status(cache)})
    if args.command == "prepare":
        require_model_assets(cache, {"rembg"})
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
    required_models = (
        {"rembg", "shape"}
        if args.shape_only
        else {"rembg", "shape", "texture", "dino", "realesrgan"}
    )
    require_model_assets(cache, required_models)
    prepared = args.output_dir / "input.rgba.png"
    print("Preparing image...", file=sys.stderr, flush=True)
    prepare_image(args.image, prepared)
    print("Generating shape...", file=sys.stderr, flush=True)
    glb = shape(prepared, args.output_dir / "shape.glb", cache, args.steps, args.seed)
    result = {"ok": True, "input": str(prepared), "shape": str(glb)}
    if not args.shape_only:
        print("Generating texture...", file=sys.stderr, flush=True)
        result["texture"] = str(texture(glb, prepared, args.output_dir / "textured", cache))
    return emit(result)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except CliError as error:
        return emit_error(error)

    try:
        if args.command == "help":
            parser.print_help()
            return 0

        cache = cache_root(args.cache_dir)
        components = validate_command(args)
        if args.command == "doctor":
            report, code = doctor_report(cache, args.output_dir)
            return emit(report, code)

        # Validation completes before this context creates the output directory
        # or runtime log. Invalid requests therefore cannot mutate output state.
        if args.command == "generate":
            with generate_runtime_log(args):
                return run_command(args, cache, components)
        return run_command(args, cache, components)
    except CliError as error:
        return emit_error(error)
    except (ImportError, ModuleNotFoundError) as error:
        return emit_error(CliError("dependency_failure", str(error), 5))
    except Exception as error:
        if args.command == "generate":
            args.output_dir.mkdir(parents=True, exist_ok=True)
            with (args.output_dir / "run.log").open("a", encoding="utf-8") as log:
                traceback.print_exc(file=StderrTee(sys.stderr, log))
        else:
            traceback.print_exc(file=sys.stderr)
        return emit_error(CliError("generation_failure", str(error), 1, {"exception": type(error).__name__}))


if __name__ == "__main__":
    raise SystemExit(main())
