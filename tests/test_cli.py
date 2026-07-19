import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
from PIL import Image

from hunyuan3d import cli


def test_cache_defaults_to_clone_local_cache(monkeypatch):
    monkeypatch.delenv("HUNYUAN3D_CACHE", raising=False)
    assert cli.cache_root(None) == cli.ROOT / ".cache/hunyuan3d"


def test_model_status_reports_missing_models(tmp_path: Path):
    assert cli.model_status(tmp_path) == {
        "shape": False,
        "texture": False,
        "dino": False,
        "realesrgan": False,
        "rembg": False,
    }


def test_model_status_uses_the_cli_local_dino_directory(tmp_path: Path):
    dino = tmp_path / "models/facebook/dinov2-giant"
    dino.mkdir(parents=True)
    (dino / "config.json").touch()

    assert cli.model_status(tmp_path)["dino"] is True


def test_model_status_uses_the_cli_local_rembg_directory(tmp_path: Path):
    rembg = tmp_path / "rembg"
    rembg.mkdir()
    (rembg / "u2net.onnx").touch()

    assert cli.model_status(tmp_path)["rembg"] is True


def complete_dependency_status() -> dict:
    return {
        group: {
            "ready": True,
            "checks": {
                name: {"ready": True, "module": module}
                for name, module in definitions.items()
            },
        }
        for group, definitions in cli.DEPENDENCY_DEFINITIONS.items()
    }


def complete_native_extension_status() -> dict:
    return {
        name: {"ready": True, "module": module}
        for name, module in cli.NATIVE_EXTENSION_DEFINITIONS.items()
    }


def test_doctor_reports_ready_workflows_from_mocked_probes(tmp_path: Path, monkeypatch):
    cache = tmp_path / "cache"
    for definition in cli.MODEL_DEFINITIONS.values():
        for relative_path in definition["files"]:
            path = cache / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    monkeypatch.setattr(
        cli,
        "gpu_status",
        lambda: {
            "available": True,
            "visible": True,
            "count": 1,
            "name": "Mock RTX",
            "driver_version": "550.54.14",
            "torch_version": "2.6.0",
            "torch_cuda_version": "12.4",
            "nvcc": "/usr/bin/nvcc",
            "available_vram_bytes": 12 * 1024**3,
            "total_vram_bytes": 12 * 1024**3,
        },
    )
    monkeypatch.setattr(cli, "dependency_status", complete_dependency_status)
    monkeypatch.setattr(
        cli, "native_extension_status", complete_native_extension_status
    )

    report, code = cli.doctor_report(cache, tmp_path / "output")

    assert code == 0
    assert report["ok"] is True
    assert report["gpu"]["name"] == "Mock RTX"
    assert report["gpu"]["available_vram_bytes"] == 12 * 1024**3
    assert all(report["readiness"].values())
    assert report["workflows"]["generate"]["blockers"] == []
    assert not (tmp_path / "output").exists()


def test_doctor_reports_insufficient_vram_as_a_stable_blocker(
    tmp_path: Path, monkeypatch
):
    cache = tmp_path / "cache"
    for definition in cli.MODEL_DEFINITIONS.values():
        for relative_path in definition["files"]:
            path = cache / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    monkeypatch.setattr(
        cli,
        "gpu_status",
        lambda: {
            "available": True,
            "visible": True,
            "count": 1,
            "name": "Mock GPU",
            "driver_version": "550.54.14",
            "torch_version": "2.6.0",
            "torch_cuda_version": "12.4",
            "nvcc": "/usr/bin/nvcc",
            "available_vram_bytes": 8 * 1024**3,
            "total_vram_bytes": 12 * 1024**3,
        },
    )
    monkeypatch.setattr(cli, "dependency_status", complete_dependency_status)
    monkeypatch.setattr(
        cli, "native_extension_status", complete_native_extension_status
    )

    report, code = cli.doctor_report(cache, tmp_path / "output")

    assert code == 4
    assert report["workflows"]["shape"]["ready"] is False
    assert report["workflows"]["shape"]["blockers"][0]["code"] == "insufficient_vram"
    assert (
        report["workflows"]["shape"]["blockers"][0]["details"]["required_bytes"]
        == 10 * 1024**3
    )


def test_doctor_reports_stable_blockers_without_creating_state(
    tmp_path: Path, monkeypatch
):
    cache = tmp_path / "missing-cache"
    output = tmp_path / "missing-output"
    dependencies = complete_dependency_status()
    dependencies["texture"]["ready"] = False
    dependencies["texture"]["checks"]["xatlas"] = {
        "ready": False,
        "module": "xatlas",
        "error": "No module named 'xatlas'",
    }
    native = complete_native_extension_status()
    native["custom_rasterizer"]["ready"] = False
    native["custom_rasterizer"]["error"] = "native module unavailable"

    monkeypatch.setattr(
        cli,
        "gpu_status",
        lambda: {
            "available": False,
            "visible": False,
            "count": 0,
            "name": None,
            "driver_version": None,
            "torch_version": "2.6.0",
            "torch_cuda_version": None,
            "nvcc": None,
            "available_vram_bytes": None,
            "total_vram_bytes": None,
        },
    )
    monkeypatch.setattr(cli, "dependency_status", lambda: dependencies)
    monkeypatch.setattr(cli, "native_extension_status", lambda: native)

    report, code = cli.doctor_report(cache, output)

    full_codes = {
        blocker["code"] for blocker in report["workflows"]["generate"]["blockers"]
    }
    assert code == 4
    assert report["ok"] is False
    assert report["readiness"]["prepare"] is False
    assert "missing_model" in full_codes
    assert "missing_dependency" in full_codes
    assert "missing_native_extension" in full_codes
    assert "cuda_unavailable" in full_codes
    assert all(
        blocker["remediation"]
        for blocker in report["workflows"]["generate"]["blockers"]
    )
    assert not cache.exists()
    assert not output.exists()


def test_configure_runtime_uses_the_selected_cache(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HF_HOME", "/global/huggingface")
    monkeypatch.setenv("HY3DGEN_MODELS", "/global/models")
    monkeypatch.setenv("U2NET_HOME", "/global/rembg")
    fake_torch = types.SimpleNamespace(__file__=str(tmp_path / "torch/__init__.py"))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    cli.configure_runtime(tmp_path)

    assert os.environ["HF_HOME"] == str(tmp_path / "huggingface")
    assert os.environ["HY3DGEN_MODELS"] == str(tmp_path / "models")
    assert os.environ["U2NET_HOME"] == str(tmp_path / "rembg")


def test_pull_models_preloads_rembg_into_the_selected_cache(
    monkeypatch, tmp_path: Path
):
    loaded = []

    class FakeBackgroundRemover:
        def __init__(self):
            loaded.append(os.environ["U2NET_HOME"])

    monkeypatch.setattr(cli, "legacy_paths", lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=lambda **kwargs: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hy3dshape.rembg",
        types.SimpleNamespace(BackgroundRemover=FakeBackgroundRemover),
    )
    monkeypatch.setenv("U2NET_HOME", str(tmp_path / "rembg"))

    assert cli.pull_models(tmp_path, {"prepare"}) == ["prepare"]
    assert loaded == [str(tmp_path / "rembg")]


def test_prepare_requires_the_cached_background_remover(tmp_path: Path):
    source = tmp_path / "input.png"
    Image.new("RGB", (1, 1), "white").save(source)

    assert (
        cli.main(
            [
                "--cache-dir",
                str(tmp_path),
                "prepare",
                str(source),
                "--output",
                str(tmp_path / "output.png"),
            ]
        )
        == 3
    )


def test_shape_loads_weights_from_the_selected_cache(monkeypatch, tmp_path: Path):
    loaded_paths = []

    class FakeMesh:
        def export(self, output):
            assert output == tmp_path / "shape.glb"

    class FakePipeline:
        _execution_device = "cuda"

        def __call__(self, **kwargs):
            return [FakeMesh()]

        def enable_model_cpu_offload(self, **kwargs):
            raise AssertionError("CPU offload should not be enabled in this test")

    class FakePipelineClass:
        @staticmethod
        def from_pretrained(path):
            loaded_paths.append(path)
            return FakePipeline()

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            get_device_properties=lambda _: types.SimpleNamespace(
                total_memory=32 * 1024**3
            )
        ),
        Generator=lambda device: types.SimpleNamespace(
            manual_seed=lambda seed: (device, seed)
        ),
    )
    monkeypatch.setattr(cli, "legacy_paths", lambda: None)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "hy3dshape.pipelines",
        types.SimpleNamespace(Hunyuan3DDiTFlowMatchingPipeline=FakePipelineClass),
    )

    assert (
        cli.shape(tmp_path / "input.png", tmp_path / "shape.glb", tmp_path, 50, None)
        == tmp_path / "shape.glb"
    )
    assert loaded_paths == [str(tmp_path / "models/tencent/Hunyuan3D-2.1")]


def test_texture_config_uses_only_the_selected_cache(monkeypatch, tmp_path: Path):
    captured = {}

    class FakeConfig:
        def __init__(self, **kwargs):
            captured["config"] = self

    class FakePipeline:
        def __init__(self, config):
            assert config is captured["config"]

        def __call__(self, **kwargs):
            captured["call"] = kwargs

    monkeypatch.setattr(cli, "legacy_paths", lambda: None)
    monkeypatch.setitem(
        sys.modules, "torchvision_fix", types.SimpleNamespace(apply_fix=lambda: None)
    )
    monkeypatch.setitem(
        sys.modules,
        "textureGenPipeline",
        types.SimpleNamespace(
            Hunyuan3DPaintConfig=FakeConfig,
            Hunyuan3DPaintPipeline=FakePipeline,
        ),
    )

    assert (
        cli.texture(
            tmp_path / "shape.glb",
            tmp_path / "input.png",
            tmp_path / "textured.glb",
            tmp_path,
        )
        == tmp_path / "textured.obj"
    )
    config = captured["config"]
    assert config.multiview_pretrained_path == str(
        tmp_path / "models/tencent/Hunyuan3D-2.1"
    )
    assert config.dino_ckpt_path == str(tmp_path / "models/facebook/dinov2-giant")
    assert config.realesrgan_ckpt_path == str(
        tmp_path / "realesrgan/RealESRGAN_x4plus.pth"
    )
    assert captured["call"]["save_glb"] is False


def test_texture_can_emit_a_glb_and_removes_obj_intermediates(
    monkeypatch, tmp_path: Path
):
    captured = {}

    class FakeConfig:
        def __init__(self, **_kwargs):
            pass

    class FakePipeline:
        def __init__(self, _config):
            pass

        def __call__(self, **kwargs):
            captured["call"] = kwargs
            obj = Path(kwargs["output_mesh_path"])
            obj.write_text("obj")
            obj.with_suffix(".mtl").write_text("mtl")
            obj.with_suffix(".jpg").write_text("albedo")
            obj.with_name(f"{obj.stem}_metallic.jpg").write_text("metallic")
            obj.with_name(f"{obj.stem}_roughness.jpg").write_text("roughness")
            obj.parent.joinpath("white_mesh_remesh.obj").write_text("remesh")
            obj.with_suffix(".glb").write_text("glb")

    monkeypatch.setattr(cli, "legacy_paths", lambda: None)
    monkeypatch.setitem(
        sys.modules, "torchvision_fix", types.SimpleNamespace(apply_fix=lambda: None)
    )
    monkeypatch.setitem(
        sys.modules,
        "textureGenPipeline",
        types.SimpleNamespace(
            Hunyuan3DPaintConfig=FakeConfig,
            Hunyuan3DPaintPipeline=FakePipeline,
        ),
    )

    output = cli.texture(
        tmp_path / "shape.glb",
        tmp_path / "input.png",
        tmp_path / "textured",
        tmp_path,
        output_format="glb",
    )

    assert output == tmp_path / "textured.glb"
    assert output.is_file()
    assert captured["call"]["save_glb"] is True
    assert not (tmp_path / "textured.obj").exists()
    assert not (tmp_path / "white_mesh_remesh.obj").exists()


def test_texture_uses_repository_absolute_config_path(tmp_path: Path, monkeypatch):
    captured = {}

    class FakeConfig:
        def __init__(self, **_kwargs):
            pass

    class FakePipeline:
        def __init__(self, config):
            captured["config"] = config

        def __call__(self, **_kwargs):
            pass

    monkeypatch.setattr(cli, "legacy_paths", lambda: None)
    monkeypatch.setitem(
        sys.modules, "torchvision_fix", types.SimpleNamespace(apply_fix=lambda: None)
    )
    monkeypatch.setitem(
        sys.modules,
        "textureGenPipeline",
        types.SimpleNamespace(
            Hunyuan3DPaintConfig=FakeConfig, Hunyuan3DPaintPipeline=FakePipeline
        ),
    )

    cli.texture(
        tmp_path / "shape.glb",
        tmp_path / "input.png",
        tmp_path / "textured",
        tmp_path / "cache",
    )

    assert captured["config"].multiview_cfg_path == str(
        cli.ROOT / "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    )


def test_multiview_pipeline_trusts_checked_in_custom_pipeline(
    tmp_path: Path, monkeypatch
):
    cli.legacy_paths()
    from utils import multiview_utils

    captured = {}

    class FakePipeline:
        scheduler = types.SimpleNamespace(config={})
        unet = types.SimpleNamespace(use_dino=False)

        def set_progress_bar_config(self, **_kwargs):
            pass

        def eval(self):
            pass

        def to(self, _device):
            return self

    monkeypatch.setattr(
        multiview_utils.huggingface_hub,
        "snapshot_download",
        lambda **_kwargs: str(tmp_path),
    )
    monkeypatch.setattr(
        multiview_utils.UniPCMultistepScheduler,
        "from_config",
        lambda _config, **_kwargs: object(),
    )

    def fake_from_pretrained(*_args, **kwargs):
        captured.update(kwargs)
        return FakePipeline()

    monkeypatch.setattr(
        multiview_utils.DiffusionPipeline, "from_pretrained", fake_from_pretrained
    )
    config = types.SimpleNamespace(
        device="cpu",
        multiview_cfg_path=str(cli.ROOT / "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"),
        multiview_pretrained_path="tencent/Hunyuan3D-2.1",
        cpu_offload=False,
    )

    multiview_utils.multiviewDiffusionNet(config)

    assert captured["trust_remote_code"] is True


def test_installed_cli_is_available_from_another_directory(tmp_path: Path):
    script_dir = Path(sys.executable).parent
    assert (script_dir / "hunyuan3d").is_file()
    env = {**os.environ, "PATH": f"{script_dir}{os.pathsep}{os.environ['PATH']}"}

    version = subprocess.run(
        ["hunyuan3d", "--version"],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert version.stdout.strip() == cli.package_version()

    doctor = subprocess.run(
        ["hunyuan3d", "doctor"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert doctor.returncode in {0, 4}
    doctor_payload = json.loads(doctor.stdout)
    assert doctor_payload["schema_version"] == 1
    assert doctor_payload["cache"] == str(cli.ROOT / ".cache/hunyuan3d")


def test_persistent_command_installer_runs_the_cli_from_another_directory(
    tmp_path: Path,
):
    command_dir = tmp_path / "bin"
    shell_rc = tmp_path / "bashrc"
    env = {
        **os.environ,
        "HUNYUAN3D_COMMAND_DIR": str(command_dir),
        "HUNYUAN3D_SHELL_RC": str(shell_rc),
    }
    subprocess.run(
        [Path(__file__).parents[1] / "install-command.sh"], env=env, check=True
    )

    assert (command_dir / "hunyuan3d").is_file()
    assert f'export PATH="{command_dir}:$PATH"' in shell_rc.read_text()
    command_env = {**env, "PATH": f"{command_dir}{os.pathsep}{os.environ['PATH']}"}
    version = subprocess.run(
        ["hunyuan3d", "--version"],
        cwd=tmp_path,
        env=command_env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert version.stdout.strip() == cli.package_version()


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    executable = Path(sys.executable).with_name("hunyuan3d")
    return subprocess.run(
        [executable, *args], text=True, capture_output=True, check=False
    )


def json_events(result: subprocess.CompletedProcess[str]) -> list[dict]:
    lines = [line for line in result.stderr.splitlines() if line.strip()]
    events = [json.loads(line) for line in lines]
    assert events
    assert all(event["schema_version"] == 1 for event in events)
    assert len({event["run_id"] for event in events}) == 1
    return events


def json_result(result: subprocess.CompletedProcess[str]) -> dict:
    if result.stderr.strip():
        json_events(result)
    assert len(result.stdout.splitlines()) == 1
    return json.loads(result.stdout)


@pytest.mark.parametrize(
    "arguments",
    [
        ("not-a-command",),
        ("shape", "--image", "input.png"),
        (
            "shape",
            "--image",
            "input.png",
            "--output",
            "output.glb",
            "--steps",
            "not-a-number",
        ),
    ],
)
def test_parse_failures_are_versioned_json_errors(arguments: tuple[str, ...]):
    result = run_cli(*arguments)
    assert result.returncode == 2
    payload = json_result(result)
    assert payload["schema_version"] == 1
    assert payload["ok"] is False
    assert payload["error"]["code"] == "invalid_arguments"


def test_missing_input_is_a_versioned_json_error(tmp_path: Path):
    result = run_cli(
        "prepare",
        str(tmp_path / "missing.png"),
        "--output",
        str(tmp_path / "output.png"),
    )
    assert result.returncode == 3
    payload = json_result(result)
    assert payload["schema_version"] == 1
    assert payload["error"]["code"] == "missing_input"
    assert payload["error"]["details"]["argument"] == "image"


def test_invalid_model_component_is_a_versioned_json_error():
    result = run_cli("models", "status", "--components", "unknown")
    assert result.returncode == 2
    assert json_result(result)["error"] == {
        "code": "invalid_arguments",
        "message": "Unknown model component.",
        "details": {"invalid": ["unknown"]},
    }


@pytest.mark.parametrize("components", ["", ",", "shape,,texture", "shape,shape"])
def test_invalid_model_component_selections_fail_without_pulling(components: str):
    result = run_cli("models", "pull", "--components", components)

    assert result.returncode == 2
    payload = json_result(result)
    assert payload["schema_version"] == 1
    assert payload["ok"] is False
    assert payload["error"]["code"] == "invalid_arguments"


def test_unreadable_image_fails_before_runtime_setup(
    tmp_path: Path, monkeypatch, capsys
):
    source = tmp_path / "input.png"
    source.write_text("not an image")
    output = tmp_path / "output.png"

    monkeypatch.setattr(
        cli, "configure_runtime", lambda _cache: pytest.fail("runtime was configured")
    )

    assert cli.main(["prepare", str(source), "--output", str(output)]) == 3
    payload = json.loads(capsys.readouterr().out)

    assert payload["error"]["code"] == "invalid_input"
    assert not output.exists()


def test_invalid_steps_fail_before_runtime_setup(tmp_path: Path, monkeypatch, capsys):
    source = tmp_path / "input.png"
    Image.new("RGB", (1, 1), "white").save(source)
    output = tmp_path / "shape.glb"

    monkeypatch.setattr(
        cli, "configure_runtime", lambda _cache: pytest.fail("runtime was configured")
    )

    assert (
        cli.main(
            ["shape", "--image", str(source), "--output", str(output), "--steps", "0"]
        )
        == 2
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["error"]["code"] == "invalid_arguments"
    assert not output.exists()


def test_unsupported_mesh_format_fails_before_runtime_setup(
    tmp_path: Path, monkeypatch, capsys
):
    mesh = tmp_path / "mesh.txt"
    mesh.write_text("not a mesh")
    image = tmp_path / "input.png"
    Image.new("RGB", (1, 1), "white").save(image)

    monkeypatch.setattr(
        cli, "configure_runtime", lambda _cache: pytest.fail("runtime was configured")
    )

    assert (
        cli.main(
            [
                "texture",
                "--mesh",
                str(mesh),
                "--image",
                str(image),
                "--output",
                str(tmp_path / "textured"),
            ]
        )
        == 3
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["error"]["code"] == "invalid_input"


def test_generate_rejects_partial_output_plan_before_runtime_setup(
    tmp_path: Path, monkeypatch, capsys
):
    source = tmp_path / "input.png"
    Image.new("RGB", (1, 1), "white").save(source)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    existing = output_dir / "shape.glb"
    existing.write_text("existing")

    monkeypatch.setattr(
        cli, "configure_runtime", lambda _cache: pytest.fail("runtime was configured")
    )

    assert (
        cli.main(
            [
                "generate",
                "--image",
                str(source),
                "--output-dir",
                str(output_dir),
                "--shape-only",
            ]
        )
        == 3
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["error"]["code"] == "output_conflict"
    assert str(existing) in payload["error"]["details"]["paths"]
    assert not (output_dir / "run.log").exists()


def test_generate_write_plan_includes_texture_artifacts(tmp_path: Path):
    planned = {
        path.name for path in cli.generate_write_plan(tmp_path / "output", False)
    }

    assert planned == {
        "input.rgba.png",
        "shape.glb",
        "run.log",
        "white_mesh_remesh.obj",
        "textured.obj",
        "textured.mtl",
        "textured.jpg",
        "textured_metallic.jpg",
        "textured_roughness.jpg",
        "textured.glb",
    }

    obj_planned = {
        path.name
        for path in cli.generate_write_plan(tmp_path / "output", False, "obj")
    }
    assert "textured.glb" not in obj_planned
    assert cli.generate_write_plan(tmp_path / "output", True, "obj") == [
        tmp_path / "output/input.rgba.png",
        tmp_path / "output/shape.glb",
        tmp_path / "output/run.log",
    ]


def test_overwrite_allows_an_existing_direct_output(
    tmp_path: Path, monkeypatch, capsys
):
    source = tmp_path / "input.png"
    Image.new("RGB", (1, 1), "white").save(source)
    output = tmp_path / "output.png"
    output.write_text("old")

    monkeypatch.setattr(cli, "configure_runtime", lambda _cache: None)
    monkeypatch.setattr(cli, "require_model_assets", lambda _cache, _components: None)

    def fake_prepare(_source: Path, destination: Path) -> Path:
        destination.write_text("new")
        return destination

    monkeypatch.setattr(cli, "prepare_image", fake_prepare)

    assert (
        cli.main(["prepare", str(source), "--output", str(output), "--overwrite"]) == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["ok"] is True
    assert output.read_text() == "new"


def test_overwrite_does_not_treat_an_output_directory_as_a_file(tmp_path: Path, capsys):
    source = tmp_path / "input.png"
    Image.new("RGB", (1, 1), "white").save(source)
    output = tmp_path / "output.png"
    output.mkdir()

    assert (
        cli.main(["prepare", str(source), "--output", str(output), "--overwrite"]) == 3
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["error"]["code"] == "invalid_output"


def test_output_parent_must_be_a_directory(tmp_path: Path, capsys):
    source = tmp_path / "input.png"
    Image.new("RGB", (1, 1), "white").save(source)
    output_parent = tmp_path / "not-a-directory"
    output_parent.write_text("file")

    assert (
        cli.main(
            [
                "generate",
                "--image",
                str(source),
                "--output-dir",
                str(output_parent),
                "--shape-only",
            ]
        )
        == 3
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["error"]["code"] == "invalid_output"


def test_success_result_is_versioned_json():
    result = run_cli("models", "status")
    assert result.returncode == 0
    payload = json_result(result)
    assert payload["schema_version"] == 1
    assert payload["ok"] is True


def test_model_pull_reports_component_progress(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "configure_runtime", lambda cache: None)

    def fake_pull(cache: Path, components: set[str], reporter: cli.ProgressReporter):
        reporter.progress("model_pull", 1, len(components), component="shape")
        reporter.progress("model_pull", 2, len(components), component="prepare")
        return ["shape", "prepare"]

    monkeypatch.setattr(cli, "pull_models", fake_pull)

    assert (
        cli.main(
            [
                "--cache-dir",
                str(tmp_path / "cache"),
                "models",
                "pull",
                "--components",
                "shape,prepare",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    events = json_events(types.SimpleNamespace(stderr=captured.err))
    progress = [event for event in events if event["event"] == "progress"]
    assert [(event["current"], event["total"]) for event in progress] == [
        (0, 2),
        (1, 2),
        (2, 2),
        (2, 2),
    ]
    assert json.loads(captured.out)["pulled"] == ["shape", "prepare"]


def test_generate_emits_ordered_jsonl_events_for_each_stage(
    tmp_path: Path, monkeypatch, capsys
):
    source = tmp_path / "input.png"
    Image.new("RGB", (1, 1), "white").save(source)
    output_dir = tmp_path / "output"

    monkeypatch.setattr(cli, "configure_runtime", lambda cache: None)
    monkeypatch.setattr(cli, "require_cuda", lambda: None)
    monkeypatch.setattr(cli, "require_model_assets", lambda cache, components: None)

    def fake_prepare(image: Path, output: Path) -> Path:
        output.touch()
        return output

    def fake_shape(
        image: Path, output: Path, cache: Path, steps: int, seed: int | None
    ) -> Path:
        output.touch()
        return output

    def fake_texture(
        mesh: Path,
        image: Path,
        output: Path,
        cache: Path,
        output_format: str,
    ) -> Path:
        return output.with_suffix(f".{output_format}")

    monkeypatch.setattr(cli, "prepare_image", fake_prepare)
    monkeypatch.setattr(cli, "shape", fake_shape)
    monkeypatch.setattr(cli, "texture", fake_texture)

    assert (
        cli.main(["generate", "--image", str(source), "--output-dir", str(output_dir)])
        == 0
    )
    captured = capsys.readouterr()
    events = json_events(types.SimpleNamespace(stderr=captured.err))

    started = [event["stage"] for event in events if event["event"] == "stage_started"]
    completed = [
        event["stage"] for event in events if event["event"] == "stage_completed"
    ]
    assert started == ["generate", "prepare", "shape", "texture"]
    assert completed == ["prepare", "shape", "texture", "generate"]
    assert [event["event"] for event in events][-1] == "run_completed"
    assert {event["stage"] for event in events if event["event"] == "progress"} == {
        "generate",
        "prepare",
        "shape",
        "texture",
    }
    assert len(captured.out.splitlines()) == 1
    payload = json.loads(captured.out)
    assert payload["ok"] is True
    assert payload["texture"] == str(output_dir / "textured.glb")


@pytest.mark.parametrize(
    ("extra_arguments", "expected_output"),
    [([], "textured.glb"), (["--output-format", "obj"], "textured.obj")],
)
def test_generate_output_format_controls_final_texture_path(
    tmp_path: Path,
    monkeypatch,
    capsys,
    extra_arguments: list[str],
    expected_output: str,
):
    source = tmp_path / "input.png"
    Image.new("RGB", (1, 1), "white").save(source)
    output_dir = tmp_path / "output"

    monkeypatch.setattr(cli, "configure_runtime", lambda cache: None)
    monkeypatch.setattr(cli, "require_cuda", lambda: None)
    monkeypatch.setattr(cli, "require_model_assets", lambda cache, components: None)
    monkeypatch.setattr(cli, "prepare_image", lambda image, output: output)
    monkeypatch.setattr(cli, "shape", lambda image, output, cache, steps, seed: output)
    monkeypatch.setattr(
        cli,
        "texture",
        lambda mesh, image, output, cache, output_format: output.with_suffix(
            f".{output_format}"
        ),
    )

    assert (
        cli.main(
            [
                "generate",
                "--image",
                str(source),
                "--output-dir",
                str(output_dir),
                *extra_arguments,
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["texture"] == str(output_dir / expected_output)


def test_failed_stage_event_is_bounded_and_redacted(
    tmp_path: Path, monkeypatch, capsys
):
    source = tmp_path / "input.png"
    Image.new("RGB", (1, 1), "white").save(source)
    output_dir = tmp_path / "output"

    monkeypatch.setattr(cli, "configure_runtime", lambda cache: None)
    monkeypatch.setattr(cli, "require_cuda", lambda: None)
    monkeypatch.setattr(cli, "require_model_assets", lambda cache, components: None)
    monkeypatch.setattr(cli, "prepare_image", lambda image, output: output)

    def fail_shape(*args, **kwargs):
        raise RuntimeError("token=super-secret " + "x" * 1000)

    monkeypatch.setattr(cli, "shape", fail_shape)

    assert (
        cli.main(
            [
                "generate",
                "--image",
                str(source),
                "--output-dir",
                str(output_dir),
                "--shape-only",
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    events = json_events(types.SimpleNamespace(stderr=captured.err))
    failures = [
        event
        for event in events
        if event["event"] == "stage_failed" and event["stage"] == "shape"
    ]
    assert failures
    message = failures[0]["error"]["message"]
    assert "super-secret" not in message
    assert len(message) <= cli.EVENT_MESSAGE_LIMIT
    assert "Traceback" not in captured.err
    assert json.loads(captured.out)["error"]["code"] == "generation_failure"


def test_generate_writes_runtime_log_and_keeps_stdout_json(
    tmp_path: Path, monkeypatch, capsys
):
    source = tmp_path / "input.png"
    Image.new("RGB", (1, 1), "white").save(source)
    output_dir = tmp_path / "output"

    monkeypatch.setattr(cli, "configure_runtime", lambda cache: None)
    monkeypatch.setattr(cli, "require_cuda", lambda: None)
    monkeypatch.setattr(cli, "require_model_assets", lambda cache, components: None)

    def fake_prepare(image: Path, output: Path) -> Path:
        print("image preparation detail", file=sys.stderr)
        output.touch()
        return output

    def fake_shape(
        image: Path, output: Path, cache: Path, steps: int, seed: int | None
    ) -> Path:
        print("shape generation detail", file=sys.stderr)
        output.touch()
        return output

    monkeypatch.setattr(cli, "prepare_image", fake_prepare)
    monkeypatch.setattr(cli, "shape", fake_shape)

    assert (
        cli.main(
            [
                "generate",
                "--image",
                str(source),
                "--output-dir",
                str(output_dir),
                "--shape-only",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()

    assert json.loads(captured.out)["ok"] is True
    log = (output_dir / "run.log").read_text()
    assert "Preparing image..." in log
    assert "image preparation detail" in log
    assert "Generating shape..." in log
    assert "shape generation detail" in log


def test_generate_runtime_log_retains_traceback(tmp_path: Path, monkeypatch, capsys):
    source = tmp_path / "input.png"
    Image.new("RGB", (1, 1), "white").save(source)
    output_dir = tmp_path / "output"

    monkeypatch.setattr(cli, "configure_runtime", lambda cache: None)
    monkeypatch.setattr(cli, "require_cuda", lambda: None)
    monkeypatch.setattr(cli, "require_model_assets", lambda cache, components: None)
    monkeypatch.setattr(cli, "prepare_image", lambda image, output: output)

    def fail_shape(*args, **kwargs):
        raise RuntimeError("shape pipeline failed")

    monkeypatch.setattr(cli, "shape", fail_shape)

    assert (
        cli.main(
            [
                "generate",
                "--image",
                str(source),
                "--output-dir",
                str(output_dir),
                "--shape-only",
            ]
        )
        == 1
    )
    captured = capsys.readouterr()

    assert json.loads(captured.out)["error"]["code"] == "generation_failure"
    log = (output_dir / "run.log").read_text()
    assert "Traceback" in log
    assert "RuntimeError: shape pipeline failed" in log


def test_help_remains_available_as_an_explicit_human_path():
    result = run_cli("help")
    assert result.returncode == 0
    assert result.stdout.startswith("usage: hunyuan3d")
    assert "--output-format glb" in result.stdout
    assert "textured.mtl" in result.stdout
