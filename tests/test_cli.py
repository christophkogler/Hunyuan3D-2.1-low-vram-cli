import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

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
    }


def test_model_status_uses_the_cli_local_dino_directory(tmp_path: Path):
    dino = tmp_path / "models/facebook/dinov2-giant"
    dino.mkdir(parents=True)
    (dino / "config.json").touch()

    assert cli.model_status(tmp_path)["dino"] is True


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
        cuda=types.SimpleNamespace(get_device_properties=lambda _: types.SimpleNamespace(total_memory=32 * 1024**3)),
        Generator=lambda device: types.SimpleNamespace(manual_seed=lambda seed: (device, seed)),
    )
    monkeypatch.setattr(cli, "legacy_paths", lambda: None)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "hy3dshape.pipelines", types.SimpleNamespace(
        Hunyuan3DDiTFlowMatchingPipeline=FakePipelineClass
    ))

    assert cli.shape(tmp_path / "input.png", tmp_path / "shape.glb", tmp_path, 50, None) == tmp_path / "shape.glb"
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
    monkeypatch.setitem(sys.modules, "torchvision_fix", types.SimpleNamespace(apply_fix=lambda: None))
    monkeypatch.setitem(sys.modules, "textureGenPipeline", types.SimpleNamespace(
        Hunyuan3DPaintConfig=FakeConfig,
        Hunyuan3DPaintPipeline=FakePipeline,
    ))

    assert cli.texture(tmp_path / "shape.glb", tmp_path / "input.png", tmp_path / "textured.glb", tmp_path) == tmp_path / "textured.obj"
    config = captured["config"]
    assert config.multiview_pretrained_path == str(tmp_path / "models/tencent/Hunyuan3D-2.1")
    assert config.dino_ckpt_path == str(tmp_path / "models/facebook/dinov2-giant")
    assert config.realesrgan_ckpt_path == str(tmp_path / "realesrgan/RealESRGAN_x4plus.pth")


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


def test_persistent_command_installer_runs_the_cli_from_another_directory(tmp_path: Path):
    command_dir = tmp_path / "bin"
    shell_rc = tmp_path / "bashrc"
    env = {
        **os.environ,
        "HUNYUAN3D_COMMAND_DIR": str(command_dir),
        "HUNYUAN3D_SHELL_RC": str(shell_rc),
    }
    subprocess.run([Path(__file__).parents[1] / "install-command.sh"], env=env, check=True)

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
    return subprocess.run([executable, *args], text=True, capture_output=True, check=False)


def json_result(result: subprocess.CompletedProcess[str]) -> dict:
    assert result.stderr == ""
    return json.loads(result.stdout)


@pytest.mark.parametrize(
    "arguments",
    [
        ("not-a-command",),
        ("shape", "--image", "input.png"),
        ("shape", "--image", "input.png", "--output", "output.glb", "--steps", "not-a-number"),
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
    result = run_cli("prepare", str(tmp_path / "missing.png"), "--output", str(tmp_path / "output.png"))
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


def test_success_result_is_versioned_json():
    result = run_cli("models", "status")
    assert result.returncode == 0
    payload = json_result(result)
    assert payload["schema_version"] == 1
    assert payload["ok"] is True


def test_help_remains_available_as_an_explicit_human_path():
    result = run_cli("help")
    assert result.returncode == 0
    assert result.stdout.startswith("usage: hunyuan3d")
