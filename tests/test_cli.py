import json
import os
import subprocess
import sys
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


def test_generate_writes_runtime_log_and_keeps_stdout_json(tmp_path: Path, monkeypatch, capsys):
    source = tmp_path / "input.png"
    source.touch()
    output_dir = tmp_path / "output"

    monkeypatch.setattr(cli, "configure_runtime", lambda cache: None)
    monkeypatch.setattr(cli, "require_cuda", lambda: None)
    monkeypatch.setattr(cli, "require_model_assets", lambda cache, components: None)

    def fake_prepare(image: Path, output: Path) -> Path:
        print("image preparation detail", file=sys.stderr)
        output.touch()
        return output

    def fake_shape(image: Path, output: Path, cache: Path, steps: int, seed: int | None) -> Path:
        print("shape generation detail", file=sys.stderr)
        output.touch()
        return output

    monkeypatch.setattr(cli, "prepare_image", fake_prepare)
    monkeypatch.setattr(cli, "shape", fake_shape)

    assert cli.main(["generate", "--image", str(source), "--output-dir", str(output_dir), "--shape-only"]) == 0
    captured = capsys.readouterr()

    assert json.loads(captured.out)["ok"] is True
    log = (output_dir / "run.log").read_text()
    assert "Preparing image..." in log
    assert "image preparation detail" in log
    assert "Generating shape..." in log
    assert "shape generation detail" in log


def test_generate_runtime_log_retains_traceback(tmp_path: Path, monkeypatch, capsys):
    source = tmp_path / "input.png"
    source.touch()
    output_dir = tmp_path / "output"

    monkeypatch.setattr(cli, "configure_runtime", lambda cache: None)
    monkeypatch.setattr(cli, "require_cuda", lambda: None)
    monkeypatch.setattr(cli, "require_model_assets", lambda cache, components: None)
    monkeypatch.setattr(cli, "prepare_image", lambda image, output: output)

    def fail_shape(*args, **kwargs):
        raise RuntimeError("shape pipeline failed")

    monkeypatch.setattr(cli, "shape", fail_shape)

    assert cli.main(["generate", "--image", str(source), "--output-dir", str(output_dir), "--shape-only"]) == 1
    captured = capsys.readouterr()

    assert json.loads(captured.out)["error"]["code"] == "generation_failure"
    log = (output_dir / "run.log").read_text()
    assert "Traceback" in log
    assert "RuntimeError: shape pipeline failed" in log


def test_help_remains_available_as_an_explicit_human_path():
    result = run_cli("help")
    assert result.returncode == 0
    assert result.stdout.startswith("usage: hunyuan3d")
