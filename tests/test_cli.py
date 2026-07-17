import json
import os
import subprocess
import sys
from pathlib import Path

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

    assert doctor.returncode in {0, 2}
    assert json.loads(doctor.stdout)["cache"] == str(cli.ROOT / ".cache/hunyuan3d")
