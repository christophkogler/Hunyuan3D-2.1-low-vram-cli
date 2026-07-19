from pathlib import Path
import sys

import pytest


def test_convert_obj_to_glb_exports_a_real_file(tmp_path: Path):
    pytest.importorskip("bpy")

    from hy3dpaint.DifferentiableRenderer.mesh_utils import convert_obj_to_glb

    obj = tmp_path / "triangle.obj"
    obj.write_text(
        "mtllib triangle.mtl\n"
        "o triangle\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "vt 0 0\n"
        "vt 1 0\n"
        "vt 0 1\n"
        "usemtl Material\n"
        "f 1/1 2/2 3/3\n"
    )
    (tmp_path / "triangle.mtl").write_text("newmtl Material\nKd 1 1 1\n")
    glb = tmp_path / "triangle.glb"

    assert convert_obj_to_glb(str(obj), str(glb)) is True
    assert glb.is_file()
    assert glb.stat().st_size > 0


def test_convert_obj_to_glb_reports_missing_bpy(monkeypatch):
    pytest.importorskip("bpy")

    from hy3dpaint.DifferentiableRenderer.mesh_utils import convert_obj_to_glb

    monkeypatch.setitem(sys.modules, "bpy", None)
    with pytest.raises(ImportError, match="GLB export requires"):
        convert_obj_to_glb("missing.obj", "missing.glb")
