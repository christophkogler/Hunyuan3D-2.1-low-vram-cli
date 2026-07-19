import sys
import types

from hunyuan3d import cli


def test_mesh_simplification_uses_explicit_target_face_count(monkeypatch, tmp_path):
    cli.legacy_paths()
    from utils import simplify_mesh_utils

    captured = {}

    class FakeMeshSet:
        def load_new_mesh(self, *_args, **_kwargs):
            pass

        def save_current_mesh(self, *_args, **_kwargs):
            pass

    class FakeMesh:
        faces = types.SimpleNamespace(shape=(40_001, 3))

        def simplify_quadric_decimation(self, **kwargs):
            captured.update(kwargs)
            return self

        def export(self, _path):
            pass

    monkeypatch.setitem(
        sys.modules, "pymeshlab", types.SimpleNamespace(MeshSet=FakeMeshSet)
    )
    monkeypatch.setattr(simplify_mesh_utils.trimesh, "load", lambda *_args, **_kwargs: FakeMesh())

    simplify_mesh_utils.mesh_simplify_trimesh(str(tmp_path / "shape.obj"), str(tmp_path / "remeshed.obj"))

    assert captured == {"face_count": 40_000}
