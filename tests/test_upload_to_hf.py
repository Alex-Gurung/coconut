import tempfile
import unittest
from pathlib import Path

from scripts.train.upload_to_hf import compute_delete_patterns, list_local_relative_files


class FakeApi:
    def __init__(self, remote_files=None, raises=False):
        self.remote_files = remote_files or []
        self.raises = raises
        self.calls = []

    def list_repo_files(self, repo_id, *, revision=None, repo_type=None):
        self.calls.append(
            {
                "repo_id": repo_id,
                "revision": revision,
                "repo_type": repo_type,
            }
        )
        if self.raises:
            raise RuntimeError("repo not ready")
        return list(self.remote_files)


class UploadToHfHelpersTest(unittest.TestCase):
    def test_list_local_relative_files_uses_posix_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "subdir").mkdir()
            (root / "README.md").write_text("x")
            (root / "subdir" / "model-00001-of-00002.safetensors").write_text("y")

            files = list_local_relative_files(root)

        self.assertEqual(
            files,
            {
                "README.md",
                "subdir/model-00001-of-00002.safetensors",
            },
        )

    def test_compute_delete_patterns_removes_remote_only_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("x")
            (root / "model-00001-of-00002.safetensors").write_text("a")
            (root / "model-00002-of-00002.safetensors").write_text("b")

            api = FakeApi(
                remote_files=[
                    ".gitattributes",
                    "README.md",
                    "model-00001-of-00002.safetensors",
                    "model-00002-of-00002.safetensors",
                    "model.safetensors",
                    "chat_template.json",
                ]
            )

            delete_patterns = compute_delete_patterns(
                api=api,
                repo_id="agurung/example",
                model_dir=root,
                revision="checkpoint_24",
            )

        self.assertEqual(delete_patterns, ["chat_template.json", "model.safetensors"])
        self.assertEqual(
            api.calls,
            [
                {
                    "repo_id": "agurung/example",
                    "revision": "checkpoint_24",
                    "repo_type": "model",
                }
            ],
        )

    def test_compute_delete_patterns_returns_empty_on_remote_listing_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("x")

            api = FakeApi(raises=True)
            delete_patterns = compute_delete_patterns(
                api=api,
                repo_id="agurung/example",
                model_dir=root,
                revision="checkpoint_24",
            )

        self.assertEqual(delete_patterns, [])


if __name__ == "__main__":
    unittest.main()
