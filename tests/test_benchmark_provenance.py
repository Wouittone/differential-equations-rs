"""The candidate run must not silently fall back to the old registry package."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "provenance", Path(__file__).resolve().parents[1] / "scripts/run_improvement_benchmarks.py")
provenance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(provenance)


class CandidateDependencyTests(unittest.TestCase):
    def test_rejects_registry_wrong_path_and_wrong_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library, host = root / "library", root / "host"
            library.mkdir()
            host.mkdir()
            (library / "Cargo.toml").write_text('[package]\nversion="1.4.1"\n', encoding="utf-8")
            for dependency in [
                '{version="=1.0.0"}',
                '{version="=1.4.1",path="../other"}',
                '{version="=1.0.0",path="../library"}',
                '{version="1.4",path="../library"}',
                '{workspace=true}',
            ]:
                (host / "Cargo.toml").write_text('[dependencies]\ndifferential-equations-rs=' + dependency, encoding="utf-8")
                with self.subTest(dependency=dependency), self.assertRaises(RuntimeError):
                    provenance.validate_library_dependency(host, library)
            (host / "Cargo.toml").write_text(
                '[dependencies]\nrenamed={package="differential-equations-rs",version="=1.4.1",path="../library"}', encoding="utf-8")
            provenance.validate_library_dependency(host, library)
            (host / "Cargo.toml").write_text(
                '[workspace.dependencies]\nrenamed={package="differential-equations-rs",version="=1.4.1",path="../library"}', encoding="utf-8")
            provenance.validate_library_dependency(host, library)


if __name__ == "__main__":
    unittest.main()
