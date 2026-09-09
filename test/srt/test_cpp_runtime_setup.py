"""Check shell runtime selection without needing Linux shared libraries or CUDA."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class CppRuntimeSetupTests(unittest.TestCase):
    def run_setup(self, prefix, setup="", probe=False):
        # Keep fake library paths inside the shell; never load them into Python.
        script = """
set -euo pipefail
source "$1/scripts/configure_cpp_runtime.sh"
prefix="$2"
python() {
    if [[ "$2" == 'import sys; print(sys.prefix)' ]]; then
        printf '%s\\n' "$prefix"
    else
        return "${PROBE_EXIT:-0}"
    fi
}
"""
        script += setup + "\nconfigure_cpp_runtime\n"
        if probe:
            script += "check_zmq_runtime\n"
        script += 'printf "SELECTED=%s\\nPRELOAD=%s\\n" "$PRELOAD_LIBSTDCXX" "${LD_PRELOAD:-}"\n'
        env = os.environ.copy()
        for key in ("PRELOAD_LIBSTDCXX", "LD_PRELOAD", "PROBE_EXIT"):
            env.pop(key, None)
        return subprocess.run(
            ["bash", "-c", script, "test", str(ROOT), str(prefix)],
            env=env,
            capture_output=True,
            text=True,
        )

    def test_environment_runtime_replaces_system_preload(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            library = prefix / "lib/libstdc++.so.6"
            library.parent.mkdir()
            library.touch()
            result = self.run_setup(
                prefix,
                'LD_PRELOAD="/usr/lib/libstdc++.so.6:/opt/libtrace.so /opt/libextra.so"',
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(
                f"PRELOAD={library}:/opt/libtrace.so:/opt/libextra.so", result.stdout
            )
            result = self.run_setup(prefix)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"PRELOAD={library}", result.stdout)
            result = self.run_setup(prefix, 'PRELOAD_LIBSTDCXX=""')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("SELECTED=\nPRELOAD=\n", result.stdout)

    def test_explicit_path_and_missing_path(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            (prefix / "custom.so").touch()
            result = self.run_setup(prefix, 'PRELOAD_LIBSTDCXX="$prefix/custom.so"')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"PRELOAD={prefix}/custom.so", result.stdout)
            result = self.run_setup(prefix, 'PRELOAD_LIBSTDCXX="$prefix/missing.so"')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("C++ runtime does not exist", result.stderr)

    def test_no_environment_library_uses_normal_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_setup(directory, probe=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("SELECTED=\nPRELOAD=\n", result.stdout)

    def test_failed_import_stops_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_setup(directory, "PROBE_EXIT=1", probe=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("ZeroMQ import failed before model startup", result.stderr)
            self.assertNotIn("SELECTED=", result.stdout)


if __name__ == "__main__":
    unittest.main()
