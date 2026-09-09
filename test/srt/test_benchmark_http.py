"""Exercise readiness decisions with deterministic curl responses, without GPUs."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class BenchmarkHttpTests(unittest.TestCase):
    def run_probe(self, code="200", exit_code="0", port="31000", base_url=None):
        with tempfile.TemporaryDirectory() as directory:
            env = os.environ.copy()
            for name in ("CLIENT_BASE_URL", "NO_PROXY", "no_proxy"):
                env.pop(name, None)
            env.update(
                PORT=port,
                NO_PROXY="internal.example",
                no_proxy="private.example",
                MOCK_CODE=code,
                MOCK_EXIT=exit_code,
            )
            if base_url is not None:
                env["CLIENT_BASE_URL"] = base_url
            script = """
set -euo pipefail
source "$1/scripts/benchmark_http.sh"
configure_benchmark_http
CURRENT_DIR="$2"
HOST=0.0.0.0
SERVER_PID=$$
SERVER_START_TIMEOUT=3
printf 'worker diagnostic\\n' > "$CURRENT_DIR/server.log"
curl() {
    printf '%s\\n' "$@" > "$CURRENT_DIR/curl_args"
    if [[ "$MOCK_EXIT" != 0 ]]; then echo 'mock connection failure' >&2; fi
    printf '%s' "$MOCK_CODE"
    return "$MOCK_EXIT"
}
sleep() { SECONDS=$((SECONDS + 2)); }
wait_for_server
"""
            result = subprocess.run(
                ["bash", "-c", script, "test", str(ROOT), directory],
                env=env,
                text=True,
                capture_output=True,
                timeout=10,
            )
            return result, (Path(directory) / "curl_args").read_text()

    def test_ready_on_configured_port_bypasses_local_proxy(self):
        result, args = self.run_probe()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("http://127.0.0.1:31000/health", args)
        self.assertIn("--noproxy\n", args)
        self.assertIn("internal.example,private.example,127.0.0.1,localhost,::1", args)
        self.assertIn("--connect-timeout\n3\n--max-time\n3\n", args)
        self.assertIn("Server ready", result.stdout)

    def test_explicit_url_is_preserved_and_slash_normalized(self):
        result, args = self.run_probe(base_url="http://localhost:32000/")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("http://localhost:32000/health", args)

    def test_unready_and_redirect_are_not_success(self):
        for code in ("503", "302"):
            result, _ = self.run_probe(code=code)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(f"HTTP {code}", result.stderr)
            self.assertIn("Timed out waiting", result.stderr)
            self.assertIn("worker diagnostic", result.stderr)
            self.assertNotIn("Server ready", result.stdout)

    def test_connection_error_is_visible(self):
        result, _ = self.run_probe(code="000", exit_code="7")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("curl exit 7", result.stderr)
        self.assertIn("mock connection failure", result.stderr)


if __name__ == "__main__":
    unittest.main()
