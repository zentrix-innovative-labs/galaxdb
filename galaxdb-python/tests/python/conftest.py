"""Shared pytest fixtures for the galaxdb Python integration suite.

These tests exercise the real PyO3 `galaxdb` module (embedded mode)
and a real `galaxdb-server` binary (remote mode). Nothing is mocked.

Build the extension before running:

    maturin develop --release -m galaxdb-python/Cargo.toml

Then:

    pytest galaxdb-python/tests/python/ -v

See task 22.6 in `.kiro/specs/galaxdb-v1-engine/tasks.md`.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

import pytest


def _repo_root() -> Path:
    """Walk up from this file to the Cargo workspace root."""
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "Cargo.toml").is_file():
            # First ancestor with a Cargo.toml that declares a workspace
            # is the root (nested crates all have Cargo.toml, so we
            # want the outermost match).
            pass
    # Walk up and pick the outermost Cargo.toml.
    current = here
    root = None
    for parent in here.parents:
        if (parent / "Cargo.toml").is_file():
            root = parent
    if root is None:
        raise RuntimeError("could not locate Cargo workspace root")
    return root


def _server_binary() -> Path:
    """Locate the release `galaxdb-server` binary.

    Prefers the workspace's standard `target/release` output. If the
    binary is missing, skip the test rather than trying to rebuild
    from inside pytest — the build is the user's responsibility and
    we don't want a pytest run to trigger a 30-minute release compile.
    """
    root = _repo_root()
    candidate = root / "target" / "release" / "galaxdb-server"
    if not candidate.is_file():
        pytest.skip(
            f"galaxdb-server binary not found at {candidate}; "
            f"run `cargo build --release -p galaxdb-server` first"
        )
    return candidate


def _pick_free_port() -> int:
    """Ask the OS for a free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(host: str, port: int, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.25)
            try:
                s.connect((host, port))
                return
            except OSError:
                time.sleep(0.05)
    raise RuntimeError(
        f"galaxdb-server did not start listening on {host}:{port} within {timeout}s"
    )


@pytest.fixture
def temp_db_dir(tmp_path: Path) -> Path:
    """A fresh on-disk directory for `galaxdb.Database(path)`."""
    d = tmp_path / "db"
    d.mkdir()
    return d


@pytest.fixture
def running_server(tmp_path: Path):
    """Spawn a real `galaxdb-server` bound to a free port.

    Yields `(dsn, data_dir)` where `dsn` is a libpq-style connection
    string that `galaxdb.connect()` can pass to the postgres client.
    The server is terminated and its data directory cleaned up when
    the test exits, regardless of outcome.
    """
    binary = _server_binary()
    data_dir = tmp_path / "server-data"
    data_dir.mkdir()
    port = _pick_free_port()

    proc = subprocess.Popen(
        [
            str(binary),
            "--port",
            str(port),
            "--data-dir",
            str(data_dir),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "RUST_LOG": "galaxdb=warn"},
    )
    try:
        _wait_for_port("127.0.0.1", port, timeout=15.0)
        dsn = (
            f"host=127.0.0.1 port={port} user=galaxdb "
            f"dbname=galaxdb sslmode=disable"
        )
        yield dsn, data_dir
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2.0)
        # Best-effort cleanup; pytest's tmp_path will get the rest.
        shutil.rmtree(data_dir, ignore_errors=True)
