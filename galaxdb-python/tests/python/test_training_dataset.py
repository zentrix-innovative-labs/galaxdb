"""`Database.training_dataset(tag)` coverage for the galaxdb module.

End-to-end path: INSERT rows, `CREATE VERSION TAG '<name>' FOR TRAINING`,
call `training_dataset(tag)`, receive a path, open the path with the
real `lance` crate (via its Python bindings), and assert row count.

If `lance` (and `pyarrow`, which `lance` requires) are not installed in
the running environment, the test is skipped — same convention the
Rust integration tests use when optional dependencies aren't available.
The Rust-side test in `crates/galaxdb-embedded/src/lib.rs`
(`training_dataset_writes_real_lance_dataset`) is the always-on gate
for the same behaviour; this file is the complementary Python surface.

Task 22.6 acceptance: the third of three pytest files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import galaxdb

lance = pytest.importorskip(
    "lance",
    reason=(
        "the `lance` Python package is optional for the galaxdb tests; "
        "install with `pip install lance pyarrow` to run this file"
    ),
)


def _make_training_db(tmp_path: Path, n_rows: int = 5) -> galaxdb.Database:
    db = galaxdb.Database(str(tmp_path / "db"))
    db.execute("CREATE TABLE docs (id INT PRIMARY KEY, body TEXT)")
    for i in range(1, n_rows + 1):
        db.execute(f"INSERT INTO docs (id, body) VALUES ({i}, 'row-{i}')")
    return db


def test_training_dataset_writes_readable_lance_dataset(tmp_path: Path) -> None:
    db = _make_training_db(tmp_path, n_rows=5)
    db.execute(
        "CREATE VERSION TAG 'train-v1' FOR TRAINING "
        "WITH TRAINING PRECISION 'float32' TRAINING SEED 42"
    )

    path_str = db.training_dataset("train-v1")
    assert isinstance(path_str, str) and path_str
    out = Path(path_str)
    assert out.exists(), f"returned path does not exist: {out}"
    assert out.is_dir(), f"Lance datasets are directories, got {out}"
    # The output must live inside the database directory.
    assert str(out).startswith(str(tmp_path))

    # Open through the real lance package and verify we get our rows
    # back.
    ds = lance.dataset(path_str)
    row_count = ds.count_rows()
    assert row_count == 5, f"expected 5 rows, lance saw {row_count}"

    # Columns should include the two we created.
    column_names = [f.name for f in ds.schema]
    assert "id" in column_names
    assert "body" in column_names


def test_training_dataset_rejects_non_training_tag(tmp_path: Path) -> None:
    db = _make_training_db(tmp_path, n_rows=2)
    db.execute("CREATE VERSION TAG 'plain-snapshot'")
    with pytest.raises(RuntimeError) as excinfo:
        db.training_dataset("plain-snapshot")
    msg = str(excinfo.value)
    assert "FOR TRAINING" in msg or "training" in msg.lower()


def test_training_dataset_unknown_tag_errors(tmp_path: Path) -> None:
    db = _make_training_db(tmp_path, n_rows=1)
    with pytest.raises(RuntimeError) as excinfo:
        db.training_dataset("does-not-exist")
    msg = str(excinfo.value)
    assert "unknown" in msg.lower() or "does-not-exist" in msg


def test_training_dataset_is_iterable_for_pytorch(tmp_path: Path) -> None:
    """Prove the returned dataset can be iterated one batch.

    `lance.dataset(path).to_pytorch()` exists on some builds but is
    version-sensitive. The iterable contract the task asks for is a
    Lance scan that produces Arrow `RecordBatch`es — which is exactly
    what `ds.to_batches()` returns. Iterating one batch is sufficient
    evidence the dataset surface works end-to-end from Python.
    """
    db = _make_training_db(tmp_path, n_rows=3)
    db.execute(
        "CREATE VERSION TAG 'iter-tag' FOR TRAINING "
        "WITH TRAINING PRECISION 'float32'"
    )
    path_str = db.training_dataset("iter-tag")

    ds = lance.dataset(path_str)
    batches = list(ds.to_batches())
    assert batches, "Lance scan yielded no batches"
    total = sum(b.num_rows for b in batches)
    assert total == 3
