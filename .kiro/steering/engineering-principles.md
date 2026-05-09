# GalaxDB Engineering Principles (always in effect)

These rules apply to every commit in this workspace. They are not suggestions. If a rule conflicts with a task description in the spec, the rule wins; raise it with the user before coding.

## 1. No mocks in production code paths

- Production code must never carry a mock, stub, fake, or fallback that silently substitutes synthetic behavior for a real operation.
- Mocks are permitted **only** inside `#[cfg(test)]` blocks or inside files under `tests/` with a comment explaining what real component they stand in for.
- Examples of what this rule forbids:
  - The sidecar's `--mock-dim` flag at runtime and the `falling back to mock mode` path when the real model fails to load. The real model must load or the process must exit with a clear error.
  - `NoOpVectorBackend` or `NoOpKeyProvider` being silently selected in production builds.
  - Any function whose body is `Err("feature not implemented")` or `Ok(dummy_value)` in a code path users will reach.

## 2. No silent fallbacks

- If a real implementation fails (model load, KMS decrypt, sidecar crash past retry budget), the engine must surface a typed error to the caller. No logging-and-returning-fake-data.
- Retry, degraded-mode, and backlog-overflow are real behaviors the spec documents. They are not fallbacks; they are designed responses with observable metrics. Those stay.

## 3. No task ticked without real implementation

- A task is complete when every acceptance criterion in its spec entry is satisfied by real code against real infrastructure, plus tests that exercise the real code.
- An abstraction boundary (trait + in-memory reference impl) is a legitimate partial deliverable **only if** the task explicitly covers only the abstraction, and the dependent tasks that provide the concrete implementation are listed as prerequisites in the consolidation tracker.
- Skipping a sub-task to get a later sub-task to compile is never acceptable. Order follows the spec unless the spec is wrong — in which case fix the spec first.

## 4. No faked benchmarks

- Every benchmark number published in `docs/BENCHMARKS.md`, `README.md`, `Evidence-Backed.md`, or anywhere else must be reproducible from a named command against a named dataset on named hardware.
- Random-vector benchmarks for HNSW are not reported. SIFT1M or equivalent ANN-benchmarks datasets only.
- Benchmarks always use `--release`. Numbers from debug builds are never published.
- If a competitor comparison is made, run the competitor on the same hardware and same dataset and publish both numbers side by side.

## 5. Cross-platform and no vendor lock-in

- Primary platforms: macOS (x86_64 and aarch64), Linux (x86_64 and aarch64), Windows.
- `#[cfg(target_os = "linux")]` is acceptable only for Linux-specific optimizations (io_uring, libnuma) with a tokio-based fallback for other platforms.
- External services are pluggable via trait. Key management supports at minimum: local file, environment variable, AWS KMS, Google Cloud KMS, Azure Key Vault, HashiCorp Vault, and an external-process provider (so any KMS exposing a CLI can be wired in without code changes).
- Embedding models are selected at runtime via HuggingFace model IDs. No model is hard-coded as a requirement.

## 6. AWS test instance discipline

- AWS instance ID `i-0b2dec9226f62db65` (c6id.4xlarge) is used for integration and benchmark runs.
- Always start with `aws ec2 start-instances`, mount NVMe, rsync code, build `--release`, run, collect results, and then stop with `aws ec2 stop-instances --instance-ids i-0b2dec9226f62db65`.
- Never leave the instance running overnight or unattended.
- Credentials and IPs never appear in code, docs, or commit messages.

## 7. Task tracker is the source of truth

- `tasks.md` is the authoritative list. It is updated **after** a task is verifiably complete, not before.
- If a task was ticked prematurely (discovered via audit), unmark it, file a consolidation task, and fix the underlying stub before ticking any dependent task.
- Every task checkbox change in this workspace requires either (a) green CI against real tests, or (b) a human acknowledgement if CI isn't available for that path.

## 8. Verification before claims

- Before stating "X works", run the code and show the output. Claims about performance, correctness, or integration require the actual command and the actual result.
- Read the file before describing what it does. Never infer implementation from function names.
