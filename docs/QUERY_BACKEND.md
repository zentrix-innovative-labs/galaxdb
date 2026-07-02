# Query Backend & DataFusion Governance

GalaxDB's relational and analytical query layer is built on
[Apache DataFusion](https://datafusion.apache.org/), confined behind an
anti-corruption boundary. This document records how that dependency is
governed, why, and what the escape hatch is. It is the operational companion
to the HTAP query-engine spec (Requirement 7).

## The boundary: `galaxdb-query`

`crates/galaxdb-query` is the **only** crate in the workspace permitted to
depend on or reference DataFusion. Its public API is GalaxDB-owned types
exclusively — `ArrowSource`, `QueryBackend`, `ScanRequest`, `ScanPredicate`,
`ReadSnapshot`, `GalaxLogicalPlan`, `QueryContext`. No `datafusion::` type
appears in any signature outside this crate, so the volatile DataFusion API
never leaks into the rest of the engine, the wire protocol, or persisted
formats.

Apache Arrow *does* cross the boundary (it is the public element type of
`ArrowSource`). Arrow is the stable columnar interchange the storage layer
produces; the boundary is specifically about the DataFusion query API, which
churns across releases.

### Enforced mechanically

- **Containment guard** — `crates/galaxdb-query/tests/containment.rs` walks
  the whole workspace and fails the build if any other crate declares a
  `datafusion*` dependency or references `datafusion::` / `use datafusion` in
  source. Runs in CI on every change.
- **Single-version ban** — `deny.toml` sets `deny-multiple-versions = true`
  for both `datafusion` and `arrow`, so a transitive bump can never silently
  introduce a duplicate major. (`cargo deny check bans`.)

## Version pin

DataFusion is pinned to an **exact** version in
`crates/galaxdb-query/Cargo.toml`:

```toml
datafusion = "=52.5.0"
```

`=52.5.0` is the version `lance` already resolves transitively, so pinning to
it keeps a single DataFusion (and single Arrow `=57.3.1`) in the graph. Arrow
is pinned in the workspace catalog (`[workspace.dependencies] arrow =
"=57.3.1"`).

### Upgrading DataFusion (deliberate, gated)

An upgrade is never incidental. The procedure (Req 7.4):

1. Bump the pin in `galaxdb-query/Cargo.toml` and the Arrow pin if required.
2. Run the SQL conformance + regression corpus
   (`galaxdb-query/tests/conformance/`, HTAP task 24). **Any** failing case
   blocks the bump — it never blocks a release.
3. Re-run the analytical benchmarks (HTAP task 27) to catch performance
   regressions.
4. Only then merge the bump.

A dedicated `datafusion-bump` CI job (HTAP task 25) runs the corpus against a
candidate version so the blast radius of an upgrade is known before merge.

## Build-time cost

A **cold** build that compiles DataFusion + Lance from scratch takes roughly
**46 minutes** locally (measured during the HTAP spike,
`CURRENT_STATE_FACTS.md §H`). Once those artifacts exist in `target/`,
incremental builds that only touch `galaxdb-query` are fast (seconds), because
DataFusion was already in the graph transitively via `lance` — adding it as a
direct dependency reuses the compiled rlibs.

### CI / local mitigation

- **`sccache`** as the Rust compiler cache so DataFusion/Lance object files
  are shared across CI jobs and clean checkouts. Configure
  `RUSTC_WRAPPER=sccache` with a shared (e.g. S3/GHA) cache backend.
- **Cached crate layer** — CI caches `~/.cargo` and `target/` keyed on
  `Cargo.lock`, so the heavy crates compile once per lockfile change, not per
  push.
- The conformance and `datafusion-bump` jobs run on top of the cached layer.

## Escape hatch (Req 7.5)

`QueryBackend` is a trait. The DataFusion implementation
(`DataFusionBackend`, HTAP task 11) is one impl behind it. If DataFusion ever
becomes untenable (licensing, a breaking change we cannot absorb, a
performance cliff), the response does not touch any other crate:

1. **Alternative backend** — implement `QueryBackend` over a different
   execution engine; nothing outside `galaxdb-query` changes.
2. **Vendored fork** — DataFusion is Apache-2.0. We can vendor and fork it
   under `galaxdb-query` and continue, pinning to our fork. The containment
   boundary means the fork's surface is contained to this one crate.

Either path is a `galaxdb-query`-local change, which is the entire point of
the anti-corruption layer.

### Implementing an alternative backend

A substitute engine implements the `QueryBackend` trait
(`crates/galaxdb-query/src/lib.rs`) — the entire surface it must satisfy:

```rust
#[async_trait::async_trait]
pub trait QueryBackend: Send + Sync {
    /// Register a table's `ArrowSource` so the backend can scan it.
    fn register(&self, table: &str, source: Arc<dyn ArrowSource>) -> GalaxResult<()>;

    /// Execute a logical plan, returning a stream of Arrow result batches.
    async fn execute(
        &self,
        plan: GalaxLogicalPlan,
        ctx: &QueryContext,
    ) -> GalaxResult<ResultStream>;
}
```

Everything the backend consumes and produces is a GalaxDB-owned or Arrow
type: it reads rows through `ArrowSource` (which the storage engine
implements via `EngineArrowSource`, yielding Arrow `RecordBatch`es at a
`ReadSnapshot`), receives a `GalaxLogicalPlan` (the validated analytical SQL
plus its referenced tables), and returns a `ResultStream` of Arrow batches.
Errors must map to `GalaxError` (the DataFusion impl does this in
`error.rs`); no engine-specific error text is allowed to escape.

**Selection point.** The concrete backend is chosen in exactly one place:
`galaxdb_query::backend::run_analytical_sql_blocking`
(`crates/galaxdb-query/src/backend.rs`), which the embedded engine calls from
its analytical path (`Database::analytical_query`). It constructs a
`DataFusionBackend`, registers an `EngineArrowSource` per referenced table at
the query snapshot, and runs the plan. Swapping backends is editing that one
constructor; the embedded engine, wire protocol, and storage layer are
untouched because they only ever see `QueryBackend`, `ArrowSource`, and Arrow.

**Checklist for a substitute:**

1. `impl QueryBackend for MyBackend` with the two methods above.
2. Map the engine's errors to `GalaxError::Query { sqlstate, message }` with
   no foreign brand text (mirror `galaxdb-query/src/error.rs`).
3. Point `run_analytical_sql_blocking` (or a sibling constructor) at
   `MyBackend`.
4. Run the SQL conformance corpus (HTAP task 24) — a substitute is only
   complete when the corpus is green, exactly as for a DataFusion bump.

The containment guard guarantees step 3 is the only wiring change: nothing
outside `galaxdb-query` can have named the old backend's types.
