# Versioning

GalaxDB follows [Semantic Versioning](https://semver.org/).

- **Current version: `0.3.0`** (pre-1.0 development).
- While on `0.x`, the public API and on-disk format may still change between
  minor versions. Breaking changes are called out in release notes.
- **`1.0.0`** will be the first release with a stability commitment. It is cut
  only after the release gate passes: green CI (build + `clippy -D warnings`
  + `cargo deny` + full test suite), the acceptance criteria of the HTAP
  query-engine spec (Req 1–7 and Req 8 AC1–3), and the published `--release`
  benchmarks on the named hardware.

## Release history and the tag ordering note

Two tags predate this policy and are **chronologically out of order**:

| Tag              | Tagged      | SemVer rank |
|------------------|-------------|-------------|
| `v1.0.0-beta.1`  | 2026-05-14  | highest     |
| `v0.2.0`         | 2026-06-17  | lower       |

`v1.0.0-beta.1` was a **premature** `1.0` pre-release: it was cut before the
engine was ready to commit to a stable API. Development then correctly reset
to the `0.x` line (`v0.2.0`, now `0.3.0`) to reflect honest pre-1.0 status.
Tagging `0.2.0` *after* `1.0.0-beta.1` looks like a version decrease, but it
is intentional: the beta was withdrawn in practice, not superseded by a
`1.0.0` final.

This is **cosmetic and does not block `1.0.0`.** Under SemVer,
`1.0.0` (final) outranks `1.0.0-beta.1` (`1.0.0-beta.1 < 1.0.0`), so the
eventual stable `1.0.0` release is valid and unambiguous regardless of the
stray beta tag.

### Recommended cleanup (requires a maintainer decision)

The `v1.0.0-beta.1` tag/release is misleading (it implies a 1.0 line that was
abandoned). Options:

1. **Leave it** — harmless; `1.0.0` will supersede it. Simplest.
2. **Delete it** — remove the local + remote tag and its GitHub release so the
   history reads `0.2.0 → 0.3.0 → … → 1.0.0`. This edits a *published* release,
   so it is a deliberate maintainer action, not done automatically:

   ```bash
   git tag -d v1.0.0-beta.1
   git push --delete origin v1.0.0-beta.1
   # then delete the GitHub Release for v1.0.0-beta.1 in the UI/gh CLI
   ```

## Cutting a release

1. Bump the source version (workspace `Cargo.toml`, the explicitly-versioned
   crates, and `galaxdb-python/pyproject.toml`). Already done for `0.3.0`.
2. Tag `vX.Y.Z` on `main` and push it. `.github/workflows/release.yml` builds
   the per-platform binaries and publishes the GitHub Release.
3. **Then** update the release-artifact pointers to the new tag + real
   artifact hashes: `Formula/galaxdb.rb` (`version`, download URLs, `sha256`
   of each published binary) and `install.sh` (`VERSION`). These track the
   *latest published* release, so they are updated only once the binaries
   exist — never ahead of the tag, or `brew install` / `curl | bash` break.
