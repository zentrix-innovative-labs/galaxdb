# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.3.x   | ✅ |

## Reporting a vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Email **security@galaxdb.com** with:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Any suggested fix

We will acknowledge receipt within 48 hours and provide a timeline for a fix within 7 days.

## Disclosure policy

We follow coordinated disclosure. We ask that you give us 90 days to address a vulnerability before public disclosure.

## Security surface and hardening

GalaxDB exposes the following security-relevant surfaces. Operators should review each before
exposing a server beyond loopback.

### Authentication (SCRAM-SHA-256)

- Enable with `--auth` or `GALAXDB_AUTH=1`. When enabled, every connection must complete a
  SCRAM-SHA-256 exchange (RFC 5802 / 7677); a wrong password or unknown role is rejected with
  SQLSTATE `28P01` and no role-enumeration signal.
- The initial superuser is provisioned from `GALAXDB_INITIAL_SUPERUSER` /
  `GALAXDB_INITIAL_SUPERUSER_PASSWORD` on first start against an empty catalog. **No default
  password ships** — if auth is enabled, the catalog is empty, and no initial superuser is
  configured, the server refuses to start.
- Passwords are never stored or logged in plaintext; only the SCRAM verifier (salt, iteration
  count, stored/server keys) is persisted.
- With auth **disabled**, the server runs in trusted-local mode and logs a loud startup warning.
  Run auth-disabled servers only on loopback or a trusted private network.

### Transport encryption (TLS 1.2/1.3)

- Configured via `tls_mode` = `disable` | `allow` | `require`, with `tls_cert_path` / `tls_key_path`
  (PEM). TLS is terminated with rustls — no OpenSSL — and restricted to TLS 1.2 and 1.3.
- In `require` mode a plaintext `StartupMessage` with no prior TLS handshake is rejected
  (SQLSTATE `08P01`); SCRAM runs **inside** the TLS channel.
- There is no self-signed fallback: a missing or malformed cert/key is a typed startup error.

### Authorization (RBAC)

- Table-level `GRANT` / `REVOKE` for `SELECT` / `INSERT` / `UPDATE` / `DELETE`, enforced at a single
  executor chokepoint before any storage access. Denied statements map to SQLSTATE `42501`.
- Only a superuser may run `CREATE/DROP/ALTER ROLE`, `GRANT`, and `REVOKE`. Grants take effect for
  subsequent statements without a restart.
- The same chokepoint covers the wire (extended protocol, `COPY`) and embedded paths.

### Audit log

- A JSONL audit sink records authentication, authorization decisions, DDL, and admin events. Point
  it at a file with `audit_log_path`; ship the file to your SIEM.

### Encryption at rest

- AES-256-GCM on every PAX block and WAL record. Key management is pluggable with no vendor lock-in:
  local file, environment variable, external command (any KMS CLI), and HashiCorp Vault Transit.
  Native cloud KMS providers (AWS/GCP/Azure over REST) are in progress — see [ROADMAP.md](ROADMAP.md).
- Master keys and credentials are sourced from the environment/config and never emitted in SQL,
  logs, or error messages.

### Reporting scope

Security reports are especially welcome for: authentication/authorization bypass, TLS downgrade,
key-management or credential leakage, WAL/SST tampering that escapes checksum verification, and any
path that returns another role's data.
