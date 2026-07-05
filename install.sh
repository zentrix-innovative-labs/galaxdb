#!/usr/bin/env bash
# GalaxDB installer — downloads the server binary for your platform
# Usage: curl -fsSL https://raw.githubusercontent.com/zentrix-innovative-labs/galaxdb/main/install.sh | bash

set -euo pipefail

REPO="zentrix-innovative-labs/galaxdb"
INSTALL_DIR="${GALAXDB_INSTALL_DIR:-/usr/local/bin}"

# Resolve the version to install. By default we ask the GitHub API for the
# latest published release so this script never needs editing on a new
# release. Pin a specific version with GALAXDB_VERSION=v0.3.0 if you need to.
detect_latest_version() {
    local api="https://api.github.com/repos/${REPO}/releases/latest"
    local tag
    # Parse "tag_name": "v0.3.0" without requiring jq.
    tag="$(curl -fsSL "$api" \
        | grep -m1 '"tag_name"' \
        | sed -E 's/.*"tag_name"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/')"
    if [ -z "$tag" ]; then
        echo "ERROR: could not determine the latest GalaxDB release from ${api}." >&2
        echo "       Set GALAXDB_VERSION=vX.Y.Z to install a specific version." >&2
        exit 1
    fi
    echo "$tag"
}

VERSION="${GALAXDB_VERSION:-$(detect_latest_version)}"

detect_platform() {
    local os arch
    os="$(uname -s)"
    arch="$(uname -m)"

    case "$os" in
        Linux)
            case "$arch" in
                x86_64) echo "linux-x86_64" ;;
                aarch64|arm64) echo "linux-aarch64" ;;
                *) echo "unsupported arch: $arch" >&2; exit 1 ;;
            esac
            ;;
        Darwin)
            case "$arch" in
                x86_64) echo "macos-x86_64" ;;
                arm64) echo "macos-arm64" ;;
                *) echo "unsupported arch: $arch" >&2; exit 1 ;;
            esac
            ;;
        *)
            echo "unsupported OS: $os" >&2
            exit 1
            ;;
    esac
}

PLATFORM="$(detect_platform)"
BASE_URL="https://github.com/${REPO}/releases/download/${VERSION}"

echo "Installing GalaxDB ${VERSION} for ${PLATFORM}..."

# Download server binary
curl -fsSL "${BASE_URL}/galaxdb-server-${PLATFORM}" -o /tmp/galaxdb-server
chmod +x /tmp/galaxdb-server

# Install
mkdir -p "$INSTALL_DIR" 2>/dev/null || true
if [ -w "$INSTALL_DIR" ]; then
    mv /tmp/galaxdb-server "${INSTALL_DIR}/galaxdb-server"
else
    sudo mkdir -p "$INSTALL_DIR"
    sudo mv /tmp/galaxdb-server "${INSTALL_DIR}/galaxdb-server"
fi

echo ""
echo "✓ GalaxDB installed to ${INSTALL_DIR}/galaxdb-server"
echo ""
echo "Quick start:"
echo "  galaxdb-server --data-dir ./mydata --port 5433"
echo ""
echo "Python client:"
echo "  pip install galaxdb"
echo ""
echo "Docs: https://github.com/${REPO}"
