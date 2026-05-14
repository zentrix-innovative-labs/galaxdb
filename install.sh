#!/usr/bin/env bash
# GalaxDB installer — downloads the server binary for your platform
# Usage: curl -fsSL https://raw.githubusercontent.com/zentrix-innovative-labs/galaxdb/main/install.sh | bash

set -euo pipefail

REPO="zentrix-innovative-labs/galaxdb"
VERSION="v1.0.0-beta.1"
INSTALL_DIR="${GALAXDB_INSTALL_DIR:-/usr/local/bin}"

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
if [ -w "$INSTALL_DIR" ]; then
    mv /tmp/galaxdb-server "${INSTALL_DIR}/galaxdb-server"
else
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
