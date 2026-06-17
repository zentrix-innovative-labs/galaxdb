class Galaxdb < Formula
  desc "AI-native database — SQL + vector search + training exports in one system"
  homepage "https://github.com/zentrix-innovative-labs/galaxdb"
  version "0.2.0"
  license "Apache-2.0"

  on_macos do
    on_intel do
      url "https://github.com/zentrix-innovative-labs/galaxdb/releases/download/v0.2.0/galaxdb-server-macos-x86_64"
      # sha256 updated after the release workflow uploads the binary
      sha256 :no_check
    end

    on_arm do
      url "https://github.com/zentrix-innovative-labs/galaxdb/releases/download/v0.2.0/galaxdb-server-macos-arm64"
      sha256 :no_check
    end
  end

  on_linux do
    on_intel do
      url "https://github.com/zentrix-innovative-labs/galaxdb/releases/download/v0.2.0/galaxdb-server-linux-x86_64"
      sha256 :no_check
    end

    on_arm do
      url "https://github.com/zentrix-innovative-labs/galaxdb/releases/download/v0.2.0/galaxdb-server-linux-aarch64"
      sha256 :no_check
    end
  end

  def install
    binary_name = stable.url.split("/").last
    bin.install binary_name => "galaxdb-server"
  end

  def caveats
    <<~EOS
      To start GalaxDB:
        galaxdb-server --data-dir ~/galaxdb-data --port 5433 --auth

      Python client (pip):
        pip install galaxdb-client

      For embedding support, also download the sidecar from the same release
      and start with:
        galaxdb-server --data-dir ~/galaxdb-data --sidecar galaxdb-sidecar \\
                       --model sentence-transformers/all-MiniLM-L6-v2

      Docs: https://github.com/zentrix-innovative-labs/galaxdb
    EOS
  end

  test do
    system "#{bin}/galaxdb-server", "--help"
  end
end
