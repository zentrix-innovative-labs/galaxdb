class Galaxdb < Formula
  desc "AI-native database — SQL + vector search + training exports in one system"
  homepage "https://github.com/zentrix-innovative-labs/galaxdb"
  version "0.2.0"
  license "Apache-2.0"

  on_macos do
    on_intel do
      url "https://github.com/zentrix-innovative-labs/galaxdb/releases/download/v0.2.0/galaxdb-server-macos-x86_64"
      sha256 "5d8838cfcd5cbaa90790c8c0667c1333dbb10d304a907977dfd1ed15e42e933d"
    end

    on_arm do
      url "https://github.com/zentrix-innovative-labs/galaxdb/releases/download/v0.2.0/galaxdb-server-macos-arm64"
      sha256 "9cdfe55ff3622e34c56961a37948062b2e4971189b3d8eb5c2d7be64d1b3f5f8"
    end
  end

  on_linux do
    on_intel do
      url "https://github.com/zentrix-innovative-labs/galaxdb/releases/download/v0.2.0/galaxdb-server-linux-x86_64"
      sha256 "a8554637dba25b2489c19453097f6ee2fcbc6195fff3c6490c44f2772ec3f1c2"
    end

    on_arm do
      url "https://github.com/zentrix-innovative-labs/galaxdb/releases/download/v0.2.0/galaxdb-server-linux-aarch64"
      sha256 "dd3e90ad7cb884441b9da639e884e02bef8534afe2e6e77a51a9283d60acb12a"
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

      Python client:
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
