class Galaxdb < Formula
  desc "AI-native database — SQL + vector search + training exports in one system"
  homepage "https://github.com/zentrix-innovative-labs/galaxdb"
  version "1.0.0-beta.1"
  license "Apache-2.0"

  on_macos do
    on_intel do
      url "https://github.com/zentrix-innovative-labs/galaxdb/releases/download/v1.0.0-beta.1/galaxdb-server-macos-x86_64"
      sha256 "d8c544abae0b4659a869db68cecd1bf0436f73b014f019b40543b79922bc55a8"
    end
  end

  on_linux do
    on_intel do
      url "https://github.com/zentrix-innovative-labs/galaxdb/releases/download/v1.0.0-beta.1/galaxdb-server-linux-x86_64"
      sha256 "346633fdd3db375195986357f05a92a51f62038f11e3edaef331a2a11a83c9c1"
    end
  end

  def install
    bin.install stable.url.split("/").last => "galaxdb-server"
  end

  def caveats
    <<~EOS
      To start GalaxDB:
        galaxdb-server --data-dir ~/galaxdb-data --port 5433

      For embedding support, also download the sidecar:
        https://github.com/zentrix-innovative-labs/galaxdb/releases/latest

      Then start with:
        galaxdb-server --data-dir ~/galaxdb-data --sidecar galaxdb-sidecar --model sentence-transformers/all-MiniLM-L6-v2
    EOS
  end

  test do
    system "#{bin}/galaxdb-server", "--help"
  end
end
