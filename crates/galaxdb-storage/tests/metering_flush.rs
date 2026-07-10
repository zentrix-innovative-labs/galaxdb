//! v0.6 E-4 metering (M.6): the storage engine persists the metering counter
//! file to the data volume on flush/checkpoint. This verifies the *wiring*
//! (Engine::flush → galaxdb_observe::flush_metering) — the counter-value
//! correctness is proven in galaxdb-observe's `metering_persist` test, and the
//! cross-process restore is proven by the v0.6 real-data script.

use galaxdb_storage::engine::{Engine, EngineConfig};

#[tokio::test]
async fn flush_persists_metering_file_on_the_volume() {
    let dir = tempfile::tempdir().unwrap();
    let config = EngineConfig {
        data_dir: dir.path().to_path_buf(),
        ..Default::default()
    };
    let engine = Engine::new(config).unwrap();
    engine.put(b"k1".to_vec(), b"v1".to_vec()).await.unwrap();
    engine.flush_memtable().await.unwrap();

    let meter = dir.path().join("metering.gmet");
    assert!(
        meter.exists(),
        "flush/checkpoint must persist the metering counter file to the data dir"
    );
    let bytes = std::fs::read(&meter).unwrap();
    assert!(bytes.len() >= 16 + 6 * 8, "metering file has header + 6 u64");
    assert_eq!(
        &bytes[0..4],
        b"GMET",
        "metering file must carry the GMET format magic"
    );
}
