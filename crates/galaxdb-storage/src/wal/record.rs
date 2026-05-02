//! WAL record format and types.
//!
//! Record wire format:
//! ```text
//! ┌──────┬──────────┬────────┬──────────────┬─────────────────┐
//! │ type │ seq_no   │ length │ xxh3_checksum│ lz4_payload     │
//! │ u8   │ u64      │ u32    │ u64          │ [u8; length]    │
//! └──────┴──────────┴────────┴──────────────┴─────────────────┘
//! ```
//!
//! - `type` (1 byte): record type discriminant
//! - `seq_no` (8 bytes): monotonically increasing sequence number
//! - `length` (4 bytes): byte length of the compressed payload
//! - `xxh3_checksum` (8 bytes): XXH3-64 over `[type | seq_no | length | lz4_payload]`
//! - `lz4_payload` (variable): LZ4-compressed original payload

use std::io::{self, Read, Write};

use xxhash_rust::xxh3::xxh3_64;

/// Size of the fixed-size WAL record header (type + seq_no + length + checksum).
pub const WAL_RECORD_HEADER_SIZE: usize = 1 + 8 + 4 + 8; // 21 bytes

/// WAL record type discriminants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum WalRecordType {
    /// Row insert or update.
    RowPut = 0x01,
    /// Row tombstone (delete).
    RowDelete = 0x02,
    /// Vector delta buffer insert.
    DeltaInsert = 0x03,
    /// Vector delta buffer tombstone.
    DeltaTombstone = 0x04,
    /// Checkpoint marker.
    Checkpoint = 0x05,
    /// Blob log reference for KV-separated values.
    BlobRef = 0x06,
}

impl WalRecordType {
    /// Convert a raw byte to a `WalRecordType`, returning `None` for unknown values.
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0x01 => Some(Self::RowPut),
            0x02 => Some(Self::RowDelete),
            0x03 => Some(Self::DeltaInsert),
            0x04 => Some(Self::DeltaTombstone),
            0x05 => Some(Self::Checkpoint),
            0x06 => Some(Self::BlobRef),
            _ => None,
        }
    }
}

/// A single WAL record with its metadata and payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalRecord {
    /// The type of this record.
    pub record_type: WalRecordType,
    /// Monotonically increasing sequence number.
    pub seq_no: u64,
    /// The uncompressed payload bytes.
    pub payload: Vec<u8>,
}

impl WalRecord {
    /// Create a new WAL record.
    pub fn new(record_type: WalRecordType, seq_no: u64, payload: Vec<u8>) -> Self {
        Self {
            record_type,
            seq_no,
            payload,
        }
    }

    /// Serialize this record to bytes (with LZ4 compression and XXH3-64 checksum).
    ///
    /// Wire format: `[type:u8][seq_no:u64][length:u32][xxh3_checksum:u64][lz4_payload:bytes]`
    ///
    /// The checksum covers `[type | seq_no | length | lz4_payload]`.
    pub fn serialize(&self) -> Vec<u8> {
        let compressed = lz4_flex::compress_prepend_size(&self.payload);
        let length = compressed.len() as u32;

        // Build the data that the checksum covers: type + seq_no + length + lz4_payload
        let mut checksum_input = Vec::with_capacity(1 + 8 + 4 + compressed.len());
        checksum_input.push(self.record_type as u8);
        checksum_input.extend_from_slice(&self.seq_no.to_le_bytes());
        checksum_input.extend_from_slice(&length.to_le_bytes());
        checksum_input.extend_from_slice(&compressed);

        let checksum = xxh3_64(&checksum_input);

        // Write the full record: type + seq_no + length + checksum + lz4_payload
        let mut buf = Vec::with_capacity(WAL_RECORD_HEADER_SIZE + compressed.len());
        buf.push(self.record_type as u8);
        buf.extend_from_slice(&self.seq_no.to_le_bytes());
        buf.extend_from_slice(&length.to_le_bytes());
        buf.extend_from_slice(&checksum.to_le_bytes());
        buf.extend_from_slice(&compressed);

        buf
    }

    /// Deserialize a WAL record from a reader, verifying the XXH3-64 checksum.
    ///
    /// Returns `Ok(Some(record))` on success, `Ok(None)` on clean EOF (no partial
    /// header), or `Err` on I/O error or checksum mismatch.
    pub fn deserialize<R: Read>(reader: &mut R) -> io::Result<Option<Self>> {
        // Read the fixed header
        let mut header = [0u8; WAL_RECORD_HEADER_SIZE];
        match reader.read_exact(&mut header) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e),
        }

        let record_type_byte = header[0];
        let seq_no = u64::from_le_bytes(header[1..9].try_into().unwrap());
        let length = u32::from_le_bytes(header[9..13].try_into().unwrap());
        let stored_checksum = u64::from_le_bytes(header[13..21].try_into().unwrap());

        let record_type = WalRecordType::from_u8(record_type_byte).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown WAL record type: {:#04x}", record_type_byte),
            )
        })?;

        // Read the compressed payload
        let mut compressed = vec![0u8; length as usize];
        reader.read_exact(&mut compressed)?;

        // Verify checksum over [type | seq_no | length | lz4_payload]
        let mut checksum_input = Vec::with_capacity(1 + 8 + 4 + compressed.len());
        checksum_input.push(record_type_byte);
        checksum_input.extend_from_slice(&seq_no.to_le_bytes());
        checksum_input.extend_from_slice(&length.to_le_bytes());
        checksum_input.extend_from_slice(&compressed);

        let computed_checksum = xxh3_64(&checksum_input);
        if computed_checksum != stored_checksum {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "WAL checksum mismatch at seq_no {}: expected {:#018x}, got {:#018x}",
                    seq_no, stored_checksum, computed_checksum
                ),
            ));
        }

        // Decompress the payload
        let payload = lz4_flex::decompress_size_prepended(&compressed).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("LZ4 decompression failed at seq_no {}: {}", seq_no, e),
            )
        })?;

        Ok(Some(Self {
            record_type,
            seq_no,
            payload,
        }))
    }

    /// Write this record's serialized bytes to a writer.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> io::Result<usize> {
        let bytes = self.serialize();
        writer.write_all(&bytes)?;
        Ok(bytes.len())
    }
}
