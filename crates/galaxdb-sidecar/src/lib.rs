//! GalaxDB Embedding Sidecar — shared library for protocol types and client.
//!
//! This crate provides:
//! - `protocol`: Wire protocol types and serialization for Unix socket communication
//! - The sidecar binary (`galaxdb-sidecar`) uses these types for the server side
//! - The engine uses these types via `SidecarClient` for the client side

pub mod manager;
pub mod protocol;
pub mod tracking;
