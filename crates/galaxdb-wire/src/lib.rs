//! GalaxDB Wire — PostgreSQL simple query wire protocol.
//!
//! Implements the PostgreSQL v3 simple query protocol (Q message flow)
//! with startup handshake, RowDescription, DataRow, CommandComplete,
//! ErrorResponse, and ReadyForQuery messages.

pub mod messages;
pub mod server;

#[cfg(test)]
mod tests;
