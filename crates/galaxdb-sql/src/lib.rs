//! GalaxDB SQL — SQL Parser (sqlparser-rs + AuroraSQL), Query Planner, Query Executor.
//!
//! The parser handles standard SQL via `sqlparser-rs` and extends it with
//! AuroraSQL syntax: EMBEDDING MODEL, SEMANTIC_MATCH, AT VERSION,
//! CREATE VERSION TAG, BULK INSERT, SHOW EMBEDDING HEALTH, BACKUP/RESTORE.

pub mod ast;
pub mod parser;

#[cfg(test)]
mod tests;
