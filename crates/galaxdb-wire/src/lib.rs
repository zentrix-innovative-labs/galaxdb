//! GalaxDB Wire — PostgreSQL simple query wire protocol + pg_catalog stubs.

pub mod messages;
pub mod pg_catalog;
pub mod server;
pub mod tls;

#[cfg(test)]
mod tests;
