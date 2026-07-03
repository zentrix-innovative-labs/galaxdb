//! GalaxDB SQL — SQL Parser, Query Planner, Query Executor, Transaction Manager.

pub mod ast;
pub mod auth_store;
pub mod classify;
pub mod columnar;
pub mod executor;
pub mod parser;
pub mod planner;
pub mod row_codec;
pub mod scalar;
pub mod secondary_index;
pub mod stmt_cache;
pub mod transaction;
pub mod types;

pub use auth_store::{AuthStore, RoleRecord};
pub use stmt_cache::{bind_placeholders, BoundValue, StatementCache};
pub use types::SqlType;
pub use executor::{
    execute_legacy, execute_with_context, is_text_column, Catalog, CatalogColumn, ExecuteResult,
    ExecutorContext, InMemorySystemColumnSink, MinHashPolicy, Row, SystemColumnSink,
    SystemColumnWrite, TableEntry, VectorSearchBackend, VectorSearchResult,
};

#[cfg(test)]
mod tests;
#[cfg(test)]
mod planner_tests;
#[cfg(test)]
mod executor_tests;
