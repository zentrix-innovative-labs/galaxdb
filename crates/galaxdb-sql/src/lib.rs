//! GalaxDB SQL — SQL Parser, Query Planner, Query Executor, Transaction Manager.

pub mod ast;
pub mod executor;
pub mod parser;
pub mod planner;
pub mod row_codec;
pub mod transaction;

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
