//! GalaxDB SQL — SQL Parser, Query Planner, Query Executor, Transaction Manager.

pub mod ast;
pub mod executor;
pub mod parser;
pub mod planner;
pub mod transaction;

pub use executor::{
    execute, execute_with_policies, is_text_column, Catalog, CatalogColumn, ExecuteResult,
    InMemorySystemColumnSink, MinHashPolicy, NoOpVectorBackend, Row, SystemColumnSink,
    SystemColumnWrite, TableEntry, VectorSearchBackend, VectorSearchResult,
};

#[cfg(test)]
mod tests;
#[cfg(test)]
mod planner_tests;
#[cfg(test)]
mod executor_tests;
