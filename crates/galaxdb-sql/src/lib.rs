//! GalaxDB SQL — SQL Parser, Query Planner, Query Executor.
//!
//! Handles standard SQL via `sqlparser-rs` and extends it with AuroraSQL syntax.

pub mod ast;
pub mod executor;
pub mod parser;
pub mod planner;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod planner_tests;
#[cfg(test)]
mod executor_tests;
