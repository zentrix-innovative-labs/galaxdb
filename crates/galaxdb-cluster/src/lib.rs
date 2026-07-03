//! GalaxDB cluster coordination seam and provider registry.
//!
//! This crate defines the **semver-stable extension boundary** that separates
//! the open-source single-node engine from an enterprise cluster layer. An
//! enterprise crate can supply its own [`ClusterCoordinator`] implementation,
//! wrap it in a [`Providers`] bundle, and pass it into the server's
//! [`ServerConfig::providers`] field — the engine never names an enterprise
//! type.
//!
//! # Coordinator
//!
//! [`ClusterCoordinator`] exposes three methods that the server consults at
//! startup and per-query:
//!
//! * [`ClusterCoordinator::get_node_role`] — returns this node's role in the
//!   cluster topology.
//! * [`ClusterCoordinator::route_query`] — returns the [`Routing`] decision
//!   for a query (which node id should handle it, plus the leader's address
//!   for redirects).
//! * [`ClusterCoordinator::is_leader`] — convenience predicate; `true` when
//!   this node may accept writes.
//!
//! The open-source bundle ships [`SingleNodeCoordinator`], which always
//! reports the node as [`NodeRole::Standalone`] and routes every query to the
//! local node. It never contacts a remote peer and never fails.
//!
//! # Providers
//!
//! [`Providers`] bundles the three extension slots the engine consults:
//!
//! * `security` — [`galaxdb_auth::SecurityProviders`] (authentication,
//!   authorization, audit).
//! * `key_spec` — `Option<`[`galaxdb_crypto::KeyProviderSpec`]`>` (encryption
//!   key management). `None` means TDE is not configured.
//! * `cluster` — `Arc<dyn ClusterCoordinator>` (topology and routing).
//!
//! Call [`Providers::single_node_default`] to obtain the all-OSS default
//! bundle without any configuration.
//!
//! # Stability
//!
//! The [`ClusterCoordinator`] trait and the [`Providers`] struct are a
//! **stable extension API**. Breaking changes follow semantic versioning so
//! that an enterprise crate pinned to a compatible version of this crate can
//! supply its own implementations without recompiling the core engine.

pub mod coordinator;
pub mod providers;

pub use coordinator::{ClusterCoordinator, NodeRole, Routing, SingleNodeCoordinator};
pub use providers::Providers;
