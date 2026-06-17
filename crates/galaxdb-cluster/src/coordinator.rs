//! Cluster coordinator trait and the single-node reference implementation.
//!
//! # Extension boundary
//!
//! [`ClusterCoordinator`] is a **semver-stable trait**. The engine uses it
//! solely through a `dyn ClusterCoordinator` reference inside
//! [`crate::Providers`], so an enterprise crate can supply any implementation
//! without changing engine code. The trait methods are kept deliberately
//! minimal: only the decisions the server actually needs to make.

use std::sync::Arc;

/// The role this node plays in the cluster topology.
///
/// The variant determines which operations the node may accept:
///
/// | Role | Accepts writes | Accepts reads |
/// |------|---------------|---------------|
/// | [`Leader`](NodeRole::Leader) | yes | yes |
/// | [`Follower`](NodeRole::Follower) | no | no (by default) |
/// | [`ReadReplica`](NodeRole::ReadReplica) | no | yes |
/// | [`Standalone`](NodeRole::Standalone) | yes | yes |
///
/// [`Standalone`](NodeRole::Standalone) is the role of a node that is not
/// part of a multi-node cluster — the open-source single-node case. The
/// engine treats `Standalone` exactly like `Leader`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeRole {
    /// This node is the Raft/Paxos leader; it accepts writes.
    Leader,
    /// This node is a follower; it replicates from the leader.
    Follower,
    /// This node is a read-only replica; it accepts read queries only.
    ReadReplica,
    /// This node is not part of a multi-node cluster (single-node mode).
    Standalone,
}

/// Describes where a query should be executed.
///
/// The server receives a [`Routing`] decision from
/// [`ClusterCoordinator::route_query`] and either executes the query locally
/// (when `node_id` matches the local node) or redirects the client
/// to `leader_addr`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Routing {
    /// Logical node identifier that should handle this query (e.g.
    /// `"local"`, `"node-1"`, a UUID string).
    pub node_id: String,
    /// The leader's network address in `host:port` form, populated when this
    /// node is not the leader and the client should be redirected. `None`
    /// when this node can handle the query directly (leader or standalone).
    pub leader_addr: Option<String>,
}

/// Determines cluster topology and query routing for the server.
///
/// # Stability
///
/// This trait is a **semver-stable extension boundary**. The open-source
/// engine instantiates it only via [`Arc<dyn ClusterCoordinator>`] inside
/// [`crate::Providers`]. Enterprise editions implement it to provide real
/// distributed consensus, leader election, and topology queries.
///
/// Breaking changes (new required methods, changed signatures) follow
/// semantic versioning of this crate.
///
/// # Implementing the trait
///
/// A correct minimal implementation must satisfy these invariants:
///
/// * If [`is_leader`] returns `true`, then [`get_node_role`] must return
///   either [`NodeRole::Leader`] or [`NodeRole::Standalone`].
/// * If [`is_leader`] returns `false`, then [`get_node_role`] must return
///   [`NodeRole::Follower`] or [`NodeRole::ReadReplica`].
/// * The [`Routing`] returned by [`route_query`] must have `leader_addr =
///   None` whenever [`is_leader`] returns `true`.
///
/// [`is_leader`]: ClusterCoordinator::is_leader
/// [`get_node_role`]: ClusterCoordinator::get_node_role
/// [`route_query`]: ClusterCoordinator::route_query
pub trait ClusterCoordinator: Send + Sync {
    /// Returns the role of this node in the cluster topology.
    ///
    /// Called once at startup (for logging) and on any topology-change
    /// notification. Results are not cached by the engine.
    fn get_node_role(&self) -> NodeRole;

    /// Returns the routing decision for a query described by `query_type`.
    ///
    /// `query_type` is a hint from the server (`"read"` or `"write"`) that
    /// allows the coordinator to direct reads to replicas when appropriate.
    /// Implementations are free to ignore it and always route locally.
    ///
    /// A [`Routing`] with `leader_addr = None` means the local node should
    /// handle the query. A non-`None` `leader_addr` is the address to which
    /// the client should be redirected.
    fn route_query(&self, query_type: &str) -> Routing;

    /// Returns `true` if this node may accept write queries.
    ///
    /// Equivalent to checking whether [`get_node_role`] returns
    /// [`NodeRole::Leader`] or [`NodeRole::Standalone`], but provided as a
    /// direct predicate for the server's hot path.
    ///
    /// [`get_node_role`]: ClusterCoordinator::get_node_role
    fn is_leader(&self) -> bool;

    /// A short, stable name for logging and diagnostics (e.g.
    /// `"single-node"`, `"raft-v1"`).
    fn name(&self) -> &str;
}

/// The open-source single-node coordinator.
///
/// This coordinator always reports the node as [`NodeRole::Standalone`],
/// always routes queries to the local node, and always reports itself as the
/// leader. It never contacts a remote peer and never fails.
///
/// Pass an instance to [`crate::Providers::new`] (or obtain one via
/// [`crate::Providers::single_node_default`]) to run GalaxDB in standalone
/// mode.
pub struct SingleNodeCoordinator;

impl ClusterCoordinator for SingleNodeCoordinator {
    fn get_node_role(&self) -> NodeRole {
        NodeRole::Standalone
    }

    fn route_query(&self, _query_type: &str) -> Routing {
        Routing {
            node_id: "local".to_string(),
            leader_addr: None,
        }
    }

    fn is_leader(&self) -> bool {
        true
    }

    fn name(&self) -> &str {
        "single-node"
    }
}

/// Convenience constructor — wraps [`SingleNodeCoordinator`] in an `Arc`
/// suitable for storing in [`crate::Providers`].
pub fn single_node_arc() -> Arc<dyn ClusterCoordinator + Send + Sync> {
    Arc::new(SingleNodeCoordinator)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_node_coordinator_is_always_leader() {
        let coord = SingleNodeCoordinator;
        assert!(coord.is_leader());
        assert_eq!(coord.get_node_role(), NodeRole::Standalone);
    }

    #[test]
    fn single_node_coordinator_routes_locally() {
        let coord = SingleNodeCoordinator;

        let read_routing = coord.route_query("read");
        assert_eq!(read_routing.node_id, "local");
        assert_eq!(read_routing.leader_addr, None);

        let write_routing = coord.route_query("write");
        assert_eq!(write_routing.node_id, "local");
        assert_eq!(write_routing.leader_addr, None);
    }

    #[test]
    fn single_node_coordinator_name() {
        assert_eq!(SingleNodeCoordinator.name(), "single-node");
    }

    #[test]
    fn node_role_standalone_is_leader_consistent() {
        let coord = SingleNodeCoordinator;
        let role = coord.get_node_role();
        let leader = coord.is_leader();
        // Invariant: is_leader ↔ role is Leader or Standalone.
        let role_says_leader = matches!(role, NodeRole::Leader | NodeRole::Standalone);
        assert_eq!(leader, role_says_leader);
    }

    #[test]
    fn routing_leader_addr_none_when_leader() {
        let coord = SingleNodeCoordinator;
        assert!(coord.is_leader());
        let r = coord.route_query("write");
        // Invariant: if is_leader() then leader_addr must be None.
        assert_eq!(r.leader_addr, None);
    }
}
