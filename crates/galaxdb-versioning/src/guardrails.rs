//! Semantic guardrails for AT VERSION + SEMANTIC_MATCH queries.
//!
//! Rules:
//! 1. AT VERSION + SEMANTIC_MATCH without CONSISTENCY mode → reject with error
//! 2. CONSISTENCY 'SEMANTIC_FRESH' → allowed (search current HNSW, warn in metadata)
//! 3. CONSISTENCY 'SEMANTIC_SNAPSHOT' → reject ("v2 feature")
//! 4. AT VERSION without SEMANTIC_MATCH → always allowed (ROW_SNAPSHOT)

use crate::tags::ConsistencyMode;

/// Validate whether a query combination is allowed.
///
/// Returns Ok(mode) if allowed, Err(message) if rejected.
pub fn validate_version_query(
    has_at_version: bool,
    has_semantic_match: bool,
    consistency: Option<ConsistencyMode>,
) -> Result<Option<ConsistencyMode>, String> {
    if !has_at_version {
        // No AT VERSION — everything is allowed
        return Ok(None);
    }

    if !has_semantic_match {
        // AT VERSION without SEMANTIC_MATCH — always allowed (ROW_SNAPSHOT)
        return Ok(Some(ConsistencyMode::RowSnapshot));
    }

    // AT VERSION + SEMANTIC_MATCH — requires explicit consistency mode
    match consistency {
        None => Err(
            "AT VERSION with SEMANTIC_MATCH requires an explicit CONSISTENCY mode. \
             Use CONSISTENCY 'SEMANTIC_FRESH' to search the current index against \
             historical rows, or remove SEMANTIC_MATCH for a pure row snapshot."
                .to_string()
        ),
        Some(ConsistencyMode::RowSnapshot) => Err(
            "ROW_SNAPSHOT consistency does not support SEMANTIC_MATCH. \
             Use CONSISTENCY 'SEMANTIC_FRESH' or remove SEMANTIC_MATCH."
                .to_string()
        ),
        Some(ConsistencyMode::SemanticFresh) => Ok(Some(ConsistencyMode::SemanticFresh)),
        Some(ConsistencyMode::SemanticSnapshot) => Err(
            "CONSISTENCY 'SEMANTIC_SNAPSHOT' is a v2 feature and is not yet implemented. \
             Use CONSISTENCY 'SEMANTIC_FRESH' instead."
                .to_string()
        ),
    }
}

/// Warning message for SEMANTIC_FRESH queries.
pub const SEMANTIC_FRESH_WARNING: &str =
    "WARNING: SEMANTIC_FRESH searches the current HNSW index against historical rows. \
     Results may include vectors that did not exist at the requested version, \
     and may miss vectors that have been deleted since.";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_at_version_always_allowed() {
        let result = validate_version_query(false, false, None);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), None);

        let result = validate_version_query(false, true, None);
        assert!(result.is_ok());
    }

    #[test]
    fn at_version_without_semantic_match_allowed() {
        let result = validate_version_query(true, false, None);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Some(ConsistencyMode::RowSnapshot));
    }

    #[test]
    fn at_version_with_semantic_match_no_consistency_rejected() {
        let result = validate_version_query(true, true, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("requires an explicit CONSISTENCY mode"));
    }

    #[test]
    fn at_version_semantic_match_row_snapshot_rejected() {
        let result = validate_version_query(true, true, Some(ConsistencyMode::RowSnapshot));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("does not support SEMANTIC_MATCH"));
    }

    #[test]
    fn at_version_semantic_match_semantic_fresh_allowed() {
        let result = validate_version_query(true, true, Some(ConsistencyMode::SemanticFresh));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Some(ConsistencyMode::SemanticFresh));
    }

    #[test]
    fn at_version_semantic_match_semantic_snapshot_rejected() {
        let result = validate_version_query(true, true, Some(ConsistencyMode::SemanticSnapshot));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("v2 feature"));
    }
}
