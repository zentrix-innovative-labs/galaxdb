//! Statement cache and prepared-statement parameter binding (Req 6, 7).
//!
//! Two pieces that together remove the per-statement SQL-parse cost that
//! caps single-row throughput:
//!
//! * [`StatementCache`] — a bounded LRU keyed by normalized SQL text that
//!   stores the parsed [`AuroraStatement`] list. On a hit the parser is
//!   skipped entirely (Req 7.2). Used by the simple-query path for
//!   repeated identical statements.
//! * [`bind_placeholders`] — substitutes positional parameters (`$1..$N`)
//!   into a statement template that was parsed **once**, so the extended
//!   query protocol's repeated `Execute`s never re-invoke the parser
//!   (Req 6 AC6 / Req 7). The values arrive already typed over the wire,
//!   so this is a pure AST rewrite, never a re-parse.
//!
//! It is purely a performance optimization: a cache hit produces the exact
//! same parsed form as a miss, and binding produces the exact same AST a
//! literal statement would parse to (Req 7.5).

use std::collections::HashMap;
use std::ops::ControlFlow;
use std::sync::Arc;

use sqlparser::ast::{Expr, Value, VisitMut, VisitorMut};

use galaxdb_common::{GalaxError, GalaxResult};

use crate::ast::AuroraStatement;
use crate::parser;

/// A bound parameter value, decoded from the wire by the server. Maps to a
/// concrete SQL literal when substituted into a prepared statement.
#[derive(Debug, Clone, PartialEq)]
pub enum BoundValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    Text(String),
    Null,
}

impl BoundValue {
    /// Convert to the sqlparser literal that replaces a `$n` placeholder.
    fn to_sql_value(&self) -> Value {
        match self {
            BoundValue::Int(i) => Value::Number(i.to_string(), false),
            BoundValue::Float(f) => {
                let mut s = f.to_string();
                // Keep it float-typed so it never collapses to an integer
                // literal in the AST.
                if f.is_finite()
                    && !s.contains('.')
                    && !s.contains('e')
                    && !s.contains('E')
                {
                    s.push_str(".0");
                }
                Value::Number(s, false)
            }
            BoundValue::Bool(b) => Value::Boolean(*b),
            BoundValue::Text(t) => Value::SingleQuotedString(t.clone()),
            BoundValue::Null => Value::Null,
        }
    }
}

/// A `VisitorMut` that rewrites `$n` placeholder expressions in place with
/// their bound values. Records the first placeholder index that has no
/// corresponding bound value so the caller can surface a typed error.
struct PlaceholderBinder<'a> {
    values: &'a [BoundValue],
    missing: Option<String>,
}

impl VisitorMut for PlaceholderBinder<'_> {
    type Break = ();

    fn pre_visit_expr(&mut self, expr: &mut Expr) -> ControlFlow<()> {
        if let Expr::Value(Value::Placeholder(p)) = expr {
            // PostgreSQL placeholders are `$1`, `$2`, … (1-based).
            match p.strip_prefix('$').and_then(|s| s.parse::<usize>().ok()) {
                Some(idx) if idx >= 1 && idx <= self.values.len() => {
                    *expr = Expr::Value(self.values[idx - 1].to_sql_value());
                }
                _ => {
                    self.missing = Some(p.clone());
                    return ControlFlow::Break(());
                }
            }
        }
        ControlFlow::Continue(())
    }
}

/// Bind positional parameters into a parsed statement template, returning a
/// concrete statement set with every `$n` replaced by its value. The
/// template is cloned (not re-parsed), so `Execute` is parser-free.
///
/// Only `Standard` SQL statements carry placeholders; the AuroraSQL
/// extension variants are passed through unchanged.
pub fn bind_placeholders(
    template: &[AuroraStatement],
    values: &[BoundValue],
) -> GalaxResult<Vec<AuroraStatement>> {
    let mut out = Vec::with_capacity(template.len());
    for stmt in template {
        let mut cloned = stmt.clone();
        if let AuroraStatement::Standard(boxed) = &mut cloned {
            let mut binder = PlaceholderBinder {
                values,
                missing: None,
            };
            let _ = boxed.as_mut().visit(&mut binder);
            if let Some(p) = binder.missing {
                return Err(GalaxError::SqlParse {
                    position: 0,
                    message: format!("no bound value for placeholder {p}"),
                });
            }
        }
        out.push(cloned);
    }
    Ok(out)
}

/// A bounded LRU cache of parsed statements keyed by normalized SQL text.
///
/// On a hit the SQL parser is not invoked (Req 7.2). Eviction is strict
/// least-recently-used once `capacity` distinct statements are cached
/// (Req 7.3). The cached value is an `Arc` so a hit clones only a pointer.
pub struct StatementCache {
    capacity: usize,
    map: HashMap<String, Arc<Vec<AuroraStatement>>>,
    /// Keys in least-recently-used order: front = oldest, back = newest.
    order: Vec<String>,
}

impl StatementCache {
    /// Create a cache holding at most `capacity` distinct statements
    /// (clamped to at least 1).
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            map: HashMap::new(),
            order: Vec::new(),
        }
    }

    /// Return the parsed form of `sql`, parsing and caching it on a miss.
    /// On a hit the parser is skipped and the statement is marked
    /// most-recently-used.
    pub fn get_or_parse(&mut self, sql: &str) -> GalaxResult<Arc<Vec<AuroraStatement>>> {
        let key = Self::normalize(sql);
        if let Some(v) = self.map.get(&key).cloned() {
            self.touch(&key);
            return Ok(v);
        }
        let parsed = Arc::new(parser::parse(sql)?);
        self.insert(key, parsed.clone());
        Ok(parsed)
    }

    /// Number of distinct statements currently cached.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Whether `sql` (normalized) is currently cached — for tests/metrics.
    pub fn contains(&self, sql: &str) -> bool {
        self.map.contains_key(&Self::normalize(sql))
    }

    /// Normalize SQL to a cache key. Trimming surrounding whitespace is
    /// safe (it never changes the parse) and lets a client's repeated,
    /// byte-identical statements share one cache entry. Internal
    /// whitespace is left untouched so string literals are never altered.
    fn normalize(sql: &str) -> String {
        sql.trim().to_string()
    }

    /// Mark `key` most-recently-used.
    fn touch(&mut self, key: &str) {
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            let k = self.order.remove(pos);
            self.order.push(k);
        }
    }

    /// Insert a freshly parsed entry, evicting the least-recently-used
    /// entry if at capacity.
    fn insert(&mut self, key: String, val: Arc<Vec<AuroraStatement>>) {
        if self.map.contains_key(&key) {
            self.map.insert(key.clone(), val);
            self.touch(&key);
            return;
        }
        if self.map.len() >= self.capacity {
            if let Some(evict) = self.order.first().cloned() {
                self.order.remove(0);
                self.map.remove(&evict);
            }
        }
        self.order.push(key.clone());
        self.map.insert(key, val);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_hit_returns_same_parsed_form() {
        let mut cache = StatementCache::new(8);
        let sql = "SELECT id, name FROM users WHERE id = 1";
        let first = cache.get_or_parse(sql).unwrap();
        let second = cache.get_or_parse(sql).unwrap();
        // Same Arc allocation → the second call did not re-parse.
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn whitespace_is_normalized_for_the_key() {
        let mut cache = StatementCache::new(8);
        let a = cache.get_or_parse("  SELECT 1  ").unwrap();
        let b = cache.get_or_parse("SELECT 1").unwrap();
        assert!(Arc::ptr_eq(&a, &b), "trimmed whitespace must hit the same entry");
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn lru_evicts_oldest_when_over_capacity() {
        let mut cache = StatementCache::new(2);
        cache.get_or_parse("SELECT 1").unwrap();
        cache.get_or_parse("SELECT 2").unwrap();
        assert_eq!(cache.len(), 2);
        // Touch "SELECT 1" so "SELECT 2" becomes the LRU victim.
        cache.get_or_parse("SELECT 1").unwrap();
        cache.get_or_parse("SELECT 3").unwrap(); // evicts "SELECT 2"
        assert_eq!(cache.len(), 2);
        assert!(cache.contains("SELECT 1"));
        assert!(cache.contains("SELECT 3"));
        assert!(!cache.contains("SELECT 2"), "LRU entry must have been evicted");
    }

    #[test]
    fn bind_substitutes_positional_parameters() {
        let template = parser::parse("INSERT INTO t (id, name, ok) VALUES ($1, $2, $3)").unwrap();
        let bound = bind_placeholders(
            &template,
            &[
                BoundValue::Int(7),
                BoundValue::Text("alice".to_string()),
                BoundValue::Bool(true),
            ],
        )
        .unwrap();
        // The bound AST must equal what the concrete literal SQL parses to.
        let expected = parser::parse("INSERT INTO t (id, name, ok) VALUES (7, 'alice', true)").unwrap();
        assert_eq!(bound, expected);
    }

    #[test]
    fn bind_substitutes_in_where_clause() {
        let template = parser::parse("SELECT id FROM t WHERE id = $1").unwrap();
        let bound = bind_placeholders(&template, &[BoundValue::Int(42)]).unwrap();
        let expected = parser::parse("SELECT id FROM t WHERE id = 42").unwrap();
        assert_eq!(bound, expected);
    }

    #[test]
    fn bind_null_and_float() {
        let template = parser::parse("INSERT INTO t (a, b) VALUES ($1, $2)").unwrap();
        let bound = bind_placeholders(&template, &[BoundValue::Null, BoundValue::Float(3.5)]).unwrap();
        let expected = parser::parse("INSERT INTO t (a, b) VALUES (NULL, 3.5)").unwrap();
        assert_eq!(bound, expected);
    }

    #[test]
    fn bind_missing_value_errors() {
        let template = parser::parse("SELECT id FROM t WHERE id = $1").unwrap();
        assert!(bind_placeholders(&template, &[]).is_err());
    }
}
