//! Scalar expressions for `INSERT`/`UPDATE` value positions.
//!
//! A `SET col = <expr>` or `VALUES (<expr>)` position is a real scalar
//! expression, not just a literal: `UPDATE t SET bal = bal - 30` must compute
//! `old_bal - 30` per row. PostgreSQL evaluates such expressions against the
//! **old (pre-update) row values** for every matching row (all SET clauses in
//! one statement see the old values), per
//! <https://www.postgresql.org/docs/current/sql-update.html>. This module is
//! the GalaxDB-owned expression IR + evaluator that implements exactly that.
//!
//! Errors are typed and carry the PostgreSQL SQLSTATE (division by zero →
//! `22012`, overflow → `22003`, non-numeric operand → `42804`). Nothing is
//! ever silently coerced to a wrong value — the previous behavior of storing
//! the literal text `"bal - 30"` was a data-corruption bug this replaces.

use galaxdb_common::{GalaxError, GalaxResult};

use crate::planner::Value;

/// Binary arithmetic / string operators supported in a value position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    /// SQL string concatenation (`||`).
    Concat,
}

/// A scalar expression evaluated against a row's columns.
#[derive(Debug, Clone, PartialEq)]
pub enum ScalarExpr {
    /// A constant literal.
    Literal(Value),
    /// A reference to a column of the row being evaluated (the *old* row for
    /// `UPDATE`). Resolving a column that the row does not have is an error.
    Column(String),
    /// `left <op> right`.
    Binary {
        op: ArithOp,
        left: Box<ScalarExpr>,
        right: Box<ScalarExpr>,
    },
    /// Unary negation (`-expr`).
    Neg(Box<ScalarExpr>),
}

impl ScalarExpr {
    /// A bare literal expression (the common case).
    pub fn lit(v: Value) -> Self {
        ScalarExpr::Literal(v)
    }

    /// Evaluate against `row` (`(column, value)` pairs). Column references
    /// resolve to the row's values; an unknown column is
    /// [`GalaxError::ColumnNotFound`]. For contexts with no row (e.g. an
    /// `INSERT ... VALUES` position, which cannot reference columns) pass an
    /// empty slice — a column reference then correctly errors.
    pub fn eval(&self, row: &[(String, Value)]) -> GalaxResult<Value> {
        match self {
            ScalarExpr::Literal(v) => Ok(v.clone()),
            ScalarExpr::Column(name) => row
                .iter()
                .find(|(k, _)| k == name)
                .map(|(_, v)| v.clone())
                .ok_or_else(|| GalaxError::ColumnNotFound(name.clone())),
            ScalarExpr::Neg(e) => negate(e.eval(row)?),
            ScalarExpr::Binary { op, left, right } => {
                apply(*op, left.eval(row)?, right.eval(row)?)
            }
        }
    }
}

/// A numeric operand: exact integer, or floating point.
enum Num {
    Int(i64),
    Flt(f64),
}

/// Coerce a `Value` to a numeric operand for arithmetic. Text that parses as
/// a number is accepted (matches how numeric-typed columns are stored as
/// text); non-numeric values are a typed `42804` error, never a silent 0.
fn to_num(v: &Value) -> GalaxResult<Num> {
    match v {
        Value::Integer(n) => Ok(Num::Int(*n)),
        Value::Float(f) => Ok(Num::Flt(*f)),
        Value::Text(s) => {
            let t = s.trim();
            if let Ok(i) = t.parse::<i64>() {
                Ok(Num::Int(i))
            } else if let Ok(f) = t.parse::<f64>() {
                Ok(Num::Flt(f))
            } else {
                Err(GalaxError::Arithmetic {
                    sqlstate: "42804",
                    message: format!("operand '{s}' is not a number in an arithmetic expression"),
                })
            }
        }
        other => Err(GalaxError::Arithmetic {
            sqlstate: "42804",
            message: format!("cannot use {other:?} as an arithmetic operand"),
        }),
    }
}

fn overflow() -> GalaxError {
    GalaxError::Arithmetic {
        sqlstate: "22003",
        message: "integer out of range".to_string(),
    }
}

fn div_zero() -> GalaxError {
    GalaxError::Arithmetic {
        sqlstate: "22012",
        message: "division by zero".to_string(),
    }
}

fn negate(v: Value) -> GalaxResult<Value> {
    match v {
        Value::Null => Ok(Value::Null),
        Value::Integer(n) => n.checked_neg().map(Value::Integer).ok_or_else(overflow),
        Value::Float(f) => Ok(Value::Float(-f)),
        other => match to_num(&other)? {
            Num::Int(n) => n.checked_neg().map(Value::Integer).ok_or_else(overflow),
            Num::Flt(f) => Ok(Value::Float(-f)),
        },
    }
}

/// Render a value for string concatenation (SQL `||`).
fn concat_text(v: &Value) -> String {
    match v {
        Value::Text(s) => s.clone(),
        Value::Integer(n) => n.to_string(),
        Value::Float(f) => f.to_string(),
        Value::Bool(b) => b.to_string(),
        other => crate::row_codec::value_display(other),
    }
}

/// Apply a binary operator with SQL semantics: NULL propagates (any NULL
/// operand yields NULL); integer arithmetic is checked (overflow → `22003`);
/// division/modulo by zero → `22012`; mixing a float promotes to float;
/// `||` concatenates text.
fn apply(op: ArithOp, l: Value, r: Value) -> GalaxResult<Value> {
    // NULL propagation (three-valued logic): NULL <op> x = NULL.
    if matches!(l, Value::Null) || matches!(r, Value::Null) {
        return Ok(Value::Null);
    }
    if op == ArithOp::Concat {
        return Ok(Value::Text(format!("{}{}", concat_text(&l), concat_text(&r))));
    }
    match (to_num(&l)?, to_num(&r)?) {
        (Num::Int(a), Num::Int(b)) => {
            let out = match op {
                ArithOp::Add => a.checked_add(b).ok_or_else(overflow)?,
                ArithOp::Sub => a.checked_sub(b).ok_or_else(overflow)?,
                ArithOp::Mul => a.checked_mul(b).ok_or_else(overflow)?,
                ArithOp::Div => {
                    if b == 0 {
                        return Err(div_zero());
                    }
                    // PostgreSQL integer division truncates toward zero, which
                    // is Rust's `/` for i64 (and `checked_div` guards the one
                    // overflow case, i64::MIN / -1).
                    a.checked_div(b).ok_or_else(overflow)?
                }
                ArithOp::Mod => {
                    if b == 0 {
                        return Err(div_zero());
                    }
                    a.checked_rem(b).ok_or_else(overflow)?
                }
                ArithOp::Concat => unreachable!("handled above"),
            };
            Ok(Value::Integer(out))
        }
        (a, b) => {
            let (x, y) = (as_f64(a), as_f64(b));
            let out = match op {
                ArithOp::Add => x + y,
                ArithOp::Sub => x - y,
                ArithOp::Mul => x * y,
                ArithOp::Div => {
                    if y == 0.0 {
                        return Err(div_zero());
                    }
                    x / y
                }
                ArithOp::Mod => {
                    if y == 0.0 {
                        return Err(div_zero());
                    }
                    x % y
                }
                ArithOp::Concat => unreachable!("handled above"),
            };
            Ok(Value::Float(out))
        }
    }
}

fn as_f64(n: Num) -> f64 {
    match n {
        Num::Int(i) => i as f64,
        Num::Flt(f) => f,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row() -> Vec<(String, Value)> {
        vec![
            ("bal".to_string(), Value::Integer(100)),
            ("name".to_string(), Value::Text("acct".to_string())),
            ("rate".to_string(), Value::Float(1.5)),
        ]
    }

    fn col(name: &str) -> ScalarExpr {
        ScalarExpr::Column(name.to_string())
    }
    fn int(n: i64) -> ScalarExpr {
        ScalarExpr::Literal(Value::Integer(n))
    }
    fn bin(op: ArithOp, l: ScalarExpr, r: ScalarExpr) -> ScalarExpr {
        ScalarExpr::Binary { op, left: Box::new(l), right: Box::new(r) }
    }

    #[test]
    fn column_minus_literal() {
        // bal - 30 = 70 (the exact case the live test caught).
        let e = bin(ArithOp::Sub, col("bal"), int(30));
        assert_eq!(e.eval(&row()).unwrap(), Value::Integer(70));
    }

    #[test]
    fn column_plus_column_and_literal_fold() {
        let e = bin(ArithOp::Add, col("bal"), int(1));
        assert_eq!(e.eval(&row()).unwrap(), Value::Integer(101));
        // constant fold with no row (INSERT VALUES position): 2 + 3 = 5.
        let f = bin(ArithOp::Add, int(2), int(3));
        assert_eq!(f.eval(&[]).unwrap(), Value::Integer(5));
    }

    #[test]
    fn int_float_promotes_to_float() {
        let e = bin(ArithOp::Mul, col("rate"), int(2));
        assert_eq!(e.eval(&row()).unwrap(), Value::Float(3.0));
    }

    #[test]
    fn integer_division_truncates() {
        assert_eq!(bin(ArithOp::Div, int(7), int(2)).eval(&[]).unwrap(), Value::Integer(3));
        assert_eq!(bin(ArithOp::Mod, int(7), int(2)).eval(&[]).unwrap(), Value::Integer(1));
    }

    #[test]
    fn division_by_zero_is_typed_error() {
        let e = bin(ArithOp::Div, int(1), int(0));
        match e.eval(&[]) {
            Err(GalaxError::Arithmetic { sqlstate, .. }) => assert_eq!(sqlstate, "22012"),
            other => panic!("expected 22012 division_by_zero, got {other:?}"),
        }
    }

    #[test]
    fn overflow_is_typed_error() {
        let e = bin(ArithOp::Mul, int(i64::MAX), int(2));
        match e.eval(&[]) {
            Err(GalaxError::Arithmetic { sqlstate, .. }) => assert_eq!(sqlstate, "22003"),
            other => panic!("expected 22003 overflow, got {other:?}"),
        }
    }

    #[test]
    fn null_propagates() {
        let e = bin(ArithOp::Add, ScalarExpr::Literal(Value::Null), int(5));
        assert_eq!(e.eval(&[]).unwrap(), Value::Null);
    }

    #[test]
    fn non_numeric_operand_is_typed_error() {
        // name + 1 → 42804 datatype_mismatch (name is 'acct').
        let e = bin(ArithOp::Add, col("name"), int(1));
        match e.eval(&row()) {
            Err(GalaxError::Arithmetic { sqlstate, .. }) => assert_eq!(sqlstate, "42804"),
            other => panic!("expected 42804, got {other:?}"),
        }
    }

    #[test]
    fn concat_builds_text() {
        let e = bin(
            ArithOp::Concat,
            col("name"),
            ScalarExpr::Literal(Value::Text("-1".to_string())),
        );
        assert_eq!(e.eval(&row()).unwrap(), Value::Text("acct-1".to_string()));
    }

    #[test]
    fn unknown_column_is_error() {
        assert!(matches!(
            col("nope").eval(&row()),
            Err(GalaxError::ColumnNotFound(_))
        ));
    }

    #[test]
    fn negation() {
        assert_eq!(ScalarExpr::Neg(Box::new(col("bal"))).eval(&row()).unwrap(), Value::Integer(-100));
    }
}
