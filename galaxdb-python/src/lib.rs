//! GalaxDB Python client — PyO3 bindings for embedded and remote modes.
//!
//! Usage from Python:
//! ```python
//! import galaxdb
//!
//! # Embedded mode: talks to a local data directory.
//! db = galaxdb.Database("/path/to/data")
//! db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")
//! db.execute("INSERT INTO users (id, name) VALUES (1, 'alice')")
//! rows = db.execute("SELECT * FROM users")
//! for row in rows:
//!     print(row)
//!
//! # Remote mode: connects to a running galaxdb-server over PostgreSQL
//! # wire protocol.
//! conn = galaxdb.connect("host=127.0.0.1 port=5433 user=galaxdb dbname=galaxdb sslmode=disable")
//! conn.execute("CREATE TABLE t (id INT PRIMARY KEY, name TEXT)")
//! conn.execute("INSERT INTO t (id, name) VALUES (1, 'alice')")
//! rows = conn.execute("SELECT * FROM t")
//! conn.close()
//! ```

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use galaxdb_embedded::{Database as RustDatabase, QueryResult as RustQueryResult};
use postgres::{Client, NoTls, SimpleQueryMessage};

// ---------------------------------------------------------------------------
// Embedded mode: galaxdb.Database(path)
// ---------------------------------------------------------------------------

/// A GalaxDB database instance (embedded mode).
#[pyclass(unsendable)]
struct Database {
    inner: RustDatabase,
}

#[pymethods]
impl Database {
    /// Open or create a database at the given path.
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let inner = RustDatabase::open(path)
            .map_err(|e| PyRuntimeError::new_err(format!("failed to open database: {}", e)))?;
        Ok(Self { inner })
    }

    /// Execute a SQL statement.
    ///
    /// Returns:
    ///   - For SELECT/SHOW: list of dicts
    ///   - For INSERT/UPDATE/DELETE: number of rows affected
    ///   - For DDL: status string
    fn execute(&mut self, py: Python<'_>, sql: &str) -> PyResult<PyObject> {
        let result = self
            .inner
            .execute(sql)
            .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))?;
        query_result_to_python(py, result)
    }

    /// Get the database path.
    #[getter]
    fn path(&self) -> String {
        self.inner.path().to_string_lossy().to_string()
    }

    /// Get the number of tables.
    #[getter]
    fn table_count(&self) -> usize {
        self.inner.table_count()
    }

    /// Check if a table exists.
    fn table_exists(&self, name: &str) -> bool {
        self.inner.table_exists(name)
    }

    fn __repr__(&self) -> String {
        format!(
            "Database(path='{}', tables={})",
            self.inner.path().display(),
            self.inner.table_count()
        )
    }
}

// ---------------------------------------------------------------------------
// Remote mode: galaxdb.connect(connstring) → Connection
// ---------------------------------------------------------------------------

/// A live connection to a galaxdb-server over the PostgreSQL wire
/// protocol. Returned by `galaxdb.connect(connstring)`.
///
/// The connection uses the blocking `postgres` crate under the hood
/// (no tokio runtime is exposed to Python). Every call to `execute`
/// drives a real `SimpleQuery` exchange against the server and maps
/// the response back into the same shape as embedded mode:
///
/// * rows → `list[dict[str, str]]`
/// * write counts → `int`
/// * DDL / status → `str`
#[pyclass(unsendable)]
struct Connection {
    client: Option<Client>,
    conn_str: String,
}

#[pymethods]
impl Connection {
    /// Execute a SQL statement against the remote server.
    fn execute(&mut self, py: Python<'_>, sql: &str) -> PyResult<PyObject> {
        let client = self
            .client
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("connection is closed"))?;

        // The server uses the simple-query protocol end-to-end and
        // emits `CommandComplete` tags like `"OK n"`, `"SELECT n"`, or
        // plain DDL messages. `simple_query` collects every message
        // from every statement in the batch so we can translate it
        // back into the embedded-mode result shape without caring
        // about the tag text.
        let messages = client
            .simple_query(sql)
            .map_err(|e| PyRuntimeError::new_err(format!("remote execute failed: {}", e)))?;

        let mut rows: Vec<Vec<(String, String)>> = Vec::new();
        let mut column_names: Vec<String> = Vec::new();
        let mut row_count: Option<u64> = None;
        let mut command_tag: Option<String> = None;

        for msg in messages {
            match msg {
                SimpleQueryMessage::RowDescription(cols) => {
                    column_names = cols.iter().map(|c| c.name().to_string()).collect();
                }
                SimpleQueryMessage::Row(row) => {
                    let mut pairs: Vec<(String, String)> =
                        Vec::with_capacity(column_names.len());
                    for (idx, name) in column_names.iter().enumerate() {
                        let val = row
                            .get(idx)
                            .map(|s| s.to_string())
                            .unwrap_or_default();
                        pairs.push((name.clone(), val));
                    }
                    rows.push(pairs);
                }
                SimpleQueryMessage::CommandComplete(n) => {
                    row_count = Some(n);
                }
                other => {
                    command_tag = Some(format!("{:?}", other));
                }
            }
        }

        if !rows.is_empty() || !column_names.is_empty() {
            // Even a zero-row result of a SELECT returns a RowDescription.
            let py_rows = PyList::empty_bound(py);
            for row in &rows {
                let dict = PyDict::new_bound(py);
                for (name, val) in row {
                    dict.set_item(name, val)?;
                }
                py_rows.append(dict)?;
            }
            return Ok(py_rows.into());
        }

        if let Some(n) = row_count {
            return Ok(n.into_pyobject(py)?.into_any().unbind());
        }

        let msg = command_tag.unwrap_or_else(|| "OK".to_string());
        Ok(msg.into_pyobject(py)?.into_any().unbind())
    }

    /// Close the connection. Further calls to `execute` raise.
    fn close(&mut self) {
        self.client.take();
    }

    /// Is the connection still open?
    #[getter]
    fn is_open(&self) -> bool {
        self.client.is_some()
    }

    fn __repr__(&self) -> String {
        let open = if self.client.is_some() { "open" } else { "closed" };
        format!("Connection(dsn='{}', state={})", self.conn_str, open)
    }
}

/// Connect to a remote galaxdb-server via the PostgreSQL wire
/// protocol.
///
/// `connstring` is a libpq-style DSN, e.g.
/// `"host=127.0.0.1 port=5433 user=galaxdb dbname=galaxdb sslmode=disable"`.
/// Plain-text connections are required because the GalaxDB wire server
/// currently reports `N` (no SSL) on the SSLRequest handshake.
#[pyfunction]
fn connect(connstring: &str) -> PyResult<Connection> {
    let client = Client::connect(connstring, NoTls).map_err(|e| {
        PyRuntimeError::new_err(format!("failed to connect to '{}': {}", connstring, e))
    })?;
    Ok(Connection {
        client: Some(client),
        conn_str: connstring.to_string(),
    })
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn query_result_to_python(py: Python<'_>, result: RustQueryResult) -> PyResult<PyObject> {
    match result {
        RustQueryResult::Rows(rows) => {
            let py_rows = PyList::empty_bound(py);
            for row in &rows {
                let dict = PyDict::new_bound(py);
                for (key, value) in &row.values {
                    dict.set_item(key, value)?;
                }
                py_rows.append(dict)?;
            }
            Ok(py_rows.into())
        }
        RustQueryResult::RowCount(n) => Ok(n.into_pyobject(py)?.into_any().unbind()),
        RustQueryResult::Ok(msg) => Ok(msg.into_pyobject(py)?.into_any().unbind()),
    }
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

#[pymodule]
fn galaxdb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Database>()?;
    m.add_class::<Connection>()?;
    m.add_function(wrap_pyfunction!(connect, m)?)?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}
