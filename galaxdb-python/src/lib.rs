//! GalaxDB Python client — PyO3 bindings for embedded mode.
//!
//! Usage from Python:
//! ```python
//! import galaxdb
//!
//! # Embedded mode
//! db = galaxdb.Database("/path/to/data")
//! db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")
//! db.execute("INSERT INTO users (id, name) VALUES (1, 'alice')")
//! rows = db.execute("SELECT * FROM users")
//! for row in rows:
//!     print(row)
//! ```

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use galaxdb_embedded::{Database as RustDatabase, QueryResult as RustQueryResult};

/// A GalaxDB database instance (embedded mode).
#[pyclass]
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
            RustQueryResult::RowCount(n) => Ok(n.to_object(py)),
            RustQueryResult::Ok(msg) => Ok(msg.to_object(py)),
        }
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

/// Connect to a remote GalaxDB server via PostgreSQL wire protocol.
///
/// Remote mode connects via the PostgreSQL wire protocol. For embedded
/// mode (no server), use galaxdb.Database(path) instead.
#[pyfunction]
fn connect(_connstring: &str) -> PyResult<PyObject> {
    Err(PyRuntimeError::new_err(
        "remote mode not yet available — use galaxdb.Database(path) for embedded mode",
    ))
}

/// Python module definition.
#[pymodule]
fn galaxdb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Database>()?;
    m.add_function(wrap_pyfunction!(connect, m)?)?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}
