//! AuroraSQL AST types — extensions beyond standard SQL.

/// A parsed AuroraSQL statement.
#[derive(Debug, Clone, PartialEq)]
pub enum AuroraStatement {
    /// Standard SQL statement (delegated to sqlparser).
    Standard(Box<sqlparser::ast::Statement>),
    /// CREATE TABLE with optional embedding columns.
    CreateTable(CreateTableStmt),
    /// SEMANTIC_MATCH query (parsed from SELECT).
    SemanticMatch(SemanticMatchExpr),
    /// SELECT ... AT VERSION with optional consistency mode.
    AtVersion(AtVersionExpr),
    /// CREATE VERSION TAG.
    CreateVersionTag(CreateVersionTagStmt),
    /// BULK INSERT.
    BulkInsert(BulkInsertStmt),
    /// SHOW EMBEDDING HEALTH.
    ShowEmbeddingHealth { table: Option<String> },
    /// BACKUP TO '/path'.
    BackupTo { path: String },
    /// RESTORE FROM '/path'.
    RestoreFrom { path: String },
    /// ANALYZE table_name.
    Analyze { table: String },
    /// CREATE ROLE name [PASSWORD '...'] [SUPERUSER].
    CreateRole(CreateRoleStmt),
    /// DROP ROLE name.
    DropRole { name: String, if_exists: bool },
    /// ALTER ROLE name PASSWORD '...'.
    AlterRolePassword { name: String, password: String },
    /// GRANT priv ON table TO role.
    Grant(GrantStmt),
    /// REVOKE priv ON table FROM role.
    Revoke(GrantStmt),
}

/// A privilege that can be granted on a table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Privilege {
    Select,
    Insert,
    Update,
    Delete,
}

/// CREATE ROLE statement.
#[derive(Debug, Clone, PartialEq)]
pub struct CreateRoleStmt {
    pub name: String,
    /// Plaintext password, if `PASSWORD '...'` was supplied. Used only to
    /// build the SCRAM verifier at execution time, then dropped — never
    /// stored.
    pub password: Option<String>,
    pub is_superuser: bool,
}

/// GRANT / REVOKE statement (same shape for both).
#[derive(Debug, Clone, PartialEq)]
pub struct GrantStmt {
    pub privilege: Privilege,
    pub table: String,
    pub role: String,
}

/// Column definition with optional embedding annotation.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnDef {
    pub name: String,
    pub data_type: String,
    pub nullable: bool,
    pub primary_key: bool,
    pub embedding: Option<EmbeddingDef>,
}

/// Embedding column annotation: EMBEDDING MODEL 'name' DIM n.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingDef {
    pub model_name: String,
    pub dimensions: Option<u32>,
}

/// CREATE TABLE with embedding columns.
#[derive(Debug, Clone, PartialEq)]
pub struct CreateTableStmt {
    pub table_name: String,
    pub columns: Vec<ColumnDef>,
    pub if_not_exists: bool,
}

/// SEMANTIC_MATCH(col, 'query', threshold).
#[derive(Debug, Clone, PartialEq)]
pub struct SemanticMatchExpr {
    pub column: String,
    pub query: String,
    pub threshold: f64,
}

/// AT VERSION timestamp_or_tag with optional consistency mode.
#[derive(Debug, Clone, PartialEq)]
pub struct AtVersionExpr {
    pub version: VersionRef,
    pub consistency: Option<ConsistencyMode>,
}

/// Version reference: timestamp or named tag.
#[derive(Debug, Clone, PartialEq)]
pub enum VersionRef {
    Timestamp(u64),
    Tag(String),
}

/// Consistency mode for AT VERSION + SEMANTIC_MATCH.
#[derive(Debug, Clone, PartialEq)]
pub enum ConsistencyMode {
    RowSnapshot,
    SemanticFresh,
}

/// CREATE VERSION TAG statement.
#[derive(Debug, Clone, PartialEq)]
pub struct CreateVersionTagStmt {
    pub name: String,
    pub for_training: bool,
    pub training_opts: Option<TrainingOpts>,
}

/// Training options for version tags.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainingOpts {
    pub precision: Option<TrainingPrecision>,
    pub seed: Option<u64>,
}

/// Training precision options.
#[derive(Debug, Clone, PartialEq)]
pub enum TrainingPrecision {
    Sq8,
    Rabitq,
    Float32,
}

/// BULK INSERT statement.
#[derive(Debug, Clone, PartialEq)]
pub struct BulkInsertStmt {
    pub table: String,
    pub columns: Vec<String>,
    pub values: Vec<Vec<String>>,
}
