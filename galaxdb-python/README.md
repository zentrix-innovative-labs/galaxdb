# galaxdb

GalaxDB Python client — AI-native embedded database.

## Installation

```bash
pip install galaxdb
```

## Usage

```python
import galaxdb

# Embedded mode — no server required
db = galaxdb.Database("/tmp/mydb")

# Create tables
db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")

# Insert data
db.execute("INSERT INTO users (id, name) VALUES (1, 'alice')")

# Query
rows = db.execute("SELECT * FROM users")
for row in rows:
    print(row)

# Check table exists
print(db.table_exists("users"))  # True
print(db.table_count)            # 1
```
