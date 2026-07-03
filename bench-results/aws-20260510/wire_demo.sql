-- Hitting the GalaxDB wire server from psql over an SSH tunnel.
-- Every response below comes from target/release/galaxdb-server running
-- on c6id.4xlarge (<redacted>), which routes through
-- galaxdb-embedded::Database → galaxdb-sql::executor::execute_legacy
-- against a real galaxdb-storage Engine.

CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price FLOAT);

INSERT INTO products (id, name, price) VALUES (1, 'espresso', 3.50);
INSERT INTO products (id, name, price) VALUES (2, 'latte', 4.25);
INSERT INTO products (id, name, price) VALUES (3, 'mocha', 4.75);

-- Plain read — should return 3 rows.
SELECT id, name, price FROM products;

-- Filter by price — this is the path the embedded probe showed as broken.
SELECT id, name FROM products WHERE price > 4.0;

-- Point lookup by PK.
SELECT id, name, price FROM products WHERE id = 2;

-- Update one row (should affect exactly 1).
UPDATE products SET price = 5.00 WHERE id = 3;
SELECT id, name, price FROM products WHERE id = 3;

-- Delete one row (should affect exactly 1).
DELETE FROM products WHERE id = 1;
SELECT id, name, price FROM products;

-- Delete non-existent (should affect 0).
DELETE FROM products WHERE id = 99;
