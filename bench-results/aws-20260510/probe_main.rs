use galaxdb_embedded::{Database, QueryResult};

fn run(db: &mut Database, sql: &str) {
    println!("SQL> {}", sql);
    match db.execute(sql) {
        Ok(QueryResult::Ok(s)) => println!("  OK: {}", s),
        Ok(QueryResult::RowCount(n)) => println!("  rows affected: {}", n),
        Ok(QueryResult::Rows(rows)) => {
            println!("  {} row(s)", rows.len());
            for r in rows.iter().take(10) {
                let pairs: Vec<String> = r
                    .values
                    .iter()
                    .map(|(c, v)| format!("{}={}", c, v))
                    .collect();
                println!("    {}", pairs.join(", "));
            }
            if rows.len() > 10 {
                println!("    ... ({} more)", rows.len() - 10);
            }
        }
        Err(e) => println!("  ERR: {}", e),
    }
}

fn main() {
    let path = "/mnt/nvme/galaxdb/probe_db";
    let _ = std::fs::remove_dir_all(path);
    let mut db = Database::open(path).expect("open db");
    println!("=== opened {} ===", path);

    run(&mut db, "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price FLOAT)");
    run(&mut db, "INSERT INTO products (id, name, price) VALUES (1, 'espresso', 3.50)");
    run(&mut db, "INSERT INTO products (id, name, price) VALUES (2, 'latte', 4.25)");
    run(&mut db, "INSERT INTO products (id, name, price) VALUES (3, 'mocha', 4.75)");
    run(&mut db, "SELECT id, name, price FROM products");
    run(&mut db, "SELECT id, name FROM products WHERE price > 4.0");
    run(&mut db, "UPDATE products SET price = 5.00 WHERE id = 3");
    run(&mut db, "SELECT id, name, price FROM products WHERE id = 3");
    run(&mut db, "DELETE FROM products WHERE id = 1");
    run(&mut db, "SELECT id, name, price FROM products");
    run(&mut db, "DELETE FROM products WHERE id = 99");
    println!(
        "=== done, table_count={}, total rows={} ===",
        db.table_count(),
        db.row_count()
    );
}
