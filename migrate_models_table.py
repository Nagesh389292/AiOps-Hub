import sqlite3

DB_PATHS = [
    r"C:\Users\TA23008\Desktop\Project\aiops_hub.db",
    r"C:\Users\TA23008\Desktop\Project\Project\aiops-hub\aiops_hub.db",
]

MIGRATIONS = [
    ("display_name", "ALTER TABLE models ADD COLUMN display_name VARCHAR(128) DEFAULT ''"),
    ("model_id",     "ALTER TABLE models ADD COLUMN model_id VARCHAR(256) DEFAULT ''"),
    ("base_url",     "ALTER TABLE models ADD COLUMN base_url VARCHAR(512) DEFAULT ''"),
    ("api_key",      "ALTER TABLE models ADD COLUMN api_key VARCHAR(512) DEFAULT ''"),
    ("is_custom",    "ALTER TABLE models ADD COLUMN is_custom INTEGER DEFAULT 0"),
]

for db_path in DB_PATHS:
    print(f"\nMigrating: {db_path}")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(models)")
    cols = [r[1] for r in cur.fetchall()]
    print("  Current columns:", cols)
    for col, sql in MIGRATIONS:
        if col not in cols:
            cur.execute(sql)
            print(f"  Added: {col}")
        else:
            print(f"  Already exists: {col}")
    conn.commit()
    conn.close()
    print("  Done.")

print("\nAll migrations complete.")
