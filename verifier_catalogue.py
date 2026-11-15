import sqlite3
from pathlib import Path

db = Path(__file__).resolve().parent / "spherex_catalog_test.sqlite"

conn = sqlite3.connect(db)

total = conn.execute(
    "SELECT COUNT(*) FROM observations"
).fetchone()[0]

ok = conn.execute(
    "SELECT COUNT(*) FROM observations WHERE status = 'OK'"
).fetchone()[0]

errors = conn.execute(
    "SELECT COUNT(*) FROM observations WHERE status != 'OK'"
).fetchone()[0]

print("================================")
print("VERIFICATION DU CATALOGUE SPHEREX")
print("================================")
print("Total :", total)
print("Fichiers OK :", ok)
print("Erreurs :", errors)

conn.close()