
import boto3
import time
from botocore import UNSIGNED
from botocore.config import Config

BUCKET = "nasa-irsa-spherex"
REGION = "us-east-1"

PREFIXES = [
    "dr1/level2/",
    "qr3/level2/",
]

PAGE_SIZE = 1000

client = boto3.client(
    "s3",
    region_name=REGION,
    config=Config(
        signature_version=UNSIGNED,
        max_pool_connections=10
    )
)

grand_total = 0

for prefix in PREFIXES:

    print("\n" + "=" * 65, flush=True)
    print("ARCHIVE :", prefix, flush=True)
    print("=" * 65, flush=True)

    paginator = client.get_paginator("list_objects_v2")

    count = 0
    pages = 0
    total_size = 0
    start = time.time()

    for page in paginator.paginate(
        Bucket=BUCKET,
        Prefix=prefix,
        PaginationConfig={"PageSize": PAGE_SIZE}
    ):

        pages += 1

        for obj in page.get("Contents", []):

            key = obj["Key"]

            if key.lower().endswith((".fits", ".fits.gz")):
                count += 1
                total_size += obj["Size"]

        elapsed = time.time() - start

        print(
            f"\rPages : {pages} | "
            f"FITS : {count:,} | "
            f"Taille : {total_size / 1e9:.2f} Go | "
            f"Temps : {elapsed / 60:.1f} min",
            end="",
            flush=True
        )

    print("\n\nRÉSULTAT :", flush=True)
    print("Fichiers FITS :", count, flush=True)
    print("Taille totale :", round(total_size / 1e9, 2), "Go", flush=True)

    grand_total += count

print("\n" + "=" * 65)
print("TOTAL DR1 + QR3 :", f"{grand_total:,}", "fichiers FITS")
print("=" * 65)
