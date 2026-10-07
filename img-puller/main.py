import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image
from io import BytesIO

NON_TAG_COLUMNS = {"id", "image_url", "relative_path", "title", "description"}
SAVE_EVERY = 100


def detect_product_type(filepath: str) -> str:
    filename = Path(filepath).stem.lower()
    for product_type in ("shoes", "tops", "pants"):
        if product_type in filename:
            return product_type
    sys.exit(f"Error: could not detect product type from filename '{Path(filepath).name}'. "
             "Expected filename containing 'shoes', 'tops', or 'pants'.")


def download_image(url: str, dest: str) -> bool:
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        img = img.resize((224, 224))
        img.save(dest)
        return True
    except requests.RequestException as e:
        print(f"  Failed to download {url}: {e}")
        return False


def write_csv(filepath: str, fieldnames: list, rows: list):
    # Write to a temp file and swap, so an interrupted run never leaves a half-written CSV.
    tmp_path = filepath + ".tmp"
    with open(tmp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, filepath)


def download_images(rows: list, filepath: str, product_type: str):
    images_dir = os.path.join(os.path.dirname(filepath), os.pardir, "data/images", product_type)
    os.makedirs(images_dir, exist_ok=True)

    # Prepare download tasks
    tasks = []
    for row in rows:
        url = row["image_url"]
        ext = os.path.splitext(urlparse(url).path)[1] or ".jpg"
        filename = f"{row['id']}{ext}"
        dest = os.path.join(images_dir, filename)
        relative_path = os.path.relpath(dest, os.path.dirname(filepath)).replace(os.sep, "/")
        tasks.append((row, url, dest, relative_path))

    # Download images in parallel
    with ThreadPoolExecutor(max_workers=25) as executor:
        futures = {
            executor.submit(download_image, url, dest): (row, relative_path)
            for row, url, dest, relative_path in tasks
        }
        for future in as_completed(futures):
            row, relative_path = futures[future]
            print(f"Downloading {row['id']}...")
            if future.result():
                row["relative_path"] = relative_path
            else:
                row["relative_path"] = ""


def describe_rows(rows: list, filepath: str, fieldnames: list, product_type: str, model: str,
                  workers: int, limit: int):
    from describer import describe_product

    # Only rows without a title, so an interrupted run picks up where it left off.
    pending = [row for row in rows if not row.get("title")]
    if limit:
        pending = pending[:limit]
    print(f"Generating title/description for {len(pending)} rows with {model}...")

    done = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                describe_product,
                row["image_url"],
                product_type,
                {k: v for k, v in row.items() if k not in NON_TAG_COLUMNS},
                model,
            ): row
            for row in pending
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                result = future.result()
                row["title"] = result["title"]
                row["description"] = result["description"]
                done += 1
            except Exception as e:
                print(f"  Failed to describe {row['id']}: {e}")
                failed += 1
            if (done + failed) % SAVE_EVERY == 0:
                write_csv(filepath, fieldnames, rows)
                print(f"  {done + failed}/{len(pending)} processed ({failed} failed), progress saved.")

    print(f"Descriptions done: {done} ok, {failed} failed.")


def main():
    parser = argparse.ArgumentParser(description="Download images from a tagged CSV and record their paths.")
    parser.add_argument("filepath", help="Path to the CSV file (e.g. data/tops_tags.csv)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Do not download images (keeps the existing relative_path values)")
    parser.add_argument("--describe", action="store_true",
                        help="Generate title and description for each product with Claude")
    parser.add_argument("--model", default=None, help="Claude model to use (default: claude-opus-5-5)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel requests to Claude (default: 8)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Describe at most N rows (useful to test on a sample first)")
    args = parser.parse_args()

    filepath = os.path.abspath(args.filepath)
    if not os.path.isfile(filepath):
        sys.exit(f"Error: file not found: {filepath}")

    product_type = detect_product_type(filepath)

    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        if "image_url" not in reader.fieldnames:
            sys.exit("Error: CSV is missing 'image_url' column.")
        rows = list(reader)
        # dict.fromkeys drops duplicated columns left by earlier runs
        fieldnames = list(dict.fromkeys(reader.fieldnames))

    if "relative_path" not in fieldnames:
        fieldnames.append("relative_path")
    if args.describe:
        for column in ("title", "description"):
            if column not in fieldnames:
                fieldnames.append(column)

    if not args.skip_download:
        download_images(rows, filepath, product_type)
        write_csv(filepath, fieldnames, rows)
        print(f"Images done. {len(rows)} rows processed. Images saved to images/{product_type}/")

    if args.describe:
        from describer import DEFAULT_MODEL
        try:
            describe_rows(rows, filepath, fieldnames, product_type, args.model or DEFAULT_MODEL,
                          args.workers, args.limit)
        finally:
            write_csv(filepath, fieldnames, rows)


if __name__ == "__main__":
    main()
