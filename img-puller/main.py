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


def main():
    parser = argparse.ArgumentParser(description="Download images from a tagged CSV and record their paths.")
    parser.add_argument("filepath", help="Path to the CSV file (e.g. data/tops_tags.csv)")
    args = parser.parse_args()

    filepath = os.path.abspath(args.filepath)
    if not os.path.isfile(filepath):
        sys.exit(f"Error: file not found: {filepath}")

    product_type = detect_product_type(filepath)
    images_dir = os.path.join(os.path.dirname(filepath), os.pardir, "data/images", product_type)
    os.makedirs(images_dir, exist_ok=True)

    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        if "image_url" not in reader.fieldnames:
            sys.exit("Error: CSV is missing 'image_url' column.")
        rows = list(reader)
        fieldnames = list(reader.fieldnames) + ["relative_path"]

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

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. {len(rows)} rows processed. Images saved to images/{product_type}/")


if __name__ == "__main__":
    main()