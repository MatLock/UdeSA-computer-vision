# UdeSa Image Puller

Downloads product images from tagged CSV files and records their local paths.

## Requirements

- Python 3.8+
- `requests` library

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py <path-to-csv>
```

### Examples

```bash
python main.py data/tops_tags.csv
python main.py data/dresses_tags.csv
python main.py data/pants_tags.csv
```

## What it does

1. Reads the CSV file and detects the product type from the filename (`tops`, `dresses`, or `pants`).
2. Downloads each image from the `image_url` column into `images/<product_type>/`.
3. Adds a `relative_path` column to the CSV with the local path to each downloaded image.

## Output structure

```
.
├── data/
│   ├── tops_tags.csv
│   ├── dresses_tags.csv
│   └── pants_tags.csv
├── images/
│   ├── tops/
│   ├── dresses/
│   └── pants/
└── main.py
```