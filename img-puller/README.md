# UdeSa Image Puller

Downloads product images from tagged CSV files and records their local paths.

## Requirements

- Python 3.10+
- `requests`, `Pillow` and `anthropic` libraries

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

## Generating titles and descriptions with Claude

With `--describe`, each product image is sent to Claude together with its tags, and two columns are added to the CSV: `title` and `description`.

```bash
export ANTHROPIC_API_KEY=your_key_here

# Try it on a few rows first, without downloading the images again
python main.py data/tops_tags.csv --skip-download --describe --limit 20

# Full run
python main.py data/tops_tags.csv --skip-download --describe
```

| Flag | Description |
| --- | --- |
| `--describe` | Generate `title` and `description` with Claude |
| `--skip-download` | Do not download the images again (keeps `relative_path`) |
| `--limit N` | Describe at most N rows |
| `--model` | Claude model (default: `claude-opus-5-5`) |
| `--workers` | Parallel requests to Claude (default: 8) |

Only rows without a `title` are processed, and the CSV is saved every 100 rows, so if a run is interrupted (or some rows fail) you can run the same command again and it will continue where it left off.

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

## Authors

Jorge flores - jfflores90@gmail.com

Hernán Marano - herchugm@gmail.com

Nicolás Velázquez - 