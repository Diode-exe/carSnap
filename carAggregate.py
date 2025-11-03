#!/usr/bin/env python3

import argparse
import pathlib
import re
import datetime
import collections
import pandas as pd
import matplotlib.pyplot as plt

# Match filenames like:
#   2025-10-22 16-33-39.jpg
#   2025-10-22_16-33-39.png
DATE_RE = re.compile(r'^(\d{4}-\d{2}-\d{2})[ _](\d{2}-\d{2}-\d{2})')

IMG_EXTS = {
    ".jpg", ".jpeg", ".png", ".heic", ".bmp", ".gif",
    ".webp", ".tiff"
}


def parse_date_from_filename(name: str):
    m = DATE_RE.match(name)
    if not m:
        return None

    date_part = m.group(1)
    time_part = m.group(2).replace("-", ":")

    try:
        dt = datetime.datetime.strptime(
            f"{date_part} {time_part}",
            "%Y-%m-%d %H:%M:%S"
        )
        return dt.date()
    except ValueError:
        return None


def find_files(root: pathlib.Path, recursive: bool):
    it = root.rglob("*") if recursive else root.iterdir()

    files = []
    for f in it:
        if not f.is_file():
            continue

        ext = f.suffix.lower()

        # Accept image formats or no extension if pattern matches
        if ext in IMG_EXTS or ext == "":
            if DATE_RE.match(f.name):
                files.append(f)

    return files


def aggregate_counts(files):
    cnt = collections.Counter()

    for f in files:
        d = parse_date_from_filename(f.name)
        if d:
            cnt[d] += 1

    return cnt


def plot_counts(counts, out_path=None):
    if not counts:
        print("No matching files found.")
        return

    s = pd.Series(counts).sort_index()
    s.index = pd.to_datetime(s.index)

    plt.figure(figsize=(10, 4))
    plt.bar(s.index.strftime("%Y-%m-%d"), s.values)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Date")
    plt.ylabel("Cars per day")
    plt.title("Car Frequency by Day")
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path)
        print(f"Saved plot to: {out_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate pictures by date based on filename format."
    )
    parser.add_argument("folder", help="Folder to scan")
    parser.add_argument("-r", "--recursive", action="store_true",
                        help="Scan subdirectories too")
    parser.add_argument("-o", "--out", help="Save plot as PNG")

    args = parser.parse_args()

    root = pathlib.Path(args.folder)
    files = find_files(root, args.recursive)

    if not files:
        print("No files matched the naming pattern.")
        return

    counts = aggregate_counts(files)

    print("\nCounts:")
    for d, c in sorted(counts.items()):
        print(d, c)

    plot_counts(counts, args.out)


if __name__ == "__main__":
    main()
