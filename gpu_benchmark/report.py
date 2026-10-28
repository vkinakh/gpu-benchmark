"""Summarize completed cases without treating failures as measurements."""

import argparse
import csv
import json
from pathlib import Path
import statistics


def summarize(root):
    records = [json.loads(p.read_text()) for p in Path(root).glob("*/result.json")]
    groups = {}
    for row in records:
        if row["status"] == "success":
            key = (row["task"], row["model"], row["precision"], row["gpus"])
            groups.setdefault(key, []).append(row)
    rows = []
    for (task, model, precision, gpus), values in sorted(groups.items()):
        rate = statistics.median(v["samples_per_second"] for v in values)
        baseline = groups.get((task, model, "fp32", gpus))
        single = groups.get((task, model, precision, 1))
        row = dict(
            task=task,
            model=model,
            precision=precision,
            gpus=gpus,
            repetitions=len(values),
            samples_per_second=rate,
            peak_memory_bytes=max(v["peak_memory_bytes"] for v in values),
            speedup_vs_fp32=(
                rate / statistics.median(v["samples_per_second"] for v in baseline)
                if baseline
                else None
            ),
            scaling_efficiency=(
                rate
                / (gpus * statistics.median(v["samples_per_second"] for v in single))
                if single
                else None
            ),
        )
        rows.append(row)
    return rows


def write_report(root):
    rows = summarize(root)
    path = Path(root) / "summary.csv"
    if rows:
        with path.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory")
    a = p.parse_args()
    print(write_report(a.directory))


if __name__ == "__main__":
    main()
