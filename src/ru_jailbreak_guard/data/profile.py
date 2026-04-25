"""Generate a self-contained HTML data quality report (no external deps)."""

from __future__ import annotations

import html
from pathlib import Path

import polars as pl


def _section(title: str, body_html: str) -> str:
    return f"<section><h2>{html.escape(title)}</h2>{body_html}</section>"


def _df_to_table(df: pl.DataFrame) -> str:
    cols = "".join(f"<th>{html.escape(c)}</th>" for c in df.columns)
    rows = []
    for row in df.iter_rows():
        cells = "".join(f"<td>{html.escape(str(v))}</td>" for v in row)
        rows.append(f"<tr>{cells}</tr>")
    return f"<table border='1'><tr>{cols}</tr>{''.join(rows)}</table>"


def write_profile_report(df: pl.DataFrame, output: Path) -> None:
    """Write an HTML report summarizing dataset composition + samples.

    Args:
        df: DataFrame conforming to CANONICAL_SCHEMA.
        output: Destination .html file (parent dir created if missing).
    """
    sections: list[str] = []
    sections.append(_section("Row count", f"<p>{df.height} rows total</p>"))

    if df.height > 0:
        class_counts = df.group_by("label").agg(pl.len().alias("count")).sort("label")
        sections.append(_section("Class balance (label)", _df_to_table(class_counts)))

        src_counts = (
            df.group_by("source").agg(pl.len().alias("count")).sort("count", descending=True)
        )
        sections.append(_section("Per-source counts", _df_to_table(src_counts)))

        df_with_len = df.with_columns(pl.col("text").str.len_chars().alias("text_chars"))
        length_stats = df_with_len.select(
            pl.col("text_chars").min().alias("min"),
            pl.col("text_chars").quantile(0.25).alias("q25"),
            pl.col("text_chars").median().alias("median"),
            pl.col("text_chars").quantile(0.75).alias("q75"),
            pl.col("text_chars").max().alias("max"),
        )
        sections.append(_section("Text length (chars)", _df_to_table(length_stats)))

        samples = df.group_by("label").head(5).select(["label", "source", "text"])
        sections.append(_section("Sample rows (5 per label)", _df_to_table(samples)))

    body = "\n".join(sections)
    page = (
        "<!DOCTYPE html><html lang='ru'><head><meta charset='utf-8'>"
        "<title>ru-jailbreak-guard data profile</title></head>"
        f"<body><h1>Data profile</h1>{body}</body></html>"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")


def main() -> None:
    """CLI entry point for DVC stage."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    df = pl.read_parquet(args.input)
    write_profile_report(df, args.output)
    print(f"Wrote profile to {args.output}")


if __name__ == "__main__":
    main()
