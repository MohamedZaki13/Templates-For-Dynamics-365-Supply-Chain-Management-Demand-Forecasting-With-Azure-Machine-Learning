"""Partition the D365 input CSV into series-preserving buckets for the parallel step.

The parallel step appends each mini-batch's output to parallel_run_step.txt with a
single storage append (max 4 MiB per append, max 50,000 appends per file). Feeding
the whole input as one mini-batch therefore fails for production-sized tenants
(RequestBodyTooLarge) and leaves all but one worker idle.

This step writes the rows to <output>/<PartitionBucket>/data.csv, keeping every
GranularityAttributeKey (forecast series) whole inside one bucket, and emits an
MLTable whose partition format exposes PartitionBucket. The parallel step uses
partition_keys=["PartitionBucket"], so each bucket becomes one mini-batch.
"""
import argparse
import heapq
import math
import os

import pandas as pd

KEY_COLUMN = "GranularityAttributeKey"
ATTRIBUTE_COLUMN = "GranularityAttribute"
HORIZON_COLUMN = "HORIZON"
BUCKET_COLUMN = "PartitionBucket"

# Conservative size of one forecast.r output line, excluding the attribute name:
# "<attr>",<datekey>,<qty>,<sigma>,<error%>,<model>
_OUTPUT_LINE_OVERHEAD_BYTES = 80

_MLTABLE = """type: mltable
paths:
- pattern: ./**/*.csv
transformations:
- read_delimited:
    delimiter: ','
    encoding: utf8
    header: all_files_same_headers
    infer_column_types: false
    include_path_column: true
- extract_columns_from_partition_format:
    partition_format: '{%s}/{file_name}.{extension}'
- drop_columns:
  - Path
  - file_name
  - extension
""" % BUCKET_COLUMN


def estimate_output_bytes(df):
    """Estimate forecast.r output size per series: HORIZON lines of roughly
    len(GranularityAttribute) + overhead bytes. Returns {key: bytes} in input order."""
    columns = [c for c in (ATTRIBUTE_COLUMN, HORIZON_COLUMN) if c in df.columns]
    # first() skips missing values, so this picks each series' parameter row values.
    first = df[[KEY_COLUMN] + columns].replace("", pd.NA).groupby(KEY_COLUMN, sort=False).first()
    if HORIZON_COLUMN in first:
        lines = pd.to_numeric(first[HORIZON_COLUMN], errors="coerce").fillna(1).clip(lower=1).astype(int)
    else:
        lines = pd.Series(1, index=first.index)
    if ATTRIBUTE_COLUMN in first:
        attribute_bytes = first[ATTRIBUTE_COLUMN].fillna("").astype(str).str.encode("utf-8").str.len()
    else:
        attribute_bytes = pd.Series(0, index=first.index)
    sizes = lines * (attribute_bytes + _OUTPUT_LINE_OVERHEAD_BYTES)
    return {key: int(size) for key, size in sizes.items()}


def assign_buckets(df, min_partitions, max_series_per_partition, max_partition_output_bytes, max_partitions):
    """Return ({key: bucket}, bucket_count, per-bucket estimated output bytes)."""
    sizes = estimate_output_bytes(df)
    series_count = len(sizes)
    total_bytes = sum(sizes.values())

    bucket_count = max(
        min_partitions,
        math.ceil(total_bytes / max_partition_output_bytes),
        math.ceil(series_count / max_series_per_partition),
    )
    bucket_count = max(1, min(bucket_count, max_partitions, series_count))

    # Largest-first into the least-loaded bucket keeps buckets balanced; ties are
    # broken by first appearance / bucket index so the layout is deterministic.
    keys = list(sizes.keys())
    order = sorted(range(series_count), key=lambda i: (-sizes[keys[i]], i))
    heap = [(0, bucket) for bucket in range(bucket_count)]
    loads = [0] * bucket_count
    assignment = {}
    for index in order:
        load, bucket = heapq.heappop(heap)
        key = keys[index]
        assignment[key] = bucket
        loads[bucket] = load + sizes[key]
        heapq.heappush(heap, (loads[bucket], bucket))
    return assignment, bucket_count, loads


def partition(input_path, output_path, min_partitions, max_series_per_partition, max_partition_output_bytes, max_partitions):
    # Read every value as text, exactly as D365 wrote it (v1 read all columns as
    # strings too); empty cells stay empty so forecast.r sees the same input.
    df = pd.read_csv(input_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if KEY_COLUMN not in df.columns:
        raise ValueError(f"Input is missing the required column '{KEY_COLUMN}'.")
    if BUCKET_COLUMN in df.columns:
        raise ValueError(f"Input must not contain the reserved column '{BUCKET_COLUMN}'.")
    if len(df) == 0:
        raise ValueError("Input contains no rows to forecast.")
    os.makedirs(output_path, exist_ok=True)

    assignment, bucket_count, loads = assign_buckets(
        df, min_partitions, max_series_per_partition, max_partition_output_bytes, max_partitions
    )
    width = len(str(bucket_count - 1))
    buckets = df[KEY_COLUMN].map(assignment)
    # groupby keeps the original row order inside each bucket, so every series keeps
    # its parameter row first (forecast.r reads parameters from the first row).
    for bucket, rows in df.groupby(buckets, sort=True):
        folder = os.path.join(output_path, str(bucket).zfill(width))
        os.makedirs(folder, exist_ok=True)
        rows.to_csv(os.path.join(folder, "data.csv"), index=False, encoding="utf-8")

    with open(os.path.join(output_path, "MLTable"), "w", encoding="utf-8") as handle:
        handle.write(_MLTABLE)

    print(
        f"Partitioned {len(df)} rows / {len(assignment)} series into {bucket_count} buckets; "
        f"largest estimated bucket output {max(loads)} bytes."
    )
    return bucket_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", required=True, help="Path to the input CSV.")
    parser.add_argument("--output_data", required=True, help="Output folder for the partitioned MLTable.")
    parser.add_argument("--min_partitions", type=int, default=1)
    parser.add_argument("--max_series_per_partition", type=int, default=100)
    parser.add_argument("--max_partition_output_bytes", type=int, default=1024 * 1024)
    parser.add_argument("--max_partitions", type=int, default=10000)
    args = parser.parse_args()
    partition(
        args.input_data,
        args.output_data,
        args.min_partitions,
        args.max_series_per_partition,
        args.max_partition_output_bytes,
        args.max_partitions,
    )


if __name__ == "__main__":
    main()
