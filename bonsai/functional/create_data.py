import logging
from pathlib import Path
from typing import Optional
import polars as pl
from bonsai.functional.features import create_features

OPTIONAL_TOKEN_COLUMNS = (
    "row_idx",
    "row_id",
    "value_normalized",
    "value_bin",
    "value_present",
)
ORDER_COLUMNS = ("row_idx", "row_id")


def drop_duplicates(df: pl.DataFrame) -> pl.DataFrame:
    pre = len(df)
    subset = ["subject_id", "code", "time"]
    subset.extend(column for column in OPTIONAL_TOKEN_COLUMNS if column in df.columns)
    df = df.unique(subset=subset, maintain_order=True)
    if pre != len(df):
        logging.info(
            f"Dropped {pre - len(df)} duplicate rows based on {subset}"
        )
    return df


def process_split(
    split,
    path_input_dir: Path,
    path_output_dir: Path,
    tokenizer,
    exclude_regex: Optional[str] = None,
):
    path_output_dir_split = path_output_dir / split
    path_output_dir_split.mkdir(parents=True, exist_ok=True)

    data_counts = {
        "loaded": 0,
        "after_duplicates": 0,
        "after_exclusion": 0,
        "after_features": 0,
    }
    ids = []
    shards = [shard for shard in (path_input_dir / split).glob("*.parquet")]
    logging.info(f"Found {len(shards)} shards to process in {split}")
    for shard_idx, shard in enumerate(shards, 1):
        logging.info(f"Processing shard {shard_idx}/{len(shards)}: {shard}")

        # Load
        shard_df = pl.read_parquet(shard)
        data_counts["loaded"] += len(shard_df)

        # Drop duplicates
        shard_df = drop_duplicates(shard_df)
        data_counts["after_duplicates"] += len(shard_df)

        # Optional: Exclude codes based on regex
        if exclude_regex is not None:
            shard_df = shard_df.filter(~pl.col("code").str.contains(exclude_regex))
        data_counts["after_exclusion"] += len(shard_df)

        # Create features
        features = create_features(shard_df)
        data_counts["after_features"] += len(features)

        # Tokenize
        tokenized = tokenizer(features)
        sort_columns = ["subject_id", "abspos"]
        sort_columns.extend(
            column for column in ORDER_COLUMNS if column in tokenized.columns
        )
        tokenized = tokenized.sort(sort_columns)

        if "value_present" not in tokenized.columns and (
            "value_normalized" in tokenized.columns or "value_bin" in tokenized.columns
        ):
            present_exprs = []
            if "value_normalized" in tokenized.columns:
                present_exprs.append(pl.col("value_normalized").is_not_null())
            if "value_bin" in tokenized.columns:
                present_exprs.append(pl.col("value_bin").is_not_null())
            value_present = present_exprs[0]
            for expr in present_exprs[1:]:
                value_present = value_present | expr
            tokenized = tokenized.with_columns(value_present=value_present)

        # Cast to correct dtypes
        columns = [
            pl.col("subject_id").cast(pl.Int64),
            pl.col("code").cast(pl.Int64),
            pl.col("age").cast(pl.Float32),
            pl.col("abspos").cast(pl.Float32),
            pl.col("segment").cast(pl.Int32),
        ]
        if "row_idx" in tokenized.columns:
            columns.append(pl.col("row_idx").cast(pl.Int64))
        if "row_id" in tokenized.columns:
            columns.append(pl.col("row_id").cast(pl.Int64))
        if "value_normalized" in tokenized.columns:
            columns.append(
                pl.col("value_normalized").fill_null(0.0).cast(pl.Float32)
            )
        if "value_bin" in tokenized.columns:
            columns.append(pl.col("value_bin").fill_null(0).cast(pl.Int64))
        if "value_present" in tokenized.columns:
            columns.append(pl.col("value_present").fill_null(False).cast(pl.Boolean))
        tokenized = tokenized.select(*columns)
        tokenized.write_parquet(path_output_dir_split / f"{shard.stem}.parquet")

        ids.extend(tokenized["subject_id"].unique())

    logging.info(
        f"Finished processing {split} \n"
        f"Total rows loaded: {data_counts['loaded']} \n"
        f"Total rows after dropping duplicates: {data_counts['after_duplicates']} \n"
        f"Total rows after exclusion: {data_counts['after_exclusion']} \n"
        f"Total rows after feature creation: {data_counts['after_features']}"
    )

    return ids
