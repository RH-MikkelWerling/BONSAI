from typing import Optional, Dict
import polars as pl


class EHRTokenizer:
    def __init__(
        self,
        vocabulary=None,
        cutoffs: Optional[Dict[str, int]] = None,
        sep_tokens: bool = True,
        vocabulary_cutoff_abspos: Optional[float] = None,
    ):
        self.hot_vocab = vocabulary is None
        if vocabulary is None:
            vocabulary = {
                "[PAD]": 0,
                "[CLS]": 1,
                "[SEP]": 2,
                "[UNK]": 3,
                "[MASK]": 4,
            }
        self.vocabulary = vocabulary

        self.cutoffs = cutoffs
        self.sep_tokens = sep_tokens
        self.vocabulary_cutoff_abspos = vocabulary_cutoff_abspos

    def __call__(self, features: pl.DataFrame) -> pl.DataFrame:
        """
        Features must retain the canonical MEDS order produced by ehr2meds.
        """
        # Apply cutoffs if needed before updating vocabulary
        if self.cutoffs is not None:
            features = features.with_columns(self.limit_code_length(pl.col("code")))

        # Update vocabulary if vocabulary is `hot`
        if self.hot_vocab:
            vocabulary_features = features
            if self.vocabulary_cutoff_abspos is not None:
                vocabulary_features = features.filter(
                    pl.col("abspos") < self.vocabulary_cutoff_abspos
                )
            self.update_vocabulary(vocabulary_features["code"])

        if self.sep_tokens:
            features = self.add_sep_tokens(features)

        # Tokenize
        features = features.with_columns(self.tokenize(pl.col("code")))

        return features

    def update_vocabulary(self, codes: pl.Series) -> None:
        """Update self.vocabulary from unique codes"""
        # Get unique codes
        unique_codes = codes.unique()

        # Add new codes
        new_codes = set(unique_codes) - set(self.vocabulary)
        if new_codes:
            start_idx = max(self.vocabulary.values()) + 1
            new_indices = range(start_idx, start_idx + len(new_codes))
            self.vocabulary.update(dict(zip(new_codes, new_indices)))

    def add_sep_tokens(self, df: pl.DataFrame) -> pl.DataFrame:
        """Insert ``[SEP]`` after each non-final segment without re-sorting.

        ehr2meds owns the canonical event order. Each input row therefore emits
        either itself or ``[event, SEP]``; exploding those small lists preserves
        source-row order and the position of simultaneous events.
        """
        df = df.with_row_index("_token_order").with_columns(
            _insert_sep=(
                (pl.col("segment") != pl.col("segment").shift(-1))
                & (pl.col("subject_id") == pl.col("subject_id").shift(-1))
            )
        )
        df = df.with_columns(
            pl.int_ranges(
                pl.lit(0),
                pl.lit(1) + pl.col("_insert_sep").cast(pl.Int64),
            ).alias("_sep_offset")
        ).explode("_sep_offset")
        sep_updates = [
            pl.when(pl.col("_sep_offset") == 1)
            .then(pl.lit("[SEP]"))
            .otherwise(pl.col("code"))
            .alias("code"),
            (pl.col("_token_order") * 2 + pl.col("_sep_offset"))
            .cast(pl.Int64)
            .alias("row_idx"),
        ]
        for column in ("value_bin", "value_normalized"):
            if column in df.columns:
                sep_updates.append(
                    pl.when(pl.col("_sep_offset") == 1)
                    .then(pl.lit(None).cast(df.schema[column]))
                    .otherwise(pl.col(column))
                    .alias(column)
                )
        if "value_present" in df.columns:
            sep_updates.append(
                pl.when(pl.col("_sep_offset") == 1)
                .then(pl.lit(False))
                .otherwise(pl.col("value_present"))
                .alias("value_present")
            )
        if "numeric_value" in df.columns:
            sep_updates.append(
                pl.when(pl.col("_sep_offset") == 1)
                .then(pl.lit(None).cast(df.schema["numeric_value"]))
                .otherwise(pl.col("numeric_value"))
                .alias("numeric_value")
            )
        return df.with_columns(sep_updates).drop(
            "_token_order", "_insert_sep", "_sep_offset"
        )

    def tokenize(self, codes: pl.Expr) -> pl.Expr:
        """Map self.vocabulary onto codes, mapping unknown codes to [UNK] token"""
        return codes.replace_strict(self.vocabulary, default=self.vocabulary["[UNK]"])

    def limit_code_length(self, codes: pl.Expr) -> pl.Expr:
        """Limit code lengths using a {prefix: length} self.cutoff dict.
        Example:
            With cutoffs={'D': 4}, 'D123456' becomes 'D1234'
        """
        for prefix, length in self.cutoffs.items():
            codes = (
                pl.when(codes.str.starts_with(prefix))
                .then(codes.str.slice(0, length))
                .otherwise(codes)
            )

        return codes

    def freeze_vocabulary(self) -> None:
        self.hot_vocab = False

    def hot_vocabulary(self) -> None:
        self.hot_vocab = True
