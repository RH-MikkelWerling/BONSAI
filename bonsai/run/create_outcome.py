import logging
from pathlib import Path
import hydra
import polars as pl
from dotenv import load_dotenv
from omegaconf import DictConfig

from hydra.core.plugins import Plugins
from bonsai.paths import get_config_path
from bonsai.functional.outcomes import (
    apply_prospective_split,
    resolve_split_contract,
    get_subject_first_row_for_conditions,
    get_date_from_absolute_date,
    get_date_from_relative_date,
    get_date_from_exposure_date,
    fill_nans_with_sampled,
    summarize_outcome_splits,
    validate_split_integrity,
)
from bonsai.modules.hydra.plugins import DataCreationSearchpathPlugin

load_dotenv()
Plugins.instance().register(DataCreationSearchpathPlugin)


@hydra.main(
    config_path=get_config_path(),
    config_name="example_outcome1",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    input_dir = Path(cfg.paths.input_dir)
    save_path = Path(cfg.paths.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    exclude = cfg.outcome.exclude
    outcome = cfg.outcome.outcome
    index = cfg.outcome.index
    censor = cfg.outcome.censor

    logging.info(f"Starting create_outcome for `{save_path.stem}`")
    logging.info(f"Excluding subjects with {exclude}")
    logging.info(f"Outcome date assigned with {outcome}")
    logging.info(f"Index date assigned with {index}")
    logging.info(f"Censor date assigned with {censor}")

    all_outcomes = []
    for split in cfg.splits:
        shards = [shard for shard in (input_dir / split).glob("*.parquet")]
        for shard in shards:
            df = pl.read_parquet(shard, columns=["subject_id", "time", "code"])

            df = df.drop_nulls(["subject_id", "time", "code"])

            # Exclude subjects matching exclude.conditions
            if exclude is not None:
                exclude_df = get_subject_first_row_for_conditions(
                    df, exclude.conditions, exclude.dependence
                )
                logging.info(f"Excluding {len(exclude_df)} subjects")
                df = df.join(
                    exclude_df.select("subject_id"),
                    on="subject_id",
                    how="anti",
                )

            # Assign the outcomes matching outcome.conditions
            outcomes = get_subject_first_row_for_conditions(
                df, outcome.conditions, outcome.dependence
            )
            logging.info(f"Matched {len(outcomes)} subjects")
            outcomes = (
                df.select("subject_id")
                .unique()
                .join(outcomes, on="subject_id", how="left")
                .drop("code")
                .rename({"time": "outcome_date"})
            )
            assert len(outcomes) == df["subject_id"].n_unique()

            # Assign index dates
            if index.type == "absolute":
                outcomes = outcomes.with_columns(
                    index_date=pl.lit(
                        get_date_from_absolute_date(
                            absolute_date=index["absolute_date"]
                        )
                    )
                )
            elif index.type == "relative":
                outcomes = outcomes.with_columns(
                    index_date=get_date_from_relative_date(
                        relative_dates=pl.col("outcome_date"),
                        relative_hour_shift=index["relative_hour_shift"],
                    )
                )
            elif index.type == "exposure":
                outcomes = outcomes.with_columns(
                    index_date=get_date_from_exposure_date(
                        subjects=outcomes.select("subject_id"),
                        df=df,
                        dependence=index["dependence"],
                        conditions=index["conditions"],
                    )
                )
            else:
                raise ValueError(
                    f"got index.type={index.type}. This is either misconfigured or not yet supported"
                )

            outcomes = outcomes.with_columns(split=pl.lit(split))
            all_outcomes.append(outcomes)

    all_outcomes = pl.concat(all_outcomes) if all_outcomes else pl.DataFrame()

    if (index_dates := all_outcomes["index_date"]).is_null().any():
        logging.warning(
            f"Found {index_dates.is_null().sum()} NaN index dates -- Replacing them..."
        )
        all_outcomes = all_outcomes.with_columns(
            index_date=fill_nans_with_sampled(all_outcomes["index_date"])
        )

    all_outcomes = all_outcomes.with_columns(
        censor_date=get_date_from_relative_date(
            relative_dates=pl.col("index_date"),  # Censoring is based on index_date
            relative_hour_shift=censor[
                "relative_hour_shift"
            ],  # 0 sets index_date=censor_date
        )
    )

    prospective_split = cfg.get("prospective_split")
    if prospective_split:
        outcome_name = prospective_split.get("outcome_name", save_path.stem)
        split_cfg = resolve_split_contract(prospective_split)
        pandas_outcomes = all_outcomes.to_pandas()
        pandas_outcomes = apply_prospective_split(
            pandas_outcomes,
            **split_cfg,
        )
        report = validate_split_integrity(
            pandas_outcomes,
            **split_cfg,
        )
        if not report["ok"]:
            raise ValueError(f"Prospective split integrity check failed: {report}")
        summary = summarize_outcome_splits(
            pandas_outcomes,
            outcome_name=outcome_name,
        )
        summary.to_csv(save_path.with_suffix(".split_summary.csv"), index=False)
        all_outcomes = pl.from_pandas(pandas_outcomes)

    logging.info(
        f"Total number of subjects: {len(all_outcomes):_} ({all_outcomes['outcome_date'].is_not_null().sum():_} positives)"
    )
    logging.info(f"Saving to {save_path}")
    all_outcomes.write_parquet(save_path)


if __name__ == "__main__":
    main()
