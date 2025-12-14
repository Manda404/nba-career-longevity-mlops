from typing import Tuple
from pyspark.sql import DataFrame

def split_train_valid_spark(
    df: DataFrame,
    valid_size: float = 0.2,
    seed: int = 42,
) -> Tuple[DataFrame, DataFrame]:
    train_df, valid_df = df.randomSplit([1.0 - valid_size, valid_size], seed=seed)
    return train_df, valid_df
