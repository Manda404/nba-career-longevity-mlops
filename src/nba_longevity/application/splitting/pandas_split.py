from pandas import DataFrame
from typing import Sequence, Mapping, Tuple
from sklearn.model_selection import train_test_split

def split_train_valid_pandas(
    rows: Sequence[Mapping[str, object]],
    target_column: str,
    valid_size: float = 0.2,
    seed: int = 42,
) -> Tuple[list[dict], list[dict]]:
    
    df = DataFrame(list(rows))

    y = df[target_column]
    train_df, valid_df = train_test_split(
        df,
        test_size=valid_size,
        random_state=seed,
        stratify=y,   # important en classification
    )

    return train_df.to_dict("records"), valid_df.to_dict("records")
