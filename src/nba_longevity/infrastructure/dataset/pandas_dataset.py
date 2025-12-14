import pandas as pd
from typing import Iterator, Mapping, Any
from nba_longevity.domain.dataset.dataset import Dataset


class PandasDataset(Dataset):
    """
    Implémentation Pandas du Dataset abstrait.
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        for row in self._df.itertuples(index=False):
            yield row._asdict()
