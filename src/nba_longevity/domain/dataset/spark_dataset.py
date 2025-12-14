from typing import Iterator, Mapping, Any
from domain.dataset.dataset import Dataset


class SparkDataset(Dataset):
    """
    Implémentation Spark du Dataset abstrait.
    """

    def __init__(self, spark_df):
        self._df = spark_df

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        # toLocalIterator = streaming ligne par ligne
        for row in self._df.toLocalIterator():
            yield row.asDict()
