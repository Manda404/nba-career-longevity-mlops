from pandas import read_csv
from nba_longevity.application.bootstrap import app_logger
from nba_longevity.domain.ports.dataset_loader_port import DatasetLoaderPort
from nba_longevity.infrastructure.dataset.pandas_dataset import PandasDataset


class CsvDatasetLoader(DatasetLoaderPort):
    """
    CSV dataset loader.

    Responsibility:
    - Load raw CSV data from disk
    - Wrap it into a PandasDataset abstraction
    - Perform NO business logic
    """

    def __init__(self, path: str):
        self.path = path

    def load(self) -> PandasDataset:
        app_logger.info(f"Loading CSV dataset from path: {self.path}")

        df = read_csv(self.path)

        app_logger.info(
            f"CSV loaded successfully | rows={df.shape[0]} | cols={df.shape[1]}"
        )

        return PandasDataset(df)
