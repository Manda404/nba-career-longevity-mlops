from domain.ports.dataset_loader_port import DatasetLoaderPort
from infrastructure.dataset.spark_dataset import SparkDataset


class SparkDatasetLoader(DatasetLoaderPort):
    """
    Charge un dataset depuis Spark.
    """

    def __init__(self, spark_session, path: str, fmt: str = "csv"):
        self.spark = spark_session
        self.path = path
        self.fmt = fmt

    def load(self):
        if self.fmt == "csv":
            df = (
                self.spark.read
                .option("header", True)
                .option("inferSchema", True)
                .csv(self.path)
            )
        else:
            raise ValueError(f"Unsupported format: {self.fmt}")

        return SparkDataset(df)