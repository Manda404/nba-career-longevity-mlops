from domain.ports.preprocessing_port import PreprocessingPort
from infrastructure.dataset.spark_dataset import SparkDataset


class SparkPreprocessingAdapter(PreprocessingPort):
    """
    Preprocessing distribué avec Spark.
    """

    def preprocess(self, dataset: SparkDataset) -> SparkDataset:
        # Ici on sait que c'est SparkDataset
        # 1. flux logique → Spark DataFrame (infra seulement)
        df = dataset._df

        # 2. règles métier (exemples)
        df = df.dropna()

        df = (
            df.withColumn("GP", df["GP"].cast("int"))
              .withColumn("MIN", df["MIN"].cast("double"))
              .withColumn("PTS", df["PTS"].cast("double"))
        )
        # 3. retour au flux logique
        return SparkDataset(df)
