from pandas import DataFrame, to_numeric
from nba_longevity.domain.ports.preprocessing_port import PreprocessingPort
from nba_longevity.domain.dataset.dataset import Dataset
from nba_longevity.infrastructure.dataset.pandas_dataset import PandasDataset
from nba_longevity.domain.preprocessing.preprocessing_rules import (
    NUMERIC_COLUMNS, TARGET_COLUMN, ID_COLUMN
)


class PandasPreprocessingAdapter(PreprocessingPort):
    """
    Nettoyage et préparation des données (Pandas).
    """

    def preprocess(self, dataset: Dataset) -> Dataset:
        df = DataFrame(list(dataset))

        # 1. Cast explicite
        for col in NUMERIC_COLUMNS:
            df[col] = to_numeric(df[col], errors="coerce")

        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

        # 2. Gestion des NaN
        # → médiane (robuste, dataset petit)
        df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].fillna(
            df[NUMERIC_COLUMNS].median()
        )

        # 3. Drop lignes sans target
        df = df.dropna(subset=[TARGET_COLUMN])

        # 4. Sécurité : aucune minute négative / nulle
        df = df[df["MinutesPerGame"] > 0]

        return PandasDataset(df)
