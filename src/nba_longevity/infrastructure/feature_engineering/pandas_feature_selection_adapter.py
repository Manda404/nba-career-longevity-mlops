from pandas import DataFrame
from nba_longevity.domain.dataset.dataset import Dataset
from nba_longevity.application.bootstrap import app_logger
from nba_longevity.domain.ports.feature_selection_port import FeatureSelectionPort
from nba_longevity.infrastructure.dataset.pandas_dataset import PandasDataset
from nba_longevity.domain.features.feature_spaces import (
    FEATURE_SPACE_MINIMAL,
    FEATURE_SPACE_EXTENDED,
    TARGET_COLUMN
)


class PandasFeatureSelectionAdapter(FeatureSelectionPort):
    """
    Implémentation Pandas de la sélection des features ML.

    Cette classe est responsable de la projection d’un dataset enrichi
    (features brutes + features dérivées) vers un espace de features
    final destiné à l’entraînement ou à l’inférence d’un modèle ML.

    Responsabilité unique :
        - sélectionner les colonnes pertinentes pour le modèle
        - ne crée aucune feature
        - ne modifie aucune valeur

    Le choix du feature space (minimal, étendu, etc.) est une décision
    métier définie dans le Domain.
    """

    def __init__(self, feature_space: list[str] | None = None):
        """
        Initialise le sélecteur de features.

        Parameters
        ----------
        feature_space : list[str] | None, optional
            Liste explicite des features à conserver.
            - Si None, le FEATURE_SPACE_MINIMAL est utilisé par défaut.
            - Permet d’injecter FEATURE_SPACE_EXTENDED ou tout autre
              espace de features versionné.
        """
        self.feature_space = feature_space or FEATURE_SPACE_EXTENDED

        app_logger.info(
            f"FeatureSelection initialisé avec "
            f"{len(self.feature_space)} features"
        )

        app_logger.debug(
            f"Espace de features sélectionné : {self.feature_space}"
        )

    def select_features(self, dataset: Dataset) -> Dataset:
        """
        Sélectionne l’espace de features final du modèle.
        """
        app_logger.info("🎯 Démarrage de la sélection des features")

        # Conversion Dataset -> DataFrame
        df = DataFrame(list(dataset))
        app_logger.debug(
            f"Dataset d’entrée : {df.shape[0]} lignes, {df.shape[1]} colonnes"
        )

        selected_cols = self.feature_space + [TARGET_COLUMN]

        # Vérification de sécurité
        missing_cols = set(selected_cols) - set(df.columns)
        if missing_cols:
            app_logger.error(
                f"Colonnes manquantes lors de la sélection : {missing_cols}"
            )
            raise ValueError(f"Colonnes manquantes : {missing_cols}")

        app_logger.info(
            f"{len(self.feature_space)} features sélectionnées + colonne cible"
        )

        app_logger.debug(
            f"Colonnes finales utilisées par le modèle : {selected_cols}"
        )

        return PandasDataset(df[selected_cols])
