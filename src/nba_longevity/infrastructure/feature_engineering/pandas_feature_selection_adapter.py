from pandas import DataFrame
from nba_longevity.domain.ports.feature_selection_port import FeatureSelectionPort
from nba_longevity.domain.dataset.dataset import Dataset
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

    def select_features(self, dataset: Dataset) -> Dataset:
        """
        Sélectionne l’espace de features final du modèle.

        Parameters
        ----------
        dataset : Dataset
            Dataset enrichi contenant :
            - les features brutes
            - les features dérivées
            - la colonne cible

        Returns
        -------
        Dataset
            Nouveau Dataset ne contenant que :
            - les features sélectionnées
            - la colonne cible

        Notes
        -----
        - Cette méthode garantit l’alignement train / inference.
        - Toute feature non sélectionnée est volontairement exclue
          du modèle afin d’éviter la redondance, le bruit ou le leakage.
        """
        df = DataFrame(list(dataset))

        selected_cols = self.feature_space + [TARGET_COLUMN]

        return PandasDataset(df[selected_cols])
