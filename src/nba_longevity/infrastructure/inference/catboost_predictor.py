from typing import Sequence
from pandas import DataFrame
from catboost import CatBoostClassifier
from nba_longevity.domain.ports.predictor_port import PredictorPort
# Feature spaces (Domain)
from nba_longevity.domain.features.feature_spaces import (
    FEATURE_SPACE_MINIMAL,
    FEATURE_SPACE_EXTENDED,
)



class CatBoostPredictor(PredictorPort):
    def __init__(self, model: CatBoostClassifier):
        self.model = model

    def predict_proba(self, rows, feature_space: str = "minimal")->Sequence[float]:
        df = DataFrame(list(rows))

        # Choix de l'espace de features
        if feature_space == "extended":
            selected_features = FEATURE_SPACE_EXTENDED
        else:
            selected_features = FEATURE_SPACE_MINIMAL

        X = df[selected_features]
        return self.model.predict_proba(X)[:, 1]
