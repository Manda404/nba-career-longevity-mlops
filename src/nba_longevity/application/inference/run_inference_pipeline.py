from domain.features.feature_contract import FEATURE_COLUMNS
from nba_longevity.domain.features.feature_spaces import (
    FEATURE_SPACE_MINIMAL,
    FEATURE_SPACE_EXTENDED,
)

def run_inference(
    predictor,
    feature_dataset,
    feature_space: str = "minimal"
):
    
    # Choix de l'espace de features
    if feature_space == "extended":
        selected_features = FEATURE_SPACE_EXTENDED
    else:
        selected_features = FEATURE_SPACE_MINIMAL

    rows = list(feature_dataset)
    proba = predictor.predict_proba(
        rows=rows,
        feature_columns=selected_features,
    )

    return [
        {
            **row,
            "proba_5yrs": float(p),
        }
        for row, p in zip(rows, proba)
    ]
