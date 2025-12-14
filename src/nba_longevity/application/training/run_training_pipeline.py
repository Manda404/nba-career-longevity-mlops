from pathlib import Path

# 🔹 Logger
from nba_longevity.application.bootstrap import app_logger

# Dataset loading
from nba_longevity.infrastructure.dataset.csv_dataset_loader import CsvDatasetLoader

# Preprocessing
from nba_longevity.infrastructure.preprocessing.pandas_preprocessing_adapter import (
    PandasPreprocessingAdapter
)

# Feature engineering
from nba_longevity.infrastructure.feature_engineering.pandas_feature_engineering_adapter import (
    PandasFeatureEngineeringAdapter
)
from nba_longevity.infrastructure.feature_engineering.pandas_feature_selection_adapter import (
    PandasFeatureSelectionAdapter
)

# Feature spaces (Domain)
from nba_longevity.domain.features.feature_spaces import (
    FEATURE_SPACE_MINIMAL,
    FEATURE_SPACE_EXTENDED,
    TARGET_COLUMN
)

# Split
from nba_longevity.application.splitting.pandas_split import split_train_valid_pandas

# Training
from nba_longevity.infrastructure.training.xgboost_trainer import XGBoostTrainer
from nba_longevity.infrastructure.training.catboost_trainer import CatBoostTrainer

# Config & utils
from nba_longevity.infrastructure.config.settings import load_infra_config
from nba_longevity.infrastructure.system_utils.root_finder import get_repository_root


def run_training(
    model_type: str = "xgboost",
    feature_space: str = "minimal",
):
    """
    Pipeline complet d'entraînement ML (Pandas backend).
    """

    app_logger.info(
        f"🚀 Starting training pipeline | model={model_type} | feature_space={feature_space}"
    )

    # 0️⃣ Initialisation environnement & config
    root_path = get_repository_root(add_to_sys_path=False)
    config_path = Path(f"{root_path}/config/infra.yaml")
    config = load_infra_config(config_path)

    app_logger.debug(f"Repository root: {root_path}")
    app_logger.debug(f"Raw data path: {config.paths.raw_data}")

    # Choix de l'espace de features
    if feature_space == "extended":
        selected_features = FEATURE_SPACE_EXTENDED
    else:
        selected_features = FEATURE_SPACE_MINIMAL

    app_logger.info(
        f"Using feature space with {len(selected_features)} features"
    )

    # 1️⃣ Chargement des données
    app_logger.info("📥 Loading raw dataset")
    loader = CsvDatasetLoader(path=config.paths.raw_data)
    dataset = loader.load()

    # 2️⃣ Preprocessing
    app_logger.info("🧹 Preprocessing dataset")
    preprocessor = PandasPreprocessingAdapter()
    clean_dataset = preprocessor.preprocess(dataset)

    # 3️⃣ Feature engineering (ajout uniquement)
    app_logger.info("🧠 Feature engineering (add features)")
    feature_engineer = PandasFeatureEngineeringAdapter()
    enriched_dataset = feature_engineer.add_features(clean_dataset)

    # 4️⃣ Feature selection (projection ML)
    app_logger.info("🎯 Feature selection (ML projection)")
    feature_selector = PandasFeatureSelectionAdapter(
        feature_space=selected_features
    )
    feature_dataset = feature_selector.select_features(enriched_dataset)

    # 5️⃣ Split train / validation
    app_logger.info("✂️ Splitting train / validation")
    train_rows, valid_rows = split_train_valid_pandas(
        rows=list(feature_dataset),
        target_column=TARGET_COLUMN,
        valid_size=0.2,
        seed=42,
    )

    app_logger.info(
        f"Train size: {len(train_rows)} | Validation size: {len(valid_rows)}"
    )

    # 6️⃣ Entraînement
    if model_type == "xgboost":
        app_logger.info("🏋️ Training XGBoost model")

        trainer = XGBoostTrainer()
        model = trainer.train(
            train_rows=train_rows,
            valid_rows=valid_rows,
            feature_columns=selected_features,
            target_column=TARGET_COLUMN,
            params={
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "max_depth": 4,
                "eta": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "num_boost_round": 1000,
                "early_stopping_rounds": 50,
            },
        )

    elif model_type == "catboost":
        app_logger.info("🏋️ Training CatBoost model")

        trainer = CatBoostTrainer()
        model = trainer.train(
            train_rows=train_rows,
            valid_rows=valid_rows,
            feature_columns=selected_features,
            target_column=TARGET_COLUMN,
            params={
                "loss_function": "Logloss",
                "eval_metric": "AUC",
                "iterations": 2000,
                "learning_rate": 0.03,
                "depth": 6,
                "early_stopping_rounds": 50,
                "verbose": False,
            },
        )

    else:
        app_logger.error(f"Unknown model_type: {model_type}")
        raise ValueError(f"Unknown model_type: {model_type}")

    app_logger.success("✅ Training pipeline completed successfully")

    return model, valid_rows