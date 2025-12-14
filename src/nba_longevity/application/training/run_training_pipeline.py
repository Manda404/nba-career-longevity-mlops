from infrastructure.dataset.csv_dataset_loader import CsvDatasetLoader
from infrastructure.preprocessing.pandas_preprocessing_adapter import (
    PandasPreprocessingAdapter,
)
from infrastructure.feature_engineering.pandas_feature_engineering_adapter import (
    PandasFeatureEngineeringAdapter,
)
from infrastructure.training.xgboost_trainer import XGBoostTrainer
from infrastructure.training.catboost_trainer import CatBoostTrainer
from application.splitting.pandas_split import split_train_valid_pandas
from domain.features.feature_contract import FEATURE_COLUMNS

TARGET_COLUMN = "TARGET_5YRS"


def run_training(
    csv_path: str,
    model_type: str = "xgboost",
):
    # 1. Load
    loader = CsvDatasetLoader(csv_path)
    dataset = loader.load()

    # 2. Preprocessing
    preprocessor = PandasPreprocessingAdapter()
    clean_dataset = preprocessor.preprocess(dataset)

    # 3. Feature engineering
    feature_engineer = PandasFeatureEngineeringAdapter()
    feature_dataset = feature_engineer.build_features(clean_dataset)

    # 4. Split train / valid
    train_rows, valid_rows = split_train_valid_pandas(
        rows=list(feature_dataset),
        target_column=TARGET_COLUMN,
        valid_size=0.2,
        seed=42,
    )

    # 5. Train
    if model_type == "xgboost":
        trainer = XGBoostTrainer()
        model = trainer.train(
            train_rows=train_rows,
            valid_rows=valid_rows,
            feature_columns=FEATURE_COLUMNS,
            target_column=TARGET_COLUMN,
            params={
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "max_depth": 4,
                "eta": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": 1.0,  # à ajuster si déséquilibre
                "num_boost_round": 1000,
                "early_stopping_rounds": 50,
            },
        )

    elif model_type == "catboost":
        trainer = CatBoostTrainer()
        model = trainer.train(
            train_rows=train_rows,
            valid_rows=valid_rows,
            feature_columns=FEATURE_COLUMNS,
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
        raise ValueError(f"Unknown model_type: {model_type}")

    return model
