import numpy as np
import pandas as pd

from scripts.prepare_data import _engineer_hai_features, _scale_hai


def _frame(labels: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=len(labels), freq="s", tz="UTC"),
            "P1_CONSTANT": [1.0] * len(labels),
            "P1_CONTINUOUS": np.arange(len(labels), dtype=float),
            "P1_BINARY": [0, 0, 1, 1, 0, 0][: len(labels)],
            "label": labels,
        }
    )


def test_hai_feature_engineering_removes_constants_and_aligns_splits() -> None:
    train = _frame([0, 0, 0, 0, 0, 0])
    test = _frame([0, 0, 1, 1, 0, 0])

    engineered_train, engineered_test, metadata = _engineer_hai_features(
        train,
        test,
        rolling_window=3,
    )

    assert engineered_train.columns.tolist() == engineered_test.columns.tolist()
    assert "P1_CONSTANT" not in engineered_train.columns
    assert "P1_CONTINUOUS__delta" in engineered_train.columns
    assert "P1_CONTINUOUS__abs_delta" in engineered_train.columns
    assert "P1_CONTINUOUS__rolling_std" in engineered_train.columns
    assert "P1_CONTINUOUS__local_deviation" in engineered_train.columns
    assert "P1_BINARY__transition" in engineered_train.columns
    assert engineered_test["label"].tolist() == [0, 0, 1, 1, 0, 0]
    assert metadata["removed_constant_features"] == ["P1_CONSTANT"]
    assert metadata["rolling_window_rows"] == 3
    assert np.isfinite(engineered_train.drop(columns=["time", "label"]).to_numpy()).all()


def test_hai_scaling_uses_engineered_feature_columns() -> None:
    train = _frame([0, 0, 0, 0, 0, 0])
    test = _frame([0, 0, 1, 1, 0, 0])
    engineered_train, engineered_test, _ = _engineer_hai_features(train, test, rolling_window=3)

    scaled_train, scaled_test = _scale_hai(engineered_train, engineered_test)
    feature_columns = [column for column in scaled_train.columns if column not in {"time", "label"}]

    assert feature_columns == [column for column in scaled_test.columns if column not in {"time", "label"}]
    assert np.isfinite(scaled_train[feature_columns].to_numpy()).all()
    assert np.isfinite(scaled_test[feature_columns].to_numpy()).all()
    assert ((scaled_train[feature_columns] >= 0) & (scaled_train[feature_columns] <= 1)).all().all()
