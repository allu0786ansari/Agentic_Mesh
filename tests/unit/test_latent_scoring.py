import numpy as np

from evaluation.inference import combined_anomaly_scores, event_level_metrics


def test_combined_scores_use_validation_statistics() -> None:
    validation_errors = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
    test_errors = np.array([1.0, 2.0])
    validation_latent = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    test_latent = np.array([0.5, 2.0])

    test_scores, validation_scores, parameters = combined_anomaly_scores(
        validation_errors,
        test_errors,
        validation_latent,
        test_latent,
        alpha=0.5,
    )

    assert test_scores.shape == (2,)
    assert validation_scores.shape == (5,)
    assert test_scores[1] > test_scores[0]
    assert parameters["alpha"] == 0.5
    assert np.isfinite(test_scores).all()


def test_event_metrics_report_detection_delay() -> None:
    result = event_level_metrics(
        raw_labels=np.array([0, 0, 1, 1, 1, 0, 0, 1, 1], dtype=np.int8),
        window_starts=np.array([0, 2, 4, 6, 8]),
        window_size=1,
        scores=np.array([0.0, 0.0, 0.0, 1.0, 1.0]),
        threshold=0.5,
    )

    assert result["attack_event_count"] == 2
    assert result["detected_attack_events"] == 1
    assert result["detected_event_delays_rows"] == [1]
    assert result["mean_detection_delay_rows"] == 1.0
