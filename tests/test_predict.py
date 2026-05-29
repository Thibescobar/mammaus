"""Tests for the prediction pipeline with a mocked classifier."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mammaus.constants import MODEL_ID
from mammaus.predict import predict_cli


@pytest.fixture()
def fake_images(tmp_path: Path) -> Path:
    """Create a minimal set of fake PNG files for testing."""
    acq_dir = tmp_path / "preprocessed" / "test_acq"
    acq_dir.mkdir(parents=True)
    for i in range(5):
        (acq_dir / f"frame_{i:03d}.png").write_bytes(b"\x89PNG\r\n")
    return tmp_path / "preprocessed"


def _make_mock_classifier():
    """Return a mock classifier that returns plausible predictions."""
    mock = MagicMock()
    mock.return_value = [
        {"label": "0", "score": 0.75},
        {"label": "2", "score": 0.20},
        {"label": "1", "score": 0.05},
    ]
    return mock


class TestPredictCli:
    @patch("mammaus.predict.pipeline")
    def test_runs_end_to_end(self, mock_pipeline, fake_images, tmp_path):
        """Full pipeline run with mocked model produces expected outputs."""
        mock_pipeline.return_value = _make_mock_classifier()
        output_dir = tmp_path / "results"

        with patch(
            "sys.argv",
            ["predict_cli", str(fake_images), "--output", str(output_dir)],
        ):
            predict_cli()

        # Check that scores were saved
        scores_dir = output_dir / "scores"
        assert scores_dir.exists()
        npz_files = list(scores_dir.glob("*.npz"))
        assert len(npz_files) == 1

        # Validate score contents
        data = np.load(npz_files[0])
        assert set(data.files) == {"benign", "malignant", "normal"}
        assert len(data["benign"]) == 5

        # Check that report and figure were generated
        reports = list(output_dir.glob("*_report.txt"))
        assert len(reports) == 1
        figures = list(output_dir.glob("*.png"))
        assert len(figures) == 1

    @patch("mammaus.predict.pipeline")
    def test_scores_values_match_mock(self, mock_pipeline, fake_images, tmp_path):
        """Saved scores match the mocked classifier output."""
        mock_pipeline.return_value = _make_mock_classifier()
        output_dir = tmp_path / "results"

        with patch(
            "sys.argv",
            ["predict_cli", str(fake_images), "--output", str(output_dir)],
        ):
            predict_cli()

        data = np.load(output_dir / "scores" / "test_acq_scores.npz")
        # Mock returns 0.75 for benign (label "0") → 75.0%
        np.testing.assert_allclose(data["benign"], [75.0] * 5, atol=0.1)
        # Mock returns 0.05 for malignant (label "1") → 5.0%
        np.testing.assert_allclose(data["malignant"], [5.0] * 5, atol=0.1)

    @patch("mammaus.predict.pipeline")
    def test_no_images_exits(self, mock_pipeline, tmp_path):
        """CLI exits with error when no PNGs are found."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with patch("sys.argv", ["predict_cli", str(empty_dir)]):
            with pytest.raises(SystemExit):
                predict_cli()


class TestModelId:
    def test_model_id_is_set(self):
        assert MODEL_ID == "hugging-science/breast-cancer-detector-2"
