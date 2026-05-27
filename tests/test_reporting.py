import numpy as np
import pytest

from mammaus.reporting import (
    compute_stats,
    find_consecutive_malignant_runs,
    find_malignant_frames,
    generate_global_text_report,
    make_acquisition_figure,
    make_global_figure,
    print_acquisition_report,
    series_display_name,
)


class TestSeriesDisplayName:
    def test_known_code(self):
        result = series_display_name("1_RAP")
        assert "Right breast" in result
        assert "Areolar" in result

    def test_unknown_code(self):
        result = series_display_name("1_UNKNOWN")
        assert "UNKNOWN" in result

    def test_no_prefix(self):
        result = series_display_name("RAP")
        assert "Right breast" in result


class TestFindMalignantFrames:
    def test_no_malignant(self):
        scores = {
            "benign": [80.0, 70.0, 90.0],
            "malignant": [5.0, 10.0, 3.0],
            "normal": [15.0, 20.0, 7.0],
        }
        top1, suspect = find_malignant_frames(scores)
        assert top1 == []
        assert suspect == []

    def test_malignant_top1(self):
        scores = {
            "benign": [10.0, 70.0],
            "malignant": [80.0, 10.0],
            "normal": [10.0, 20.0],
        }
        top1, suspect = find_malignant_frames(scores)
        assert len(top1) == 1
        assert top1[0][0] == 0  # frame index
        assert top1[0][1] == 80.0

    def test_suspect_above_threshold(self):
        scores = {
            "benign": [50.0],
            "malignant": [35.0],
            "normal": [15.0],
        }
        top1, suspect = find_malignant_frames(scores, threshold=30.0)
        assert top1 == []
        assert len(suspect) == 1

    def test_custom_threshold(self):
        scores = {
            "benign": [50.0],
            "malignant": [25.0],
            "normal": [25.0],
        }
        top1, suspect = find_malignant_frames(scores, threshold=20.0)
        assert len(suspect) == 1


class TestFindConsecutiveMalignantRuns:
    def test_no_runs(self):
        scores = {
            "benign": [80.0, 80.0, 80.0, 80.0],
            "malignant": [5.0, 5.0, 5.0, 5.0],
            "normal": [15.0, 15.0, 15.0, 15.0],
        }
        runs = find_consecutive_malignant_runs(scores, min_run=3)
        assert runs == []

    def test_single_run(self):
        scores = {
            "benign": [10.0, 10.0, 10.0, 80.0],
            "malignant": [80.0, 80.0, 80.0, 5.0],
            "normal": [10.0, 10.0, 10.0, 15.0],
        }
        runs = find_consecutive_malignant_runs(scores, min_run=3)
        assert len(runs) == 1
        assert runs[0] == (0, 2, 3)

    def test_isolated_not_counted(self):
        scores = {
            "benign": [10.0, 80.0, 10.0, 80.0],
            "malignant": [80.0, 5.0, 80.0, 5.0],
            "normal": [10.0, 15.0, 10.0, 15.0],
        }
        runs = find_consecutive_malignant_runs(scores, min_run=3)
        assert runs == []

    def test_run_at_end(self):
        scores = {
            "benign": [80.0, 10.0, 10.0, 10.0],
            "malignant": [5.0, 80.0, 80.0, 80.0],
            "normal": [15.0, 10.0, 10.0, 10.0],
        }
        runs = find_consecutive_malignant_runs(scores, min_run=3)
        assert len(runs) == 1
        assert runs[0] == (1, 3, 3)


class TestComputeStats:
    def test_basic_stats(self):
        scores = {
            "benign": np.array([80.0, 70.0, 90.0, 85.0]),
            "malignant": np.array([5.0, 10.0, 3.0, 8.0]),
            "normal": np.array([15.0, 20.0, 7.0, 7.0]),
        }
        stats = compute_stats(scores)
        assert stats["n_frames"] == 4
        assert stats["counts"]["benign"] == 4
        assert stats["counts"]["malignant"] == 0
        assert stats["to_check"] is False
        assert stats["pct"]["benign"] == 100.0
        assert pytest.approx(stats["mean_malig"], abs=0.1) == 6.5
        assert stats["max_malig"] == 10.0

    def test_with_malignant_run(self):
        scores = {
            "benign": np.array([10.0, 10.0, 10.0]),
            "malignant": np.array([80.0, 80.0, 80.0]),
            "normal": np.array([10.0, 10.0, 10.0]),
        }
        stats = compute_stats(scores, min_run=3)
        assert stats["to_check"] is True
        assert stats["n_malig_top1"] == 3
        assert len(stats["consecutive_runs"]) == 1


# --- Fixtures for report/figure tests ---

def _sample_scores():
    """Return sample scores dict for 10 frames (mostly benign)."""
    return {
        "benign": np.array([80.0, 75.0, 70.0, 85.0, 90.0, 78.0, 72.0, 88.0, 82.0, 76.0]),
        "malignant": np.array([5.0, 8.0, 10.0, 3.0, 2.0, 7.0, 12.0, 4.0, 6.0, 9.0]),
        "normal": np.array([15.0, 17.0, 20.0, 12.0, 8.0, 15.0, 16.0, 8.0, 12.0, 15.0]),
    }


def _sample_scores_malignant():
    """Return sample scores with malignant dominant frames."""
    return {
        "benign": np.array([10.0, 10.0, 10.0, 80.0, 80.0]),
        "malignant": np.array([80.0, 80.0, 80.0, 5.0, 5.0]),
        "normal": np.array([10.0, 10.0, 10.0, 15.0, 15.0]),
    }


class TestGenerateGlobalTextReport:
    def test_reassuring_report(self, tmp_path):
        stats = compute_stats(_sample_scores())
        all_stats = {"test_acq": stats}
        generate_global_text_report(all_stats, tmp_path, n_acq=1, n_frames=10)
        report = (tmp_path / "global_report.txt").read_text()
        assert "GLOBAL REPORT" in report
        assert "reassuring" in report.lower()
        assert "test_acq" in report

    def test_to_check_report(self, tmp_path):
        stats = compute_stats(_sample_scores_malignant())
        all_stats = {"acq_mal": stats}
        generate_global_text_report(all_stats, tmp_path, n_acq=1, n_frames=5)
        report = (tmp_path / "global_report.txt").read_text()
        assert "TO CHECK" in report
        assert "require review" in report


class TestPrintAcquisitionReport:
    def test_generates_report_file(self, tmp_path):
        scores = _sample_scores()
        print_acquisition_report("1_RAP", scores, tmp_path)
        report_path = tmp_path / "1_RAP_report.txt"
        assert report_path.exists()
        text = report_path.read_text()
        assert "RESULT BY CATEGORY" in text
        assert "OVERALL REASSURING" in text

    def test_report_with_malignant_frames(self, tmp_path):
        scores = _sample_scores_malignant()
        print_acquisition_report("1_TEST", scores, tmp_path)
        text = (tmp_path / "1_TEST_report.txt").read_text()
        assert "TO CHECK" in text
        assert "SUSPICIOUS FRAMES" in text


class TestMakeAcquisitionFigure:
    def test_generates_png(self, tmp_path):
        scores = _sample_scores()
        fig_path = make_acquisition_figure("1_RAP", scores, tmp_path)
        assert fig_path.exists()
        assert fig_path.suffix == ".png"
        assert fig_path.stat().st_size > 1000

    def test_with_malignant_data(self, tmp_path):
        scores = _sample_scores_malignant()
        fig_path = make_acquisition_figure("1_MAL", scores, tmp_path)
        assert fig_path.exists()


class TestMakeGlobalFigure:
    def test_generates_global_png(self, tmp_path):
        s1 = _sample_scores()
        s2 = _sample_scores_malignant()
        all_data = {"1_RAP": s1, "1_RMED": s2}
        all_stats = {
            "1_RAP": compute_stats(s1),
            "1_RMED": compute_stats(s2),
        }
        fig_path = make_global_figure(all_data, all_stats, tmp_path)
        assert fig_path.exists()
        assert fig_path.name == "global_report.png"
        assert fig_path.stat().st_size > 1000
