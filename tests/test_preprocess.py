
import numpy as np

from mammaus.preprocess import (
    apply_windowing,
    crop_ultrasound_region,
    get_patient_id,
    get_series_info,
)


class TestApplyWindowing:
    def test_normalizes_to_uint8(self):
        arr = np.array([[0, 100, 200]], dtype=np.float32)

        class FakeDS:
            pass

        result = apply_windowing(arr, FakeDS())
        assert result.dtype == np.uint8
        assert result.min() == 0
        assert result.max() == 255

    def test_with_window_center_width(self):
        arr = np.array([[0, 50, 100, 150, 200]], dtype=np.float32)

        class FakeDS:
            WindowCenter = 100.0
            WindowWidth = 100.0

        result = apply_windowing(arr, FakeDS())
        assert result.dtype == np.uint8
        assert result[0, 0] == 0
        assert result[0, -1] == 255


class TestCropUltrasoundRegion:
    def test_crops_black_borders(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        img[20:80, 30:70] = 128
        cropped = crop_ultrasound_region(img)
        assert cropped.shape[0] < 100
        assert cropped.shape[1] < 100

    def test_all_black_returns_original(self):
        img = np.zeros((50, 50), dtype=np.uint8)
        cropped = crop_ultrasound_region(img)
        assert cropped.shape == (50, 50)

    def test_rgb_input(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[20:80, 30:70, :] = 128
        cropped = crop_ultrasound_region(img)
        assert cropped.ndim == 3
        assert cropped.shape[0] < 100


class TestGetSeriesInfo:
    def test_with_description_and_number(self):
        class FakeDS:
            SeriesDescription = "Breast Left"
            SeriesNumber = 3

        assert get_series_info(FakeDS()) == "3_Breast_Left"

    def test_with_description_only(self):
        class FakeDS:
            SeriesDescription = "Breast Right"

        assert get_series_info(FakeDS()) == "Breast_Right"

    def test_with_number_only(self):
        class FakeDS:
            SeriesNumber = 7

        assert get_series_info(FakeDS()) == "7"

    def test_no_metadata(self):
        class FakeDS:
            pass

        assert get_series_info(FakeDS()) == "unknown"


class TestGetPatientId:
    def test_with_patient_name(self):
        class FakeDS:
            PatientName = "DOE^JANE"

        assert get_patient_id(FakeDS()) == "DOE_JANE"

    def test_with_patient_id_fallback(self):
        class FakeDS:
            PatientID = "P12345"

        assert get_patient_id(FakeDS()) == "P12345"

    def test_no_metadata(self):
        class FakeDS:
            pass

        assert get_patient_id(FakeDS()) == "patient"
