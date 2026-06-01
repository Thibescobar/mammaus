# Common constants and mappings for mammaus

import logging

MODEL_ID = "hugging-science/breast-cancer-detector-2"
DEFAULT_MIN_RUN = 3
DEFAULT_MALIGNANT_THRESHOLD = 30.0


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the mammaus CLI."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=level,
    )

LABELS = {
    "0": "benign",
    "1": "malignant",
    "2": "normal",
}

LABEL_COLORS = {
    "benign": "#2ecc71",
    "malignant": "#e74c3c",
    "normal": "#3498db",
}

LABEL_EN = {
    "benign": "Benign",
    "malignant": "Malignant (suspicious)",
    "normal": "Normal",
}

SERIES_NAMES_EN = {
    "RAP":  "Right breast — Areolar / Periareolar",
    "RLAT": "Right breast — Lateral",
    "RLOQ": "Right breast — Lower Outer Quadrant",
    "RMED": "Right breast — Medial",
    "RSUP": "Right breast — Superior",
    "LAP":  "Left breast — Areolar / Periareolar",
    "LLAT": "Left breast — Lateral",
    "LLOQ": "Left breast — Lower Outer Quadrant",
    "LMED": "Left breast — Medial",
    "LSUP": "Left breast — Superior",
}
