# Common constants and mappings for mammaus

MODEL_ID = "hugging-science/breast-cancer-detector-2"

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
