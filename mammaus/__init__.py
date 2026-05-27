"""mammaus — AI-powered ultrasound video analysis pipeline."""

__version__ = "1.0.0"
__all__ = ["preprocess_cli", "predict_cli", "report_global_cli"]


def __getattr__(name: str):
    if name == "preprocess_cli":
        from mammaus.preprocess import preprocess_cli
        return preprocess_cli
    if name == "predict_cli":
        from mammaus.predict import predict_cli
        return predict_cli
    if name == "report_global_cli":
        from mammaus.reporting import report_global_cli
        return report_global_cli
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
