"""Prediction observations are explicitly images or clips; videos aggregate them."""


def validate_prediction_unit(value):
    if not isinstance(value, str) or value not in {"image", "clip"}:
        raise ValueError("Prediction unit must be image or clip")
    return value
