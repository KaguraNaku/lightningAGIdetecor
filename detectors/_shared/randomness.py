"""Serializable native Python/NumPy random state without detector dependencies."""

import hashlib
import json
import random

import numpy as np


class RandomState:
    """Observe RNG state without drawing samples or changing either generator."""

    @staticmethod
    def capture():
        numpy = np.random.get_state()
        return {
            "python": random.getstate(),
            "numpy": [numpy[0], numpy[1].tolist(), *numpy[2:]],
        }

    @classmethod
    def digest(cls):
        return hashlib.sha256(json.dumps(cls.capture()).encode()).hexdigest()


class ArrayIdentity:
    """Byte identity of a contiguous array; no pixel normalization is applied."""

    @staticmethod
    def digest(pixels):
        return hashlib.sha256(
            memoryview(np.ascontiguousarray(pixels)).cast("B")
        ).hexdigest()
