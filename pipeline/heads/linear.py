"""Linear heads on the image embedding, saved as plain `.npz` arrays (no pickle).

- `LinearHead`: logistic regression + isotonic calibration + gating thresholds (binary).
- `SoftmaxHead`: multinomial logistic regression (shot type).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from pipeline.files import atomic_write
from pipeline.heads.metrics import Thresholds, pick_thresholds

MAX_ITER = 2000


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.asarray(1 / (1 + np.exp(-z)))


@dataclass(frozen=True, eq=False)
class LinearHead:
    name: str
    coef: np.ndarray  # (d,)
    intercept: float
    iso_x: np.ndarray  # isotonic calibration knots: raw sigmoid score -> probability
    iso_y: np.ndarray
    # On the raw score, not the calibrated probability: isotonic output is a step function, and
    # a threshold inside a tie block would move the whole block (it is monotone, so the order is
    # the same).
    thresholds: Thresholds

    def raw(self, x: np.ndarray) -> np.ndarray:
        return _sigmoid(x @ self.coef + self.intercept)

    def zone(self, x: np.ndarray) -> np.ndarray:
        """0 pass, 1 escalate, 2 reject."""
        return self.thresholds.zone(self.raw(x))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(np.interp(self.raw(x), self.iso_x, self.iso_y))

    def save(self, out_dir: Path) -> Path:
        path = out_dir / f"{self.name}.npz"

        def write(tmp: Path) -> None:
            with tmp.open("wb") as handle:
                np.savez(
                    handle,
                    kind=np.array("linear"),
                    name=np.array(self.name),
                    coef=self.coef,
                    intercept=np.array(self.intercept),
                    iso_x=self.iso_x,
                    iso_y=self.iso_y,
                    thresholds=np.array([self.thresholds.low, self.thresholds.high]),
                )

        atomic_write(path, write)
        return path


@dataclass(frozen=True, eq=False)
class SoftmaxHead:
    name: str
    classes: np.ndarray  # (k,) str
    coef: np.ndarray  # (k, d)
    intercept: np.ndarray  # (k,)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = x @ self.coef.T + self.intercept
        z = np.exp(z - z.max(axis=1, keepdims=True))
        return np.asarray(z / z.sum(axis=1, keepdims=True))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.classes[np.argmax(x @ self.coef.T + self.intercept, axis=1)])

    def save(self, out_dir: Path) -> Path:
        path = out_dir / f"{self.name}.npz"

        def write(tmp: Path) -> None:
            with tmp.open("wb") as handle:
                np.savez(
                    handle,
                    kind=np.array("softmax"),
                    name=np.array(self.name),
                    classes=self.classes.astype(str),
                    coef=self.coef,
                    intercept=self.intercept,
                )

        atomic_write(path, write)
        return path


def load_head(path: Path) -> LinearHead | SoftmaxHead:
    with np.load(path, allow_pickle=False) as z:
        name = str(z["name"])
        if str(z["kind"]) == "softmax":
            return SoftmaxHead(name, z["classes"], z["coef"], z["intercept"])
        low, high = (float(t) for t in z["thresholds"])
        return LinearHead(
            name, z["coef"], float(z["intercept"]), z["iso_x"], z["iso_y"], Thresholds(low, high)
        )


def fit_linear_head(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    *,
    precision: float,
    recall: float,
    c: float = 1.0,
) -> LinearHead:
    """Class-balanced logistic regression on the train split; isotonic calibration and gating
    thresholds on the calibration split."""
    model = LogisticRegression(C=c, class_weight="balanced", max_iter=MAX_ITER)
    model.fit(x_train, y_train)
    coef = model.coef_[0].astype(np.float64)
    intercept = float(model.intercept_[0])
    raw = _sigmoid(x_cal @ coef + intercept)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw, y_cal)
    iso_x, iso_y = iso.X_thresholds_, iso.y_thresholds_
    thresholds = pick_thresholds(raw, y_cal, precision=precision, recall=recall)
    return LinearHead(name, coef, intercept, iso_x, iso_y, thresholds)


def fit_softmax_head(name: str, x: np.ndarray, y: np.ndarray, *, c: float = 1.0) -> SoftmaxHead:
    model = LogisticRegression(C=c, class_weight="balanced", max_iter=MAX_ITER)
    model.fit(x, y)
    return SoftmaxHead(
        name,
        np.asarray(model.classes_).astype(str),
        model.coef_.astype(np.float64),
        model.intercept_.astype(np.float64),
    )
