"""Truncated-SVD helpers for LoRA-XS initialization.

scikit-learn is imported lazily, inside :func:`run_svd`. At module scope it was an import-time
dependency of ``export/training_export.py`` (via ``initialization_utils``), so **every** training-stage
export — including plain LoRA and MARS, which never touch SVD — died with `ModuleNotFoundError: No
module named 'sklearn'` before doing any work. scikit-learn is not in the `ort-training-local` profile;
only the LoRA-XS path actually needs it, and that path now says so with a message naming the fix.
"""

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from sklearn.decomposition import TruncatedSVD


def run_svd(
    input_matrix: np.ndarray, rank: int, n_iter: int, random_state: int
) -> tuple[np.ndarray, "TruncatedSVD"]:
    try:
        from sklearn.decomposition import TruncatedSVD
    except ImportError as exc:  # pragma: no cover - depends on the active profile
        raise ImportError(
            "LoRA-XS initialization needs scikit-learn, which is not part of the training profile. "
            "Install it alongside the profile (`uv pip install scikit-learn`) or choose --peft lora/mars."
        ) from exc

    svd = TruncatedSVD(n_components=rank, n_iter=n_iter, random_state=random_state)
    svd.fit(input_matrix)
    reduced_matrix = svd.transform(input_matrix)
    return reduced_matrix, svd


def get_linear_rec_svd(
    input_matrix: np.ndarray, rank: int, n_iter: int, random_state: int
) -> tuple[np.ndarray, np.ndarray, Any]:
    reduced_matrix, svd = run_svd(input_matrix, rank, n_iter, random_state)

    reconstructed_matrix = svd.inverse_transform(reduced_matrix)
    return reconstructed_matrix, reduced_matrix, svd.components_
