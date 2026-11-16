"""Speech MFCC + zero-crossing-rate analytic for cepstral delivery summaries.

Emits interpretable scalar summaries (per-coefficient mean and standard
deviation) of the first 13 Mel-frequency cepstral coefficients, plus
zero-crossing-rate mean and standard deviation. These are the hand-crafted
features consumed by the SVM/Random-Forest baselines.

Design note
-----------
Following the speech-lane house style established by ``speech_spectral`` —
"coverage is intentionally limited to interpretable scalar summaries" — this
analytic deliberately does NOT export the full time-resolved MFCC matrix or a
log-Mel spectrogram. Temporal dynamics (delta / delta-delta information) are
captured by the deep model's spectrogram input at training time, not stored as
near-constant scalar summaries (the mean of a first difference tends to ~0 over
an utterance and carries little discriminative signal).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from src.schemas.clvad_schema import AnalyticID, AnalyticOutput

try:
    import librosa as _librosa
    _LIBROSA_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-specific
    _librosa = None  # type: ignore[assignment]
    _LIBROSA_AVAILABLE = False

_ANALYTIC_ID = AnalyticID.SPEECH_MFCC
_HOP_LENGTH = 512
_N_FFT = 2048
_N_MFCC = 13


def run(processed_input: Any) -> AnalyticOutput:
    """Return per-coefficient MFCC summaries and ZCR summaries for speech audio."""
    skip_reason = _check_inputs(processed_input)
    if skip_reason is not None:
        return AnalyticOutput(
            analytic_id=_ANALYTIC_ID,
            status="skipped",
            fields={},
            skip_reason=skip_reason,
        )

    try:
        waveform = _prepare_waveform(processed_input.normalized_waveform)
        sample_rate = _resolve_sample_rate(processed_input)
        if not bool(np.any(waveform != 0.0)):
            return AnalyticOutput(
                analytic_id=_ANALYTIC_ID,
                status="skipped",
                fields={},
                skip_reason="speech_mfcc: normalized_waveform is all-zero",
            )

        fields: Dict[str, float] = {}

        mfccs = _librosa.feature.mfcc(
            y=waveform,
            sr=sample_rate,
            n_mfcc=_N_MFCC,
            n_fft=_N_FFT,
            hop_length=_HOP_LENGTH,
        )  # shape: (n_mfcc, n_frames)
        for coeff_index in range(_N_MFCC):
            coeff_number = coeff_index + 1
            row = mfccs[coeff_index]
            mean_value = _mean_or_none(row)
            std_value = _std_or_none(row)
            if mean_value is not None:
                fields[f"mfcc_{coeff_number}_mean"] = mean_value
            if std_value is not None:
                fields[f"mfcc_{coeff_number}_std"] = std_value

        zcr = _librosa.feature.zero_crossing_rate(
            y=waveform,
            frame_length=_N_FFT,
            hop_length=_HOP_LENGTH,
        )[0]
        zcr_mean = _mean_or_none(zcr)
        zcr_std = _std_or_none(zcr)
        if zcr_mean is not None:
            fields["zero_crossing_rate_mean"] = zcr_mean
        if zcr_std is not None:
            fields["zero_crossing_rate_std"] = zcr_std

        if not fields:
            return AnalyticOutput(
                analytic_id=_ANALYTIC_ID,
                status="skipped",
                fields={},
                skip_reason="speech_mfcc: no finite cepstral frames",
            )
        return AnalyticOutput(
            analytic_id=_ANALYTIC_ID,
            status="success",
            fields=fields,
        )
    except ValueError:
        raise
    except Exception as exc:  # pragma: no cover - defensive engine boundary
        return AnalyticOutput(
            analytic_id=_ANALYTIC_ID,
            status="failed",
            fields={},
            error_message=f"speech_mfcc: {type(exc).__name__}: {exc}",
        )


def _check_inputs(processed_input: Any) -> Optional[str]:
    if not _LIBROSA_AVAILABLE:
        return "speech_mfcc: librosa unavailable"
    waveform = getattr(processed_input, "normalized_waveform", None)
    if waveform is None:
        return "speech_mfcc: normalized_waveform unavailable"
    try:
        if len(waveform) == 0:
            return "speech_mfcc: normalized_waveform is empty"
    except TypeError:
        return "speech_mfcc: normalized_waveform is not array-like"
    if getattr(processed_input, "sample_rate", None) is None:
        return "speech_mfcc: sample_rate unavailable"
    return None


def _prepare_waveform(waveform: Any) -> np.ndarray:
    prepared = np.asarray(waveform, dtype=float)
    if prepared.ndim != 1:
        raise ValueError("normalized_waveform must be a mono one-dimensional array")
    if prepared.size == 0:
        raise ValueError("normalized_waveform must contain at least one sample")
    if not bool(np.all(np.isfinite(prepared))):
        raise ValueError("normalized_waveform contains non-finite samples")
    return prepared


def _resolve_sample_rate(processed_input: Any) -> int:
    sample_rate = int(getattr(processed_input, "sample_rate"))
    if sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    return sample_rate


def _mean_or_none(values: Any) -> Optional[float]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def _std_or_none(values: Any) -> Optional[float]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return None
    return float(np.std(finite))


__all__ = ["run", "_mean_or_none", "_std_or_none"]
