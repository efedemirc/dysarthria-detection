"""Unit tests for the speech_mfcc temporal analytic."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from src.features.temporal import speech_mfcc
from src.schemas.clvad_schema import AnalyticID


def _sine(sample_rate: int = 16000, freq: float = 220.0, seconds: float = 1.0) -> np.ndarray:
    time_axis = np.linspace(0.0, seconds, int(sample_rate * seconds), endpoint=False)
    return np.sin(2.0 * np.pi * freq * time_axis).astype(float)


class TestSpeechMfcc(unittest.TestCase):
    def test_sine_wave_produces_all_28_fields(self) -> None:
        if not speech_mfcc._LIBROSA_AVAILABLE:
            self.skipTest("librosa unavailable")
        sample_rate = 16000
        result = speech_mfcc.run(SimpleNamespace(
            normalized_waveform=_sine(sample_rate),
            sample_rate=sample_rate,
        ))

        self.assertEqual(result.status, "success")
        self.assertEqual(result.analytic_id, AnalyticID.SPEECH_MFCC)
        self.assertEqual(len(result.fields), 28)
        for i in range(1, 14):
            self.assertIn(f"mfcc_{i}_mean", result.fields)
            self.assertIn(f"mfcc_{i}_std", result.fields)
            self.assertGreaterEqual(result.fields[f"mfcc_{i}_std"], 0.0)
        self.assertIn("zero_crossing_rate_mean", result.fields)
        self.assertIn("zero_crossing_rate_std", result.fields)

    def test_mfcc_mean_can_be_negative(self) -> None:
        # MFCC means are not bounded below at zero; the schema must allow this.
        if not speech_mfcc._LIBROSA_AVAILABLE:
            self.skipTest("librosa unavailable")
        result = speech_mfcc.run(SimpleNamespace(
            normalized_waveform=_sine(),
            sample_rate=16000,
        ))
        self.assertEqual(result.status, "success")
        # The 0th cepstral coefficient (energy term) is typically strongly negative
        # for a low-amplitude normalized sine; at minimum, some mean is < 0.
        means = [v for k, v in result.fields.items() if k.endswith("_mean") and k.startswith("mfcc_")]
        self.assertTrue(any(m < 0.0 for m in means))

    def test_zcr_mean_in_unit_interval(self) -> None:
        if not speech_mfcc._LIBROSA_AVAILABLE:
            self.skipTest("librosa unavailable")
        result = speech_mfcc.run(SimpleNamespace(
            normalized_waveform=_sine(),
            sample_rate=16000,
        ))
        zcr = result.fields["zero_crossing_rate_mean"]
        self.assertGreaterEqual(zcr, 0.0)
        self.assertLessEqual(zcr, 1.0)

    def test_all_zero_waveform_skips_without_null_fields(self) -> None:
        result = speech_mfcc.run(SimpleNamespace(
            normalized_waveform=np.zeros(4096, dtype=float),
            sample_rate=16000,
        ))
        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.fields, {})
        self.assertIn("all-zero", result.skip_reason)

    def test_missing_sample_rate_skips(self) -> None:
        result = speech_mfcc.run(SimpleNamespace(
            normalized_waveform=np.ones(4096, dtype=float),
            sample_rate=None,
        ))
        self.assertEqual(result.status, "skipped")
        self.assertIn("sample_rate unavailable", result.skip_reason)

    def test_deterministic_for_identical_input(self) -> None:
        if not speech_mfcc._LIBROSA_AVAILABLE:
            self.skipTest("librosa unavailable")
        wave = _sine(seconds=0.5)
        first = speech_mfcc.run(SimpleNamespace(normalized_waveform=wave, sample_rate=16000))
        second = speech_mfcc.run(SimpleNamespace(normalized_waveform=wave, sample_rate=16000))
        self.assertEqual(first.fields, second.fields)


if __name__ == "__main__":
    unittest.main()
