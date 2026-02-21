import sys
import os
import numpy as np
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def make_fake_audio(duration_sec=2, sr=22050):
    t = np.linspace(0, duration_sec, int(sr * duration_sec))
    y = (np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return y, sr


class TestExtractFeatures:

    @patch("librosa.load")
    @patch("librosa.beat.beat_track")
    @patch("librosa.feature.mfcc")
    @patch("librosa.feature.chroma_stft")
    @patch("librosa.feature.spectral_centroid")
    @patch("librosa.feature.zero_crossing_rate")
    def test_returns_numpy_array(self, mock_zcr, mock_sc, mock_chroma, mock_mfcc, mock_beat, mock_load):
        y, sr = make_fake_audio()
        mock_load.return_value = (y, sr)
        mock_beat.return_value = (np.array([120.0]), np.array([10]))
        mock_mfcc.return_value = np.random.rand(13, 100)
        mock_chroma.return_value = np.random.rand(12, 100)
        mock_sc.return_value = np.random.rand(1, 100)
        mock_zcr.return_value = np.random.rand(1, 100)

        from features import extract_features
        result = extract_features("fake.wav")

        assert isinstance(result, np.ndarray)
        assert result.ndim == 1

    @patch("librosa.load")
    @patch("librosa.beat.beat_track")
    @patch("librosa.feature.mfcc")
    @patch("librosa.feature.chroma_stft")
    @patch("librosa.feature.spectral_centroid")
    @patch("librosa.feature.zero_crossing_rate")
    def test_feature_vector_length(self, mock_zcr, mock_sc, mock_chroma, mock_mfcc, mock_beat, mock_load):
        y, sr = make_fake_audio()
        mock_load.return_value = (y, sr)
        mock_beat.return_value = (np.array([120.0]), np.array([10]))
        mock_mfcc.return_value = np.random.rand(13, 100)
        mock_chroma.return_value = np.random.rand(12, 100)
        mock_sc.return_value = np.random.rand(1, 100)
        mock_zcr.return_value = np.random.rand(1, 100)

        from features import extract_features
        result = extract_features("fake.wav")

        assert len(result) == 41

    @patch("librosa.load", side_effect=Exception("Dosya bulunamadı"))
    def test_raises_on_bad_file(self, mock_load):
        from features import extract_features

        with pytest.raises(Exception):
            extract_features("olmayan.wav")