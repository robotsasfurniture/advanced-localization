import numpy as np
import torch
import torchaudio
from abc import ABC, abstractmethod

from xsrp_project.xsrp.grids import Grid

class XSrp(ABC):
    def __init__(self, fs: float, mic_positions=None, room_dims=None, c=343.0):
        self.fs = fs
        self.mic_positions = mic_positions
        self.room_dims = room_dims
        self.c = c

        self.n_mics = len(mic_positions)

        # 0. Create the initial grid of candidate positions
        self.candidate_grid = self.create_initial_candidate_grid(room_dims)

        # ---- Load Silero VAD model here ----
        self.vad_model, self.vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            source='github'
        )
        (self.get_speech_timestamps, _, _, _) = self.vad_utils

    def smooth_srp_map(self, srp_map, window_size:int=5):
        """
        Apply a moving average to the SRP map to smooth it.
        Since srp_map might be 1D or a flattened 2D, adjust as needed.
        """
        window = np.ones(window_size) / window_size
        return np.convolve(srp_map, window, 'valid')

    def apply_gsc(self, data, peaks, sidelobe_reduction=0.5):
        """
        Apply Generalized Sidelobe Canceller to boost signal at peaks and reduce elsewhere.
        """
        mask = np.zeros_like(data)
        for peak in peaks:
            mask[peak] = 1

        mask = self.smooth_srp_map(mask, window_size=10)  # Smooth the mask to create transition regions
        return data * (1 + mask * sidelobe_reduction)

    def align_peaks_with_audio(self, srp_map, mic_signals, candidate_grid, n_best=4, segment_duration=0.5):
        """
        Align top peaks in the SRP map with audio segments from the microphone signals.

        Args:
            srp_map (np.array): The SRP map values.
            mic_signals (list): List of microphone signals.
            candidate_grid (Grid): The spatial candidate grid.
            n_best (int): Number of top peaks to align.
            segment_duration (float): Duration of the audio segment in seconds.

        Returns:
            tuple: A tuple containing the top indices and the audio segments.
        """
        # Get the top n peak indices from the SRP map
        top_indices = np.argsort(srp_map)[-n_best:]
        top_candidates = candidate_grid[top_indices]

        audio_segments = []
        half_segment_samples = int((segment_duration * self.fs) / 2)

        for candidate in top_candidates:
            for mic_signal in mic_signals:
                # Calculate the delay for this candidate and microphone
                delay = np.linalg.norm(candidate - self.mic_positions[0]) / self.c
                delay_samples = int(delay * self.fs)

                # Extract the segment around the delay
                start = max(0, delay_samples - half_segment_samples)
                end = min(len(mic_signal), delay_samples + half_segment_samples)
                segment = mic_signal[start:end]
                audio_segments.append(segment)

        return top_indices, audio_segments

    def classify_candidates_with_vad(self, audio_segments):
        """
        Classify each candidate as speech or non-speech using silero-vad.

        Args:
            audio_segments (list): List of audio segments aligned with SRP map peaks.

        Returns:
            list: A list of binary labels (True for speech, False for non-speech) for each segment.
        """
        vad_labels = []

        for segment in audio_segments:
            # Convert the audio segment to a torch.Tensor if it is not already
            if not isinstance(segment, torch.Tensor):
                segment = torch.from_numpy(segment.astype(np.float32))

            # Apply VAD
            speech_timestamps = self.get_speech_timestamps(segment, self.vad_model, sampling_rate=self.fs)

            # Label segment as speech or non-speech based on VAD
            is_speech = True if len(speech_timestamps) > 0 else False
            vad_labels.append(is_speech)

        return vad_labels

    def forward(
        self, mic_signals, mic_positions=None, room_dims=None, n_best:int=4
    ) -> tuple[np.array, np.array, Grid]:
        if mic_positions is None:
            mic_positions = self.mic_positions
        if room_dims is None:
            room_dims = self.room_dims

        if mic_positions is None:
            raise ValueError(
                "mic_positions and room_dims must be specified either in the constructor or in the forward method"
            )

        candidate_grid = self.candidate_grid

        estimated_positions = np.array([])

        # 1. Compute the signal features (e.g., GCC-PHAT)
        signal_features = self.compute_signal_features(mic_signals)

        while True:
            # 2. Create the SRP map
            srp_map = self.create_srp_map(
                mic_positions, candidate_grid, signal_features
            )

            # Smooth SRP map
            srp_map = self.smooth_srp_map(srp_map=srp_map)

            # Align peaks with audio
            top_indices, audio_segments = self.align_peaks_with_audio(srp_map, mic_signals, candidate_grid, n_best=n_best, segment_duration=0.5)

            # Apply GSC
            srp_map = self.apply_gsc(srp_map, top_indices)

            # ---- Integrate VAD Classification ----
            # Classify the top candidates as speech or non-speech (True as speech, False as non speech)
            vad_labels = self.classify_candidates_with_vad(audio_segments)

            # Here you can use vad_labels to filter non-speech candidates
            # For example, remove candidates that are non-speech or reduce their SRP scores
            # This is just an example:
            for idx, label in enumerate(vad_labels):
                if not label:
                    # Reduce SRP score for non-speech candidates
                    srp_map[top_indices[idx]] *= 0.1

            # 3. Grid search step (refine candidate grid)
            estimated_positions, new_candidate_grid, signal_features = self.grid_search(
                candidate_grid, srp_map, estimated_positions, signal_features
            )

            # 4. Update candidate grid
            if len(new_candidate_grid) == 0:
                # If no new candidates, we're done
                break
            else:
                candidate_grid = new_candidate_grid

        return estimated_positions, srp_map, candidate_grid

    @abstractmethod
    def compute_signal_features(self, mic_signals):
        pass

    @abstractmethod
    def create_initial_candidate_grid(self, room_dims):
        pass

    @abstractmethod
    def create_srp_map(
        self, mic_positions: np.array, candidate_grid: Grid, signal_features: np.array
    ):
        pass

    @abstractmethod
    def grid_search(
        self, candidate_grid, srp_map, estimated_positions, signal_features
    ) -> tuple[np.array, np.array]:
        pass