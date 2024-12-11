import numpy as np

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
    
    def smooth_srp_map(self, srp_map, window_size:int=5):
        """
        Apply a moving average to the SRP map to smooth it.
        """
        window = np.ones(window_size) / window_size
        # TODO: Might have to use 2D moving average
        # return convolve2d(srp_map, kernel, mode='same')
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
            list: List of audio segments corresponding to the top peaks.
        """
        # Get the top n peak indices from the SRP map
        top_indices = np.argsort(srp_map)[-n_best:]
        top_candidates = candidate_grid[top_indices]

        audio_segments = []
        half_segment_samples = int((segment_duration * self.fs) / 2)

        for candidate in top_candidates:
            for mic_signal in mic_signals:
                # Calculate the delay for this candidate and microphone
                # Only need to calculate the delay from one microphone position to approximate the delay as the srp_map implicitly accounts for the delays
                delay = np.linalg.norm(candidate - self.mic_positions[0]) / self.c
                delay_samples = int(delay * self.fs)

                # Extract the segment around the delay
                start = max(0, delay_samples - half_segment_samples)
                end = min(len(mic_signal), delay_samples + half_segment_samples)
                segment = mic_signal[start:end]
                audio_segments.append(segment)

        return audio_segments
    
    # def apply_gsc(self, mic_signals, top_candidates, mic_positions):
    #     """
    #     Apply a Generalized Sidelobe Canceller (GSC) to focus on the selected n best candidate locations.
    #     This will boost the signals from the desired locations and reduce the signals from other locations.
    #     """
    #     # Calculate the delay for each microphone signal based on the top candidate's position
    #     delays = [np.linalg.norm(mic_position - top_candidates[0]) / self.c for mic_position in mic_positions]

    #     # Apply the delay to each microphone signal
    #     delayed_signals = [np.roll(mic_signal, int(delay * self.fs)) for mic_signal, delay in zip(mic_signals, delays)]

    #     # Beamformer: sum the delayed signals
    #     beamformer_output = np.sum(delayed_signals, axis=0)

    #     # Blocking matrix: subtract the mean of the delayed signals from each delayed signal
    #     mean_signal = np.mean(delayed_signals, axis=0)
    #     blocking_matrix_output = [delayed_signal - mean_signal for delayed_signal in delayed_signals]

    #     # GSC output: subtract the blocking matrix output from the beamformer output
    #     gsc_output = beamformer_output - np.sum(blocking_matrix_output, axis=0)

    #     return gsc_output

    def forward(
        self, mic_signals, mic_positions=None, room_dims=None, n_best:int=4
    ) -> tuple[set, np.array]:
        if mic_positions is None:
            mic_positions = self.mic_positions
        if room_dims is None:
            room_dims = self.room_dims

        if mic_positions is None:
            raise ValueError(
                """
                mic_positions and room_dims must be specified
                either in the constructor or in the forward method
                """
            )

        candidate_grid = self.candidate_grid

        estimated_positions = np.array([])

        # 1. Compute the signal features (usually, GCC-PHAT)
        signal_features = self.compute_signal_features(mic_signals)

        while True:
            # 2. Project the signal features into space, i.e., create the SRP map
            srp_map = self.create_srp_map(
                mic_positions, candidate_grid, signal_features
            )

            top_indices = np.argsort(srp_map)[-n_best:]
            top_candidates = candidate_grid[top_indices]

            audio_segments  = self.align_peaks_with_audio(srp_map=srp_map, mic_signals=mic_signals, candidate_grid=candidate_grid)

            
            srp_map = self.smooth_srp_map(srp_map=srp_map)

            # Apply GSC to focus on the selected n best candidate locations
            srp_map = self.apply_gsc(mic_signals, top_candidates, mic_positions)

            # 



            # 3. Find the source position in the SRP map
            estimated_positions, new_candidate_grid, signal_features = self.grid_search(
                candidate_grid, srp_map, estimated_positions, signal_features
            )

            # 4. Update the candidate grid
            if len(new_candidate_grid) == 0:
                # If the new candidate grid is empty, we have found all the sources
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
