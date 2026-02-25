import librosa
import numpy as np
import pretty_midi
from scipy.signal import find_peaks
from dataclasses import dataclass

@dataclass
class TranscriberParams:
    """
    Master configuration object containing ALL tunable parameters.
    """
    # Audio Processing
    sr: int = 22050
    hop_length: int = 256

    # CQT (Frequency Analysis)
    fmin_note: str = 'A0'
    bins_per_octave: int = 36

    # Onset Detection
    onset_delta: float = 0.07 # Threshold for onset det. lower means more sensitive
    onset_backtrack: bool = True
    pre_max: int = 3
    post_max: int = 3
    pre_avg: int = 3
    post_avg: int = 3
    wait: int = 1

    # Peak Picking
    slice_window: int = 7                   # Frames to check after onset
    peak_height_db: float = -30             # Minimum volume to consider note
    peak_prominence: float = 10             # How much peak must stick out from noise
    peak_distance_semitones: float = 0.9    # Minimum distance between notes (semitones)
    
    # ReTrigger Protection
    energy_slope_thresh: float = 2.0
    max_polyphony: int = 7                 # How many notes can start at most on the same onset

    # Harmonic Filtering
    harmonic_threshold: float = 5           # If harmonic quiter than fund. freq. remove it
    harmonic_3rd_threshold: float = 10
    harmonic_5th_threshold: float = 8
    freq_tolerance: float = 0.05            # tolerance for inharmonicity

    # Offset Detection
    abs_threshold_db: float = -55           # Silence threshold
    rel_threshold_db: float = 20            # How many dB's must it drom from peak to stop
    min_duration_sec: float = 0.2
    max_duration_sec: float = 3.0

    # Velocity
    vel_min_db: float = -30
    vel_max_db: float = 0
    vel_min_midi: int = 40
    vel_max_midi: int = 120

    # Cleaning
    skip_cleaning: bool = False
    global_polyphony_limit: int = 12
    global_polyphony_timestep: float = 0.1
    
    # MIDI
    midi_program: int = 0 # Acoustic Grand Piano


class PianoTranscriber:
    def __init__(self, audio_path, params: TranscriberParams = None):
        self.audio_path = audio_path
        # If no params given, load the default ones
        self.params = params if params else TranscriberParams()

        self.y = None
        self.y_detection = None
        self.cqt_db = None
        self.onset_frames = []
        self.onset_times = []
        self.notes = []

    def load_and_preprocess(self):
        print(f"Loading {self.audio_path}...")
        # Load Audio
        raw_y, _ = librosa.load(self.audio_path, sr=self.params.sr)
        # Clean audio for Pitch Detection
        clean_max = np.percentile(np.abs(raw_y), 99.0)
        if clean_max > 0:
            self.y = np.clip(raw_y / clean_max, -1.0, 1.0)
        else:
            self.y = librosa.util.normalize(raw_y)

        # Compute CQT on clean y
        # CQT (High resolution for pitch accuracy)
        # 8 octaves * bins per octave
        n_bins = 8 * self.params.bins_per_octave

        cqt = np.abs(librosa.cqt(self.y,
                                 sr=self.params.sr,
                                 hop_length=self.params.hop_length,
                                 n_bins=n_bins,
                                 bins_per_octave=self.params.bins_per_octave,
                                 fmin=librosa.note_to_hz(self.params.fmin_note)))
        self.cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)

        # Hard normalization for onset detection
        detect_max = np.percentile(np.abs(raw_y), 90.0)
        if detect_max > 0:
            self.y_detection = np.clip(raw_y / detect_max, -1.0, 1.0)
        else:
            self.y_detection = self.y

    def detect_onsets(self):
        print("Detecting Onsets...")
        onset_envelope = librosa.onset.onset_strength(
            y=self.y_detection,
            sr=self.params.sr,
            hop_length=self.params.hop_length,
            aggregate=np.mean
        )
        # Find the peaks in the previous novelty function. These are the note onset times
        self.onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_envelope,
            sr=self.params.sr,
            hop_length=self.params.hop_length,
            backtrack=self.params.onset_backtrack,
            delta=self.params.onset_delta,
            pre_max=self.params.pre_max,
            post_max=self.params.post_max,
            pre_avg=self.params.pre_avg,
            post_avg=self.params.post_avg,
            wait=self.params.wait
        )
        self.onset_times = librosa.frames_to_time(
            self.onset_frames,
            sr=self.params.sr,
            hop_length=self.params.hop_length
        )

    def filter_harmonics(self, peaks, spectrum_slice):
        """
        Removes peaks that appear to be harmonics
        """
        if len(peaks) < 2:
            return peaks

        fmin = librosa.note_to_hz(self.params.fmin_note)
        peak_freqs = fmin * (2 ** (peaks / self.params.bins_per_octave))
        # List of peaks to keep
        # True = Keep, False = Remove
        keep_mask = np.ones(len(peaks), dtype=bool)

        # Double check: erase both normal (higher) and sub (lower) harmonics
        for i in range(len(peaks)):
            if not keep_mask[i]: continue

            freq_i = peak_freqs[i]
            amp_i = spectrum_slice[peaks[i]]

            for j in range(i + 1, len(peaks)):
                if not keep_mask[j]: continue

                freq_j = peak_freqs[j]
                amp_j = spectrum_slice[peaks[j]]

                ratio = freq_j / freq_i
                closest_integer = round(ratio)

                # If ratio is close to an integer
                if abs(ratio - closest_integer) < self.params.freq_tolerance:
                    if closest_integer == 3:
                        threshold = self.params.harmonic_3rd_threshold
                    elif closest_integer == 5:
                        threshold = self.params.harmonic_5th_threshold
                    else:
                        threshold = self.params.harmonic_threshold
                    
                    if amp_i - amp_j > threshold:
                        keep_mask[j] = False

        return peaks[keep_mask]

    def detect_offset_and_save(self, note_data):
        """
        Tracks the note decay to find the end time and calculates velocity
        """
        start_frame = note_data['start_frame']
        bin_idx = note_data['bin_idx']
        initial_energy = note_data['initial_energy']

        max_frames = int(self.params.max_duration_sec * self.params.sr / self.params.hop_length)
        current_frame = start_frame + 1
        end_frame = start_frame + 1

        # Forward tracking: stop if threshold passed OR max time is out
        while current_frame < self.cqt_db.shape[1] and (current_frame - start_frame) < max_frames:
            # Take the energy on that bin and the neighbouring ones (in case frequency moves slightly)
            b_min = max(0, bin_idx - 1)
            b_max = min(self.cqt_db.shape[0], bin_idx + 2)
            energy_slice = self.cqt_db[ b_min:b_max, current_frame]
            current_energy = np.max(energy_slice) if len(energy_slice) > 0 else -80

            # Stop conditions
            # Condition 1: Absolute silence
            cond1 = current_energy < self.params.abs_threshold_db
            # Condition 2: Relative silence
            cond2 = current_energy < (initial_energy - self.params.rel_threshold_db)
            if cond1 or cond2:
                end_frame = current_frame
                break
            # Continue on next frame
            end_frame = current_frame
            current_frame += 1

        end_time = librosa.frames_to_time(end_frame, sr=self.params.sr, hop_length=self.params.hop_length)
        duration = end_time - note_data['start_time']
        # Delete very short notes
        if duration < self.params.min_duration_sec:
            return

        # Velocity detection
        # Map dB range to velocity (linear interpolation)
        velocity = np.interp(
            initial_energy,
            [self.params.vel_min_db, self.params.vel_max_db],
            [self.params.vel_min_midi, self.params.vel_max_midi]
        )
        final_velocity = int(np.clip(velocity, 0, 127))
        final_pitch = int(np.clip(note_data['midi_note'], 0, 127))
        
        # Create PrettyMidi note
        note = pretty_midi.Note(
            velocity=final_velocity,
            pitch=final_pitch,
            start=float(note_data['start_time']),
            end=float(end_time)
        )

        # Append to self.notes
        self.notes.append(note)

    def extract_notes(self):
        print("Extracting notes from detected onsets...")
        bins_per_semitone = self.params.bins_per_octave / 12
        calc_distance = max(1, int(self.params.peak_distance_semitones * bins_per_semitone))

        for i, start_frame in enumerate(self.onset_frames):
            # Check boundaries
            window_end = min(start_frame + self.params.slice_window, self.cqt_db.shape[1])
            cqt_window = self.cqt_db[:, start_frame:window_end]
            onset_spectrum = np.max(cqt_window, axis=1)
            raw_peaks, _ = find_peaks(
                onset_spectrum,
                height=self.params.peak_height_db,
                prominence=self.params.peak_prominence,
                distance=calc_distance
            )

            valid_peaks = []
            slope_window = 3
            for bin_idx in raw_peaks:
                prev_start = max(0, start_frame - slope_window)
                post_end = min(self.cqt_db.shape[1], start_frame + slope_window)

                # Accept the note if its on the start of the song
                if start_frame < slope_window:
                    valid_peaks.append(bin_idx)
                    continue

                # Calculate avg energy before and after onset
                pre_energy = np.mean(self.cqt_db[bin_idx, prev_start:start_frame])
                post_energy = np.mean(self.cqt_db[bin_idx, start_frame:post_end])
                diff = post_energy - pre_energy
                if diff > self.params.energy_slope_thresh:
                    valid_peaks.append(bin_idx)

            filtered_peaks = np.array(valid_peaks)

            # Filter harmonics
            peaks = self.filter_harmonics(filtered_peaks, onset_spectrum)

            # Max polyphony constraint
            if len(peaks) > self.params.max_polyphony:
                # Find the dB of the peaks selected
                peak_amps = onset_spectrum[peaks]
                # Sort indices wrt amplitude low to high
                sorted_order = np.argsort(peak_amps)
                # keep the last (loudest)
                top_indices = sorted_order[-self.params.max_polyphony:]
                # Update peaks
                peaks = peaks[top_indices]
                # Re sort wrt frequency
                peaks = np.sort(peaks)

            # Process found peaks
            for bin_idx in peaks:
                # Convert Bin in MIDI note num.
                note_float = 21 + (bin_idx / bins_per_semitone)
                midi_note = int(np.round(note_float))
                # Check boundaries
                if not (21 <= midi_note <= 108):
                    continue

                note_data = {
                    'midi_note': midi_note,
                    'start_frame': start_frame,
                    'start_time': librosa.frames_to_time(start_frame, sr=self.params.sr, hop_length=self.params.hop_length),
                    'initial_energy': onset_spectrum[bin_idx],
                    'bin_idx': bin_idx # to track decay
                }

                self.detect_offset_and_save(note_data)

    def clean_global_polyphony(self):
        print("Cleaning MIDI file...")
        if self.params.skip_cleaning:
            return
        limit = self.params.global_polyphony_limit
        if limit is None or not self.notes:
            return
        self.notes.sort(key=lambda x: x.start)
        notes_to_remove = set()
        last_time = max(n.end for n in self.notes)
        time_step = self.params.global_polyphony_timestep
        current_time = 0.0

        while current_time < last_time:
            active_notes = [n for n in self.notes if n.start <= current_time < n.end]
            if len(active_notes) > limit:
                excess_count = len(active_notes) - limit
                active_notes.sort(key=lambda x: (x.start > current_time - 0.1, x.velocity))
                for i in range(excess_count):
                    notes_to_remove.add(active_notes[i])
            current_time += time_step
        
        original_count = len(self.notes)
        self.notes = [n for n in self.notes if n not in notes_to_remove]
        print(f"Removed {original_count - len(self.notes)} notes due to global polyphony.")

    def save_midi(self, output_filename):
        # Create a prettyMIDI object
        pm = pretty_midi.PrettyMIDI()

        # Create instrument
        instrument = pretty_midi.Instrument(program=self.params.midi_program)

        # Add notes to instrument
        sorted_notes = sorted(self.notes, key=lambda x: x.start)
        instrument.notes.extend(sorted_notes)

        pm.instruments.append(instrument)

        # Write to MIDI file
        try:
            pm.write(output_filename)
            print(f"MIDI successfully saved in: {output_filename}")
        except Exception as e:
            print(f"Error writing MIDI file: {e}")


# --- WRAPPER FUNCTION (For easy integration) ---
def transcribe_audio_dsp(audio_path, output_midi_path, params=None):
    """
    High-level function to run the entire DSP transcription pipeline.
    This makes it easy to call from gui/app.py.
    """
    try:
        transcriber = PianoTranscriber(audio_path, params=params)
        transcriber.load_and_preprocess()
        transcriber.detect_onsets()
        transcriber.extract_notes()
        transcriber.clean_global_polyphony()
        transcriber.save_midi(output_midi_path)
        return True
    except Exception as e:
        print(f"DSP Transcription failed: {e}")
        return False