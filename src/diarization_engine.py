"""
Diarization Engine - Speaker diarization using pyannote-audio
"""
from typing import List, Tuple, Optional, Callable


class DiarizationEngine:
    """Wraps pyannote-audio for speaker diarization and alignment with whisper segments"""

    def __init__(self, hf_token: str, device: str = "auto"):
        self.hf_token = hf_token
        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        self.pipeline = None

    PIPELINE_ID = "pyannote/speaker-diarization-3.1"

    @classmethod
    def _token_help(cls, exc) -> str:
        return (
            f"could not load the pyannote diarization pipeline: {exc}\n"
            "pyannote models are gated on Hugging Face - one-time setup:\n"
            "  1. Create a free token at https://huggingface.co/settings/tokens\n"
            f"  2. Accept the conditions at https://huggingface.co/{cls.PIPELINE_ID}\n"
            "     and https://huggingface.co/pyannote/segmentation-3.0\n"
            "  3. Put the token in settings.json as \"hf_token\"")

    def _load_pipeline(self):
        """Load the pyannote speaker-diarization pipeline (lazy, first call only)."""
        if self.pipeline is not None:
            return

        import torch
        from pyannote.audio import Pipeline

        print(f"Loading {self.PIPELINE_ID} pipeline...")
        try:
            # pyannote.audio >= 4 takes token=; None falls back to any
            # cached `hf auth login` credential.
            pipeline = Pipeline.from_pretrained(
                self.PIPELINE_ID, token=self.hf_token or None)
        except TypeError:
            # pyannote.audio < 4 named the parameter use_auth_token
            pipeline = Pipeline.from_pretrained(
                self.PIPELINE_ID, use_auth_token=self.hf_token or True)
        except Exception as exc:
            raise RuntimeError(self._token_help(exc)) from exc
        if pipeline is None:
            # from_pretrained returns None when the gated-model conditions
            # have not been accepted for this token
            raise RuntimeError(self._token_help("access not granted"))
        self.pipeline = pipeline

        if self.device == "cuda" and torch.cuda.is_available():
            self.pipeline.to(torch.device("cuda"))

        print("Diarization pipeline loaded.")

    def diarize(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> List[Tuple[float, float, str]]:
        """
        Run speaker diarization on an audio file.

        Args:
            audio_path: Path to the audio file.
            num_speakers: Optional hint for number of speakers.
            progress_callback: Optional callback for status updates.

        Returns:
            List of (start_seconds, end_seconds, speaker_label) turns.
        """
        self._load_pipeline()

        if progress_callback:
            progress_callback("Running speaker diarization...")

        kwargs = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers

        diarization = self.pipeline(audio_path, **kwargs)
        # pyannote 4 pipelines return an output object holding the
        # Annotation in .speaker_diarization; pyannote 3 returned the
        # Annotation directly.
        annotation = getattr(diarization, "speaker_diarization", diarization)

        turns: List[Tuple[float, float, str]] = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            turns.append((turn.start, turn.end, speaker))

        if progress_callback:
            progress_callback(f"Diarization complete: {len(turns)} turns, "
                              f"{len(set(t[2] for t in turns))} speakers")

        return turns

    @staticmethod
    def align_with_transcript(
        whisper_segments: List[dict],
        diarization_turns: List[Tuple[float, float, str]],
    ) -> List[dict]:
        """
        Assign speaker labels to whisper segments by maximum temporal overlap.

        Args:
            whisper_segments: List of dicts with 'start', 'end', 'text' keys.
            diarization_turns: List of (start, end, speaker_label) from diarize().

        Returns:
            New list of segment dicts with added 'speaker' key.
        """
        result = []
        for seg in whisper_segments:
            seg_start = seg["start"]
            seg_end = seg["end"]
            best_speaker = "UNKNOWN"
            best_overlap = 0.0

            for turn_start, turn_end, speaker in diarization_turns:
                overlap_start = max(seg_start, turn_start)
                overlap_end = min(seg_end, turn_end)
                overlap = max(0.0, overlap_end - overlap_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = speaker

            result.append({
                **seg,
                "speaker": best_speaker,
            })

        return result

    @staticmethod
    def merge_consecutive(segments: List[dict]) -> List[dict]:
        """
        Merge adjacent segments that share the same speaker.

        Args:
            segments: List of dicts with 'start', 'end', 'text', 'speaker'.

        Returns:
            Merged list of segment dicts.
        """
        if not segments:
            return []

        merged = [segments[0].copy()]
        for seg in segments[1:]:
            prev = merged[-1]
            if seg["speaker"] == prev["speaker"]:
                prev["end"] = seg["end"]
                prev["text"] = prev["text"] + " " + seg["text"]
            else:
                merged.append(seg.copy())

        return merged
