from cog import BasePredictor, Path, Input
from pathlib import Path as PathlibPath
import torchaudio
import tempfile
import base64
from pipeline_ace_step import ACEStepPipeline


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory once when the container starts"""
        self.model = ACEStepPipeline()

    def predict(
        self,
        src_audio: Path = Input(description="Input audio file to edit"),
        original_prompt: str = Input(
            description="Original prompt used for generating the song",
            default="",
        ),
        new_lyrics: str = Input(description="New lyrics to guide the edit", default=""),
        edit_mode: str = Input(
            description="Edit mode: only_lyrics or remix",
            choices=["only_lyrics", "remix"],
            default="only_lyrics",
        ),
        edit_n_min: float = Input(
            description="Edit strength minimum (lower is more aggressive)", default=None
        ),
        edit_n_max: float = Input(description="Edit strength maximum", default=None),
        format: str = Input(
            description="Output format", choices=["wav", "flac"], default="wav"
        ),
    ) -> Path:
        """Run an edit on the model"""

        # Load duration
        waveform, sample_rate = torchaudio.load(src_audio)
        duration = waveform.shape[1] / sample_rate

        # Set edit strength defaults
        if edit_mode == "only_lyrics":
            edit_n_min = 0.6 if edit_n_min is None else edit_n_min
            edit_n_max = 1.0 if edit_n_max is None else edit_n_max
        elif edit_mode == "remix":
            edit_n_min = 0.2 if edit_n_min is None else edit_n_min
            edit_n_max = 0.4 if edit_n_max is None else edit_n_max

        # Run inference
        output_paths = self.model(
            task="edit",
            prompt="",
            lyrics="",
            src_audio_path=str(src_audio),
            edit_target_prompt=str(original_prompt),
            edit_target_lyrics=str(new_lyrics),
            edit_n_min=edit_n_min,
            edit_n_max=edit_n_max,
            audio_duration=duration,
            infer_step=60,
            guidance_scale=15.0,
            batch_size=1,
            save_path="/tmp",
            format=format,
        )

        return (
            Path(output_paths[0])
            if isinstance(output_paths[0], PathlibPath)
            else Path(PathlibPath(output_paths[0]))
        )
