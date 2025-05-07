import fal
import base64
import tempfile
import torchaudio
from pathlib import Path
from pydantic import BaseModel, Field


class GenerateInput(BaseModel):
    prompt: str
    lyrics: str = ""
    duration: float = 60.0


class EditInput(BaseModel):
    src_audio_base64: str
    original_prompt: str
    new_lyrics: str = ""
    edit_mode: str = "only_lyrics"
    edit_n_min: float | None = None
    edit_n_max: float | None = None


class RepaintInput(BaseModel):
    src_audio_base64: str
    original_prompt: str
    lyrics: str = ""
    repaint_start: float = 0.0
    repaint_end: float = 10.0
    retake_variance: float = 0.5


class MusicOutput(BaseModel):
    base64_audio: str
    mime_type: str
    metadata: dict


class ACEStepApp(fal.App, name="ace-step-music", keep_alive=300):
    machine_type = "GPU-A100"
    requirements = [
        "fal",
        "torchaudio",
        # Add any other dependencies here
    ]

    def setup(self):
        from pipeline_ace_step import ACEStepPipeline

        self.model = ACEStepPipeline()

    def encode_audio(self, audio_path: str) -> tuple[str, str]:
        ext = Path(audio_path).suffix.lstrip(".")
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        return base64.b64encode(audio_bytes).decode("utf-8"), f"audio/{ext}"

    def save_audio(self, base64_audio: str) -> tuple[str, float]:
        tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_file.write(base64.b64decode(base64_audio))
        tmp_file.close()
        waveform, sample_rate = torchaudio.load(tmp_file.name)
        duration = waveform.shape[1] / sample_rate
        return tmp_file.name, duration

    @fal.endpoint("/generate")
    def generate(self, request: GenerateInput) -> MusicOutput:
        output_paths = self.model(
            task="text2music",
            prompt=request.prompt,
            lyrics=request.lyrics,
            audio_duration=request.duration,
            infer_step=60,
            guidance_scale=15.0,
            batch_size=1,
            save_path="./outputs",
            format="flac",
        )
        base64_audio, mime = self.encode_audio(output_paths[0])
        return MusicOutput(
            base64_audio=base64_audio,
            mime_type=mime,
            metadata={"duration": request.duration, "mode": "generate"},
        )

    @fal.endpoint("/edit")
    def edit(self, request: EditInput) -> MusicOutput:
        src_path, duration = self.save_audio(request.src_audio_base64)
        if request.edit_mode == "only_lyrics":
            n_min = 0.6 if request.edit_n_min is None else request.edit_n_min
            n_max = 1.0 if request.edit_n_max is None else request.edit_n_max
        else:
            n_min = 0.2 if request.edit_n_min is None else request.edit_n_min
            n_max = 0.4 if request.edit_n_max is None else request.edit_n_max

        output_paths = self.model(
            task="edit",
            src_audio_path=src_path,
            edit_target_prompt=request.original_prompt,
            edit_target_lyrics=request.new_lyrics,
            edit_n_min=n_min,
            edit_n_max=n_max,
            audio_duration=duration,
            infer_step=60,
            guidance_scale=15.0,
            batch_size=1,
            save_path="./outputs",
            format="flac",
        )
        base64_audio, mime = self.encode_audio(output_paths[0])
        return MusicOutput(
            base64_audio=base64_audio,
            mime_type=mime,
            metadata={
                "duration": duration,
                "mode": "edit",
                "edit_n_min": n_min,
                "edit_n_max": n_max,
            },
        )

    @fal.endpoint("/repaint")
    def repaint(self, request: RepaintInput) -> MusicOutput:
        src_path, duration = self.save_audio(request.src_audio_base64)

        output_paths = self.model(
            task="repaint",
            src_audio_path=src_path,
            prompt=request.original_prompt,
            lyrics=request.lyrics,
            repaint_start=request.repaint_start,
            repaint_end=request.repaint_end,
            retake_variance=request.retake_variance,
            audio_duration=duration,
            infer_step=60,
            guidance_scale=15.0,
            batch_size=1,
            save_path="./outputs",
            format="flac",
        )
        base64_audio, mime = self.encode_audio(output_paths[0])
        return MusicOutput(
            base64_audio=base64_audio,
            mime_type=mime,
            metadata={
                "duration": duration,
                "mode": "repaint",
                "repaint_start": request.repaint_start,
                "repaint_end": request.repaint_end,
                "retake_variance": request.retake_variance,
            },
        )
