import os
import json
# import argparse # No longer used for direct execution, config dict is used.
import logging
from datetime import datetime
import torch
import matplotlib
import torch.nn.functional as F
import torch.utils.data
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.core import LightningModule
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

# Assuming 'acestep' is an installed package or accessible in PYTHONPATH
try:
    from acestep.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )
    from acestep.text2music_dataset import Text2MusicDataset # Ensure this is correctly located/installed
    from acestep.pipeline_ace_step import ACEStepPipeline # Used by TrainingPipeline
    from acestep.apg_guidance import apg_forward, MomentumBuffer # Used in diffusion_process
except ImportError as e:
    # This path should ideally not be taken if environment is set up
    logger.error(f"Critical Import Error: Missing acestep components: {e}. Training will likely fail.")
    # Dummy classes for structural integrity only if imports fail, not for functionality
    class FlowMatchEulerDiscreteScheduler: pass
    class Text2MusicDataset: pass
    class ACEStepPipeline: pass
    class MomentumBuffer: pass # Dummy for MomentumBuffer
    def apg_forward(pred_cond, pred_uncond, guidance_scale, momentum_buffer): return pred_cond # Dummy for apg_forward


# Diffusers imports (ensure installed: pip install diffusers transformers)
try:
    from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
        retrieve_timesteps, # Used in diffusion_process
    )
    from diffusers.utils.torch_utils import randn_tensor # Used in diffusion_process
    from transformers import AutoModel, Wav2Vec2FeatureExtractor
    import torchaudio
except ImportError as e:
    logger.error(f"Critical Import Error: Missing diffusers or transformers: {e}. Training will likely fail.")
    # Dummy functions/classes if needed
    def retrieve_timesteps(scheduler, num_inference_steps, device, timesteps): return torch.linspace(0, scheduler.config.num_train_timesteps -1, num_inference_steps).long().flip(0), num_inference_steps
    def randn_tensor(*args, **kwargs): return torch.randn(*args, **kwargs)
    class AutoModel: pass
    class Wav2Vec2FeatureExtractor: pass
    import torchaudio # Should be available if torch is

# Setup basic logging
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
training_log_file = os.path.join(LOGS_DIR, "training.log")

# Configure root logger if not already configured, or get existing logger
# This ensures PL and other libraries also use this logging setup if they use standard logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s:%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(training_log_file, mode='a'), # Append mode
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__) # Logger for this module

matplotlib.use("Agg")
# Global torch settings from original trainer.py - can be moved to start_training or made configurable
# torch.backends.cudnn.benchmark = False 
# torch.set_float32_matmul_precision("high")


class TrainingPipeline(LightningModule):
    def __init__(
        self, learning_rate: float = 1e-4, num_workers: int = 4, T: int = 1000,
        weight_decay: float = 1e-2, every_plot_step: int = 2000, shift: float = 3.0,
        logit_mean: float = 0.0, logit_std: float = 1.0, timestep_densities_type: str = "logit_normal",
        ssl_coeff: float = 1.0, base_checkpoint_dir=None, max_steps: int = 200000,
        warmup_steps: int = 10, dataset_path: str = "./data/your_dataset_path",
        lora_config_path: str = None, adapter_name: str = "lora_adapter",
    ):
        super().__init__()
        self.save_hyperparameters() # Saves all __init__ args to self.hparams
        self.T = T # Used for internal scheduler init, self.hparams.T also available

        self.scheduler = self.get_scheduler() # Initialize own scheduler for noise/sigmas

        logger.info(f"Loading base ACEStep model from base_checkpoint_dir: {self.hparams.base_checkpoint_dir}")
        # ACEStepPipeline is expected to load its components (transformer, dcae, text_encoder)
        acestep_pipeline = ACEStepPipeline(self.hparams.base_checkpoint_dir)
        acestep_pipeline.load_checkpoint(acestep_pipeline.checkpoint_dir) # ACEStep's internal checkpoint loading

        transformers_model = acestep_pipeline.ace_step_transformer.float()
        transformers_model.enable_gradient_checkpointing()

        if self.hparams.lora_config_path:
            logger.info(f"Applying LoRA configuration from: {self.hparams.lora_config_path}")
            try: from peft import LoraConfig
            except ImportError: raise ImportError("PEFT library not found. Install with `pip install peft` for LoRA training.")
            
            with open(self.hparams.lora_config_path, encoding="utf-8") as f:
                lora_config_dict = json.load(f)
            lora_config = LoraConfig(**lora_config_dict)
            transformers_model.add_adapter(adapter_config=lora_config, adapter_name=self.hparams.adapter_name)
            self.adapter_name = self.hparams.adapter_name # Store for saving adapter later
            logger.info(f"LoRA adapter '{self.adapter_name}' added to transformer model.")
        else:
            logger.warning("No LoRA config path provided. Training full model or using pre-existing adapters if any.")
            self.adapter_name = None

        self.transformers = transformers_model
        self.dcae = acestep_pipeline.music_dcae.float()
        self.dcae.requires_grad_(False)
        self.text_encoder_model = acestep_pipeline.text_encoder_model.float()
        self.text_encoder_model.requires_grad_(False)
        self.text_tokenizer = acestep_pipeline.text_tokenizer
        
        # SSL Models setup
        ssl_cache_dir = os.path.join(self.hparams.base_checkpoint_dir, "ssl_models_cache")
        os.makedirs(ssl_cache_dir, exist_ok=True)
        logger.info(f"SSL models will use cache directory: {ssl_cache_dir}")
        self._initialize_ssl_models(ssl_cache_dir)
        
        logger.info("TrainingPipeline initialized successfully.")

    def _initialize_ssl_models(self, cache_dir):
        # MERT Model
        try:
            self.mert_model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True, cache_dir=cache_dir).eval()
        except Exception as e:
            logger.warning(f"Initial MERT load failed: {e}. Attempting config fix...")
            try:
                # Attempt MERT config fix (this part is fragile and environment-dependent)
                # Construct path to potential config.json within snapshot structure
                mert_snapshot_base = os.path.join(cache_dir, "models--m-a-p--MERT-v1-330M", "snapshots")
                if os.path.exists(mert_snapshot_base):
                    snapshot_dirs = [d for d in os.listdir(mert_snapshot_base) if os.path.isdir(os.path.join(mert_snapshot_base, d))]
                    if snapshot_dirs:
                        latest_snapshot_dir = max(snapshot_dirs, key=lambda d: os.path.getmtime(os.path.join(mert_snapshot_base, d)))
                        mert_config_path = os.path.join(mert_snapshot_base, latest_snapshot_dir, "config.json")
                        if os.path.exists(mert_config_path):
                            logger.info(f"Found MERT config for potential fix: {mert_config_path}")
                            with open(mert_config_path) as f: mert_cfg = json.load(f)
                            mert_cfg["conv_pos_batch_norm"] = False
                            with open(mert_config_path, "w") as f: json.dump(mert_cfg, f)
                            self.mert_model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True, cache_dir=cache_dir).eval()
                            logger.info("MERT model loaded successfully after config fix.")
                        else: raise FileNotFoundError("MERT config.json not found in latest snapshot.")
                    else: raise FileNotFoundError("No MERT snapshots found.")
                else: raise FileNotFoundError(f"MERT snapshot base directory not found: {mert_snapshot_base}")
            except Exception as e_fix:
                logger.error(f"MERT model config fix failed: {e_fix}. MERT SSL will be unavailable.")
                self.mert_model = None
        if self.mert_model: self.mert_model.requires_grad_(False)
        self.resampler_mert = torchaudio.transforms.Resample(orig_freq=48000, new_freq=24000)
        try: self.processor_mert = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True, cache_dir=cache_dir)
        except Exception as e: logger.error(f"Failed to load MERT processor: {e}"); self.processor_mert = None

        # mHuBERT Model
        try:
            self.hubert_model = AutoModel.from_pretrained("utter-project/mHuBERT-147", cache_dir=cache_dir).eval()
            if self.hubert_model: self.hubert_model.requires_grad_(False)
            self.resampler_mhubert = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)
            self.processor_mhubert = Wav2Vec2FeatureExtractor.from_pretrained("utter-project/mHuBERT-147", cache_dir=cache_dir)
        except Exception as e:
            logger.error(f"Failed to load mHuBERT model or processor: {e}. mHuBERT SSL will be unavailable.")
            self.hubert_model = None; self.processor_mhubert = None
            
    def infer_mert_ssl(self, target_wavs, wav_lengths): # Copied from original trainer.py Pipeline
        if not self.mert_model or not self.processor_mert: return None
        # Ensure model is on the correct device (PL should handle this for nn.Modules)
        # self.mert_model.to(self.device) 
        
        # Input is N x 2 x T (48kHz), convert to N x T (24kHz), mono
        # Make sure resampler is on the same device as data if it has state
        mert_input_wavs_mono_24k = self.resampler_mert.to(target_wavs.device)(target_wavs.mean(dim=1))
        bsz = target_wavs.shape[0]
        actual_lengths_24k = wav_lengths // 2

        means = torch.stack([mert_input_wavs_mono_24k[i, : actual_lengths_24k[i]].mean() for i in range(bsz)])
        vars_val = torch.stack([mert_input_wavs_mono_24k[i, : actual_lengths_24k[i]].var() for i in range(bsz)])
        mert_input_wavs_mono_24k = (mert_input_wavs_mono_24k - means.view(-1, 1)) / torch.sqrt(vars_val.view(-1, 1) + 1e-7)

        chunk_size = 24000 * 5
        all_chunks, chunk_actual_lengths, num_chunks_per_audio = [], [], []
        for i in range(bsz):
            audio, actual_length = mert_input_wavs_mono_24k[i], actual_lengths_24k[i]
            num_chunks = (actual_length + chunk_size - 1) // chunk_size
            num_chunks_per_audio.append(num_chunks)
            for start_idx in range(0, actual_length, chunk_size):
                end_idx = min(start_idx + chunk_size, actual_length)
                chunk = audio[start_idx:end_idx]
                if len(chunk) < chunk_size: chunk = F.pad(chunk, (0, chunk_size - len(chunk)))
                all_chunks.append(chunk)
                chunk_actual_lengths.append(end_idx - start_idx)
        
        if not all_chunks: return None # Or handle empty list appropriately
        all_chunks_stacked = torch.stack(all_chunks, dim=0)
        
        with torch.no_grad():
            mert_ssl_hidden_states = self.mert_model(all_chunks_stacked).last_hidden_state
        
        chunk_num_features = [(length + 319) // 320 for length in chunk_actual_lengths]
        chunk_hidden_states_trimmed = [mert_ssl_hidden_states[i, :chunk_num_features[i], :] for i in range(len(all_chunks))]
        
        mert_ssl_list_final = []
        current_chunk_idx = 0
        for num_chunks in num_chunks_per_audio:
            audio_specific_chunks = chunk_hidden_states_trimmed[current_chunk_idx : current_chunk_idx + num_chunks]
            if audio_specific_chunks: # Ensure there are chunks to concat
                 mert_ssl_list_final.append(torch.cat(audio_specific_chunks, dim=0))
            else: # Handle case where an audio file might produce no chunks (e.g., too short)
                 # This might require returning a placeholder tensor of correct shape or None for this item
                 # For now, append None or a zero tensor, but this needs careful handling in the main logic
                 logger.warning(f"Audio item produced no MERT SSL chunks. Length: {actual_lengths_24k[len(mert_ssl_list_final)]}")
                 # Placeholder: append a tensor of zeros. Shape might need adjustment based on downstream use.
                 # Example: torch.zeros((0, mert_ssl_hidden_states.size(-1)), device=self.device, dtype=self.dtype)
                 # This needs to be robustly handled based on how proj_losses are computed.
                 # For now, this case might lead to errors if not handled by the caller or proj_loss.
                 # A simple fix is to ensure all_ssl_hiden_states only contains valid tensors.
                 pass # Let the list be shorter if an item has no SSL features. This needs careful handling in run_step.

            current_chunk_idx += num_chunks
        return mert_ssl_list_final if any(item is not None for item in mert_ssl_list_final) else None


    def infer_mhubert_ssl(self, target_wavs, wav_lengths): # Copied from original trainer.py Pipeline
        if not self.hubert_model or not self.processor_mhubert: return None
        # self.hubert_model.to(self.device)
        
        mhubert_input_wavs_mono_16k = self.resampler_mhubert.to(target_wavs.device)(target_wavs.mean(dim=1))
        bsz = target_wavs.shape[0]
        actual_lengths_16k = wav_lengths // 3

        means = torch.stack([mhubert_input_wavs_mono_16k[i, :actual_lengths_16k[i]].mean() for i in range(bsz)])
        vars_val = torch.stack([mhubert_input_wavs_mono_16k[i, :actual_lengths_16k[i]].var() for i in range(bsz)])
        mhubert_input_wavs_mono_16k = (mhubert_input_wavs_mono_16k - means.view(-1, 1)) / torch.sqrt(vars_val.view(-1, 1) + 1e-7)

        chunk_size = 16000 * 30
        all_chunks, chunk_actual_lengths, num_chunks_per_audio = [], [], []
        for i in range(bsz):
            audio, actual_length = mhubert_input_wavs_mono_16k[i], actual_lengths_16k[i]
            num_chunks = (actual_length + chunk_size - 1) // chunk_size
            num_chunks_per_audio.append(num_chunks)
            for start_idx in range(0, actual_length, chunk_size):
                end_idx = min(start_idx + chunk_size, actual_length)
                chunk = audio[start_idx:end_idx]
                if len(chunk) < chunk_size: chunk = F.pad(chunk, (0, chunk_size - len(chunk)))
                all_chunks.append(chunk)
                chunk_actual_lengths.append(end_idx - start_idx)

        if not all_chunks: return None
        all_chunks_stacked = torch.stack(all_chunks, dim=0)

        with torch.no_grad():
            mhubert_ssl_hidden_states = self.hubert_model(all_chunks_stacked).last_hidden_state
        
        chunk_num_features = [(length + 319) // 320 for length in chunk_actual_lengths]
        chunk_hidden_states_trimmed = [mhubert_ssl_hidden_states[i, :chunk_num_features[i], :] for i in range(len(all_chunks))]
        
        mhubert_ssl_list_final = []
        current_chunk_idx = 0
        for num_chunks in num_chunks_per_audio:
            audio_specific_chunks = chunk_hidden_states_trimmed[current_chunk_idx : current_chunk_idx + num_chunks]
            if audio_specific_chunks:
                 mhubert_ssl_list_final.append(torch.cat(audio_specific_chunks, dim=0))
            else: # Similar handling as in MERT SSL for potentially empty items
                 logger.warning(f"Audio item produced no mHuBERT SSL chunks. Length: {actual_lengths_16k[len(mhubert_ssl_list_final)]}")
                 pass
            current_chunk_idx += num_chunks
        return mhubert_ssl_list_final if any(item is not None for item in mhubert_ssl_list_final) else None


    def get_text_embeddings(self, texts, text_max_length=256): # Removed device, use self.device
        inputs = self.text_tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=text_max_length,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        # self.text_encoder_model already on self.device via PL
        with torch.no_grad(): outputs = self.text_encoder_model(**inputs)
        return outputs.last_hidden_state, inputs["attention_mask"]

    def preprocess(self, batch, train=True): # Adapted from original
        target_wavs = batch["target_wavs"].to(self.device)
        wav_lengths = batch["wav_lengths"].to(self.device) # Ensure lengths are also on device if used in tensor ops
        dtype = self.dtype # Use PL managed dtype
        bs = target_wavs.shape[0]

        mert_ssl_hidden_states_list = None
        mhubert_ssl_hidden_states_list = None
        if train and self.training: # self.training is PL's flag
            with torch.amp.autocast(device_type=self.device.type, dtype=dtype, enabled=(self.device.type == 'cuda')):
                if self.mert_model: mert_ssl_hidden_states_list = self.infer_mert_ssl(target_wavs, wav_lengths)
                if self.hubert_model: mhubert_ssl_hidden_states_list = self.infer_mhubert_ssl(target_wavs, wav_lengths)
        
        texts = batch["prompts"]
        encoder_text_hidden_states, text_attention_mask = self.get_text_embeddings(texts, text_max_length=256) # device handled by get_text_embeddings
        encoder_text_hidden_states = encoder_text_hidden_states.to(dtype)

        # self.dcae should be on self.device by PL
        target_latents, _ = self.dcae.encode(target_wavs, wav_lengths)
        attention_mask = torch.ones(bs, target_latents.shape[-1], device=self.device, dtype=dtype)
        
        speaker_embds = batch["speaker_embs"].to(self.device).to(dtype)
        keys = batch["keys"]
        lyric_token_ids = batch["lyric_token_ids"].to(self.device)
        lyric_mask = batch["lyric_masks"].to(self.device)

        if train and self.training:
            text_cfg_mask = (torch.rand(bs, device=self.device) >= 0.15).unsqueeze(1).unsqueeze(2)
            encoder_text_hidden_states = encoder_text_hidden_states * text_cfg_mask
            text_attention_mask = text_attention_mask * text_cfg_mask.squeeze(-1)

            speaker_cfg_mask = (torch.rand(bs, device=self.device) >= 0.50).unsqueeze(1)
            speaker_embds = speaker_embds * speaker_cfg_mask
            
            lyrics_cfg_mask = (torch.rand(bs, device=self.device) >= 0.15).unsqueeze(1)
            lyric_token_ids = lyric_token_ids * lyrics_cfg_mask
            lyric_mask = lyric_mask * lyrics_cfg_mask
            
        return (keys, target_latents, attention_mask, encoder_text_hidden_states,
                text_attention_mask, speaker_embds, lyric_token_ids, lyric_mask,
                mert_ssl_hidden_states_list, mhubert_ssl_hidden_states_list)


    def get_scheduler(self): # From original Pipeline, uses self.hparams
        return FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=self.hparams.T, shift=self.hparams.shift,
        )

    def configure_optimizers(self): # From original Pipeline, uses self.hparams
        trainable_params = [p for p in self.transformers.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params=[{"params": trainable_params}], lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay, betas=(0.8, 0.9),
        )
        
        def lr_lambda(current_step):
            if current_step < self.hparams.warmup_steps:
                return float(current_step) / float(max(1, self.hparams.warmup_steps))
            progress = float(current_step - self.hparams.warmup_steps) / float(
                max(1, self.hparams.max_steps - self.hparams.warmup_steps)
            )
            return max(0.0, 1.0 - progress)

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def train_dataloader(self): # From original Pipeline, uses self.hparams
        self.train_dataset = Text2MusicDataset(
            train=True, train_dataset_path=self.hparams.dataset_path,
        )
        return DataLoader(
            self.train_dataset, shuffle=True, num_workers=self.hparams.num_workers,
            pin_memory=True, collate_fn=self.train_dataset.collate_fn,
        )

    def get_sd3_sigmas(self, timesteps, n_dim=4): # Adapted from original, uses self.scheduler, self.device, self.dtype
        sigmas_val = self.scheduler.sigmas.to(device=self.device, dtype=self.dtype)
        schedule_timesteps = self.scheduler.timesteps.to(self.device)
        timesteps = timesteps.to(self.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas_val[step_indices].flatten()
        while len(sigma.shape) < n_dim: sigma = sigma.unsqueeze(-1)
        return sigma

    def get_timestep(self, bsz): # Adapted from original, uses self.hparams, self.scheduler, self.device
        if self.hparams.timestep_densities_type == "logit_normal":
            u = torch.normal(mean=self.hparams.logit_mean, std=self.hparams.logit_std, size=(bsz,), device="cpu")
            u = torch.sigmoid(u)
            indices = (u * self.scheduler.config.num_train_timesteps).long()
            indices = torch.clamp(indices, 0, self.scheduler.config.num_train_timesteps - 1)
            timesteps = self.scheduler.timesteps[indices].to(self.device)
        else:
            raise ValueError(f"Unknown timestep_densities_type: {self.hparams.timestep_densities_type}")
        return timesteps

    def run_step(self, batch, batch_idx): # Adapted from original
        (keys, target_latents, attention_mask, encoder_text_hidden_states,
         text_attention_mask, speaker_embds, lyric_token_ids, lyric_mask,
         mert_ssl_hidden_states_list, mhubert_ssl_hidden_states_list) = self.preprocess(batch, train=self.training)

        target_image = target_latents
        noise = torch.randn_like(target_image, device=self.device)
        bsz = target_image.shape[0]
        timesteps = self.get_timestep(bsz)
        sigmas = self.get_sd3_sigmas(timesteps=timesteps, n_dim=target_image.ndim)
        noisy_image = sigmas * noise + (1.0 - sigmas) * target_image
        flow_target = target_image

        all_ssl_features = []
        if mert_ssl_hidden_states_list: all_ssl_features.append(mert_ssl_hidden_states_list)
        if mhubert_ssl_hidden_states_list: all_ssl_features.append(mhubert_ssl_hidden_states_list)
        
        transformer_output = self.transformers(
            hidden_states=noisy_image, attention_mask=attention_mask,
            encoder_text_hidden_states=encoder_text_hidden_states, text_attention_mask=text_attention_mask,
            speaker_embeds=speaker_embds, lyric_token_idx=lyric_token_ids, lyric_mask=lyric_mask,
            timestep=timesteps.to(self.dtype), ssl_hidden_states=all_ssl_features,
        )
        model_pred = transformer_output.sample
        proj_losses = transformer_output.proj_losses # This should be a list of (name, loss_value) tuples

        model_pred = model_pred * (-sigmas) + noisy_image # Preconditioning

        loss_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand_as(target_image)
        masked_model_pred = (model_pred * loss_mask).view(bsz, -1)
        masked_target = (flow_target * loss_mask).view(bsz, -1)
        
        denoising_loss = F.mse_loss(masked_model_pred, masked_target, reduction="none").mean(1)
        denoising_loss = (denoising_loss * loss_mask.view(bsz, -1).mean(1)).mean()

        prefix = "train" if self.training else "val"
        self.log(f"{prefix}/denoising_loss", denoising_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        total_proj_loss_value = 0.0
        if proj_losses and isinstance(proj_losses, list): # Ensure proj_losses is a list of tuples as expected
            num_valid_proj_losses = 0
            for loss_item in proj_losses:
                if isinstance(loss_item, tuple) and len(loss_item) == 2 and isinstance(loss_item[1], torch.Tensor):
                    k_loss_name, v_loss_tensor = loss_item
                    self.log(f"{prefix}/{k_loss_name}_loss", v_loss_tensor, on_step=True, on_epoch=True, prog_bar=True, logger=True)
                    total_proj_loss_value += v_loss_tensor.mean() # Ensure it's a scalar
                    num_valid_proj_losses +=1
                # else: logger.warning(f"Skipping invalid proj_loss item: {loss_item}") # Removed due to potential log spam
            if num_valid_proj_losses > 0: total_proj_loss_value /= num_valid_proj_losses
        
        final_loss = denoising_loss + total_proj_loss_value * self.hparams.ssl_coeff
        self.log(f"{prefix}/total_loss", final_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        if self.training: # Log learning rate only during training
            lr_scheduler = self.lr_schedulers()
            if lr_scheduler: self.log("lr", lr_scheduler.get_last_lr()[0], on_step=True, prog_bar=False, logger=True)
        return final_loss

    def training_step(self, batch, batch_idx): # From original Pipeline
        loss = self.run_step(batch, batch_idx)
        if self.global_step > 0 and self.global_step % self.hparams.every_plot_step == 0:
            if self.trainer.is_global_zero:
                 self.plot_audio_samples(batch, "train") # Adapted plot_step
        return loss

    def on_save_checkpoint(self, checkpoint): # From original Pipeline, adapted
        if self.adapter_name and hasattr(self.transformers, 'save_lora_adapter') and self.trainer.is_global_zero:
            # Save LoRA adapter only on global rank 0 to avoid race conditions if on shared filesystem
            # dirpath is from ModelCheckpoint callback
            lora_save_path = os.path.join(self.trainer.checkpoint_callback.dirpath, f"lora_adapter_{self.adapter_name}_step{self.global_step}")
            os.makedirs(lora_save_path, exist_ok=True)
            self.transformers.save_lora_adapter(lora_save_path, adapter_name=self.adapter_name)
            logger.info(f"Saved LoRA adapter '{self.adapter_name}' to {lora_save_path}")
        # Note: PL takes care of saving the full model checkpoint. This is for separate LoRA adapter files.

    @torch.no_grad()
    def diffusion_process( # Copied from original Pipeline, adapted
        self, duration, encoder_text_hidden_states, text_attention_mask, speaker_embds,
        lyric_token_ids, lyric_mask, random_generators=None, infer_steps=60,
        guidance_scale=15.0, omega_scale=10.0,
    ):
        do_classifier_free_guidance = guidance_scale > 1.0 # Simplified check
        device = self.device; dtype = self.dtype
        bsz = encoder_text_hidden_states.shape[0]

        # This scheduler instance is for inference, potentially different settings from self.scheduler used for training sigmas
        eval_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0) # Or use self.scheduler if config is same

        frame_length = int(duration * 44100 / 512 / 8) # This calculation seems specific
        timesteps, num_inference_steps = retrieve_timesteps(eval_scheduler, num_inference_steps=infer_steps, device=device)

        latents = randn_tensor((bsz, 8, 16, frame_length), generator=random_generators, device=device, dtype=dtype)
        attention_mask_eval = torch.ones(bsz, frame_length, device=device, dtype=dtype)

        if do_classifier_free_guidance:
            attention_mask_eval = torch.cat([attention_mask_eval] * 2)
            encoder_text_hidden_states = torch.cat([encoder_text_hidden_states, torch.zeros_like(encoder_text_hidden_states)])
            text_attention_mask = torch.cat([text_attention_mask, torch.zeros_like(text_attention_mask)])
            speaker_embds = torch.cat([speaker_embds, torch.zeros_like(speaker_embds)])
            lyric_token_ids = torch.cat([lyric_token_ids, torch.zeros_like(lyric_token_ids)])
            lyric_mask = torch.cat([lyric_mask, torch.zeros_like(lyric_mask)])

        momentum_buffer = MomentumBuffer() # Assuming MomentumBuffer is defined or imported

        for t in tqdm(timesteps, desc="Diffusion Process"):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            timestep_expanded = t.expand(latent_model_input.shape[0])
            
            noise_pred_out = self.transformers(
                hidden_states=latent_model_input, attention_mask=attention_mask_eval,
                encoder_text_hidden_states=encoder_text_hidden_states, text_attention_mask=text_attention_mask,
                speaker_embeds=speaker_embds, lyric_token_idx=lyric_token_ids, lyric_mask=lyric_mask,
                timestep=timestep_expanded.to(dtype),
            ).sample # Assuming .sample is the direct output

            if do_classifier_free_guidance:
                pred_cond, pred_uncond = noise_pred_out.chunk(2)
                noise_pred = apg_forward(pred_cond, pred_uncond, guidance_scale, momentum_buffer) # Assuming apg_forward is defined/imported
            else:
                noise_pred = noise_pred_out
            
            latents = eval_scheduler.step(noise_pred, t, latents, return_dict=False, omega=omega_scale)[0]
        return latents

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0): # Standard PL signature
        (keys, _, _, encoder_text_hidden_states, text_attention_mask, speaker_embds,
         lyric_token_ids, lyric_mask, _, _) = self.preprocess(batch, train=False)

        # These should ideally come from config or be fixed for prediction
        infer_steps = self.hparams.get("predict_infer_steps", 60) 
        guidance_scale = self.hparams.get("predict_guidance_scale", 15.0)
        omega_scale = self.hparams.get("predict_omega_scale", 10.0)
        duration = self.hparams.get("predict_duration", 10) # seconds
        
        bsz = encoder_text_hidden_states.shape[0]
        random_generators = [torch.Generator(device=self.device).manual_seed(random.randint(0, 2**32 - 1)) for _ in range(bsz)]
        seeds = [gen.initial_seed() for gen in random_generators]

        pred_latents = self.diffusion_process(
            duration=duration, encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask, speaker_embds=speaker_embds,
            lyric_token_ids=lyric_token_ids, lyric_mask=lyric_mask,
            random_generators=random_generators, infer_steps=infer_steps,
            guidance_scale=guidance_scale, omega_scale=omega_scale,
        )
        
        # self.dcae should be on self.device
        _, pred_wavs = self.dcae.decode(pred_latents, audio_lengths=None, sr=48000) # audio_lengths might be needed if varying
        
        return {"keys": keys, "pred_wavs": pred_wavs, "prompts": batch["prompts"], 
                "candidate_lyric_chunks": batch.get("candidate_lyric_chunks"), "sr": 48000, "seeds": seeds}

    def construct_lyrics(self, candidate_lyric_chunk): # Copied from original Pipeline
        if not candidate_lyric_chunk: return ""
        return "\n".join([chunk["lyric"] for chunk in candidate_lyric_chunk if "lyric" in chunk])

    @torch.no_grad()
    def plot_audio_samples(self, batch, stage="train"): # Adapted from original plot_step
        if not self.trainer.is_global_zero: return
        logger.info(f"Plotting audio samples for stage '{stage}' at global step {self.global_step}...")
        
        results = self.predict_step(batch, 0) # batch_idx=0, dataloader_idx=0 for calling predict_step internally
        
        target_wavs = batch.get("target_wavs") # Target might not always be present (e.g. pure inference)
        pred_wavs = results["pred_wavs"]
        keys = results["keys"]
        prompts = results["prompts"]
        candidate_lyric_chunks = results.get("candidate_lyric_chunks") # Might be None
        sr = results["sr"]
        seeds = results["seeds"]

        log_dir = self.trainer.logger.log_dir
        save_dir = os.path.join(log_dir, f"{stage}_audio_samples", f"step_{self.global_step}")
        os.makedirs(save_dir, exist_ok=True)

        for i in range(len(keys)):
            key = keys[i]; prompt = prompts[i]; seed = seeds[i]
            pred_wav = pred_wavs[i] # Should be (channels, time)
            
            lyric_text = ""
            if candidate_lyric_chunks and i < len(candidate_lyric_chunks):
                lyric_text = self.construct_lyrics(candidate_lyric_chunks[i])

            filename_prefix = f"{key.replace('/', '_')}_idx{i}"
            torchaudio.save(os.path.join(save_dir, f"{filename_prefix}_pred.wav"), pred_wav.cpu(), sr)
            if target_wavs is not None and i < len(target_wavs):
                torchaudio.save(os.path.join(save_dir, f"{filename_prefix}_target.wav"), target_wavs[i].cpu(), sr)
            
            with open(os.path.join(save_dir, f"{filename_prefix}_info.txt"), "w", encoding="utf-8") as f:
                f.write(f"# KEY: {key}\n# PROMPT: {prompt}\n# LYRICS:\n{lyric_text}\n# SEED: {seed}\n")
        logger.info(f"Saved audio samples and info to {save_dir}")


def start_training(config: dict):
    logger.info("Starting training process with configuration:")
    logger.info(json.dumps(config, indent=2, sort_keys=True))

    # Global torch settings from config or defaults
    torch.backends.cudnn.benchmark = config.get("cudnn_benchmark", False)
    torch.set_float32_matmul_precision(config.get("matmul_precision", "high"))

    pipeline_args = {
        "learning_rate": config.get("learning_rate", 1e-4),
        "num_workers": config.get("num_workers", 4),
        "T": config.get("T_scheduler", 1000),
        "weight_decay": config.get("weight_decay", 1e-2),
        "every_plot_step": config.get("every_plot_step", 2000),
        "shift": config.get("scheduler_shift", 3.0),
        "logit_mean": config.get("logit_mean", 0.0),
        "logit_std": config.get("logit_std", 1.0),
        "timestep_densities_type": config.get("timestep_densities_type", "logit_normal"),
        "ssl_coeff": config.get("ssl_coeff", 1.0),
        "base_checkpoint_dir": config.get("base_model_checkpoint_dir"),
        "max_steps": config.get("max_steps", 200000),
        "warmup_steps": config.get("warmup_steps", 10),
        "dataset_path": config.get("dataset_path"),
        "lora_config_path": config.get("lora_config_path"),
        "adapter_name": config.get("adapter_name", "lora_adapter"),
    }
    for required_arg in ["base_model_checkpoint_dir", "dataset_path", "lora_config_path"]:
        if not pipeline_args[required_arg]:
            msg = f"Configuration error: '{required_arg}' is required."
            logger.error(msg)
            return msg, None
    try: model = TrainingPipeline(**pipeline_args)
    except Exception as e: logger.error(f"Error initializing TrainingPipeline: {e}", exc_info=True); return f"Init Pipeline Error: {e}", None

    exp_name = config.get("exp_name", "my_experiment")
    logger_save_dir = config.get("logger_dir", os.path.join(os.path.dirname(__file__), "training_logs"))
    run_checkpoint_dir = os.path.join(logger_save_dir, exp_name, "checkpoints")
    os.makedirs(run_checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=run_checkpoint_dir,
        filename=config.get("checkpoint_filename_format", "{epoch}-{step}-{" + f"{config.get('checkpoint_monitor_metric','train/total_loss').replace('/','_')}" + ":.2f}"),
        every_n_train_steps=config.get("every_n_train_steps", 1000),
        save_top_k=config.get("save_top_k", -1), save_last=True,
        monitor=config.get("checkpoint_monitor_metric", "train/total_loss"), # Ensure TrainingPipeline logs this exact key
        mode=config.get("checkpoint_monitor_mode", "min"),
    )
    tensorboard_logger = TensorBoardLogger(
        save_dir=logger_save_dir, name=exp_name, version=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{exp_name}"
    )
    logger.info(f"TensorBoard logs: {tensorboard_logger.log_dir}. Checkpoints: {run_checkpoint_dir}")

    trainer_params = {
        "accelerator": config.get("accelerator", "gpu" if torch.cuda.is_available() else "cpu"),
        "devices": config.get("devices", 1), "num_nodes": config.get("num_nodes", 1),
        "precision": config.get("precision", "32-true"),
        "accumulate_grad_batches": config.get("accumulate_grad_batches", 1),
        "strategy": config.get("strategy", "auto"), "max_epochs": config.get("max_epochs", -1),
        "max_steps": config.get("max_steps", 200000),
        "log_every_n_steps": config.get("log_every_n_steps", 50),
        "logger": tensorboard_logger, "callbacks": [checkpoint_callback],
        "gradient_clip_val": config.get("gradient_clip_val", 0.5),
        "gradient_clip_algorithm": config.get("gradient_clip_algorithm", "norm"),
        "reload_dataloaders_every_n_epochs": 1 if config.get("reload_dataloaders_every_n_epochs", False) else 0,
        "val_check_interval": config.get("val_check_interval", 1.0),
    }
    if trainer_params["strategy"] == "ddp_find_unused_parameters_true": # Compatibility with original
        trainer_params["strategy"] = "ddp"; trainer_params["find_unused_parameters"] = True
    
    try: trainer = Trainer(**trainer_params)
    except Exception as e: logger.error(f"Error initializing PL Trainer: {e}", exc_info=True); return f"Init PL Trainer Error: {e}", None

    try:
        logger.info("Starting model training fit...")
        trainer.fit(model, ckpt_path=config.get("ckpt_path_resume"))
        logger.info("Training finished.")
        final_ckpt_path = os.path.join(run_checkpoint_dir, "final_model_completed.ckpt")
        trainer.save_checkpoint(final_ckpt_path)
        logger.info(f"Final checkpoint saved to {final_ckpt_path}")
        return f"Training completed. Final Ckpt: {final_ckpt_path}", tensorboard_logger.log_dir
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        return f"Training Runtime Error: {e}", tensorboard_logger.log_dir if tensorboard_logger else None


if __name__ == '__main__':
    # This block is for direct testing of trainer_utils.py.
    # It requires dummy files and a functional ACEStep environment or robust dummy classes.
    logger.info("--- Running trainer_utils.py directly for testing ---")
    
    # Simplified dummy file setup - ACEStepPipeline and Text2MusicDataset might need more.
    test_files_root = os.path.join(os.getcwd(), "temp_trainer_test_files")
    os.makedirs(test_files_root, exist_ok=True)
    dummy_base_ckpt_dir = os.path.join(test_files_root, "dummy_acestep_base_model")
    os.makedirs(os.path.join(dummy_base_ckpt_dir, "ace_step_transformer"), exist_ok=True) # etc.
    dummy_dataset_path = os.path.join(test_files_root, "dummy_dataset")
    os.makedirs(dummy_dataset_path, exist_ok=True)
    with open(os.path.join(dummy_dataset_path, "train_data.jsonl"), "w") as f:
        json.dump({"prompt": "test", "audio_path": "a.wav", "speaker_emb_path": "s.pt", "lyric_path": "l.txt"},f)
    dummy_lora_config_path = os.path.join(test_files_root, "dummy_lora_config.json")
    with open(dummy_lora_config_path, "w") as f: json.dump({"r": 4, "lora_alpha": 8}, f)

    test_config = {
        "base_model_checkpoint_dir": dummy_base_ckpt_dir, # Needs valid ACEStep model parts
        "dataset_path": dummy_dataset_path, # Needs valid dataset structure for Text2MusicDataset
        "lora_config_path": dummy_lora_config_path,
        "exp_name": "trainer_utils_direct_test",
        "logger_dir": os.path.join(test_files_root, "test_logs"),
        "max_steps": 5, "warmup_steps": 1, "every_n_train_steps": 2, "log_every_n_steps":1,
        "accelerator": "cpu", "devices":1, "precision": "32-true", "num_workers":0,
        "every_plot_step": 3,
    }
    logger.info(f"Test config: {json.dumps(test_config, indent=2)}")
    
    # Note: This test will likely fail if ACEStepPipeline or Text2MusicDataset cannot initialize
    # with the highly simplified dummy files, or if the imported acestep components are not available.
    # The primary purpose of this test is to ensure the start_training function itself can be called
    # and the PL Trainer attempts initialization.
    try:
        status_msg, result_path = start_training(test_config)
        logger.info(f"Direct test run status: {status_msg}")
        logger.info(f"Result path: {result_path}")
    except Exception as e:
        logger.error(f"Direct test run failed critically: {e}", exc_info=True)
    finally:
        logger.info("To clean up, manually delete: " + test_files_root)
        logger.info("--- trainer_utils.py direct test finished ---")
