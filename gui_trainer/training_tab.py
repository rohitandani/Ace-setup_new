import gradio as gr
from .trainer_utils import start_training
from .model_manager import list_checkpoints # load_model is not directly used for starting training
import os
import json
import logging

logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL_CHECKPOINT_DIR = "checkpoints/" # For listing base models
DEFAULT_OUTPUT_CHECKPOINT_DIR = "./gui_trainer_checkpoints/"
DEFAULT_TB_LOG_DIR = "./gui_trainer_tb_logs/"
DEFAULT_LORA_CONFIG_PATH = "config/lora_config_example.json" # Example path

def create_training_tab():
    with gr.Blocks() as training_tab_interface:
        gr.Markdown("# Training Configuration Tab")

        # Populate base model checkpoints
        try:
            if not os.path.exists(DEFAULT_BASE_MODEL_CHECKPOINT_DIR):
                 os.makedirs(DEFAULT_BASE_MODEL_CHECKPOINT_DIR, exist_ok=True)
                 # Create a dummy base model dir for UI population if none exist
                 dummy_base_model_sub_dir = os.path.join(DEFAULT_BASE_MODEL_CHECKPOINT_DIR, "dummy_base_model")
                 if not os.path.exists(dummy_base_model_sub_dir): os.makedirs(dummy_base_model_sub_dir)
                 with open(os.path.join(dummy_base_model_sub_dir, "model_info.txt"), "w") as f: f.write("dummy base model")
            
            checkpoint_list = list_checkpoints(DEFAULT_BASE_MODEL_CHECKPOINT_DIR)
            # Filter for directories, as base models are dirs; list_checkpoints might return files too.
            checkpoint_list = [c for c in checkpoint_list if os.path.isdir(os.path.join(DEFAULT_BASE_MODEL_CHECKPOINT_DIR, c))]
            if not checkpoint_list:
                checkpoint_list = ["No base model directories found in " + DEFAULT_BASE_MODEL_CHECKPOINT_DIR]
        except Exception as e:
            logger.error(f"Error listing base model checkpoints: {e}", exc_info=True)
            checkpoint_list = ["Error loading base models"]
        
        with gr.Accordion("Paths and Names", open=True):
            with gr.Row():
                base_model_checkpoint_dir_dropdown = gr.Dropdown(
                    label="Base Model (ACE-Step Checkpoint Directory)",
                    choices=checkpoint_list,
                    value=checkpoint_list[0] if checkpoint_list else None,
                    interactive=True
                )
                dataset_path_textbox = gr.Textbox(
                    label="Dataset Path (from Dataset Creation Tab)",
                    placeholder="/path/to/your/huggingface_dataset",
                    interactive=True
                )
            with gr.Row():
                lora_config_path_textbox = gr.Textbox(
                    label="LoRA Config Path (JSON file)",
                    value=DEFAULT_LORA_CONFIG_PATH,
                    interactive=True
                )
                exp_name_textbox = gr.Textbox(
                    label="Experiment Name",
                    value="my_acestep_lora_exp",
                    interactive=True
                )
            with gr.Row():
                output_checkpoint_dir_textbox = gr.Textbox(
                    label="Output Checkpoint Directory (for this run)",
                    value=DEFAULT_OUTPUT_CHECKPOINT_DIR,
                    interactive=True
                )
                tb_log_dir_textbox = gr.Textbox(
                    label="TensorBoard Log Directory (Base)",
                    value=DEFAULT_TB_LOG_DIR,
                    interactive=True
                )
            resume_from_ckpt_path_textbox = gr.Textbox(
                label="Resume from Checkpoint Path (Optional, .ckpt)",
                placeholder="/path/to/lightning_checkpoint.ckpt",
                interactive=True
            )
            adapter_name_textbox = gr.Textbox( # Added for clarity, maps to adapter_name in TrainingPipeline
                label="LoRA Adapter Name (for saving config/weights)",
                value="lora_adapter",
                interactive=True
            )


        with gr.Accordion("Training Parameters", open=True):
            with gr.Row():
                learning_rate_slider = gr.Slider(minimum=1e-7, maximum=1e-3, value=1e-4, step=1e-7, label="Learning Rate", interactive=True, scale=2)
                max_steps_number = gr.Number(label="Max Training Steps", value=100000, minimum=1, interactive=True, scale=1)
                warmup_steps_number = gr.Number(label="Warmup Steps", value=1000, minimum=0, interactive=True, scale=1)
            with gr.Row():
                batch_size_slider = gr.Slider(minimum=1, maximum=64, value=4, step=1, label="Batch Size per Device", interactive=True)
                accumulate_grad_batches_number = gr.Number(label="Accumulate Grad Batches", value=1, minimum=1, interactive=True)
                num_workers_number = gr.Number(label="Num Workers (Dataloader)", value=2, minimum=0, interactive=True)
            with gr.Row():
                precision_dropdown = gr.Dropdown(label="Precision", choices=["32", "16-mixed", "bf16", "32-true", "bf16-mixed"], value="32-true", interactive=True)
                # Note: "32" and "16" are shorthands. PL recommends "32-true", "16-mixed", "bf16-mixed".
                num_devices_number = gr.Number(label="Num Devices (GPUs/CPUs)", value=1, minimum=1, interactive=True)
                gradient_clip_val_number = gr.Number(label="Gradient Clip Value (e.g., 0.5, 1.0)", value=0.5, minimum=0.0, interactive=True)
        
        with gr.Accordion("Checkpointing", open=False):
             with gr.Row():
                checkpoint_interval_number = gr.Number(label="Checkpoint Interval (steps)", value=1000, minimum=100, interactive=True)
                checkpoint_monitor_metric_textbox = gr.Textbox(label="Metric to Monitor for Best Ckpt", value="train/total_loss", interactive=True) # Example metric
                save_top_k_number = gr.Number(label="Save Top K Checkpoints", value=3, minimum=-1, step=1, interactive=True, info="-1 saves all.")


        start_training_button = gr.Button("Start Training", variant="primary")
        training_status_textbox = gr.Textbox(
            label="Training Status & Logs", interactive=False, lines=15, autoscroll=True,
            placeholder="Training progress and status will appear here..."
        )

        # --- Define Button Click Handler ---
        def handle_start_training(
            base_model_checkpoint_dir, dataset_path, lora_config_path, exp_name,
            output_checkpoint_dir, tb_log_dir, resume_from_ckpt_path, adapter_name,
            lr, max_steps_val, warmup_steps_val, batch_size_val, accumulate_grad_val, num_workers_val,
            precision_val, num_devices_val, grad_clip_val,
            ckpt_interval_val, ckpt_monitor_val, save_top_k_val
        ):
            status_updates = []
            def log_status(msg):
                status_updates.append(msg)
                logger.info(msg) # Also log to actual logger
                # This return is for Gradio's live update, which might be too frequent here.
                # Consider updating textbox less frequently or at the end.
                # For now, let's accumulate and return at end.

            log_status("Preparing training configuration...")

            if not base_model_checkpoint_dir or base_model_checkpoint_dir.startswith("No base model") or base_model_checkpoint_dir.startswith("Error loading"):
                return "Error: Please select a valid Base Model Checkpoint Directory."
            if not dataset_path: return "Error: Dataset Path is required."
            if not lora_config_path: return "Error: LoRA Config Path is required."
            if not exp_name: return "Error: Experiment Name is required."

            # Construct the config dictionary for trainer_utils.start_training
            config = {
                "base_model_checkpoint_dir": os.path.join(DEFAULT_BASE_MODEL_CHECKPOINT_DIR, base_model_checkpoint_dir) if not os.path.isabs(base_model_checkpoint_dir) else base_model_checkpoint_dir,
                "dataset_path": dataset_path,
                "lora_config_path": lora_config_path,
                "adapter_name": adapter_name, # For TrainingPipeline to know which adapter to train/save under PEFT
                
                "exp_name": exp_name,
                "logger_dir": tb_log_dir, # Base for TensorBoard logs
                # trainer_utils.start_training constructs specific checkpoint path using logger_dir and exp_name
                # So, output_checkpoint_dir_textbox might be redundant if trainer_utils standardizes it.
                # For now, let's pass it for ModelCheckpoint's dirpath if trainer_utils uses it.
                # Decision: trainer_utils.start_training will derive checkpoint dir from logger_dir/exp_name.
                # So, output_checkpoint_dir_textbox is more for user info or if they want to override.
                # For now, let's assume trainer_utils handles this. The UI field `output_checkpoint_dir_textbox`
                # will effectively be `os.path.join(tb_log_dir, exp_name, "checkpoints")`
                
                "learning_rate": lr,
                "max_steps": int(max_steps_val),
                "warmup_steps": int(warmup_steps_val),
                "batch_size": int(batch_size_val), # This is per-device batch size
                "accumulate_grad_batches": int(accumulate_grad_val),
                "num_workers": int(num_workers_val),
                
                "precision": precision_val,
                "devices": int(num_devices_val),
                "gradient_clip_val": float(grad_clip_val) if grad_clip_val else None, # PL handles None as no clip
                
                "every_n_train_steps": int(ckpt_interval_val), # For ModelCheckpoint
                "checkpoint_monitor_metric": ckpt_monitor_val,
                "save_top_k": int(save_top_k_val),

                "ckpt_path_resume": resume_from_ckpt_path if resume_from_ckpt_path and resume_from_ckpt_path.strip() else None,
                
                # Parameters from original trainer.py main() that might be needed by TrainingPipeline
                # These are now mostly hparams in TrainingPipeline or Trainer args
                "T_scheduler": 1000, # Example, make configurable if needed
                "scheduler_shift": 3.0, # Example
                "weight_decay": 1e-2, # Example
                "every_plot_step": 2000, # Example
                "ssl_coeff": 1.0, # Example
                 # "matmul_precision" (e.g. "high") can be added to config for torch.set_float32_matmul_precision
                "matmul_precision": "high" 
            }
            log_status(f"Training config compiled: {json.dumps(config, indent=2, sort_keys=True)}")
            
            status_updates.append("\nStarting training via trainer_utils.start_training...")
            try:
                # Call the actual training function
                result_message, log_path = start_training(config)
                log_status(f"Training process finished.")
                log_status(f"Result: {result_message}")
                if log_path:
                    log_status(f"TensorBoard logs may be available at: {log_path}")
                    log_status(f"You can try running: tensorboard --logdir \"{os.path.abspath(log_path)}\"")
                
            except Exception as e:
                logger.error(f"An error occurred during training: {e}", exc_info=True)
                log_status(f"An critical error occurred: {str(e)}")
            
            return "\n".join(status_updates)

        start_training_button.click(
            fn=handle_start_training,
            inputs=[
                base_model_checkpoint_dir_dropdown, dataset_path_textbox, lora_config_path_textbox, exp_name_textbox,
                output_checkpoint_dir_textbox, tb_log_dir_textbox, resume_from_ckpt_path_textbox, adapter_name_textbox,
                learning_rate_slider, max_steps_number, warmup_steps_number, batch_size_slider, accumulate_grad_batches_number, num_workers_number,
                precision_dropdown, num_devices_number, gradient_clip_val_number,
                checkpoint_interval_number, checkpoint_monitor_metric_textbox, save_top_k_number
            ],
            outputs=training_status_textbox
        )
    return training_tab_interface

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    module_dir = os.path.dirname(__file__) if __file__ else "."
    
    # Create dummy trainer_utils.py
    trainer_utils_path = os.path.join(module_dir, "trainer_utils.py")
    if not os.path.exists(trainer_utils_path):
        with open(trainer_utils_path, "w") as f:
            f.write("""
import logging, time, os, json
logger = logging.getLogger(__name__)
def start_training(config):
    logger.info(f"Dummy start_training called with config: {json.dumps(config, indent=2)}")
    dummy_log_dir = os.path.join(config.get('logger_dir', '.'), config.get('exp_name', 'dummy_exp'), 'version_dummy')
    os.makedirs(dummy_log_dir, exist_ok=True)
    # Simulate some training
    for i in range(5):
        logger.info(f"Dummy training step {i+1}/5...")
        # time.sleep(0.1) # Simulate work
    return f"Dummy training completed for {config.get('exp_name')}. Checkpoints might be in {os.path.join(dummy_log_dir, 'checkpoints')}", dummy_log_dir
""")
        logger.info(f"Created dummy trainer_utils.py at {trainer_utils_path}")

    # Create dummy model_manager.py
    model_manager_path = os.path.join(module_dir, "model_manager.py")
    if not os.path.exists(model_manager_path):
        with open(model_manager_path, "w") as f:
            f.write("""
import os, logging
logger = logging.getLogger(__name__)
def list_checkpoints(checkpoint_dir):
    logger.debug(f"Dummy list_checkpoints for dir: {checkpoint_dir}")
    if not os.path.exists(checkpoint_dir): return [f"Directory not found: {checkpoint_dir}"]
    try:
        # Only list directories as base models for this dummy
        return [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
    except Exception as e: return [f"Error listing: {e}"]
def load_model(*args, **kwargs): logger.debug("Dummy load_model called."); return "DummyModelInstance"
""")
        logger.info(f"Created dummy model_manager.py at {model_manager_path}")

    # Ensure DEFAULT_BASE_MODEL_CHECKPOINT_DIR exists for the test for list_checkpoints
    if not os.path.exists(DEFAULT_BASE_MODEL_CHECKPOINT_DIR):
        os.makedirs(DEFAULT_BASE_MODEL_CHECKPOINT_DIR, exist_ok=True)
        # Create a dummy base model inside for dropdown population
        dummy_model_for_dropdown = os.path.join(DEFAULT_BASE_MODEL_CHECKPOINT_DIR, "example_base_model_dir")
        if not os.path.exists(dummy_model_for_dropdown):
            os.makedirs(dummy_model_for_dropdown)
            with open(os.path.join(dummy_model_for_dropdown, "dummy.txt"), "w") as f: f.write("dummy base")
        logger.info(f"Ensured dummy base model dir exists: {dummy_model_for_dropdown}")
        
    # Create dummy LoRA config if it doesn't exist at default path for testing
    if not os.path.exists(DEFAULT_LORA_CONFIG_PATH):
        os.makedirs(os.path.dirname(DEFAULT_LORA_CONFIG_PATH), exist_ok=True)
        with open(DEFAULT_LORA_CONFIG_PATH, "w") as f:
            json.dump({"r": 8, "lora_alpha": 16, "target_modules": ["q_proj", "v_proj"]}, f, indent=2)
        logger.info(f"Created dummy LoRA config at {DEFAULT_LORA_CONFIG_PATH}")


    tab_interface = create_training_tab()
    tab_interface.launch(debug=True)
