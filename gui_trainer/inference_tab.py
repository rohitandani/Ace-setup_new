import gradio as gr
from .inference_utils import run_inference
from .model_manager import list_checkpoints, load_model
import os
import tempfile
import logging
import json # For parsing seeds

logger = logging.getLogger(__name__)

# CHECKPOINT_DIR_ROOT is the base directory where models (subdirs) and LoRAs (.safetensors) are stored.
# model_manager.list_checkpoints scans this.
# model_manager.load_model resolves paths relative to this if checkpoint_root_dir is passed.
CHECKPOINT_DIR_ROOT = "checkpoints/" 
DEFAULT_INFERENCE_OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "gui_trainer_audio_outputs")
AVAILABLE_SCHEDULERS_INFERENCE = ["euler", "heun", "pingpong", "dpm++"] # Match ACEStepPipeline if possible
AVAILABLE_CFG_TYPES = ["apg", "cfg", "cfg_star"] # Match ACEStepPipeline
AVAILABLE_OUTPUT_FORMATS = ["wav", "mp3", "flac"]


def create_inference_tab():
    os.makedirs(DEFAULT_INFERENCE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR_ROOT, exist_ok=True) # Ensure root checkpoint dir exists

    # For populating dropdowns - try to list actual models and LoRAs
    base_model_choices = ["None"]
    lora_choices = ["None"]
    try:
        all_ckpt_items = list_checkpoints(CHECKPOINT_DIR_ROOT)
        # Base models are directories, LoRAs are .safetensors files
        base_model_choices.extend([item for item in all_ckpt_items if os.path.isdir(os.path.join(CHECKPOINT_DIR_ROOT, item))])
        lora_choices.extend([item for item in all_ckpt_items if item.endswith(".safetensors")])
        if len(base_model_choices) == 1 and "None" in base_model_choices : # Still just "None"
             base_model_choices.append(f"No models found in {CHECKPOINT_DIR_ROOT}")
        if len(lora_choices) == 1 and "None" in lora_choices:
             lora_choices.append(f"No LoRAs (.safetensors) found in {CHECKPOINT_DIR_ROOT}")
    except Exception as e:
        logger.error(f"Error listing models/LoRAs from '{CHECKPOINT_DIR_ROOT}': {e}", exc_info=True)
        base_model_choices = ["Error loading models"]
        lora_choices = ["Error loading LoRAs"]


    with gr.Blocks() as inference_tab_interface:
        gr.Markdown("# Audio Generation / Inference Tab")
        
        with gr.Row():
            model_dropdown = gr.Dropdown(label="Select Base Model", choices=base_model_choices, value=base_model_choices[0], interactive=True)
            lora_module_dropdown = gr.Dropdown(label="Select LoRA Module (Optional)", choices=lora_choices, value=lora_choices[0], interactive=True)
        
        lora_weight_slider = gr.Slider(label="LoRA Weight", minimum=0.0, maximum=2.0, step=0.05, value=1.0, interactive=True)
        
        prompt_textbox = gr.Textbox(label="Prompt", placeholder="Describe the audio you want to generate...", lines=3, interactive=True)
        lyrics_textbox = gr.Textbox(label="Lyrics (Optional)", placeholder="Enter lyrics here, if applicable...", lines=5, interactive=True)

        with gr.Accordion("Generation Parameters", open=True):
            with gr.Row():
                audio_duration_slider = gr.Slider(label="Audio Duration (seconds)", minimum=1, maximum=300, value=10, step=1, interactive=True)
                infer_steps_slider = gr.Slider(label="Inference Steps", minimum=10, maximum=200, value=50, step=1, interactive=True)
            with gr.Row():
                guidance_scale_slider = gr.Slider(label="Guidance Scale", minimum=0.0, maximum=30.0, value=7.5, step=0.5, interactive=True)
                scheduler_dropdown_inf = gr.Dropdown(label="Scheduler Type", choices=AVAILABLE_SCHEDULERS_INFERENCE, value="euler", interactive=True)
            with gr.Row():
                cfg_type_dropdown = gr.Dropdown(label="CFG Type", choices=AVAILABLE_CFG_TYPES, value="apg", interactive=True)
                omega_scale_number = gr.Number(label="Omega Scale", value=1.0, interactive=True) # Default 10 in original trainer, 1.0 might be safer general
        
        with gr.Accordion("Output Settings & Advanced", open=False):
            manual_seeds_textbox = gr.Textbox(label="Manual Seeds (optional, comma-separated or single integer)", placeholder="e.g., 12345 or 1,2,3", interactive=True)
            output_format_dropdown = gr.Dropdown(label="Output Format", choices=AVAILABLE_OUTPUT_FORMATS, value="wav", interactive=True)
            output_path_textbox = gr.Textbox(label="Output Directory", value=DEFAULT_INFERENCE_OUTPUT_DIR, interactive=True)

        generate_audio_button = gr.Button("Generate Audio", variant="primary")
        
        gr.Markdown("### Output")
        generated_audio_output = gr.Audio(label="Generated Audio", type="filepath", interactive=False)
        inference_status_textbox = gr.Textbox(label="Inference Status", lines=5, interactive=False, placeholder="Inference log and status...")

        def handle_run_inference_wrapper(
            selected_base_model_name, selected_lora_name, lora_weight_val,
            prompt_text, lyrics_text,
            duration_val, steps_val, guidance_val, scheduler_val,
            cfg_type_val, omega_val,
            seeds_str, format_val, output_dir_val
        ):
            status_updates = []
            if selected_base_model_name == "None" or selected_base_model_name.startswith("No models") or selected_base_model_name.startswith("Error loading"):
                return "Error: Please select a valid base model.", None
            if not prompt_text:
                return "Error: Prompt cannot be empty.", None
            
            status_updates.append(f"Selected Base Model: {selected_base_model_name}")
            actual_lora_name = None
            if selected_lora_name != "None" and not selected_lora_name.startswith("No LoRAs") and not selected_lora_name.startswith("Error loading"):
                actual_lora_name = selected_lora_name
                status_updates.append(f"Selected LoRA: {actual_lora_name} (Weight: {lora_weight_val})")
            else:
                status_updates.append("No LoRA module selected or LoRA not found.")

            # Load model (base + optional LoRA)
            status_updates.append("Loading model...")
            # model_manager.load_model expects base_model_path to be the name of the dir/file within checkpoint_root_dir
            # and lora_adapter_filename to also be a name within checkpoint_root_dir
            pipeline_instance = load_model(
                base_model_path=selected_base_model_name, 
                lora_adapter_filename=actual_lora_name, 
                lora_weight=float(lora_weight_val),
                checkpoint_root_dir=CHECKPOINT_DIR_ROOT 
            )

            if not pipeline_instance : # load_model returns None on error
                status_updates.append("Failed to load the model or LoRA adapter.")
                return "\n".join(status_updates), None
            
            status_updates.append("Model loaded successfully.")

            parsed_seeds = None
            if seeds_str and seeds_str.strip():
                try:
                    parsed_seeds = [int(s.strip()) for s in seeds_str.split(',') if s.strip()]
                    if not parsed_seeds: parsed_seeds = None # Handle if all were empty strings
                except ValueError:
                    status_updates.append("Warning: Could not parse seeds. Using random seeds.")
                    parsed_seeds = None
            
            inference_params = {
                "prompt": prompt_text,
                "lyrics": lyrics_text if lyrics_text and lyrics_text.strip() else None,
                "audio_duration": float(duration_val),
                "infer_step": int(steps_val),
                "guidance_scale": float(guidance_val),
                "scheduler_type": scheduler_val,
                "cfg_type": cfg_type_val,
                "omega_scale": float(omega_val),
                "manual_seeds": parsed_seeds, # List of ints or None
                "lora_name_or_path": os.path.join(CHECKPOINT_DIR_ROOT, actual_lora_name) if actual_lora_name else None, # Pass full path if LoRA selected
                "lora_weight": float(lora_weight_val),
                "format": format_val
            }
            status_updates.append(f"Inference parameters: {json.dumps(inference_params, indent=2)}")

            status_updates.append("Starting inference process...")
            audio_file_result, error_msg = run_inference(pipeline_instance, inference_params, output_dir_val)

            if error_msg:
                status_updates.append(f"Inference Error: {error_msg}")
                return "\n".join(status_updates), None
            
            status_updates.append(f"Inference successful! Output: {audio_file_result}")
            return "\n".join(status_updates), audio_file_result

        generate_audio_button.click(
            fn=handle_run_inference_wrapper,
            inputs=[
                model_dropdown, lora_module_dropdown, lora_weight_slider,
                prompt_textbox, lyrics_textbox,
                audio_duration_slider, infer_steps_slider, guidance_scale_slider, scheduler_dropdown_inf,
                cfg_type_dropdown, omega_scale_number,
                manual_seeds_textbox, output_format_dropdown, output_path_textbox
            ],
            outputs=[inference_status_textbox, generated_audio_output]
        )
    return inference_tab_interface

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger.info("--- Running inference_tab.py standalone for testing ---")

    module_dir = os.path.dirname(__file__) if __file__ else "."
    
    # Create dummy inference_utils.py
    inference_utils_path = os.path.join(module_dir, "inference_utils.py")
    if not os.path.exists(inference_utils_path):
        with open(inference_utils_path, "w") as f:
            f.write("""
import os, logging, json, datetime
logger = logging.getLogger(__name__)
def run_inference(pipeline_instance, inference_params, output_dir):
    logger.info(f"Dummy run_inference called. Output_dir: {output_dir}, Params: {inference_params}")
    os.makedirs(output_dir, exist_ok=True)
    prompt_snip = inference_params.get('prompt', 'default')[:10].replace(' ','_')
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{prompt_snip}.{inference_params.get('format','wav')}"
    dummy_audio_path = os.path.join(output_dir, filename)
    with open(dummy_audio_path, "w") as f: f.write("dummy audio data")
    logger.info(f"Dummy audio file created: {dummy_audio_path}")
    return dummy_audio_path, None
""")
        logger.info(f"Created dummy inference_utils.py at {inference_utils_path}")

    # Create dummy model_manager.py
    model_manager_path = os.path.join(module_dir, "model_manager.py")
    if not os.path.exists(model_manager_path):
        with open(model_manager_path, "w") as f:
            f.write("""
import os, logging
logger = logging.getLogger(__name__)

class DummyACEStepPipeline: # Mock pipeline for testing load_model
    def __init__(self, checkpoint_dir, *args, **kwargs): 
        logger.info(f"DummyACEStepPipeline initialized for: {checkpoint_dir}")
    def __call__(self, **kwargs): # To be callable if run_inference expects it
        logger.info(f"DummyACEStepPipeline called with {kwargs}")
        return (["dummy_output_from_pipeline.wav", "dummy_params.json"], {})
    def load_lora(self, lora_name_or_path, lora_weight):
        logger.info(f"DummyACEStepPipeline.load_lora: {lora_name_or_path}, weight: {lora_weight}")


def list_checkpoints(checkpoint_dir_root):
    logger.info(f"Dummy list_checkpoints called for: {checkpoint_dir_root}")
    if not os.path.exists(checkpoint_dir_root): return ["Error: Root checkpoint dir not found"]
    items = []
    try:
        for item in os.listdir(checkpoint_dir_root):
            if os.path.isdir(os.path.join(checkpoint_dir_root, item)) or item.endswith(".safetensors"):
                items.append(item)
    except Exception as e: return [f"Error listing: {e}"]
    return items if items else ["No items found"]

def load_model(base_model_path, lora_adapter_filename=None, lora_weight=1.0, checkpoint_root_dir=None):
    full_base_path = os.path.join(checkpoint_root_dir, base_model_path) if checkpoint_root_dir else base_model_path
    logger.info(f"Dummy load_model: base='{full_base_path}', lora_file='{lora_adapter_filename}', weight={lora_weight}, root='{checkpoint_root_dir}'")
    if not os.path.exists(full_base_path) and base_model_path != "None": # "None" is a valid choice in UI
        logger.error(f"Dummy load_model: Base model path {full_base_path} does not exist.")
        return None # Simulate error
    
    # Simulate successful load by returning a dummy pipeline instance
    pipeline = DummyACEStepPipeline(checkpoint_dir=full_base_path)
    if lora_adapter_filename and lora_adapter_filename != "None":
        lora_full_path = os.path.join(checkpoint_root_dir, lora_adapter_filename) if checkpoint_root_dir else lora_adapter_filename
        if not os.path.exists(lora_full_path):
            logger.warning(f"Dummy load_model: LoRA path {lora_full_path} does not exist. Returning base model only.")
        else:
            pipeline.load_lora(lora_full_path, lora_weight) # Call dummy load_lora
    return pipeline
""")
        logger.info(f"Created dummy model_manager.py at {model_manager_path}")

    # Ensure CHECKPOINT_DIR_ROOT exists and has some dummy content for dropdowns
    if not os.path.exists(CHECKPOINT_DIR_ROOT): os.makedirs(CHECKPOINT_DIR_ROOT)
    dummy_model_dir = os.path.join(CHECKPOINT_DIR_ROOT, "dummy_base_model_v1")
    if not os.path.exists(dummy_model_dir): os.makedirs(dummy_model_dir)
    dummy_lora_file = os.path.join(CHECKPOINT_DIR_ROOT, "dummy_lora_adapter.safetensors")
    if not os.path.exists(dummy_lora_file): 
        with open(dummy_lora_file, "w") as f: f.write("dummy lora content")
    logger.info(f"Ensured dummy model structure in {CHECKPOINT_DIR_ROOT}")
    
    if not os.path.exists(DEFAULT_INFERENCE_OUTPUT_DIR):
        os.makedirs(DEFAULT_INFERENCE_OUTPUT_DIR, exist_ok=True)

    tab_interface = create_inference_tab()
    tab_interface.launch(debug=True)
