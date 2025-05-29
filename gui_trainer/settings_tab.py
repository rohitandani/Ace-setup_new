import gradio as gr
from .model_manager import list_checkpoints 
import os
import json
import logging

logger = logging.getLogger(__name__)

SETTINGS_FILE_PATH = os.path.join(os.path.dirname(__file__), "app_settings.json")
DEFAULT_MODEL_CHECKPOINT_DIR_SETTINGS = "checkpoints/" 
AVAILABLE_GPUS = ["Auto", "GPU 0", "GPU 1", "GPU 2", "CPU"]

DEFAULT_SETTINGS = {
    "detailed_logging": False,
    "default_model": "None",
    "preferred_gpu": "Auto"
}

def _save_settings_to_file(settings_dict, file_path):
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(settings_dict, f, indent=4)
        logger.info(f"Settings saved to {file_path}")
        return True, f"Settings saved successfully to {file_path}"
    except IOError as e:
        logger.error(f"Error saving settings to {file_path}: {e}", exc_info=True)
        return False, f"Error saving settings: {e}"
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving settings: {e}", exc_info=True)
        return False, f"Unexpected error saving settings: {e}"

def _load_settings_from_file(file_path):
    if not os.path.exists(file_path):
        logger.info(f"Settings file {file_path} not found. Returning default settings.")
        return DEFAULT_SETTINGS.copy() # Return a copy to avoid modifying the original
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            loaded_settings = json.load(f)
        logger.info(f"Settings loaded from {file_path}")
        # Ensure all default keys are present
        final_settings = DEFAULT_SETTINGS.copy()
        final_settings.update(loaded_settings) # Override defaults with loaded values
        return final_settings
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {file_path}: {e}. Returning default settings.")
        return DEFAULT_SETTINGS.copy()
    except IOError as e:
        logger.error(f"Error loading settings from {file_path}: {e}. Returning default settings.", exc_info=True)
        return DEFAULT_SETTINGS.copy()
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading settings: {e}. Returning default settings.", exc_info=True)
        return DEFAULT_SETTINGS.copy()


def create_settings_tab():
    initial_settings = _load_settings_from_file(SETTINGS_FILE_PATH)
    logger.debug(f"Initial settings loaded for UI: {initial_settings}")

    with gr.Blocks() as settings_tab_interface:
        gr.Markdown("# Application Settings Tab")

        checkpoint_list = ["None"] # Start with None
        try:
            if not os.path.exists(DEFAULT_MODEL_CHECKPOINT_DIR_SETTINGS):
                 os.makedirs(DEFAULT_MODEL_CHECKPOINT_DIR_SETTINGS, exist_ok=True)
            
            # Filter for directories as base models, and .safetensors for LoRAs (though settings tab focuses on base models)
            raw_checkpoints = list_checkpoints(DEFAULT_MODEL_CHECKPOINT_DIR_SETTINGS)
            model_dirs = [item for item in raw_checkpoints if os.path.isdir(os.path.join(DEFAULT_MODEL_CHECKPOINT_DIR_SETTINGS, item))]
            
            if model_dirs:
                checkpoint_list.extend(model_dirs)
            elif not model_dirs and raw_checkpoints: # If list_checkpoints returned something, but no dirs
                 checkpoint_list.append(f"No model directories found in {DEFAULT_MODEL_CHECKPOINT_DIR_SETTINGS}")
            # else: (if raw_checkpoints is empty, "None" is already there)
            
        except Exception as e:
            logger.error(f"Error listing checkpoints for settings: {e}", exc_info=True)
            checkpoint_list.append("Error loading models")
        
        # Ensure loaded default model is valid, otherwise revert to "None" or first valid
        loaded_default_model = initial_settings.get("default_model", "None")
        if loaded_default_model not in checkpoint_list and loaded_default_model != "None":
            logger.warning(f"Saved default model '{loaded_default_model}' not found in available choices. Reverting to 'None'.")
            loaded_default_model = "None"
        elif loaded_default_model == "None" and len(checkpoint_list) > 1 and checkpoint_list[0] == "None":
             pass # "None" is a valid choice
        elif not checkpoint_list or checkpoint_list == ["None"]: # If no models found at all
            loaded_default_model = "None"


        with gr.Column():
            gr.Markdown("## General Settings")
            detailed_logging_checkbox = gr.Checkbox(
                label="Enable Detailed Logging (More verbose console/file logs)", 
                value=initial_settings.get("detailed_logging", False),
                interactive=True
            )
            
            gr.Markdown("## Model Settings")
            default_model_dropdown = gr.Dropdown(
                label="Default Base Model (for Inference/Training startup)",
                choices=checkpoint_list,
                value=loaded_default_model,
                interactive=True
            )

            gr.Markdown("## Hardware Settings")
            preferred_gpu_radio = gr.Radio(
                label="Preferred Compute Device (Requires app restart to take full effect)",
                choices=AVAILABLE_GPUS,
                value=initial_settings.get("preferred_gpu", "Auto"),
                interactive=True
            )
            
        save_settings_button = gr.Button("Save Settings")
        status_textbox = gr.Textbox(label="Status", interactive=False, placeholder="Settings status...")

        def handle_save_settings_click(logging_enabled, default_model_val, selected_gpu_val):
            current_settings = {
                "detailed_logging": logging_enabled,
                "default_model": default_model_val if default_model_val != "None" else None, # Store None if "None" string
                "preferred_gpu": selected_gpu_val
            }
            logger.debug(f"Attempting to save settings: {current_settings}")
            success, message = _save_settings_to_file(current_settings, SETTINGS_FILE_PATH)
            return message

        save_settings_button.click(
            fn=handle_save_settings_click,
            inputs=[detailed_logging_checkbox, default_model_dropdown, preferred_gpu_radio],
            outputs=status_textbox
        )
        
        # Optionally, refresh UI components if settings could be changed elsewhere (not strictly needed here)
        # def refresh_ui_from_loaded_settings():
        #     loaded = _load_settings_from_file(SETTINGS_FILE_PATH)
        #     return {
        #         detailed_logging_checkbox: gr.update(value=loaded.get("detailed_logging")),
        #         default_model_dropdown: gr.update(value=loaded.get("default_model") or "None"), # Ensure "None" string if key is missing or None
        #         preferred_gpu_radio: gr.update(value=loaded.get("preferred_gpu"))
        #     }
        # settings_tab_interface.load(fn=refresh_ui_from_loaded_settings, outputs=[detailed_logging_checkbox, default_model_dropdown, preferred_gpu_radio])


    return settings_tab_interface

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger.info("--- Testing settings_tab.py ---")
    
    module_dir = os.path.dirname(__file__) if __file__ else "."
    
    # Dummy model_manager.py for list_checkpoints
    dummy_mm_path = os.path.join(module_dir, "model_manager.py")
    if not os.path.exists(dummy_mm_path):
        with open(dummy_mm_path, "w") as f:
            f.write("""
import os, logging
logger = logging.getLogger(__name__)
def list_checkpoints(checkpoint_dir):
    logger.debug(f"Dummy list_checkpoints for settings tab, dir: {checkpoint_dir}")
    if not os.path.exists(checkpoint_dir): return []
    try: return [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))] + [f for f in os.listdir(checkpoint_dir) if f.endswith('.safetensors')]
    except Exception as e: logger.error(f"Dummy list_checkpoints error: {e}"); return []
def load_model(p, lora_adapter_filename=None, lora_weight=1.0, checkpoint_root_dir=None): pass # Not used by settings tab directly
""")
        logger.info(f"Created dummy model_manager.py at {dummy_mm_path}")

    # Ensure DEFAULT_MODEL_CHECKPOINT_DIR_SETTINGS exists and has some content
    if not os.path.exists(DEFAULT_MODEL_CHECKPOINT_DIR_SETTINGS):
        os.makedirs(DEFAULT_MODEL_CHECKPOINT_DIR_SETTINGS, exist_ok=True)
    
    dummy_model_dir_for_test = os.path.join(DEFAULT_MODEL_CHECKPOINT_DIR_SETTINGS, "test_model_settings")
    if not os.path.exists(dummy_model_dir_for_test):
        os.makedirs(dummy_model_dir_for_test)
        with open(os.path.join(dummy_model_dir_for_test, "config.json"), "w") as f: json.dump({"name":"test"},f)
    logger.info(f"Ensured dummy model dir for settings test: {dummy_model_dir_for_test}")


    # Test saving settings
    logger.info("Testing settings save...")
    test_settings_save = {"detailed_logging": True, "default_model": "test_model_settings", "preferred_gpu": "GPU 1"}
    _save_settings_to_file(test_settings_save, SETTINGS_FILE_PATH)

    # Test loading settings
    logger.info("Testing settings load...")
    loaded_s = _load_settings_from_file(SETTINGS_FILE_PATH)
    logger.info(f"Loaded settings for test: {loaded_s}")
    assert loaded_s["detailed_logging"] == True
    assert loaded_s["default_model"] == "test_model_settings"
    assert loaded_s["preferred_gpu"] == "GPU 1"

    # Test loading non-existent file (should return defaults)
    logger.info("Testing load from non-existent file...")
    non_existent_path = os.path.join(module_dir, "non_existent_settings.json")
    if os.path.exists(non_existent_path): os.remove(non_existent_path) # Ensure it doesn't exist
    default_s = _load_settings_from_file(non_existent_path)
    logger.info(f"Settings from non-existent file (defaults): {default_s}")
    assert default_s == DEFAULT_SETTINGS

    logger.info("Launching Settings Tab UI for manual inspection...")
    # The UI will load settings from SETTINGS_FILE_PATH (which now has test_settings_save content)
    tab_interface = create_settings_tab()
    tab_interface.launch(debug=True)

    # Clean up dummy settings file after test if needed
    # if os.path.exists(SETTINGS_FILE_PATH):
    #     os.remove(SETTINGS_FILE_PATH)
    #     logger.info(f"Cleaned up test settings file: {SETTINGS_FILE_PATH}")
    # if os.path.exists(dummy_model_dir_for_test): # Clean up dummy model dir
    #     import shutil
    #     shutil.rmtree(dummy_model_dir_for_test)
    #     logger.info(f"Cleaned up dummy model directory: {dummy_model_dir_for_test}")
    # if os.path.exists(dummy_mm_path): os.remove(dummy_mm_path)
    
    logger.info("Settings tab test finished. Press Ctrl+C to exit Gradio if it's running.")
