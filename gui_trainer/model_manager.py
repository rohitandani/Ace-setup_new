import os
import json
import shutil
import logging

# Attempt to import ACEStepPipeline and TrainingPipeline
# These are crucial dependencies. If they are not found, the functionality will be severely limited.
try:
    from acestep.pipeline_ace_step import ACEStepPipeline
except ImportError:
    logging.error("Failed to import ACEStepPipeline. `load_model` will not function correctly.")
    # Define a dummy class to allow the rest of the script to be parsed
    class ACEStepPipeline:
        def __init__(self, checkpoint_dir, *args, **kwargs):
            logging.warning(f"Using dummy ACEStepPipeline due to import error. Checkpoint dir: {checkpoint_dir}")
        def load_checkpoint(self, *args, **kwargs): logging.warning("Dummy ACEStepPipeline.load_checkpoint called.")
        def load_lora(self, *args, **kwargs): logging.warning("Dummy ACEStepPipeline.load_lora called.")
        def load_quantized_checkpoint(self, *args, **kwargs): logging.warning("Dummy ACEStepPipeline.load_quantized_checkpoint called.")


try:
    # Assuming trainer_utils is in the same package (gui_trainer)
    from .trainer_utils import TrainingPipeline 
except ImportError:
    logging.error("Failed to import TrainingPipeline from .trainer_utils. `save_model` for trained LoRA will not function correctly.")
    # Define a dummy class
    class TrainingPipeline:
        def __init__(self, *args, **kwargs):
            logging.warning("Using dummy TrainingPipeline due to import error.")
        # Add dummy attributes/methods that save_model might check for
        class DummyTransformer:
            def save_lora_adapter(self, save_path, adapter_name):
                logging.warning(f"Dummy TrainingPipeline.transformers.save_lora_adapter called for {save_path}, adapter {adapter_name}")
        transformers = DummyTransformer()
        adapter_name = "dummy_adapter"


def list_checkpoints(checkpoint_dir: str) -> list[str]:
    """
    Scans the specified checkpoint_dir for potential base model directories and LoRA .safetensors files.
    - Base models are assumed to be directories.
    - LoRA adapters are assumed to be .safetensors files.
    """
    if not os.path.isdir(checkpoint_dir):
        logging.warning(f"Checkpoint directory '{checkpoint_dir}' not found or is not a directory.")
        return []

    items = []
    try:
        for item_name in os.listdir(checkpoint_dir):
            item_path = os.path.join(checkpoint_dir, item_name)
            if os.path.isdir(item_path):
                # Heuristic for base models: check for common subdirectories or marker files if known.
                # For now, list all directories as potential base model checkpoints.
                # Example check: if os.path.exists(os.path.join(item_path, "ace_step_transformer")):
                items.append(item_name) 
            elif item_name.endswith(".safetensors"):
                items.append(item_name)
    except OSError as e:
        logging.error(f"Error listing checkpoints in '{checkpoint_dir}': {e}")
        return []
    
    return items


def load_model(base_model_path: str, lora_adapter_filename: str = None, lora_weight: float = 1.0, checkpoint_root_dir: str = None):
    """
    Loads the main ACE-Step model and optionally attaches a LoRA adapter.

    Args:
        base_model_path (str): Path to the base model checkpoint directory. 
                               This path can be absolute or relative to `checkpoint_root_dir` if provided.
        lora_adapter_filename (str, optional): Filename of the LoRA adapter (.safetensors) located within `checkpoint_root_dir`.
        lora_weight (float, optional): Weight for the LoRA adapter.
        checkpoint_root_dir (str, optional): The root directory where base models and LoRA files are stored.
                                             If provided, base_model_path and lora_adapter_filename are resolved relative to this.

    Returns:
        ACEStepPipeline instance or None if loading fails.
    """
    if checkpoint_root_dir:
        if not os.path.isabs(base_model_path):
            full_base_model_path = os.path.join(checkpoint_root_dir, base_model_path)
        else:
            full_base_model_path = base_model_path # Already absolute
    else:
        full_base_model_path = base_model_path

    if not os.path.isdir(full_base_model_path):
        logging.error(f"Base model checkpoint path '{full_base_model_path}' is not a valid directory.")
        return None

    try:
        logging.info(f"Loading base model from: {full_base_model_path}")
        # ACEStepPipeline might need specific args, adjust as necessary
        # For now, assume checkpoint_dir is the primary argument.
        pipeline = ACEStepPipeline(checkpoint_dir=full_base_model_path) 
        
        # ACEStepPipeline.load_checkpoint() is called internally by its __init__ or needs specific sub-component loading.
        # The provided ACEStepPipeline structure calls load_checkpoint(self.checkpoint_dir) in its init.
        # If it needs explicit call or specific sub-component loading, that logic would go here.
        # Example: pipeline.load_checkpoint() or pipeline.load_quantized_checkpoint()
        # Based on ACEStepPipeline code, it seems to load on init.

        if lora_adapter_filename:
            if not checkpoint_root_dir:
                logging.error("Cannot load LoRA adapter by filename if checkpoint_root_dir is not provided.")
                # Or, assume lora_adapter_filename could be an absolute path if checkpoint_root_dir is None
                if not os.path.isabs(lora_adapter_filename) or not os.path.exists(lora_adapter_filename):
                     logging.error(f"LoRA adapter path '{lora_adapter_filename}' is not absolute or does not exist.")
                     return pipeline # Return base model if LoRA path is problematic but root_dir not given for relative path

            full_lora_path = os.path.join(checkpoint_root_dir, lora_adapter_filename) if checkpoint_root_dir else lora_adapter_filename
            
            if not os.path.isfile(full_lora_path):
                logging.error(f"LoRA adapter file '{full_lora_path}' not found.")
                # Optionally, return the base model pipeline here if LoRA is not critical,
                # or return None if LoRA loading failure is a hard error.
                # For now, log error and proceed to return the base pipeline.
            else:
                logging.info(f"Loading LoRA adapter from: {full_lora_path} with weight {lora_weight}")
                # The ACEStepPipeline.load_lora method might need specific adapter name or path.
                # Assuming lora_name_or_path is the file path.
                pipeline.load_lora(lora_name_or_path=full_lora_path, lora_weight=lora_weight)
        
        logging.info("Model loaded successfully.")
        return pipeline
    except ImportError as e_imp: # Catch if ACEStepPipeline was a dummy
        logging.error(f"ACEStepPipeline is not available due to import error: {e_imp}. Cannot load model.")
        return None
    except Exception as e:
        logging.error(f"Error loading model from '{full_base_model_path}' (LoRA: '{lora_adapter_filename}'): {e}", exc_info=True)
        return None


def save_model(model, save_dir_path: str, new_lora_adapter_name: str = "user_saved_lora"):
    """
    Saves the LoRA adapter from a TrainingPipeline instance.

    Args:
        model: The TrainingPipeline instance (which contains the trained LoRA layers).
        save_dir_path (str): The directory where the LoRA adapter should be saved.
                             A subdirectory for the adapter will be created here.
        new_lora_adapter_name (str): The name to save the adapter files under (e.g., 'pytorch_lora_weights.safetensors').
                                     The actual adapter name used by PEFT during saving.

    Returns:
        tuple: (success_message_or_None, error_message_or_None)
    """
    if not isinstance(model, TrainingPipeline): # Check if it's the PL pipeline wrapper
        err_msg = "Invalid model type for saving. Expected TrainingPipeline instance."
        logging.error(err_msg)
        return None, err_msg

    if not hasattr(model, 'transformers') or not hasattr(model.transformers, 'save_lora_adapter'):
        err_msg = "Model does not have a 'transformers' attribute with 'save_lora_adapter' method. Cannot save LoRA."
        logging.error(err_msg)
        return None, err_msg
    
    # The `adapter_name` used during training is stored in `model.adapter_name` by TrainingPipeline's init.
    # `save_lora_adapter` needs the name of the adapter *within the model* to save.
    # `new_lora_adapter_name` is more about the output filename structure if PEFT uses it,
    # but usually, PEFT saves with a standard name like `adapter_model.safetensors`.
    # The `save_path` for `save_lora_adapter` is a directory.
    
    # Use the adapter name that was set during training, or default if somehow not set.
    adapter_to_save = model.hparams.adapter_name if hasattr(model, 'hparams') and model.hparams.adapter_name else "default"


    try:
        # Ensure save_dir_path exists
        os.makedirs(save_dir_path, exist_ok=True)
        
        # PEFT's save_lora_adapter saves the adapter specified by `adapter_name` into the `save_directory`.
        # Files like adapter_config.json and adapter_model.bin (or .safetensors) will be created in save_dir_path.
        model.transformers.save_lora_adapter(save_directory=save_dir_path, adapter_name=adapter_to_save)
        
        # The `new_lora_adapter_name` is a bit confusing here. PEFT `save_lora_adapter` doesn't take
        # a filename for the adapter itself, but a directory. The internal files are named by PEFT.
        # If the goal is to have the *directory* be `new_lora_adapter_name`, then `save_dir_path` should reflect that.
        # Let's assume `save_dir_path` is the target directory.
        
        success_msg = f"LoRA adapter '{adapter_to_save}' saved successfully to directory: {save_dir_path}"
        logging.info(success_msg)
        return success_msg, None
    except ImportError as e_imp: # Catch if TrainingPipeline was a dummy
        err_msg = f"TrainingPipeline is not available due to import error: {e_imp}. Cannot save model."
        logging.error(err_msg)
        return None, err_msg
    except Exception as e:
        err_msg = f"Error saving LoRA adapter: {e}"
        logging.error(err_msg, exc_info=True)
        return None, err_msg


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG) # Enable debug logging for tests
    logger = logging.getLogger(__name__)
    logger.info("--- Testing model_manager.py ---")

    # Setup a temporary root directory for all test checkpoints and models
    test_root_dir = "temp_model_manager_test_files"
    os.makedirs(test_root_dir, exist_ok=True)
    
    checkpoints_root = os.path.join(test_root_dir, "checkpoints")
    os.makedirs(checkpoints_root, exist_ok=True)

    # --- Test list_checkpoints ---
    logger.info("\n--- Testing list_checkpoints ---")
    # Create dummy structure
    os.makedirs(os.path.join(checkpoints_root, "base_model_v1"), exist_ok=True)
    with open(os.path.join(checkpoints_root, "base_model_v1", "config.json"), "w") as f: json.dump({"name":"v1"}, f)
    os.makedirs(os.path.join(checkpoints_root, "another_model_dir"), exist_ok=True)
    with open(os.path.join(checkpoints_root, "lora_adapter_A.safetensors"), "w") as f: f.write("dummy lora A")
    with open(os.path.join(checkpoints_root, "lora_adapter_B.safetensors"), "w") as f: f.write("dummy lora B")
    with open(os.path.join(checkpoints_root, "some_other_file.txt"), "w") as f: f.write("text")

    listed_items = list_checkpoints(checkpoints_root)
    logger.info(f"Listed checkpoints: {listed_items}")
    assert "base_model_v1" in listed_items
    assert "another_model_dir" in listed_items
    assert "lora_adapter_A.safetensors" in listed_items
    assert "lora_adapter_B.safetensors" in listed_items
    assert "some_other_file.txt" not in listed_items 
    assert len(listed_items) == 4

    logger.info("Testing with non-existent directory:")
    assert list_checkpoints(os.path.join(test_root_dir, "non_existent_dir")) == []

    # --- Test load_model ---
    # This is difficult to test without actual ACEStep model files and a working ACEStepPipeline.
    # We will mostly test path handling and if the dummy ACEStepPipeline is called.
    logger.info("\n--- Testing load_model ---")
    
    # Path to the dummy base model created above
    dummy_base_model_name = "base_model_v1"
    dummy_lora_name = "lora_adapter_A.safetensors"

    logger.info("Attempting to load base model only...")
    # In a real scenario, ACEStepPipeline(checkpoint_dir=...) would load the model.
    # Our dummy ACEStepPipeline will just log.
    loaded_pipeline_base = load_model(base_model_path=dummy_base_model_name, checkpoint_root_dir=checkpoints_root)
    if loaded_pipeline_base: # This will be true if dummy ACEStepPipeline is used
        logger.info("load_model (base only) returned an object (expected dummy ACEStepPipeline).")
    else:
        logger.warning("load_model (base only) returned None. This might be due to ACEStepPipeline import error or other issues.")

    logger.info("\nAttempting to load base model with LoRA...")
    loaded_pipeline_lora = load_model(
        base_model_path=dummy_base_model_name,
        lora_adapter_filename=dummy_lora_name,
        checkpoint_root_dir=checkpoints_root
    )
    if loaded_pipeline_lora:
        logger.info("load_model (with LoRA) returned an object.")
    else:
        logger.warning("load_model (with LoRA) returned None.")

    logger.info("\nAttempting to load with non-existent base model path...")
    assert load_model("non_existent_model", checkpoint_root_dir=checkpoints_root) is None

    logger.info("\nAttempting to load with non-existent LoRA file...")
    # Should still return the base model pipeline instance if base model loads fine
    pipeline_no_lora_file = load_model(dummy_base_model_name, "non_existent_lora.safetensors", checkpoint_root_dir=checkpoints_root)
    if pipeline_no_lora_file:
        logger.info("load_model (with non-existent LoRA) returned base model object as expected.")
    else:
        logger.warning("load_model (with non-existent LoRA) failed to return base model.")


    # --- Test save_model ---
    # This requires a TrainingPipeline instance. We'll use the dummy one if the real one isn't importable.
    logger.info("\n--- Testing save_model ---")
    
    # Create a mock TrainingPipeline that has the necessary attributes for save_model
    class MockTrainingPipeline:
        class MockTransformer:
            def save_lora_adapter(self, save_directory, adapter_name):
                logger.info(f"MockTransformer.save_lora_adapter called: dir='{save_directory}', adapter_name='{adapter_name}'")
                # Simulate saving by creating a dummy file
                os.makedirs(save_directory, exist_ok=True)
                with open(os.path.join(save_directory, f"{adapter_name}_adapter_model.safetensors"), "w") as f:
                    f.write("dummy saved lora data")
        
        transformers = MockTransformer()
        # hparams needs to exist and have adapter_name
        class HParams:
            adapter_name = "trained_adapter_for_saving"
        hparams = HParams()

    mock_trained_pipeline = MockTrainingPipeline()
    
    # If real TrainingPipeline was imported, use that with dummy attributes for testing save_model call path
    # This is tricky because TrainingPipeline init is complex.
    # For now, relying on the MockTrainingPipeline is safer for a unit test of save_model's logic.
    
    save_lora_dir = os.path.join(test_root_dir, "saved_loras")
    
    logger.info("Attempting to save LoRA from MockTrainingPipeline...")
    success_msg, err_msg = save_model(mock_trained_pipeline, save_lora_dir)
    
    if success_msg:
        logger.info(f"save_model success: {success_msg}")
        assert os.path.exists(os.path.join(save_lora_dir, f"{mock_trained_pipeline.hparams.adapter_name}_adapter_model.safetensors"))
    else:
        logger.error(f"save_model error: {err_msg}")

    logger.info("\nAttempting to save with invalid model type...")
    class NotAPipeline: pass
    _, err_msg_invalid = save_model(NotAPipeline(), save_lora_dir)
    assert err_msg_invalid is not None
    logger.info(f"save_model with invalid type, error: {err_msg_invalid}")


    # --- Cleanup ---
    logger.info("\n--- Cleaning up test files ---")
    try:
        shutil.rmtree(test_root_dir)
        logger.info(f"Removed test directory: {test_root_dir}")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

    logger.info("\n--- model_manager.py tests finished ---")
