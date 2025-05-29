import os
import json
import logging
import datetime
import tempfile
import shutil # For cleanup

# Attempt to import ACEStepPipeline
try:
    from acestep.pipeline_ace_step import ACEStepPipeline
except ImportError:
    logging.error("Failed to import ACEStepPipeline. `run_inference` will use a mock object for testing if run directly.")
    # Define a dummy/mock class for testing if the real one isn't available
    class ACEStepPipeline:
        def __init__(self, checkpoint_dir=None, *args, **kwargs):
            self.checkpoint_dir = checkpoint_dir
            logging.info(f"Dummy ACEStepPipeline initialized with checkpoint_dir: {checkpoint_dir}")

        def __call__(self, save_path:str = None, format:str = "wav", **inference_params):
            logging.info(f"Dummy ACEStepPipeline.__call__ invoked with params: {inference_params}")
            logging.info(f"Dummy ACEStepPipeline.__call__ save_path (base): {save_path}, format: {format}")
            
            # Simulate file creation
            # save_path is base, pipeline appends format
            simulated_audio_filename = f"{os.path.basename(save_path)}.{format}"
            # Ensure the directory for save_path exists if save_path includes directory parts
            simulated_audio_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else "."
            # If save_path was just a filename, place it in a temp dir for the mock
            if not os.path.dirname(save_path): # e.g. save_path = "my_audio"
                simulated_audio_dir = tempfile.mkdtemp(prefix="dummy_inference_out_")
                # Need to reconstruct full path
                full_simulated_audio_path = os.path.join(simulated_audio_dir, simulated_audio_filename)
            else: # save_path = "some/dir/my_audio"
                os.makedirs(simulated_audio_dir, exist_ok=True)
                full_simulated_audio_path = f"{save_path}.{format}"


            with open(full_simulated_audio_path, "w") as f:
                f.write("dummy audio data")
            
            simulated_params_path = f"{save_path}_params.json"
            if not os.path.dirname(save_path): # if save_path was just filename
                 simulated_params_path = os.path.join(simulated_audio_dir, f"{os.path.basename(save_path)}_params.json")

            with open(simulated_params_path, "w") as f:
                json.dump(inference_params, f, indent=2)
            
            logging.info(f"Dummy ACEStepPipeline created: {full_simulated_audio_path} and {simulated_params_path}")
            return ([full_simulated_audio_path, simulated_params_path], inference_params)


def run_inference(
    pipeline_instance: ACEStepPipeline, 
    inference_params: dict, 
    output_dir: str
) -> tuple[str | None, str | None]:
    """
    Runs inference using a pre-loaded ACEStepPipeline instance.

    Args:
        pipeline_instance: An initialized instance of ACEStepPipeline.
        inference_params: Dictionary of parameters for ACEStepPipeline.__call__.
                          Example keys: 'prompt', 'lyrics', 'audio_duration', 'infer_step', 
                                        'guidance_scale', 'scheduler_type', 'cfg_type', 
                                        'omega_scale', 'manual_seeds', 'lora_name_or_path', 
                                        'lora_weight', 'format'.
        output_dir: Directory to save the generated audio and parameters JSON.

    Returns:
        A tuple (audio_file_path, error_message). 
        If successful, audio_file_path is populated and error_message is None.
        If an error occurs, audio_file_path is None and error_message contains the error.
    """
    if not isinstance(pipeline_instance, ACEStepPipeline):
        err_msg = "Invalid pipeline_instance: Not an ACEStepPipeline object."
        logging.error(err_msg)
        return None, err_msg

    if not os.path.isdir(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            logging.info(f"Created output directory: {output_dir}")
        except OSError as e:
            err_msg = f"Failed to create output directory '{output_dir}': {e}"
            logging.error(err_msg)
            return None, err_msg

    # Generate a unique base filename for the output files
    # Using timestamp and a snippet of the prompt (if available)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_snippet = inference_params.get('prompt', 'noprompt')[:30].replace(" ", "_").replace("/", "_")
    base_filename = f"{timestamp}_{prompt_snippet}"
    
    # ACEStepPipeline.__call__ expects `save_path` to be the path *without* extension.
    # It will append ".{format}" and "_params.json" itself.
    full_save_path_base = os.path.join(output_dir, base_filename)

    # Ensure 'format' is in inference_params, default to 'wav' if not.
    if 'format' not in inference_params:
        inference_params['format'] = 'wav'
    
    logging.info(f"Running inference. Output base path: {full_save_path_base}. Params: {inference_params}")

    try:
        # The __call__ method of ACEStepPipeline is expected to handle LoRA application internally
        # if lora_name_or_path is provided in inference_params, based on its own logic.
        # The pipeline_instance should ideally already be configured with the correct LoRA by model_manager.
        
        # Pass the base path for saving; the pipeline will add extensions.
        result_paths, _ = pipeline_instance(**inference_params, save_path=full_save_path_base)
        
        if not result_paths or not isinstance(result_paths, list) or len(result_paths) < 1:
            err_msg = "Inference call did not return expected output paths."
            logging.error(err_msg)
            return None, err_msg

        audio_output_path = result_paths[0] # First item is expected to be the audio path

        if not os.path.exists(audio_output_path):
            err_msg = f"Inference call reported success, but output audio file '{audio_output_path}' not found."
            logging.error(err_msg)
            # Attempt to find it if naming was slightly different
            expected_audio_file = f"{full_save_path_base}.{inference_params['format']}"
            if os.path.exists(expected_audio_file):
                audio_output_path = expected_audio_file
                logging.info(f"Found audio file at expected location: {audio_output_path}")
            else:
                return None, err_msg
        
        logging.info(f"Inference successful. Audio saved to: {audio_output_path}")
        return audio_output_path, None

    except Exception as e:
        err_msg = f"An error occurred during inference: {e}"
        logging.error(err_msg, exc_info=True)
        return None, err_msg


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG) # Enable debug logging for tests
    logger = logging.getLogger(__name__)
    logger.info("--- Testing inference_utils.py ---")

    # Create a temporary output directory for generated files
    temp_output_dir = tempfile.mkdtemp(prefix="inference_test_output_")
    logger.info(f"Test output directory: {temp_output_dir}")

    # Instantiate the pipeline (real or mock)
    # If ACEStepPipeline was imported, this will be the dummy. Otherwise, it might be the real one if `acestep` is installed.
    # For a controlled test, we rely on the dummy defined above if the real one isn't available or for simplicity.
    
    # Scenario 1: Using the (potentially dummy) ACEStepPipeline
    # No real checkpoint needed if using the dummy defined in this file.
    # If real ACEStepPipeline is used, it needs a valid checkpoint_dir.
    # For this test, we assume the dummy will be used if real one fails to import.
    test_pipeline = ACEStepPipeline(checkpoint_dir="dummy_checkpoint_for_test")

    # Define sample inference parameters
    sample_params = {
        "prompt": "A beautiful orchestral piece",
        "lyrics": "la la la", # Optional
        "audio_duration": 5.0, # seconds
        "infer_step": 20,
        "guidance_scale": 7.5,
        "scheduler_type": "euler", # Ensure this matches what ACEStepPipeline expects
        "cfg_type": "apg",
        "omega_scale": 1,
        "manual_seeds": [12345],
        # "lora_name_or_path": None, # Test without LoRA first
        # "lora_weight": 0.8,
        "format": "mp3" # Test with mp3
    }

    logger.info(f"\n--- Test 1: Running inference with sample parameters (format: {sample_params['format']}) ---")
    audio_file, error = run_inference(test_pipeline, sample_params.copy(), temp_output_dir)

    if error:
        logger.error(f"Test 1 Inference failed: {error}")
    else:
        logger.info(f"Test 1 Inference successful. Audio file: {audio_file}")
        assert audio_file is not None
        assert os.path.exists(audio_file), f"Output audio file {audio_file} does not exist."
        assert audio_file.endswith(f".{sample_params['format']}")
        
        # Check for the associated params file (assuming ACEStepPipeline creates it like name_params.json)
        expected_params_file_base = os.path.splitext(audio_file)[0] # remove .mp3
        expected_params_file = f"{expected_params_file_base}_params.json"
        if not os.path.exists(expected_params_file) and isinstance(test_pipeline, ACEStepPipeline) and not hasattr(test_pipeline, '_is_dummy_placeholder'):
             # If it's the dummy defined in THIS file, the params file path logic is slightly different
             # The dummy saves params as {save_path}_params.json, where save_path itself has no extension
             # So, if audio_file is .../output/file.mp3, save_path was .../output/file
             # then params file is .../output/file_params.json
             base_name_from_audio = os.path.basename(os.path.splitext(audio_file)[0])
             expected_params_file = os.path.join(os.path.dirname(audio_file), f"{base_name_from_audio}_params.json")


        assert os.path.exists(expected_params_file), f"Params JSON file {expected_params_file} does not exist."
        logger.info(f"Params JSON file found: {expected_params_file}")


    # Test with a different format (wav)
    sample_params_wav = sample_params.copy()
    sample_params_wav["format"] = "wav"
    sample_params_wav["prompt"] = "Another test prompt for wav"
    logger.info(f"\n--- Test 2: Running inference with sample parameters (format: {sample_params_wav['format']}) ---")
    audio_file_wav, error_wav = run_inference(test_pipeline, sample_params_wav, temp_output_dir)
    if error_wav:
        logger.error(f"Test 2 Inference failed: {error_wav}")
    else:
        logger.info(f"Test 2 Inference successful. Audio file: {audio_file_wav}")
        assert audio_file_wav is not None
        assert os.path.exists(audio_file_wav)
        assert audio_file_wav.endswith(f".{sample_params_wav['format']}")


    # Test with invalid pipeline instance
    logger.info("\n--- Test 3: Running inference with invalid pipeline instance ---")
    class NotAPipeline: pass
    _, error_invalid_pipeline = run_inference(NotAPipeline(), sample_params, temp_output_dir)
    assert error_invalid_pipeline is not None
    logger.info(f"Test 3 (invalid pipeline) error: {error_invalid_pipeline}")
    

    # Clean up
    logger.info(f"\n--- Cleaning up test output directory: {temp_output_dir} ---")
    try:
        shutil.rmtree(temp_output_dir)
        logger.info("Test output directory cleaned up.")
    except Exception as e:
        logger.error(f"Error cleaning up test directory: {e}")

    logger.info("\n--- inference_utils.py tests finished ---")
