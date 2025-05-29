import os
import shutil
import tempfile
from datasets import Dataset
import time # For unique temp subdirectories

# (Old parameters: audio_files, tags, common_tags_mode, tag_input, shuffle_tags)
# New signature: create_dataset(audio_files_temp_list, processed_tags_dict, output_dataset_name, base_output_dir)

def get_stable_file_paths_from_temp(temp_file_wrappers, desired_base_temp_dir_name="stable_audio_copies"):
    """
    Copies Gradio's temporary files to a new, more stable temporary location,
    preserving their original names. This helps in managing files, especially if
    they need to be accessed multiple times or by different processes.

    Args:
        temp_file_wrappers (list): List of tempfile._TemporaryFileWrapper objects from Gradio.
        desired_base_temp_dir_name (str): A prefix for the new temporary directory.

    Returns:
        tuple: (path_to_new_temp_dir, dict_of_new_paths)
               - path_to_new_temp_dir: The path to the directory containing the copied files.
               - dict_of_new_paths: A dictionary mapping original filenames to their new full paths.
                                    Returns None for path_to_new_temp_dir if input is empty or None.
    """
    if not temp_file_wrappers:
        return None, {}

    # Create a unique subdirectory within the system's temp directory
    # This helps in organizing files if this function is called multiple times
    # and makes cleanup easier.
    # Using time to ensure uniqueness if called rapidly.
    unique_subdir = f"{desired_base_temp_dir_name}_{int(time.time_ns())}"
    stable_temp_dir = os.path.join(tempfile.gettempdir(), unique_subdir)
    os.makedirs(stable_temp_dir, exist_ok=True)
    
    new_file_paths_map = {}
    for temp_file_wrapper in temp_file_wrappers:
        if hasattr(temp_file_wrapper, 'name') and hasattr(temp_file_wrapper, 'orig_name'):
            original_filename = os.path.basename(temp_file_wrapper.orig_name) # Get the original uploaded filename
            new_file_path = os.path.join(stable_temp_dir, original_filename)
            
            try:
                # temp_file_wrapper.name is the path to the temp file on disk
                shutil.copy2(temp_file_wrapper.name, new_file_path)
                new_file_paths_map[original_filename] = new_file_path
            except Exception as e:
                print(f"Error copying temp file {original_filename} to {new_file_path}: {e}")
                # Potentially skip this file or raise an error
        else:
            print(f"Warning: Skipping an item in temp_file_wrappers that lacks 'name' or 'orig_name' attributes: {temp_file_wrapper}")
            
    return stable_temp_dir, new_file_paths_map


def create_dataset(audio_files_temp_list, processed_tags_dict, output_dataset_name, base_output_dir="created_datasets"):
    """
    Creates a Hugging Face dataset from a list of audio files and processed tags.
    The dataset is structured with audio, prompt (tags), and empty lyrics files.

    Args:
        audio_files_temp_list (list): List of Gradio's tempfile._TemporaryFileWrapper objects.
        processed_tags_dict (dict): Dictionary mapping original audio filenames to their processed tag strings.
        output_dataset_name (str): Name for the output dataset (e.g., "my_awesome_songs").
        base_output_dir (str): The base directory where the dataset folder will be created.

    Returns:
        tuple: (status_message, path_to_created_dataset_or_None)
    """
    if not audio_files_temp_list:
        return "Error: No audio files provided.", None
    if not output_dataset_name:
        return "Error: Output dataset name cannot be empty.", None

    # 1. Get stable copies of uploaded audio files
    copied_audio_temp_dir, stable_audio_paths_map = get_stable_file_paths_from_temp(audio_files_temp_list)
    
    if not stable_audio_paths_map:
        if copied_audio_temp_dir: # Cleanup if dir was made but no files copied
            shutil.rmtree(copied_audio_temp_dir)
        return "Error: Could not process any of the uploaded audio files.", None

    # 2. Define the final dataset output path
    dataset_output_path = os.path.join(base_output_dir, output_dataset_name)
    os.makedirs(dataset_output_path, exist_ok=True) # Ensure it exists

    dataset_items_list = [] # List of dictionaries for Dataset.from_list

    try:
        # 3. For each audio file, prepare its entry for the dataset
        for original_filename, copied_audio_path in stable_audio_paths_map.items():
            if not os.path.exists(copied_audio_path):
                print(f"Warning: Copied audio file {copied_audio_path} (original: {original_filename}) not found. Skipping.")
                continue

            # Retrieve tags for this audio file
            tags_string = processed_tags_dict.get(original_filename)
            if tags_string is None:
                print(f"Warning: No tags found for {original_filename}. Using empty tags.")
                tags_string = ""

            # Define paths for files within the target dataset directory structure
            # We'll copy the audio file and create text files directly in the dataset_output_path
            # to keep things simple for save_to_disk.
            # We can use subdirectories per item if preferred, but flat structure is also common.
            
            base_name_for_files = os.path.splitext(original_filename)[0]

            # Paths for files that will be part of the dataset structure
            target_audio_filename = f"{base_name_for_files}{os.path.splitext(original_filename)[1]}"
            target_prompt_filename = f"{base_name_for_files}_prompt.txt"
            target_lyrics_filename = f"{base_name_for_files}_lyrics.txt"

            final_audio_path = os.path.join(dataset_output_path, target_audio_filename)
            final_prompt_path = os.path.join(dataset_output_path, target_prompt_filename)
            final_lyrics_path = os.path.join(dataset_output_path, target_lyrics_filename)

            # Copy audio file to its final destination
            shutil.copy2(copied_audio_path, final_audio_path)

            # Create _prompt.txt file
            with open(final_prompt_path, "w", encoding="utf-8") as f_prompt:
                f_prompt.write(tags_string)

            # Create empty _lyrics.txt file
            with open(final_lyrics_path, "w", encoding="utf-8") as f_lyrics:
                f_lyrics.write("") # Empty lyrics

            # Add entry to dataset list. Paths should be relative to the dataset root if possible,
            # or HF Datasets will handle absolute paths too. For save_to_disk, it's often simpler
            # if the files are already in the target structure.
            # Here, we use paths relative to dataset_output_path for clarity.
            dataset_items_list.append({
                "audio": target_audio_filename,  # Path relative to dataset_output_path
                "prompt": target_prompt_filename, # Path relative to dataset_output_path
                "lyrics": target_lyrics_filename  # Path relative to dataset_output_path
            })
            
        if not dataset_items_list:
            return "Error: No items could be prepared for the dataset.", None

        # 4. Create Hugging Face Dataset
        # The 'audio' column needs to be cast to datasets.Audio if not automatically inferred.
        # For from_list with file paths, it usually infers correctly or loads on access.
        # If using audio data directly, casting with sampling_rate is needed.
        # Here, we are providing paths, so direct casting might not be immediately necessary
        # until data loading.
        
        # We need to make sure the paths in from_list are resolvable when the dataset is loaded.
        # By saving to disk, HF datasets will often copy or ensure files are within the saved structure.
        # The paths added to dataset_items_list are already relative to dataset_output_path.
        # We will load the dataset using the dataset_output_path as the base.
        
        # Create a metadata file (e.g., dataset.jsonl or metadata.csv) that from_list can use.
        # This is a common pattern.
        metadata_filepath = os.path.join(dataset_output_path, "metadata.jsonl")
        with open(metadata_filepath, 'w', encoding='utf-8') as f_meta:
            for item in dataset_items_list:
                import json
                f_meta.write(json.dumps(item) + '\n')
        
        # Now, load the dataset from this metadata file.
        # The paths in metadata.jsonl are relative to dataset_output_path.
        # So, data_files path should be metadata_filepath, and data_dir (or equivalent) should be dataset_output_path.
        # Or, more simply, allow from_list to take absolute paths, then save_to_disk.
        # Let's adjust dataset_items_list to use absolute paths for from_list,
        # then save_to_disk should handle organizing them.

        absolute_path_dataset_items = []
        for item in dataset_items_list:
            absolute_path_dataset_items.append({
                "audio": os.path.join(dataset_output_path, item["audio"]),
                "prompt": os.path.join(dataset_output_path, item["prompt"]),
                "lyrics": os.path.join(dataset_output_path, item["lyrics"])
            })

        hf_dataset = Dataset.from_list(absolute_path_dataset_items)
        
        # Cast audio column to Audio feature type
        # This is important for correct interpretation by other tools/Viewer
        from datasets import Audio
        # Assuming common sample rate, or it will be inferred per file.
        # If files might have different rates, this needs careful handling or resampling.
        # For now, let HF Datasets infer it.
        hf_dataset = hf_dataset.cast_column("audio", Audio())


        # 5. Save the dataset to disk
        # save_to_disk will save the dataset info (dataset_info.json, state.json)
        # and Arrow data files. It will use the files already copied into dataset_output_path.
        hf_dataset.save_to_disk(dataset_output_path)
        
        status_message = f"Dataset '{output_dataset_name}' created successfully at {dataset_output_path}. Contains {len(dataset_items_list)} items."
        return status_message, dataset_output_path

    except Exception as e:
        # Basic error handling
        error_message = f"An error occurred during dataset creation: {e}"
        print(error_message)
        # Clean up dataset_output_path if it was partially created
        if os.path.exists(dataset_output_path):
            # shutil.rmtree(dataset_output_path) # Or leave for inspection
            print(f"Partial dataset may exist at {dataset_output_path} due to error.")
        return error_message, None
    finally:
        # 6. Clean up the temporary directory for copied audio files
        if copied_audio_temp_dir and os.path.exists(copied_audio_temp_dir):
            try:
                shutil.rmtree(copied_audio_temp_dir)
                print(f"Cleaned up temporary audio copy directory: {copied_audio_temp_dir}")
            except Exception as e_clean:
                print(f"Error cleaning up temp directory {copied_audio_temp_dir}: {e_clean}")


if __name__ == '__main__':
    print("--- Testing dataset_creator.py ---")

    # Create dummy dependencies (tag_generator.py) if not present for testing
    # This module (dataset_creator) doesn't directly call tag_generator,
    # but the calling tab (dataset_tab) would.
    # For this test, we just need dummy inputs.

    # 1. Setup dummy inputs
    # Create some dummy audio files as if they were uploaded via Gradio
    temp_dir_for_uploads = tempfile.mkdtemp(prefix="gradio_uploads_test_")
    
    class MockGradioTempFile:
        def __init__(self, file_path, original_name):
            self.name = file_path # Path to the temp file on disk
            self.orig_name = original_name # Original uploaded filename

    dummy_audio_files_temp_list = []
    dummy_audio_content = b"dummy_wav_data" # In reality, this would be actual audio data
    
    audio_file_names = ["song1.wav", "song2.mp3", "track_alpha.flac"]
    for i, fname in enumerate(audio_file_names):
        temp_f_path = os.path.join(temp_dir_for_uploads, f"gradio_temp_{i}_{fname}")
        with open(temp_f_path, "wb") as f:
            f.write(dummy_audio_content)
        dummy_audio_files_temp_list.append(MockGradioTempFile(temp_f_path, fname))
        print(f"Created dummy uploaded file: {temp_f_path} (orig: {fname})")

    # Dummy processed_tags_dict (output from tag_generator.process_tags)
    sample_processed_tags_dict = {
        "song1.wav": "vocals, female, pop, high quality",
        "song2.mp3": "instrumental, guitar, acoustic, relaxing",
        "track_alpha.flac": "electronic, synth, upbeat, 80s"
    }

    sample_output_dataset_name = "my_test_audio_dataset"
    sample_base_output_dir = os.path.join(os.getcwd(), "test_datasets_output") # Create in CWD for easy inspection

    # Ensure base output dir exists for testing, or create_dataset will make it
    if not os.path.exists(sample_base_output_dir):
        os.makedirs(sample_base_output_dir, exist_ok=True)
    
    print(f"\nAttempting to create dataset '{sample_output_dataset_name}' in '{sample_base_output_dir}'...")

    # 2. Call create_dataset
    status_msg, created_path = create_dataset(
        dummy_audio_files_temp_list,
        sample_processed_tags_dict,
        sample_output_dataset_name,
        sample_base_output_dir
    )

    print(f"\n--- create_dataset Result ---")
    print(f"Status: {status_msg}")
    print(f"Created Path: {created_path}")

    # 3. Verification (basic)
    if created_path and os.path.exists(created_path):
        print(f"\nDataset directory '{created_path}' exists.")
        print("Files in dataset directory:")
        for root, _, files in os.walk(created_path):
            for name in files:
                print(os.path.join(root, name))
        
        # Try loading it back (optional, but good test)
        try:
            print(f"\nAttempting to load dataset from {created_path}...")
            loaded_ds = Dataset.load_from_disk(created_path)
            print(f"Successfully loaded dataset: {loaded_ds}")
            print(f"Features: {loaded_ds.features}")
            if len(loaded_ds) > 0:
                print(f"First item: {loaded_ds[0]}")
        except Exception as e_load:
            print(f"Error loading dataset back: {e_load}")

    else:
        print("Dataset creation seems to have failed (path does not exist or was not returned).")

    # 4. Cleanup dummy uploaded files and output test datasets
    try:
        shutil.rmtree(temp_dir_for_uploads)
        print(f"\nCleaned up dummy uploads temp dir: {temp_dir_for_uploads}")
        # You might want to inspect sample_base_output_dir before cleaning it up
        # shutil.rmtree(sample_base_output_dir)
        # print(f"Cleaned up dummy output datasets dir: {sample_base_output_dir}")
        print(f"Test output dataset located at: {os.path.join(sample_base_output_dir, sample_output_dataset_name)}")
        print("Please inspect and manually delete this directory when done.")

    except Exception as e_clean:
        print(f"Error during cleanup: {e_clean}")

    print("\n--- Test run finished ---")
