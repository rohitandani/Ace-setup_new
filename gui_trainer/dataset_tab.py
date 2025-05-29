import gradio as gr
from .dataset_creator import create_dataset 
from .tag_generator import process_tags
# from .audio_processor import trim_audio, normalize_volume # Deferred for now
import os
import logging

logger = logging.getLogger(__name__)

def create_dataset_tab():
    with gr.Blocks() as dataset_tab_interface:
        gr.Markdown("# Dataset Creation Tab")

        with gr.Row():
            audio_files_input = gr.File(
                label="Upload Audio Files",
                file_count="multiple",
                file_types=["audio"] 
            )
        
        with gr.Row():
            output_dataset_name_textbox = gr.Textbox(
                label="Output Dataset Name",
                value="new_audio_dataset",
                interactive=True
            )
            base_output_dir_textbox = gr.Textbox(
                label="Base Output Directory",
                value="./datasets/", # Default relative to where app is run
                interactive=True
            )

        tagging_mode_radio = gr.Radio(
            label="Tagging Mode",
            choices=["Append Common", "Prepend Common", "Individual"],
            value="Append Common",
            interactive=True
        )

        common_tags_textbox = gr.Textbox(
            label="Common Tags (comma-separated)",
            placeholder="e.g., music, instrumental, high quality. Used if not 'Individual' mode.",
            visible=True, 
            interactive=True
        )
        
        individual_tags_info = gr.Markdown(
            "Individual mode: Tags per file are not currently set via UI. "
            "If selected, only shuffling of (non-existent) initial tags will occur, "
            "unless future UI enhancements allow per-file tag input.",
            visible=False 
        )

        shuffle_tags_checkbox = gr.Checkbox(label="Shuffle Tags", value=False, interactive=True)
        
        generate_button = gr.Button("Generate Dataset")
        output_status_textbox = gr.Textbox(label="Output Status", interactive=False, lines=10, autoscroll=True)

        # --- Define UI interactions ---
        def update_tag_input_visibility(mode_selected):
            is_common_mode = mode_selected in ["Append Common", "Prepend Common"]
            return {
                common_tags_textbox: gr.update(visible=is_common_mode, interactive=is_common_mode),
                individual_tags_info: gr.update(visible=not is_common_mode)
            }

        tagging_mode_radio.change(
            fn=update_tag_input_visibility,
            inputs=tagging_mode_radio,
            outputs=[common_tags_textbox, individual_tags_info]
        )

        # --- Define Button Click Handler ---
        def handle_generate_dataset(
            audio_files_temp_list, # List of TemporaryFileWrapper from gr.File
            ui_tag_mode, 
            common_tags_str, 
            shuffle_tags_bool,
            output_dataset_name_str,
            base_output_dir_str
        ):
            if not audio_files_temp_list:
                return "Error: No audio files uploaded."
            if not output_dataset_name_str.strip():
                return "Error: Output Dataset Name cannot be empty."
            if not base_output_dir_str.strip():
                return "Error: Base Output Directory cannot be empty."

            status_messages = ["Starting dataset generation..."]
            logger.info("Dataset generation started from UI.")

            try:
                # 1. Construct initial_tags_dict for process_tags
                # This dictionary maps original filenames to their initial tag strings.
                # For now, initial tags are empty as we don't have per-file input yet.
                initial_tags_dict = {}
                if audio_files_temp_list:
                    for file_obj in audio_files_temp_list:
                        # file_obj.name is the path to the temp file
                        # file_obj.orig_name is the original uploaded filename
                        original_filename = os.path.basename(file_obj.orig_name)
                        initial_tags_dict[original_filename] = "" # Empty initial tags
                
                status_messages.append(f"Prepared initial tags for {len(initial_tags_dict)} files.")

                # 2. Map UI tag mode to process_tags mode
                tag_processing_mode_map = {
                    "Append Common": "common_append",
                    "Prepend Common": "common_prepend",
                    "Individual": "individual"
                }
                actual_processing_mode = tag_processing_mode_map.get(ui_tag_mode, "individual")
                
                # If mode is "Individual", common_tags_str should be ignored by process_tags.
                # process_tags itself handles this: if mode is 'individual', it doesn't use common_tag_input.
                # So, we can pass common_tags_str as is.
                
                status_messages.append(f"Processing tags with mode: {actual_processing_mode}, shuffle: {shuffle_tags_bool}")
                processed_tags_dict = process_tags(
                    tags_dict=initial_tags_dict,
                    mode=actual_processing_mode,
                    common_tag_input=common_tags_str if ui_tag_mode in ["Append Common", "Prepend Common"] else "",
                    shuffle_tags=shuffle_tags_bool
                )
                status_messages.append(f"Tags processed. Result: {json.dumps(processed_tags_dict, indent=2)}")
                logger.info(f"Tags processed: {processed_tags_dict}")

                # 3. Call create_dataset
                # create_dataset expects: (audio_files_temp_list, processed_tags_dict, output_dataset_name, base_output_dir)
                status_messages.append(f"Creating dataset '{output_dataset_name_str}' in '{base_output_dir_str}'...")
                
                # The create_dataset function now directly handles the list of Gradio TemporaryFileWrappers
                dataset_status_msg, created_dataset_path = create_dataset(
                    audio_files_temp_list=audio_files_temp_list,
                    processed_tags_dict=processed_tags_dict,
                    output_dataset_name=output_dataset_name_str,
                    base_output_dir=base_output_dir_str
                )
                
                if created_dataset_path:
                    status_messages.append(f"Dataset creation successful! Path: {created_dataset_path}")
                    status_messages.append(f"Message: {dataset_status_msg}")
                    logger.info(f"Dataset created: {created_dataset_path}. Message: {dataset_status_msg}")
                else:
                    status_messages.append(f"Dataset creation failed or path not returned. Message: {dataset_status_msg}")
                    logger.error(f"Dataset creation failed. Message: {dataset_status_msg}")

            except Exception as e:
                logger.error(f"Error during dataset generation: {e}", exc_info=True)
                status_messages.append(f"An unexpected error occurred: {str(e)}")
            
            return "\n".join(status_messages)

        generate_button.click(
            fn=handle_generate_dataset,
            inputs=[
                audio_files_input,
                tagging_mode_radio,
                common_tags_textbox,
                shuffle_tags_checkbox,
                output_dataset_name_textbox,
                base_output_dir_textbox
            ],
            outputs=output_status_textbox
        )

    return dataset_tab_interface

if __name__ == "__main__":
    # Configure basic logging for testing this tab standalone
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s:%(name)s] %(message)s")

    # Create dummy versions of imported functions if they don't exist (e.g. running outside full app context)
    module_dir = os.path.dirname(__file__) if __file__ else "." # Handle if __file__ is not defined (e.g. REPL)
    
    # Dummy dataset_creator.py
    dc_path = os.path.join(module_dir, "dataset_creator.py")
    if not os.path.exists(dc_path):
        with open(dc_path, "w") as f:
            f.write("""
import os, shutil, tempfile
def get_stable_file_paths_from_temp(temp_file_wrappers, base_dir="dummy_stable"): # Dummy helper
    stable_temp_dir = os.path.join(tempfile.gettempdir(), base_dir)
    os.makedirs(stable_temp_dir, exist_ok=True)
    new_paths = {}
    if temp_file_wrappers:
        for tfw in temp_file_wrappers:
            orig_name = os.path.basename(tfw.orig_name)
            new_path = os.path.join(stable_temp_dir, orig_name)
            shutil.copy2(tfw.name, new_path) # Simulate copy
            new_paths[orig_name] = new_path
    return stable_temp_dir, new_paths

def create_dataset(audio_files_temp_list, processed_tags_dict, output_dataset_name, base_output_dir):
    print(f'Dummy create_dataset called with: audio_files_temp_list count={len(audio_files_temp_list) if audio_files_temp_list else 0}, processed_tags_dict={processed_tags_dict}, name={output_dataset_name}, dir={base_output_dir}')
    # Simulate file handling from temp list
    if audio_files_temp_list:
        stable_dir, stable_map = get_stable_file_paths_from_temp(audio_files_temp_list)
        print(f"Stable files created at {stable_dir} with map {stable_map}")
        if stable_dir and os.path.exists(stable_dir): shutil.rmtree(stable_dir) # Clean up dummy stable copies

    dummy_dataset_path = os.path.join(base_output_dir, output_dataset_name)
    os.makedirs(dummy_dataset_path, exist_ok=True) # Simulate dataset dir creation
    with open(os.path.join(dummy_dataset_path, "info.txt"), "w") as f_info:
        f_info.write(f"Dummy dataset {output_dataset_name} created with tags: {processed_tags_dict}")
    return f'Dummy dataset {output_dataset_name} created successfully at {dummy_dataset_path}', dummy_dataset_path
""")
        logger.info(f"Created dummy file: {dc_path}")

    # Dummy tag_generator.py
    tg_path = os.path.join(module_dir, "tag_generator.py")
    if not os.path.exists(tg_path):
        with open(tg_path, "w") as f:
            f.write("""
import random
def process_tags(tags_dict, mode, common_tag_input, shuffle_tags):
    print(f'Dummy process_tags called: mode={mode}, common_input={common_tag_input}, shuffle={shuffle_tags}, input_tags_dict={tags_dict}')
    output_dict = {}
    common_tags_list = [t.strip() for t in common_tag_input.split(',') if t.strip()]
    for fname, initial_tags_str in tags_dict.items():
        current_tags = [t.strip() for t in initial_tags_str.split(',') if t.strip()]
        if mode == "common_append" and common_tags_list:
            final_tags = current_tags + [ct for ct in common_tags_list if ct not in current_tags]
        elif mode == "common_prepend" and common_tags_list:
            final_tags = [ct for ct in common_tags_list if ct not in current_tags] + current_tags
        else: # individual
            final_tags = current_tags
        if shuffle_tags: random.shuffle(final_tags)
        output_dict[fname] = ", ".join(final_tags)
    return output_dict
""")
        logger.info(f"Created dummy file: {tg_path}")
    
    # Create and launch the tab interface
    # This setup allows running `python gui_trainer/dataset_tab.py`
    # if the dummy files are created in the same directory.
    # For proper package execution (e.g. `python -m gui_trainer.dataset_tab`),
    # the imports `.dataset_creator` should work if `gui_trainer` is part of PYTHONPATH.
    
    # Ensure the directory for dummy files is correct if __file__ is not defined
    if not __file__ and not module_dir == ".": # Likely means we are in a context where __file__ is not set.
        logger.warning("`__file__` is not defined. Dummy files might not be created in the correct location for standalone testing if not run as a script.")

    tab_interface = create_dataset_tab()
    tab_interface.launch(debug=True)
