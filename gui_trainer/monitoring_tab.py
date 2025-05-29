import gradio as gr
import os
import logging

logger = logging.getLogger(__name__)
DEFAULT_LOG_FILE = os.path.join(os.path.dirname(__file__), "logs", "training.log")

def create_monitoring_tab():
    with gr.Blocks() as monitoring_tab_interface:
        gr.Markdown("# Training Monitoring Tab")

        with gr.Row():
            log_file_path_textbox = gr.Textbox(
                label="Log File Path",
                value=DEFAULT_LOG_FILE, # Default path
                interactive=True,
                scale=3
            )
            load_logs_button = gr.Button("Load/Refresh Logs", scale=1)
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Training Logs")
                live_logs_textbox = gr.Textbox(
                    label="Logs",
                    lines=30, # Increased lines
                    max_lines=2000, # Allow more lines for scrolling
                    interactive=False, # Logs are display-only
                    autoscroll=True,
                    show_copy_button=True
                )
            with gr.Column(scale=1):
                gr.Markdown("### Training Metrics (Placeholder)")
                current_metrics_textbox = gr.Textbox(
                    label="Current Status",
                    lines=5,
                    interactive=False,
                    placeholder="Metrics from logs not yet parsed.",
                    show_copy_button=True
                )
                gr.Markdown("### Visualizations (Placeholder)")
                loss_curve_placeholder = gr.Markdown(
                    "Loss curves and other visualizations (e.g., from TensorBoard) will be linked or displayed here in the future."
                )
                # Example: gr.HTML(value="<p>Point to your TensorBoard instance or embed plots.</p>")

        # --- Define Log Display Logic ---
        def display_logs(log_file_path_str):
            log_content = ""
            if not log_file_path_str or not log_file_path_str.strip():
                return "Error: Log file path cannot be empty."
            
            try:
                # Ensure the directory for the default log exists if we are trying to read it
                # This is more for robustness if the default path is used before training starts.
                log_dir = os.path.dirname(log_file_path_str)
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                    logger.info(f"Created log directory (if it was default): {log_dir}")
                
                if os.path.exists(log_file_path_str):
                    with open(log_file_path_str, "r", encoding="utf-8") as f:
                        # Read last N lines or some tailing logic might be better for very large files
                        # For now, read full content. Gradio textbox handles scrolling.
                        log_content = f.read()
                    if not log_content.strip():
                        return f"Log file '{log_file_path_str}' is empty."
                    return log_content
                else:
                    return f"Log file not found at: {log_file_path_str}\n\n(Note: Training must be started via the Training Tab for logs to be generated at the default path. If using a custom path, ensure it's correct.)"
            except Exception as e:
                logger.error(f"Error reading log file '{log_file_path_str}': {e}", exc_info=True)
                return f"Error reading log file: {str(e)}"

        # Connect button to display_logs function
        load_logs_button.click(
            fn=display_logs,
            inputs=[log_file_path_textbox],
            outputs=[live_logs_textbox]
        )
        
        # Optionally, load logs when the tab is selected/interface loads, using the default path
        # This provides an initial view without needing an immediate button click.
        monitoring_tab_interface.load(
            fn=display_logs,
            inputs=[log_file_path_textbox], # Will use the default value of the textbox on initial load
            outputs=[live_logs_textbox]
        )

    return monitoring_tab_interface

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # Create a dummy log file for testing
    dummy_log_dir_for_test = os.path.join(os.path.dirname(__file__), "logs_test_monitoring_tab")
    os.makedirs(dummy_log_dir_for_test, exist_ok=True)
    dummy_log_file_path = os.path.join(dummy_log_dir_for_test, "dummy_training.log")

    with open(dummy_log_file_path, "w", encoding="utf-8") as f:
        f.write("--- Dummy Training Log Start ---\n")
        for i in range(1, 21):
            f.write(f"[2024-08-15 10:{i:02d}:00] [INFO] Epoch {i//5 + 1}/5, Step {i*100}/2000, Loss: {1.0/i:.4f}\n")
        f.write("--- Dummy Training Log End ---\n")
    
    logger.info(f"Dummy log file created at: {dummy_log_file_path}")

    # Temporarily override DEFAULT_LOG_FILE for this test run
    # This is a bit of a hack. A better way would be to pass it to create_monitoring_tab if it accepted arguments.
    # Or, the user of the test can paste the dummy_log_file_path into the UI.
    # For automatic loading in test, we can try to set the value of the textbox.
    # However, Gradio's direct component value manipulation before launch is not straightforward.
    # The UI will default to DEFAULT_LOG_FILE. For testing, manually paste dummy_log_file_path.
    
    # To make the test auto-load the dummy log, we can redefine the component with the dummy path as default.
    # This is only for the __main__ block.
    original_default_log_file = DEFAULT_LOG_FILE
    global DEFAULT_LOG_FILE_OVERRIDE_FOR_TEST 
    DEFAULT_LOG_FILE_OVERRIDE_FOR_TEST = dummy_log_file_path

    def create_monitoring_tab_for_test():
        # Use a global or a passed-in default for the log file path textbox for testing
        # For this test, we'll rely on the UI textbox defaulting and the user pasting if needed,
        # or the default DEFAULT_LOG_FILE (which won't be the dummy one unless we change it globally here)
        
        # Re-defining just the textbox with new default for testing:
        # This is not ideal as it duplicates UI definition. The best way is to make create_monitoring_tab accept a default path.
        # For now, let's just say: The test will work if user pastes the path, or if the default path is created.
        # The `monitoring_tab_interface.load` will try to load the default path.
        
        # Let's try to create the default log file with some content IF it's the one being tested.
        if not os.path.exists(DEFAULT_LOG_FILE):
             os.makedirs(os.path.dirname(DEFAULT_LOG_FILE), exist_ok=True)
             with open(DEFAULT_LOG_FILE, "w") as f: f.write("This is the default log. Please paste a specific log path if needed for testing.")
             logger.info(f"Created placeholder default log at {DEFAULT_LOG_FILE} for initial load test.")
        
        return create_monitoring_tab() # Call original function which uses the global DEFAULT_LOG_FILE


    app_blocks = gr.Blocks()
    with app_blocks:
        # create_monitoring_tab()
        # To make the test load the dummy log automatically, we'd need to adjust how default path is set.
        # Simplest for now: User pastes path from log into UI.
        # Or, we can just test the display_logs function directly.
        
        # For UI testing:
        logger.info(f"To test, paste this path into the 'Log File Path' textbox: {os.path.abspath(dummy_log_file_path)}")
        tab = create_monitoring_tab_for_test() # This will use the DEFAULT_LOG_FILE
        
    app_blocks.launch(debug=True)

    # Cleanup dummy log file after test (optional, user might want to see it)
    # print(f"Test complete. You can manually remove: {dummy_log_dir_for_test}")
    # import shutil
    # shutil.rmtree(dummy_log_dir_for_test)
