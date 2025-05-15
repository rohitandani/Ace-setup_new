import os
import click

@click.command()
@click.option(
    "--checkpoint_path",
    type=str,
    default="",
    help="Path to the checkpoint directory. Downloads automatically if empty.",
)
@click.option(
    "--server_name",
    type=str,
    default="127.0.0.1",
    help="The server name to use for the Gradio app.",
)
@click.option(
    "--port", type=int, default=7865, help="The port to use for the Gradio app."
)
@click.option("--device_id", type=int, default=0, help="The CUDA device ID to use.")
@click.option(
    "--share",
    type=click.BOOL,
    default=False,
    help="Whether to create a public, shareable link for the Gradio app.",
    show_default=True,
)
@click.option(
    "--bf16",
    type=click.BOOL,
    default=True,
    help="Whether to use bfloat16 precision. Turn off if using MPS.",
    show_default=True,
)
@click.option(
    "--torch_compile", type=click.BOOL, default=False, help="Whether to use torch.compile.", show_default=True,
)
@click.option(
    "--cpu_offload", type=bool, default=False, help="Whether to use CPU offloading (only load current stage's model to GPU)", show_default=True,
)
@click.option(
    "--overlapped_decode", type=bool, default=False, help="Whether to use overlapped decoding (run dcae and vocoder using sliding windows)", show_default=True,
)
@click.option( # New option
    "--open_browser",
    type=click.BOOL,
    default=True, # Default to True to open the browser
    help="Whether to automatically open the Gradio app in your default browser.",
    show_default=True,
)
def main(
    checkpoint_path,
    server_name,
    port,
    device_id,
    share,
    bf16,
    torch_compile,
    cpu_offload,
    overlapped_decode,
    open_browser, # Add new parameter here
):
    """
    Main function to launch the ACE Step pipeline demo.
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    from acestep.ui.components import create_main_demo_ui
    from acestep.pipeline_ace_step import ACEStepPipeline
    from acestep.data_sampler import DataSampler

    # It's good practice to import Gradio if you're directly using its launch parameters
    # though it's implicitly used by acestep.ui.components.create_main_demo_ui
    # import gradio as gr # Not strictly necessary if create_main_demo_ui returns a Gradio object

    model_demo = ACEStepPipeline(
        checkpoint_dir=checkpoint_path,
        dtype="bfloat16" if bf16 else "float32",
        torch_compile=torch_compile,
        cpu_offload=cpu_offload,
        overlapped_decode=overlapped_decode
    )
    data_sampler = DataSampler()

    demo = create_main_demo_ui(
        text2music_process_func=model_demo.__call__,
        sample_data_func=data_sampler.sample,
        load_data_func=data_sampler.load_json,
    )
    demo.launch(
        server_name=server_name,
        server_port=port,
        share=share,
        inbrowser=open_browser, # Pass the new parameter to Gradio's launch
    )


if __name__ == "__main__":
    main()