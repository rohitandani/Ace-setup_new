import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from click.testing import CliRunner

# Assuming gui.py is in the acestep directory, and tests is a sibling to acestep
# Adjust the import path if your project structure is different.
from acestep.gui import main as gui_main

class TestGui(unittest.TestCase):

    @patch('acestep.gui.ACEStepPipeline')
    @patch('acestep.gui.DataSampler')
    @patch('acestep.gui.create_main_demo_ui')
    def test_main_with_checkpoint_path_none_programmatic(self, mock_create_demo, mock_data_sampler, mock_pipeline):
        """
        Test main function called programmatically with checkpoint_path=None.
        This directly tests the fix for the maintainer's concern.
        """
        mock_demo_instance = MagicMock()
        mock_create_demo.return_value = mock_demo_instance

        # Call main directly, not through CLI, to pass None explicitly
        # Provide default values for other required arguments by main
        gui_main.callback(
            checkpoint_path=None,
            server_name="127.0.0.1",
            port=7865,
            device_id=0,
            share=False,
            bf16=True,
            torch_compile=False
        )

        mock_pipeline.assert_called_once()
        args, kwargs = mock_pipeline.call_args
        self.assertIsNone(kwargs.get('checkpoint_dir'), "ACEStepPipeline should be called with checkpoint_dir=None")
        mock_create_demo.assert_called_once()
        mock_demo_instance.launch.assert_called_once()

    @patch('acestep.gui.ACEStepPipeline')
    @patch('acestep.gui.DataSampler')
    @patch('acestep.gui.create_main_demo_ui')
    def test_main_cli_default_checkpoint_path(self, mock_create_demo, mock_data_sampler, mock_pipeline):
        """Test CLI call with default (empty string) checkpoint_path."""
        mock_demo_instance = MagicMock()
        mock_create_demo.return_value = mock_demo_instance
        
        runner = CliRunner()
        result = runner.invoke(gui_main, [])

        self.assertEqual(result.exit_code, 0, f"CLI command failed: {result.output}")
        mock_pipeline.assert_called_once()
        args, kwargs = mock_pipeline.call_args
        # Default for checkpoint_path in click is "", so Path("") is expected
        self.assertEqual(kwargs.get('checkpoint_dir'), Path(""), "ACEStepPipeline should be called with Path('') for default CLI")

    @patch('acestep.gui.ACEStepPipeline')
    @patch('acestep.gui.DataSampler')
    @patch('acestep.gui.create_main_demo_ui')
    def test_main_cli_empty_checkpoint_path(self, mock_create_demo, mock_data_sampler, mock_pipeline):
        """Test CLI call with --checkpoint_path ""."""
        mock_demo_instance = MagicMock()
        mock_create_demo.return_value = mock_demo_instance

        runner = CliRunner()
        result = runner.invoke(gui_main, ['--checkpoint_path', ''])

        self.assertEqual(result.exit_code, 0, f"CLI command failed: {result.output}")
        mock_pipeline.assert_called_once()
        args, kwargs = mock_pipeline.call_args
        self.assertEqual(kwargs.get('checkpoint_dir'), Path(""), "ACEStepPipeline should be called with Path('')")

    @patch('acestep.gui.ACEStepPipeline')
    @patch('acestep.gui.DataSampler')
    @patch('acestep.gui.create_main_demo_ui')
    def test_main_cli_valid_checkpoint_path(self, mock_create_demo, mock_data_sampler, mock_pipeline):
        """Test CLI call with a specified checkpoint_path."""
        mock_demo_instance = MagicMock()
        mock_create_demo.return_value = mock_demo_instance

        runner = CliRunner()
        dummy_path = "/test/path"
        result = runner.invoke(gui_main, ['--checkpoint_path', dummy_path])

        self.assertEqual(result.exit_code, 0, f"CLI command failed: {result.output}")
        mock_pipeline.assert_called_once()
        args, kwargs = mock_pipeline.call_args
        self.assertEqual(kwargs.get('checkpoint_dir'), Path(dummy_path), f"ACEStepPipeline should be called with Path('{dummy_path}')")

if __name__ == '__main__':
    unittest.main()
