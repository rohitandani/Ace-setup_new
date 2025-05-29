import os
from pydub import AudioSegment
from pydub.exceptions import PydubException

def parse_time_to_ms(time_str):
    """
    Parses a time string (MM:SS or SS or S) into milliseconds.
    Returns None if parsing fails.
    """
    if isinstance(time_str, (int, float)): # Already seconds
        return int(time_str * 1000)
    if not isinstance(time_str, str):
        return None
        
    parts = time_str.split(':')
    try:
        if len(parts) == 2: # MM:SS
            minutes = int(parts[0])
            seconds = float(parts[1])
            return int((minutes * 60 + seconds) * 1000)
        elif len(parts) == 1: # SS or S (float or int)
            seconds = float(parts[0])
            return int(seconds * 1000)
        else:
            return None
    except ValueError:
        return None

def trim_audio(audio_file_path, start_time_str, end_time_str):
    """
    Trims an audio file using pydub.

    Args:
        audio_file_path (str): Path to the input audio file.
        start_time_str (str/int/float): Start time (e.g., "MM:SS", "SS", or seconds as float/int).
        end_time_str (str/int/float): End time (e.g., "MM:SS", "SS", or seconds as float/int).

    Returns:
        str: Path to the trimmed audio file, or None if an error occurred.
    """
    if not os.path.exists(audio_file_path):
        print(f"Error: Audio file not found at {audio_file_path}")
        return None

    start_ms = parse_time_to_ms(start_time_str)
    end_ms = parse_time_to_ms(end_time_str)

    if start_ms is None:
        print(f"Error: Invalid start time format: {start_time_str}")
        return None
    if end_ms is None:
        print(f"Error: Invalid end time format: {end_time_str}")
        return None

    if start_ms < 0:
        print("Error: Start time cannot be negative.")
        return None
    if end_ms <= start_ms:
        print(f"Error: End time ({end_ms}ms) must be after start time ({start_ms}ms).")
        return None

    try:
        audio = AudioSegment.from_file(audio_file_path)
    except PydubException as e:
        print(f"Error loading audio file {audio_file_path}: {e}")
        return None
    except FileNotFoundError: # pydub might raise this directly if ffmpeg is missing for specific formats
        print(f"Error: File not found or ffmpeg might be missing for handling {audio_file_path}. Details: {e}")
        return None


    if end_ms > len(audio):
        print(f"Warning: End time ({end_ms}ms) exceeds audio duration ({len(audio)}ms). Trimming to end of audio.")
        end_ms = len(audio)
    
    if start_ms >= len(audio):
        print(f"Error: Start time ({start_ms}ms) is beyond audio duration ({len(audio)}ms). Cannot trim.")
        return None


    trimmed_audio = audio[start_ms:end_ms]

    base, ext = os.path.splitext(audio_file_path)
    output_path = f"{base}_trimmed{ext}"
    
    try:
        trimmed_audio.export(output_path, format=ext.lstrip('.'))
        print(f"Trimmed audio saved to: {output_path}")
        return output_path
    except PydubException as e:
        print(f"Error exporting trimmed audio: {e}")
        return None
    except Exception as e: # Catch other potential errors during export
        print(f"An unexpected error occurred during export: {e}")
        return None


def normalize_volume(audio_file_path, target_db_str):
    """
    Normalizes the volume of an audio file to a target dBFS using pydub.

    Args:
        audio_file_path (str): Path to the input audio file.
        target_db_str (str/float): Target volume in dBFS (e.g., "-6.0" or -6.0).

    Returns:
        str: Path to the normalized audio file, or None if an error occurred.
    """
    if not os.path.exists(audio_file_path):
        print(f"Error: Audio file not found at {audio_file_path}")
        return None

    try:
        target_db = float(target_db_str)
    except ValueError:
        print(f"Error: Invalid target dB value: {target_db_str}. Must be a number.")
        return None

    try:
        audio = AudioSegment.from_file(audio_file_path)
    except PydubException as e:
        print(f"Error loading audio file {audio_file_path}: {e}")
        return None
    except FileNotFoundError: 
        print(f"Error: File not found or ffmpeg might be missing for handling {audio_file_path}. Details: {e}")
        return None


    if audio.dBFS == float('-inf'): # Check for silence
        print(f"Warning: Audio file {audio_file_path} appears to be silent. Normalization may not be effective or desired.")
        # Decide if to proceed or return early for silent audio
        # For now, let's proceed, but gain might be huge or pydub might handle it.
    
    change_in_dBFS = target_db - audio.dBFS
    normalized_audio = audio.apply_gain(change_in_dBFS)

    base, ext = os.path.splitext(audio_file_path)
    output_path = f"{base}_normalized{ext}"

    try:
        normalized_audio.export(output_path, format=ext.lstrip('.'))
        print(f"Normalized audio saved to: {output_path}")
        return output_path
    except PydubException as e:
        print(f"Error exporting normalized audio: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during export: {e}")
        return None


if __name__ == '__main__':
    print("--- Testing audio_processor.py ---")
    
    # Create a dummy WAV file for testing (sine wave, 5 seconds, 440Hz)
    # This ensures we have a valid audio file that pydub can process.
    sample_rate = 44100
    duration_s = 5
    frequency = 440
    
    # Create a silent audio segment first
    dummy_audio_segment = AudioSegment.silent(duration=duration_s * 1000, frame_rate=sample_rate)
    # Generate a sine wave (this is a bit more involved with raw samples, pydub makes it easy)
    # For simplicity, we'll create a basic silent track, as generating tones requires numpy or math
    # and the focus is on pydub's functions, not tone generation.
    # A simple silent track is enough to test file operations, trimming, and normalization (dBFS will be -inf).
    
    # Let's create a non-silent dummy file for better testing of normalization.
    # We can make a simple 1-second stereo file with some gain.
    try:
        silence = AudioSegment.silent(duration=1000, frame_rate=sample_rate) # 1 second
        # Add some simple tones or noise if possible, or just export silence with gain
        # For a truly testable normalization, we need non-silent audio.
        # Let's layer two silences with different gains - this is a hack but creates non-uniform samples somewhat
        dummy_audio_segment = silence.overlay(silence.apply_gain(-10), position=500) 
        dummy_audio_segment = dummy_audio_segment.set_channels(2) # Stereo
    except Exception as e_gen:
        print(f"Could not generate complex dummy audio, using simple silence: {e_gen}")
        dummy_audio_segment = AudioSegment.silent(duration=duration_s * 1000, frame_rate=sample_rate)


    test_audio_filename = "test_audio_sample.wav"
    test_audio_mp3_filename = "test_audio_sample.mp3" # Also test mp3
    
    created_files = []
    processed_files_to_clean = []

    try:
        dummy_audio_segment.export(test_audio_filename, format="wav")
        created_files.append(test_audio_filename)
        print(f"Created dummy WAV: {test_audio_filename}, Duration: {dummy_audio_segment.duration_seconds}s, dBFS: {dummy_audio_segment.dBFS}")
        
        # Also create an MP3 version for format variety
        dummy_audio_segment.export(test_audio_mp3_filename, format="mp3")
        created_files.append(test_audio_mp3_filename)
        print(f"Created dummy MP3: {test_audio_mp3_filename}")


        # --- Test trim_audio ---
        print("\n--- Testing trim_audio (WAV) ---")
        start_trim = "0:01" # 1 second
        end_trim = "0:03"   # 3 seconds
        trimmed_wav_path = trim_audio(test_audio_filename, start_trim, end_trim)
        if trimmed_wav_path:
            processed_files_to_clean.append(trimmed_wav_path)
            trimmed_segment = AudioSegment.from_file(trimmed_wav_path)
            expected_duration_ms = parse_time_to_ms(end_trim) - parse_time_to_ms(start_trim)
            print(f"Trimmed WAV: {trimmed_wav_path}, Expected duration: {expected_duration_ms/1000.0}s, Actual duration: {trimmed_segment.duration_seconds}s")
            assert abs(trimmed_segment.duration_seconds * 1000 - expected_duration_ms) < 50, "Trimmed duration mismatch" # Allow small diff

        print("\n--- Testing trim_audio (MP3) ---")
        trimmed_mp3_path = trim_audio(test_audio_mp3_filename, 1.5, 3.5) # Using float seconds
        if trimmed_mp3_path:
            processed_files_to_clean.append(trimmed_mp3_path)
            trimmed_mp3_segment = AudioSegment.from_file(trimmed_mp3_path)
            expected_duration_mp3_ms = (3.5 - 1.5) * 1000
            print(f"Trimmed MP3: {trimmed_mp3_path}, Expected duration: {expected_duration_mp3_ms/1000.0}s, Actual duration: {trimmed_mp3_segment.duration_seconds}s")
            assert abs(trimmed_mp3_segment.duration_seconds * 1000 - expected_duration_mp3_ms) < 50, "Trimmed MP3 duration mismatch"

        # Test invalid trim cases
        print("\n--- Testing invalid trim cases ---")
        trim_audio(test_audio_filename, "invalid_time", "0:03")
        trim_audio(test_audio_filename, "0:03", "0:01")
        trim_audio(test_audio_filename, "0:01", "0:08") # End time beyond duration (should warn and clip)
        trim_audio("non_existent_file.wav", "0:01", "0:03")


        # --- Test normalize_volume ---
        print("\n--- Testing normalize_volume (WAV) ---")
        target_db_wav = -10.0
        normalized_wav_path = normalize_volume(test_audio_filename, str(target_db_wav))
        if normalized_wav_path:
            processed_files_to_clean.append(normalized_wav_path)
            normalized_segment = AudioSegment.from_file(normalized_wav_path)
            print(f"Normalized WAV: {normalized_wav_path}, Target dBFS: {target_db_wav}, Actual dBFS: {normalized_segment.dBFS:.2f}")
            # For very quiet or generated audio, dBFS might not hit target perfectly but should be close if not silent.
            if dummy_audio_segment.dBFS > -float('inf'): # Only check if original wasn't silent
                 assert abs(normalized_segment.dBFS - target_db_wav) < 1.5, "Normalized WAV dBFS mismatch"


        print("\n--- Testing normalize_volume (MP3) ---")
        target_db_mp3 = -3.5
        normalized_mp3_path = normalize_volume(test_audio_mp3_filename, target_db_mp3) # Using float
        if normalized_mp3_path:
            processed_files_to_clean.append(normalized_mp3_path)
            normalized_mp3_segment = AudioSegment.from_file(normalized_mp3_path)
            print(f"Normalized MP3: {normalized_mp3_path}, Target dBFS: {target_db_mp3}, Actual dBFS: {normalized_mp3_segment.dBFS:.2f}")
            if dummy_audio_segment.dBFS > -float('inf'):
                 assert abs(normalized_mp3_segment.dBFS - target_db_mp3) < 1.5, "Normalized MP3 dBFS mismatch"

        # Test invalid normalize cases
        print("\n--- Testing invalid normalize cases ---")
        normalize_volume(test_audio_filename, "not_a_float")
        normalize_volume("non_existent_file.wav", -6.0)

    except PydubException as e:
        print(f"Pydub specific error during testing: {e}")
        print("This might indicate issues with ffmpeg/libav or file format support.")
    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}")
    finally:
        print("\n--- Cleaning up test files ---")
        all_files_to_clean = created_files + processed_files_to_clean
        for f_path in all_files_to_clean:
            if os.path.exists(f_path):
                try:
                    os.remove(f_path)
                    print(f"Removed: {f_path}")
                except Exception as e_clean:
                    print(f"Error removing {f_path}: {e_clean}")
    
    print("\n--- audio_processor.py tests finished ---")
