import random

def process_tags(tags_dict, mode, common_tag_input, shuffle_tags):
    """
    Placeholder function for processing tags for audio files.
    Eventually, this function will handle tag assignment, common tag appending/prepending, and shuffling.

    Args:
        tags_dict (dict): A dictionary where keys are audio file paths and values are their initial tags (string, comma-separated).
        mode (str): Tagging mode. Can be 'common_append', 'common_prepend', or 'individual'.
        common_tag_input (str): Comma-separated string of common tags to be used if mode is common.
        shuffle_tags (bool): Whether to shuffle the tags for each file.

    Returns:
        dict: A dictionary with processed tags.
    """
    print("--- Processing Tags ---")
    print(f"Input tags_dict: {tags_dict}")
    print(f"Mode: {mode}")
    print(f"Common tag input: {common_tag_input}")
    print(f"Shuffle tags: {shuffle_tags}")

    processed_tags_dict = {}
    common_tags_list = [tag.strip() for tag in common_tag_input.split(',') if tag.strip()] if common_tag_input else []

    for audio_file, file_tags_str in tags_dict.items():
        current_tags = [tag.strip() for tag in file_tags_str.split(',') if tag.strip()]

        if mode == "common_append" and common_tags_list:
            final_tags = current_tags + [ct for ct in common_tags_list if ct not in current_tags]
        elif mode == "common_prepend" and common_tags_list:
            final_tags = [ct for ct in common_tags_list if ct not in current_tags] + current_tags
        else: # 'individual' mode or no common tags
            final_tags = current_tags
        
        if shuffle_tags:
            random.shuffle(final_tags)
            print(f"Shuffled tags for {audio_file}: {final_tags}")

        processed_tags_dict[audio_file] = ", ".join(final_tags)

    print(f"Processed tags_dict: {processed_tags_dict}")
    return processed_tags_dict

if __name__ == '__main__':
    # Example usage (optional, for testing)
    sample_tags_input = {
        "audio1.wav": "vocals, female, pop",
        "audio2.mp3": "instrumental, guitar, acoustic",
        "audio3.ogg": "vocals, male, rock, loud"
    }

    print("\n--- Test Case 1: Individual mode, no shuffle ---")
    processed1 = process_tags(sample_tags_input, "individual", "", False)
    # Expected: Original tags, possibly with re-joined spacing
    print(f"Result 1: {processed1}")
    assert processed1["audio1.wav"] == "vocals, female, pop"

    print("\n--- Test Case 2: Common append mode, with shuffle ---")
    common_tags_append = "music, audio"
    processed2 = process_tags(sample_tags_input, "common_append", common_tags_append, True)
    print(f"Result 2: {processed2}")
    # Expected: Common tags appended (if not present), order shuffled.
    # Check if all original and common tags are present for one file
    tags_audio1_case2 = set(processed2["audio1.wav"].split(", "))
    assert "vocals" in tags_audio1_case2
    assert "female" in tags_audio1_case2
    assert "pop" in tags_audio1_case2
    assert "music" in tags_audio1_case2
    assert "audio" in tags_audio1_case2
    assert len(tags_audio1_case2) == 5


    print("\n--- Test Case 3: Common prepend mode, no shuffle ---")
    common_tags_prepend = "HQ, studio"
    # Modifying one entry to have a common tag already
    sample_tags_input_modified = sample_tags_input.copy()
    sample_tags_input_modified["audio2.mp3"] = "instrumental, guitar, acoustic, HQ"
    processed3 = process_tags(sample_tags_input_modified, "common_prepend", common_tags_prepend, False)
    # Expected: Common tags prepended (if not present).
    print(f"Result 3: {processed3}")
    assert processed3["audio1.wav"] == "HQ, studio, vocals, female, pop"
    assert processed3["audio2.mp3"] == "studio, instrumental, guitar, acoustic, HQ" # HQ not duplicated at start, studio prepended
    assert processed3["audio3.ogg"] == "HQ, studio, vocals, male, rock, loud"


    print("\n--- Test Case 4: Individual mode, with shuffle ---")
    processed4 = process_tags(sample_tags_input, "individual", "", True)
    print(f"Result 4: {processed4}")
    tags_audio1_case4 = set(processed4["audio1.wav"].split(", "))
    assert "vocals" in tags_audio1_case4 and "female" in tags_audio1_case4 and "pop" in tags_audio1_case4
    assert len(tags_audio1_case4) == 3

    print("\n--- Test Case 5: Common append, empty common tags, no shuffle ---")
    processed5 = process_tags(sample_tags_input, "common_append", "", False)
    print(f"Result 5: {processed5}")
    assert processed5 == processed1 # Should be identical to individual, no shuffle

    print("\n--- Test Case 6: Common prepend, some tags already exist, with shuffle ---")
    sample_tags_input_overlap = {
        "trackA.flac": "vocals, pop, studio",
        "trackB.wav": "instrumental, HQ"
    }
    common_tags_overlap = "HQ, studio, processed"
    processed6 = process_tags(sample_tags_input_overlap, "common_prepend", common_tags_overlap, True)
    # Expected for trackA (example): 'HQ, processed, vocals, pop, studio' (order shuffled, studio not duplicated at start, processed added)
    # Expected for trackB (example): 'studio, processed, instrumental, HQ' (order shuffled, HQ not duplicated at start, processed and studio added)
    print(f"Result 6: {processed6}")
    tags_trackA_case6 = set(processed6["trackA.flac"].split(", "))
    assert "vocals" in tags_trackA_case6 and "pop" in tags_trackA_case6 and "studio" in tags_trackA_case6 and "HQ" in tags_trackA_case6 and "processed" in tags_trackA_case6
    assert len(tags_trackA_case6) == 5

    tags_trackB_case6 = set(processed6["trackB.wav"].split(", "))
    assert "instrumental" in tags_trackB_case6 and "HQ" in tags_trackB_case6 and "studio" in tags_trackB_case6 and "processed" in tags_trackB_case6
    assert len(tags_trackB_case6) == 4
    
    print("\nAll tag_generator tests passed (simulated).")
