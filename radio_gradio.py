import argparse
import os
import time
import json
import random
import gradio as gr
import requests
import traceback
import threading
import queue
import gc
import re
import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum, auto
from acestep.pipeline_ace_step import ACEStepPipeline

# Constants and Configuration
SUPPORTED_LANGUAGES = {
    "English": "English",
    "Spanish": "Spanish", 
    "French": "French",
    "German": "German",
    "Italian": "Italian",
    "Portuguese": "Portuguese",
    "Japanese": "Japanese",
    "Chinese": "Chinese",
    "Russian": "Russian",
    "Arabic": "Arabic",
    "Hindi": "Hindi",
    "Korean": "Korean",
    "Finnish": "Finnish"
}

# Genre-specific configurations
GENRE_DURATIONS = {
    "pop": 180,
    "rock": 210,
    "hip hop": 210,
    "electronic": 300,
    "lofi": 120,
    "jazz": 240,
    "classical": 360,
    "ambient": 420,
    "country": 210,
    "default": 180
}

GENRE_TEMPOS = {
    "pop": 120,
    "rock": 140,
    "hip hop": 95,
    "electronic": 128,
    "lofi": 85,
    "jazz": 110,
    "classical": 80,
    "ambient": 70,
    "country": 100,
    "default": 120
}

THEME_SUGGESTIONS = {
    "pop": ["summer love", "heartbreak", "dancing", "celebrity crush", "night out"],
    "rock": ["rebellion", "road trip", "broken dreams", "rock and roll lifestyle"],
    "hip hop": ["street life", "hustle", "success story", "city nights"],
    "electronic": ["neon lights", "cosmic journey", "digital dreams", "midnight drive"],
    "lofi": ["rainy day", "coffee shop", "study session", "chill vibes"],
    "jazz": ["smooth nights", "blue mood", "speakeasy", "saxophone dreams"],
    "classical": ["moonlight", "spring morning", "winter tale", "royal ball"],
    "ambient": ["ocean waves", "forest walk", "mountain sunrise", "deep space"],
    "country": ["small town", "truck driving", "lost love", "whiskey nights", "backroads"],
    "default": ["love", "dreams", "adventure", "nostalgia"]
}

# Enums and Data Classes
class RadioState(Enum):
    STOPPED = auto()
    BUFFERING = auto()
    PLAYING = auto()
    PAUSED = auto()

@dataclass
class Song:
    title: str
    artist: str
    genre: str
    theme: str
    duration: float
    lyrics: str
    language: str 
    prompt: str
    audio_path: str
    generation_time: float
    timestamp: float
    metadata: dict

@dataclass
class StationIdentity:
    name: str
    slogan: str
    color_scheme: tuple
    logo_style: str
    
    @classmethod
    def generate_identity(cls, genre: str, theme: str):
        """Generate cohesive station branding based on genre and theme"""
        # Format the station name
        theme = theme.replace(" radio", "").replace("Radio", "").title()
        suffixes = {
            "hip hop": "FM",
            "electronic": "Radio",
            "classical": "Classical",
            "jazz": "Jazz",
            "lofi": "Lo-Fi",
            "country": "Country",
            "default": "Radio"
        }
        suffix = suffixes.get(genre.lower(), suffixes["default"])
        name = f"{genre.title()} {theme} {suffix}"
        
        # Generate slogan
        slogans = {
            "pop": f"The hottest {theme} hits!",
            "rock": f"Loud {theme} anthems 24/7",
            "hip hop": f"Real {theme} sounds",
            "electronic": f"{theme} frequencies",
            "lofi": f"{theme} beats to relax to",
            "jazz": f"Smoothest {theme} in town",
            "classical": f"{theme} masterpieces",
            "ambient": f"{theme} soundscapes",
            "country": f"Real {theme} country music",
            "default": f"Your {theme} soundtrack"
        }
        slogan = slogans.get(genre.lower(), slogans["default"])
        
        # Generate color scheme based on genre
        color_schemes = {
            "pop": ("#FF6B6B", "#4ECDC4"),  # Bright colors
            "rock": ("#2F2F2F", "#F8F8F8"),  # Black and white
            "hip hop": ("#1A1A2E", "#E94560"),  # Dark with accent
            "electronic": ("#0F3460", "#E94560"),  # Dark blue with pink
            "lofi": ("#B7C4CF", "#967E76"),  # Muted tones
            "jazz": ("#2C3639", "#DCD7C9"),  # Dark with cream
            "classical": ("#3F4E4F", "#DCD7C9"),  # Elegant tones
            "ambient": ("#1B2430", "#D6D5A8"),  # Dark with soft accent
            "country": ("#5F8670", "#FF9800"),  # Earthy green with orange accent
            "default": ("#2C3639", "#A27B5C")  # Neutral tones
        }
        color_scheme = color_schemes.get(genre.lower(), color_schemes["default"])
        
        # Logo style
        logo_styles = {
            "pop": "modern",
            "rock": "grunge",
            "hip hop": "urban",
            "electronic": "futuristic",
            "lofi": "minimal",
            "jazz": "vintage",
            "classical": "elegant",
            "ambient": "abstract",
            "country": "rustic",
            "default": "modern"
        }
        logo_style = logo_styles.get(genre.lower(), logo_styles["default"])
        
        return cls(name, slogan, color_scheme, logo_style)

class AIRadioStation:
    def __init__(self, ace_step_pipeline: ACEStepPipeline, model_path: str = "E:/ubuntusta/gemma-3-4b-abliterated.Q4_K_M.gguf"):
        """
        Initialize the AI Radio Station with continuous generation.
        
        Args:
            ace_step_pipeline: Initialized ACEStepPipeline for music generation
            model_path: Path to LLM model for lyric generation
        """
        self._pipeline = ace_step_pipeline  # Store the original pipeline reference
        self.random_mode = False 
        self.llm_model_path = model_path
        self.llm = None
        self._first_play = True  
        self.pipeline_args = {
            'checkpoint_dir': ace_step_pipeline.checkpoint_dir,
            'dtype': "bfloat16",
            'torch_compile': ace_step_pipeline.torch_compile
        }
        self.current_pipeline = None  # Will be initialized when needed
        self.identity: Optional[StationIdentity] = None
        self.current_song: Optional[Song] = None
        self.playlist: List[Song] = []
        self.history: List[Song] = []
        self.song_queue = queue.Queue()
        self.state = RadioState.STOPPED
        self.stop_event = threading.Event()
        self.min_buffer_size = 1
        self.generation_thread = None
        self.playback_thread = None
        self.playback_paused = threading.Event()
        self.max_history_size = 50
        self.cache_cleanup_interval = 300
        self.last_cache_cleanup = time.time()
        self.language = "English"
        self.tempo = 120
        self.intensity = "medium"
        self.mood = "upbeat"
        self.generation_progress = 0.0
        self.current_song_elapsed = 0.0
        
        # Initialize LLM for lyric generation
        try:
            from llama_cpp import Llama
            self.llm = Llama(
                model_path=model_path,
                n_ctx=4048,
                n_threads=4,
                n_gpu_layers=-1,
            )
        except ImportError:
            print("Warning: llama-cpp-python not installed, using simple lyric generation")
            self.llm = None

    def load_llm(self):
        """Load the LLM model into memory"""
        if self.llm is None:
            print("Loading LLM model...")
            try:
                from llama_cpp import Llama
                self.llm = Llama(
                    model_path=self.llm_model_path,
                    n_ctx=4048,
                    n_threads=4,
                    n_gpu_layers=-1,
                )
            except ImportError:
                print("Warning: llama-cpp-python not installed, using simple lyric generation")
                self.llm = None

    def unload_llm(self):
        """Unload the LLM model from memory"""
        if self.llm is not None:
            print("Unloading LLM model...")
            del self.llm
            self.llm = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def get_pipeline(self):
        """Lazily initialize the pipeline when needed"""
        if self.current_pipeline is None:
            print("Initializing pipeline...")
            self.current_pipeline = ACEStepPipeline(**self.pipeline_args)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f"Pipeline dtype: {self.current_pipeline.dtype if self.current_pipeline else 'Not initialized'}")
        return self.current_pipeline

    def release_pipeline(self):
        """Clean up pipeline resources"""
        if self.current_pipeline is not None:
            print("Releasing pipeline...")
            del self.current_pipeline
            self.current_pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def _playback_worker(self):
        """Background worker for continuous playback"""
        self.state = RadioState.BUFFERING
        print("Playback worker started - buffering songs...")
        
        while not self.stop_event.is_set():
            try:
                # Clean cache periodically
                if time.time() - self.last_cache_cleanup > self.cache_cleanup_interval:
                    self._cleanup_cache()
                
                # Handle paused state
                if self.playback_paused.is_set():
                    self.state = RadioState.PAUSED
                    time.sleep(1)
                    continue
                
                # Only check buffer size on first play
                if self._first_play and self.state == RadioState.BUFFERING:
                    if self.song_queue.qsize() >= self.min_buffer_size:
                        print(f"Initial buffer filled with {self.song_queue.qsize()} songs - starting playback!")
                        self.state = RadioState.PLAYING
                        self._first_play = False
                    else:
                        print(f"Initial buffering... ({self.song_queue.qsize()}/{self.min_buffer_size})")
                        time.sleep(0.5)
                        continue
                
                # Get and play next song
                try:
                    song = self.song_queue.get(block=False)
                    self.current_song = song
                    self.history.append(song)
                    self.current_song_start_time = time.time()
                    self.current_song_elapsed = 0.0
                    
                    # Enforce history size limit
                    if len(self.history) > self.max_history_size:
                        self.history = self.history[-self.max_history_size:]
                    
                    print(f"\n=== Now Playing ===\n"
                        f"Title: {song.title}\n"
                        f"Artist: {song.artist}\n"
                        f"Duration: {song.duration:.1f}s\n")
                    
                    # Play the song
                    remaining_time = song.duration
                    while remaining_time > 0 and not self.stop_event.is_set() and not self.playback_paused.is_set():
                        chunk_time = min(0.1, remaining_time)
                        time.sleep(chunk_time)
                        remaining_time -= chunk_time
                        self.current_song_elapsed = song.duration - remaining_time
                    
                    if self.playback_paused.is_set():
                        continue
                        
                    # Save state periodically
                    if len(self.history) % 5 == 0:
                        self._save_state()
                        
                except queue.Empty:
                    if not self._first_play:  # Only show empty queue message after first play
                        print("Queue empty - waiting for next song...")
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"Error in playback worker: {e}")
                traceback.print_exc()
                time.sleep(1)
        
        print("Playback worker stopped")
        self.state = RadioState.STOPPED
        
        print("Playback worker stopped")
        self.state = RadioState.STOPPED

    def _cleanup_cache(self):
        """More aggressive VRAM cleanup"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        gc.collect()
        
        # Force Python to release memory
        for _ in range(3):
            gc.collect()
        
        # Clean up temporary files
        try:
            temp_dir = os.path.join(os.getcwd(), "temp")
            if os.path.exists(temp_dir):
                for file in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, file)
                    file_age = time.time() - os.path.getmtime(file_path)
                    if file_age > 3600 and os.path.isfile(file_path):
                        try:
                            os.remove(file_path)
                        except Exception:
                            pass
        except Exception:
            pass
        
        self.last_cache_cleanup = time.time()

    def start_radio(self, genre: str, theme: str, duration: Optional[float] = None, 
                buffer_size: int = 1, language: str = "English", max_history: int = 50, 
                tempo: Optional[int] = None, intensity: Optional[str] = None, 
                mood: Optional[str] = None, random_mode: bool = False):
        """Start the radio station with smart defaults"""
        self.stop_radio()
        self.stop_event.clear()
        self.playback_paused.clear()
        
        # Set smart defaults if parameters not provided
        duration = duration or GENRE_DURATIONS.get(genre.lower(), GENRE_DURATIONS["default"])
        tempo = tempo or GENRE_TEMPOS.get(genre.lower(), GENRE_TEMPOS["default"])
        intensity = intensity or "medium"
        mood = mood or self._detect_mood(theme)
        
        # Store the random mode setting
        self.random_mode = random_mode
        

        print(f"\n=== Starting song generation ===")
        print(f"Genre: {genre}, Theme: {theme}, Duration: {duration}s")
        if tempo is not None:
            print(f"Tempo: {tempo} BPM")
        if intensity is not None:
            print(f"Intensity: {intensity}")
        if mood is not None:
            print(f"Mood: {mood}")
        
        # Store parameters for lyric generation
        self.tempo = tempo or self.tempo
        self.intensity = intensity or self.intensity
        self.mood = mood or self.mood
        # Generate station identity (use original genre/theme even in random mode for branding)
        self.identity = StationIdentity.generate_identity(genre, theme)
        self.min_buffer_size = buffer_size
        self.max_history_size = max_history
        self.language = language
        self.playlist = []
        self.history = []
        self.state = RadioState.BUFFERING
        self.generation_progress = 0.0
        
        print(f"\n=== Starting {self.identity.name} ===")
        print(f"Slogan: {self.identity.slogan}")
        print(f"Random Mode: {'ON' if random_mode else 'OFF'}")
        print(f"Buffering {buffer_size} songs before playback...")
        print(f"Language: {SUPPORTED_LANGUAGES.get(language, language)}")
        print(f"Tempo: {tempo} BPM, Intensity: {intensity}, Mood: {mood}")
        
        # Start background workers
        self.generation_thread = threading.Thread(
            target=self._generate_song_worker,
            args=(genre, theme, duration),
            daemon=True
        )
        self.playback_thread = threading.Thread(
            target=self._playback_worker,
            daemon=True
        )
        
        self.generation_thread.start()
        self.playback_thread.start()

    def _detect_mood(self, theme: str) -> str:
        """Detect mood from theme keywords"""
        theme_lower = theme.lower()
        if any(word in theme_lower for word in ["love", "happy", "celebrate", "party"]):
            return "happy"
        elif any(word in theme_lower for word in ["sad", "heartbreak", "lonely", "tears"]):
            return "sad"
        elif any(word in theme_lower for word in ["angry", "rage", "fight", "protest"]):
            return "angry"
        elif any(word in theme_lower for word in ["chill", "relax", "calm", "peace"]):
            return "chill"
        elif any(word in theme_lower for word in ["dream", "thought", "memory", "reflect"]):
            return "reflective"
        return "upbeat"

    def generate_lyrics_and_prompt(self, genre: str, theme: str, language: str = "English") -> Tuple[str, str]:
        """Generate song lyrics with genre-specific structures"""
        structures = {
            "pop": (
                "[Verse 1]\n{lyrics}\n\n"
                "[Pre-Chorus]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Pre-Chorus]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Bridge]\n{lyrics}\n\n"
                "[Final Chorus] (with ad-libs)"
            ),
            "rock": (
                "[Guitar Intro]\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Guitar Solo] (8-16 bars)\n\n"
                "[Bridge]\n{lyrics}\n\n"
                "[Double Chorus] (big finish)"
            ),
            "hip hop": (
                "[Intro Hook]\n{lyrics}\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Bridge] (optional rap breakdown)\n\n"
                "[Outro] (fade with ad-libs)"
            ),
            "electronic": (
                "[Atmospheric Intro] (16 bars)\n\n"
                "[Build-Up]\n{lyrics}\n\n"
                "[Drop]\n{lyrics}\n\n"
                "[Breakdown Verse]\n{lyrics}\n\n"
                "[Build-Up]\n{lyrics}\n\n"
                "[Drop]\n{lyrics}\n\n"
                "[Outro] (beat fade)"
            ),
            "country": (
                "[Steel Guitar Intro]\n\n"
                "[Verse 1] (storytelling)\n{lyrics}\n\n"
                "[Chorus] (big melody)\n{lyrics}\n\n"
                "[Verse 2] (develop story)\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Fiddle Solo] (8 bars)\n\n"
                "[Bridge] (emotional peak)\n{lyrics}\n\n"
                "[Double Chorus] (with harmonies)"
            ),
            "country": (
                "[Steel Guitar Intro]\n\n"
                "[Verse 1] (storytelling)\n{lyrics}\n\n"
                "[Chorus] (big melody)\n{lyrics}\n\n"
                "[Verse 2] (develop story)\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Fiddle Solo] (8 bars)\n\n"
                "[Bridge] (emotional peak)\n{lyrics}\n\n"
                "[Double Chorus] (with harmonies)"
            ),
            "default": (
                "[Intro]\n{lyrics}\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Bridge/Middle 8]\n{lyrics}\n\n"
                "[Chorus] (variation)\n\n"
                "[Outro] (optional vamp)"
            )
        }

        structure = structures.get(genre.lower(), structures["default"])
        
        prompt_addons = {
            "pop": "radio-ready, catchy hooks, polished production",
            "rock": "electric guitars, driving drums, raw energy",
            "hip hop": "punchy beats, rhythmic flow, urban vibe",
            "electronic": "synthesizers, pulsing bass, euphoric drops", 
            "country": "steel guitar, fiddle, storytelling, twangy vocals",
            "default": "melodic, emotionally expressive, professional mix"
        }
        
        intensity_modifiers = {
            "soft": "gentle, subtle, delicate, mellow, calm",
            "medium": "balanced, moderate, steady",
            "high": "energetic, powerful, intense, loud, aggressive"
        }
        
        mood_modifiers = {
            "happy": "uplifting, joyful, positive, cheerful",
            "sad": "melancholic, sorrowful, emotional, touching",
            "reflective": "thoughtful, introspective, contemplative",
            "angry": "passionate, intense, raw, powerful",
            "upbeat": "lively, optimistic, vibrant, spirited",
            "chill": "relaxed, laid-back, peaceful, smooth"
        }

        prompt = (
            f"Write a {genre} song about '{theme}' with this structure:\n"
            f"{structure}\n\n"
            "Key requirements:\n"
            f"- Use {language} language\n"  
            f"- {prompt_addons.get(genre.lower(), prompt_addons['default'])}\n"
            f"- {intensity_modifiers.get(self.intensity, intensity_modifiers['medium'])} feel\n"
            f"- {mood_modifiers.get(self.mood, mood_modifiers['upbeat'])} mood\n"
            "- Include vivid imagery and emotional resonance\n"
            "- Match the rhythm and phrasing to the genre\n"
            "- Do not write any other texts outside the lyrics\n"
            f"{'- Add country idioms (e.g. dusty roads, neon signs)' if genre.lower() == 'country' else ''}"
            f"{'- Include rap flow patterns' if genre.lower() == 'hip hop' else ''}"
        )
        
        print(f"\nLyric generation prompt:\n{prompt}")
        
        lyrics = ""
        if self.llm_model_path:  # Only try if we have a model path
            try:
                # Load model right before use
                self.load_llm()
                
                if self.llm:  # Check if load was successful
                    print("Using LLM for lyric generation...")
                    output = self.llm(
                        prompt,
                        max_tokens=1500,
                        temperature=0.7,
                        top_p=0.9,
                        repeat_penalty=1.1,
                        stop=["[End]", "\n\n\n"],
                        echo=False
                    )
                    
                    lyrics = output["choices"][0]["text"].strip()
                    print(f"Generated lyrics:\n{lyrics}")
                    
                    if not lyrics or len(lyrics.splitlines()) < 6:
                        raise ValueError("Lyrics too short or empty")
                        
                    lyrics = lyrics.replace("[Inst]", "").strip()
            except Exception as e:
                print(f"LLM generation failed: {str(e)}")
                lyrics = self._fallback_lyrics(genre, theme)
            finally:
                # Always unload after generation
                self.unload_llm()
        else:
            lyrics = self._fallback_lyrics(genre, theme)
        
        music_prompt = (
            f"{genre} song about {theme}, professional production, "
            f"high quality, {self.tempo} BPM, "
            f"{self.intensity} intensity, {self.mood} mood, "
            "clear vocals, catchy melody"
        )
        
        lyrics = self._clean_lyrics(lyrics)
        return lyrics, music_prompt

    def _clean_lyrics(self, lyrics_text: str) -> str:
        """Clean lyrics by removing instructional text and formatting"""
        cleaned = re.sub(r'\([^)]*\)', '', lyrics_text)  # Remove (instructions)
        cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)  # Normalize line breaks
        return cleaned.strip()

    def _fallback_lyrics(self, genre: str, theme: str) -> str:
        """Simple fallback lyric generation if LLM fails"""
        return (
            f"[Verse 1]\nSinging about {theme} in {genre} style\n"
            "The AI is making music all the while\n\n"
            f"[Chorus]\nThis is {genre} radio\n"
            "Generated just for you\n"
            "Listen close and you might know\n"
            "What these lyrics try to do\n\n"
            f"[Verse 2]\n{theme} is what we're singing\n"
            "In this digital creation\n"
            "No human hands were wringing\n"
            "Just pure AI inspiration"
        )

    def _generate_song_worker(self, genre: str, theme: str, duration: float):
        """Background worker for continuous song generation"""
        while not self.stop_event.is_set():
            try:
                # Always generate if queue is below buffer size
                if self.song_queue.qsize() < self.min_buffer_size:
                    # If in random mode, generate random parameters for each song
                    if self.random_mode:
                        genres = list(THEME_SUGGESTIONS.keys())
                        current_genre = random.choice(genres)
                        current_theme = random.choice(THEME_SUGGESTIONS.get(current_genre, THEME_SUGGESTIONS["default"]))
                        current_tempo = GENRE_TEMPOS.get(current_genre, GENRE_TEMPOS["default"])
                        current_intensity = random.choice(["soft", "medium", "high"])
                        current_mood = random.choice(["happy", "sad", "reflective", "upbeat", "chill"])
                        
                        song = self.generate_song(
                            genre=current_genre,
                            theme=current_theme,
                            tempo=current_tempo,
                            intensity=current_intensity,
                            mood=current_mood
                        )
                    else:
                        song = self.generate_song(genre, theme, duration)
                        
                    self.song_queue.put(song)
                    self.playlist.append(song)
                    
                    if len(self.playlist) % 5 == 0:
                        self._save_state()
                    
                    # Small delay between generations
                    time.sleep(1)
                else:
                    # If buffer is full, check more frequently
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"Error in generation worker: {e}")
                traceback.print_exc()
                # Ensure cleanup on error
                self.unload_llm()
                # Longer delay on error
                time.sleep(10)

    def _save_state(self):
        """Save current radio state to disk"""
        state = {
            "current_station": self.identity.name if self.identity else "",
            "current_song": self.current_song.title if self.current_song else None,
            "playlist": [song.title for song in self.playlist],
            "history": [song.title for song in self.history],
            "state": self.state.name,
            "timestamp": time.time()
        }
        
        with open("radio_state.json", "w") as f:
            json.dump(state, f, indent=2)

    def stop_radio(self):
        """Stop all radio operations and clean up resources"""
        self.stop_event.set()
        self.playback_paused.clear()
        self.state = RadioState.STOPPED
        self._first_play = True  # Reset the flag
        
        # Clean up both pipeline and LLM
        self.release_pipeline()
        self.unload_llm()
        
        if self.generation_thread and self.generation_thread.is_alive():
            self.generation_thread.join(timeout=5)
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=5)
        
        self._save_state()

    def pause_radio(self):
        """Pause playback (keeps generating in background)"""
        if self.state == RadioState.PLAYING:
            self.playback_paused.set()
            self.state = RadioState.PAUSED
            return True
        return False

    def resume_radio(self):
        """Resume paused playback"""
        if self.state == RadioState.PAUSED:
            self.playback_paused.clear()
            self.state = RadioState.PLAYING
            return True
        return False

    def generate_song(self, genre: str, theme: str, duration: float = 120.0, 
                 tempo: Optional[int] = None, intensity: Optional[str] = None,
                 mood: Optional[str] = None) -> Song:
        """
        Generate a complete song with lyrics and music.
        
        Args:
            genre: Music genre (pop, rock, etc.)
            theme: Song theme/topic
            duration: Song length in seconds
            
        Returns:
            Song: Generated song object
            
        Raises:
            Exception: If generation fails at any stage
        """
        print(f"\n=== Starting song generation ===")
        print(f"Genre: {genre}, Theme: {theme}, Duration: {duration}s")
        
        # Initialize progress tracking
        self.generation_progress = 0.0
        song = None
        
        try:
            # Stage 1: Generate lyrics
            print("\n[1/3] Generating lyrics...")
            self.generation_progress = 0.33
            lyrics, music_prompt = self.generate_lyrics_and_prompt(genre, theme, self.language)
            
            # Stage 2: Generate music
            print("\n[2/3] Generating music with ACEStepPipeline...")
            self.generation_progress = 0.66
            start_time = time.time()
            
            # Get pipeline instance (initializes if needed)
            pipeline = self.get_pipeline()
            
            try:
                results = pipeline(
                    audio_duration=duration,
                    prompt=music_prompt,
                    lyrics=lyrics,
                    infer_step=27,
                    guidance_scale=15.0,
                    scheduler_type="euler",
                    cfg_type="apg",
                    omega_scale=10.0,
                    batch_size=1
                )
                
                audio_path = results[0]
                metadata = results[-1]
                
                # Stage 3: Finalize song
                generation_time = time.time() - start_time
                print(f"\n[3/3] Song generated in {generation_time:.2f} seconds")
                self.generation_progress = 1.0
                
                song = Song(
                    title=f"{theme.title()} {random.randint(1, 100)}",
                    artist="AI Radio",
                    genre=genre,
                    theme=theme,
                    duration=duration,
                    lyrics=lyrics,
                    language=self.language, 
                    prompt=music_prompt,
                    audio_path=audio_path,
                    generation_time=generation_time,
                    timestamp=time.time(),
                    metadata=metadata
                )
                
                return song
                
            except Exception as e:
                print(f"Error during music generation: {e}")
                traceback.print_exc()
                raise
            finally:
                # Always release pipeline resources after use
                self.release_pipeline()
                
        except Exception as e:
            print(f"Error during song generation: {e}")
            traceback.print_exc()
            raise
        finally:
            # Final cleanup regardless of success/failure
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Ensure progress is marked complete if we have a song
            if song is not None:
                self.generation_progress = 1.0
            else:
                self.generation_progress = 0.0

def create_radio_interface(radio: AIRadioStation):
    """Create Gradio interface for the AI Radio Station"""
    vram_display = gr.JSON(label="VRAM Usage")
    
    def update_display():
        """Update the UI display with current radio status"""
        current_song = radio.current_song
        
        # Format audio for playback if playing
        song_audio = gr.Audio(
            value=current_song.audio_path if current_song and radio.state == RadioState.PLAYING else None,
            autoplay=True if current_song and radio.state == RadioState.PLAYING else False
        )


        # Format history for display - NOW INCLUDING LANGUAGE
        history_data = [
            [song.title, song.genre, song.theme, song.language, f"{song.duration:.1f}s", 
            time.strftime('%H:%M:%S', time.localtime(song.timestamp))]
            for song in radio.history[-10:]  # Show last 10 songs
        ]
        
        # Create status messages
        buffer_msg = (
            f"Buffering: {radio.song_queue.qsize()}/{radio.min_buffer_size} songs" 
            if radio.state == RadioState.BUFFERING 
            else f"Buffer: {radio.song_queue.qsize()} songs"
        )
        
        # Calculate playback progress
        if current_song and hasattr(radio, 'current_song_elapsed') and radio.state == RadioState.PLAYING:
            playback_percent = min(100, (radio.current_song_elapsed / current_song.duration) * 100)
            playback_time = f"{int(radio.current_song_elapsed)}s / {int(current_song.duration)}s"
        else:
            playback_percent = 0
            playback_time = "0s / 0s"
        
        # generation_percent = radio.generation_progress * 100
        # vram_info = radio.get_vram_usage() if hasattr(radio, 'get_vram_usage') else {}
    
        # Get song data
        song_lyrics = current_song.lyrics if current_song else "No song playing"
        song_metadata = current_song.metadata if current_song else {}
        
        return [
            radio.identity.name if radio.identity else "",
            f"{radio.state.name} - {radio.song_queue.qsize()} songs queued",
            radio.song_queue.qsize(),
            radio.state.name,
            song_audio,
            song_lyrics,
            song_metadata,
            playback_time,
            buffer_msg,
            playback_percent,
            history_data
        ]

    def start_radio(genre, theme, duration, buff_size, model_path, language, max_history, tempo, intensity, mood, random_mode):
        """Start the radio with the specified parameters"""
        # Update model if changed
        if model_path and hasattr(radio, 'llm') and radio.llm and model_path != radio.llm.model_path:
            try:
                from llama_cpp import Llama
                radio.llm = Llama(
                    model_path=model_path,
                    n_ctx=4048,
                    n_threads=4,
                    n_gpu_layers=-1,
                )
                print(f"Loaded new LLM model from: {model_path}")
            except Exception as e:
                print(f"Failed to load new LLM model: {e}")
        
        # Start the radio station
        radio.start_radio(
            genre=genre,
            theme=theme,
            duration=duration,
            buffer_size=buff_size,
            language=language,
            max_history=max_history,
            tempo=tempo,
            intensity=intensity,
            mood=mood,
            random_mode=random_mode
        )
        
        return update_display()
    
    def surprise_me():
        """Generate random genre and theme combinations"""
        genres = list(THEME_SUGGESTIONS.keys())
        genre = random.choice(genres)
        theme = random.choice(THEME_SUGGESTIONS.get(genre, THEME_SUGGESTIONS["default"]))
        
        # duration = GENRE_DURATIONS.get(genre, GENRE_DURATIONS["default"])
        # buffer_size = 1
        language = random.choice(list(SUPPORTED_LANGUAGES.keys()))  
        # max_history = 50
        tempo = GENRE_TEMPOS.get(genre, GENRE_TEMPOS["default"])
        intensity = random.choice(["soft", "medium", "high"])
        mood = random.choice(["happy", "sad", "reflective", "upbeat", "chill"])
        
        current_model_path = radio.llm.model_path if hasattr(radio, 'llm') and radio.llm else "gemma-3-12b-it-abliterated.q4_k_m.gguf"
        
        return (
            genre, theme, current_model_path, language, tempo, intensity, mood,
            *update_display()
        )
    
    def stop_radio(self):
        """Stop all radio operations and clean up resources"""
        self.stop_event.set()
        self.playback_paused.clear()
        self.state = RadioState.STOPPED
        
        # Clean up pipeline
        self.release_pipeline()
        
        if self.generation_thread and self.generation_thread.is_alive():
            self.generation_thread.join(timeout=5)
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=5)
        
        self._save_state()

    def pause_playback():
        """Pause radio playback"""
        radio.pause_radio()
        return update_display()
    
    def resume_playback():
        """Resume radio playback"""
        radio.resume_radio()
        return update_display()
    
    def update_theme_suggestions(genre):
        """Update theme suggestions based on selected genre"""
        return gr.Dropdown(
            choices=THEME_SUGGESTIONS.get(genre.lower(), THEME_SUGGESTIONS["default"]),
            value=random.choice(THEME_SUGGESTIONS.get(genre.lower(), THEME_SUGGESTIONS["default"]))
        )
    
    # Create the Gradio interface
    with gr.Blocks(title="AI Radio Station", theme="soft", css="""
        .custom-btn {
            background: var(--button-secondary-background-fill);
        }
        .custom-btn:hover {
            background: var(--button-secondary-background-fill-hover) !important;
        }
        .station-header {
            font-size: 1.5em;
            font-weight: bold;
            color: var(--primary-500);
        }
        .progress-bar {
            margin-top: 5px;
        }
    """) as demo:
        gr.Markdown("# 🎵 AI Radio Station")
        gr.Markdown("Continuous AI-powered music generation using ACE")
        
        # Add a timer component for automatic updates
        timer = gr.Timer(0.5, active=True)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Station Controls
                with gr.Group():
                    with gr.Tab("Basic Settings"):
                        genre_input = gr.Dropdown(
                            choices=list(THEME_SUGGESTIONS.keys()),
                            value="pop",
                            label="Music Genre"
                        )
                        theme_input = gr.Dropdown(
                            choices=THEME_SUGGESTIONS["pop"],
                            value="love",
                            label="Station Theme"
                        )
                        duration_input = gr.Slider(30, 600, value=120, label="Song Duration (seconds)")
                        buffer_size = gr.Slider(1, 10, value=1, step=1, label="Buffer Size (songs)")
                        random_mode = gr.Checkbox(label="Continuous Random Mode", value=False)
                        model_path_input = gr.File(
                            label="GGUF Model File",
                            file_types=[".gguf"],
                            value="gemma-3-12b-it-abliterated.q4_k_m.gguf"
                        )
                    
                    with gr.Tab("Advanced Settings"):
                        language_input = gr.Dropdown(
                            choices=list(SUPPORTED_LANGUAGES.keys()),
                            value="English",
                            label="Lyrics Language"
                        )
                        max_history = gr.Slider(10, 200, value=50, step=10, label="Max History Size")
                        tempo_input = gr.Slider(60, 200, value=120, step=5, label="Tempo (BPM)")
                        intensity_input = gr.Dropdown(
                            choices=["soft", "medium", "high"],
                            value="medium",
                            label="Intensity"
                        )
                        mood_input = gr.Dropdown(
                            choices=["happy", "sad", "reflective", "angry", "upbeat", "chill"],
                            value="upbeat",
                            label="Mood"
                        )
                    
                    with gr.Row():
                        start_btn = gr.Button("Start Radio", elem_classes="custom-btn")
                        stop_btn = gr.Button("Stop", elem_classes="custom-btn")
                        pause_btn = gr.Button("Pause", elem_classes="custom-btn")
                        resume_btn = gr.Button("Resume", elem_classes="custom-btn")
                    
                    with gr.Row():
                        refresh_btn = gr.Button("Refresh Display")
                        surprise_btn = gr.Button("Surprise Me!")
                
                # Status Display
                with gr.Group():
                    station_output = gr.Textbox(label="Current Station", elem_classes="station-header")
                    status_output = gr.Textbox(label="Status")
                    queue_size = gr.Number(label="Songs in Queue", visible=False)
                    state_display = gr.Textbox(label="Player State", visible=False)
                    playback_pos = gr.Textbox(label="Playback Position")
                    buffer_status = gr.Textbox(label="Buffer Status", visible=False)
                    
                    # gr.Markdown("### Generation Progress")
                    generation_progress = gr.Slider(0, 100, value=0, interactive=False, elem_classes="progress-bar", visible=False)
            
            with gr.Column(scale=2):
                # Now Playing Display
                with gr.Group():
                    current_song_output = gr.Audio(label="Now Playing", interactive=False, autoplay=False)
                    with gr.Tabs():
                        with gr.TabItem("Lyrics"):
                            lyrics_output = gr.Textbox(label="Lyrics", lines=10, interactive=False)
                        with gr.TabItem("Song Details"):
                            song_info = gr.JSON(label="Song Info")
                        with gr.TabItem("History"):
                            history_display = gr.Dataframe(
                                headers=["Title", "Genre", "Theme", "Language", "Duration", "Generated"],
                                interactive=False,
                                label="Recent Songs"
                            )

        # Define output components list
        output_components = [
            station_output, 
            status_output, 
            queue_size, 
            state_display,
            current_song_output,
            lyrics_output,
            song_info,
            playback_pos,
            buffer_status,
            generation_progress,
            history_display
        ]

        # Connect UI elements
        start_btn.click(
            fn=start_radio,
            inputs=[genre_input, theme_input, duration_input, buffer_size, model_path_input, 
                language_input, max_history, tempo_input, intensity_input, mood_input, random_mode],
            outputs=output_components
        )
        
        stop_btn.click(
            fn=stop_radio,
            outputs=output_components
        )
        
        pause_btn.click(
            fn=pause_playback,
            outputs=output_components
        )
        
        resume_btn.click(
            fn=resume_playback,
            outputs=output_components
        )
        
        refresh_btn.click(
            fn=update_display,
            outputs=output_components
        )
        
        surprise_btn.click(
            fn=surprise_me,
            outputs=[genre_input, theme_input, model_path_input,
                    language_input, tempo_input, intensity_input, mood_input,
                    *output_components]
        )
        
        # Update theme suggestions when genre changes
        genre_input.change(
            fn=update_theme_suggestions,
            inputs=genre_input,
            outputs=theme_input
        )
        
        # Set up automatic updates
        timer.tick(
            fn=update_display,
            outputs=output_components
        )
        
        # Initial state
        demo.load(
            fn=lambda: ["", "STOPPED - 0 songs queued", 0, "STOPPED", 
                    None, "No song playing", {}, "0s / 0s", "Buffer: 0 songs", 0, []],  # Add empty list for history
            outputs=output_components
        )
        
    return demo

def main():
    parser = argparse.ArgumentParser(description="AI Radio Station using ACE")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints",
                       help="Path to ACEStepPipeline checkpoints")
    parser.add_argument("--model_path", type=str, default="gemma-3-12b-it-abliterated.q4_k_m.gguf",
                       help="Path to LLM model for lyric generation")
    parser.add_argument("--server_name", type=str, default="0.0.0.0",
                       help="Server address to bind to")
    parser.add_argument("--port", type=int, default=7865,
                       help="Port to run the server on")
    parser.add_argument("--device_id", type=int, default=0,
                       help="GPU device ID to use")
    parser.add_argument("--share", default=False,
                       help="Share the Gradio interface publicly")
    parser.add_argument("--bf16", default=True,
                       help="Use bfloat16 precision")
    parser.add_argument("--torch_compile", default=False,
                       help="Enable torch compilation for faster inference")
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    
    # Initialize ACE-Step pipeline
    print("Initializing ACEStepPipeline...")
    pipeline = ACEStepPipeline(
        checkpoint_dir=args.checkpoint_path,
        dtype="bfloat16" if args.bf16 else "float32",
        torch_compile=args.torch_compile
    )
    

    # Initialize AI Radio Station
    print("Initializing AI Radio Station...")
    radio = AIRadioStation(
        ace_step_pipeline=pipeline,
        model_path=args.model_path
    )
    
    # Create and launch interface
    print("Launching interface...")
    interface = create_radio_interface(radio)
    interface.queue()
    interface.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share
    )

if __name__ == "__main__":
    main()
