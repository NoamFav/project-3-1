import os
import subprocess
import json
import requests
import librosa
import numpy as np
import signal
import sys
from pathlib import Path
from typing import Dict, Any

# Service endpoints
SERVICES = {
    'asr': 'http://127.0.0.1:8001',
    'diarization': 'http://127.0.0.1:8002',
    'emotion': 'http://127.0.0.1:8003',
    'nlp': 'http://127.0.0.1:8004',
    'nonverb': 'http://127.0.0.1:8005',
    'robot_data': 'http://127.0.0.1:8006',
    'robot_speed': 'http://127.0.0.1:8007'
}


def find_session_directories(input_dir: str) -> list[str]:
    """Traverse directory tree and find all directories containing 'session.log'."""
    session_dirs = []
    
    # Walk through all directories
    for root, dirs, files in os.walk(input_dir):
        if 'session.log' in files:
            # Convert to relative path from input_dir
            session_dirs.append(root)
    
    return session_dirs


def extract_audio_segment(video_path, output_path, start_time, end_time=None):
    """Extract audio segment from video using FFmpeg."""
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-ss', str(start_time),      # Start time in seconds
    ]
    
    # Only add end time if specified
    if end_time is not None:
        cmd.extend(['-to', str(end_time)])
    
    cmd.extend([
        '-vn',                       # No video
        '-acodec', 'pcm_s16le',           # Copy audio codec (no re-encoding)
        '-ar', '16000',              # Sample rate (16kHz for speech)
        '-ac', '1',                  # Mono channel
        output_path
    ])
    
    subprocess.run(cmd, check=True)
    

def extract_robot_data_features(session_dir):
    timeline_csv = f"{session_dir}/robot_data.csv"
    
    response = requests.post(
        f"{SERVICES['robot_data']}/parse_logs",
        json={
            'log_dir': session_dir,
            'output_csv': timeline_csv
        }
    )
    
    robot_interaction_features = requests.post(
        f"{SERVICES['robot_data']}/extract_features",
        json={"timeline_csv": timeline_csv}
    ).json()
    
    return robot_interaction_features


def run_asr_and_diarization(audio_path: str) -> Dict[str, Any]:
    """Run ASR and diarization on the full audio file."""
    print(f"  Running ASR and diarization on full audio...")
    
    results = {}
    
    response = requests.post(
        f"{SERVICES['asr']}/transcribe",
        json=audio_path
    )
    results['asr'] = response.json()
    
    response = requests.post(
        f"{SERVICES['diarization']}/diarize",
        json=audio_path
    )
    results['diarization'] = response.json()
    
    return results


def get_window_segments(full_asr: Dict, full_diarization: Dict, window_start: float, window_end: float) -> tuple[Dict[str, Any], float]:
    """
    Extract ASR and diarization segments for the window with flexible boundaries.
    If a speaker's turn starts within the window (>= window_start and < window_end), 
    include the entire turn even if it extends beyond window_end.
    Returns: (window_segments, actual_window_end)
    """
    window_asr_segments = []
    window_diar_segments = []
    actual_end = window_end
    
    # Process ASR segments
    for seg in full_asr['segments']:
        if window_start <= seg['start'] < window_end:
            # Include this segment if it starts within the window
            window_asr_segments.append(seg)
            # Extend window boundary if segment extends beyond
            if seg['end'] > actual_end:
                actual_end = seg['end']
        elif seg['start'] >= window_end:
            # Stop when we reach segments that start after window
            break
    
    # Process diarization segments (speaker turns)
    for seg in full_diarization['segments']:
        if window_start <= seg['start'] < window_end:
            # Include this speaker turn if it starts within the window
            window_diar_segments.append(seg)
            # Extend window boundary if turn extends beyond
            if seg['end'] > actual_end:
                actual_end = seg['end']
        elif seg['start'] >= window_end:
            # Stop when we reach turns that start after window
            break
    
    return {
        'asr': {'segments': window_asr_segments, 'language': full_asr['language']},
        'diarization': {'segments': window_diar_segments, 'num_speakers': full_diarization['num_speakers']}
    }, actual_end


def extract_features_for_window(window: np.ndarray, window_segments: Dict[str, Any], window_index: int, window_len: int, sr: int = 16000) -> Dict[str, Any]:
    """Extract emotion, NLP, and non-verbal features for a single audio window."""
    features = {}
    
    asr_result = window_segments['asr']
    diarization_result = window_segments['diarization']
    
    # 3. Emotion - Detect emotions from transcribed text
    full_text = ' '.join([seg['text'] for seg in asr_result['segments']])
    
    if full_text.strip():  # Only process if there's text
        response = requests.post(
            f"{SERVICES['emotion']}/detect",
            json={'text': full_text}
        )
        features['emotion'] = response.json()
        
        # 4. NLP - Extract keywords and sentiment
        # Preprocess
        response = requests.post(
            f"{SERVICES['nlp']}/preprocess",
            json={'text': full_text}
        )
        features['nlp'] = {'preprocess': response.json()}
        
        # Keywords
        response = requests.post(
            f"{SERVICES['nlp']}/keywords",
            json={'text': full_text, 'top_n': 5}
        )
        features['nlp']['keywords'] = response.json()
        
        # Sentiment
        response = requests.post(
            f"{SERVICES['nlp']}/sentiment",
            json={'text': full_text}
        )
        features['nlp']['sentiment'] = response.json()
    else:
        features['emotion'] = None
        features['nlp'] = None
    
    # 5. Non-verbal features
    y_list = window.tolist()
    
    # Basic metrics
    response = requests.post(
        f"{SERVICES['nonverb']}/basic_metrics",
        json={
            'diarization': diarization_result,
            'conv_length': len(window) / sr
        }
    )
    features['nonverbal'] = {'basic_metrics': response.json()}
    
    # Pitch features
    response = requests.post(
        f"{SERVICES['nonverb']}/pitch_features",
        json={
            'diarization': diarization_result,
            'y': y_list,
            'sr': sr
        }
    )
    features['nonverbal']['pitch'] = response.json()
    
    # Loudness features
    response = requests.post(
        f"{SERVICES['nonverb']}/loudness_features",
        json={
            'diarization': diarization_result,
            'y': y_list,
            'sr': sr
        }
    )
    features['nonverbal']['loudness'] = response.json()
    
    # Tempo features
    response = requests.post(
        f"{SERVICES['nonverb']}/tempo_features",
        json={
            'diarization': diarization_result,
            'y': y_list,
            'sr': sr
        }
    )
    features['nonverbal']['tempo'] = response.json()
    
    return features


def extract_audio_features(audio_path: str, window_len: int, output_dir: str, asr_diar_path: str = None, sr: int = 16000) -> str:
    """Process a single audio file: run ASR/diarization first, then split into windows and extract features."""
    print(f"Processing {audio_path}...")
    
    # Step 1: Run ASR and diarization on full audio or load from file
    if asr_diar_path:
        print(f"  Loading ASR and diarization results from {asr_diar_path}...")
        with open(asr_diar_path, 'r') as f:
            full_results = json.load(f)
    else:
        full_results = run_asr_and_diarization(audio_path)
    
    # Step 2: Load audio
    y, _ = librosa.load(audio_path, sr=sr)
    window_samples = window_len * sr
    total_duration = len(y) / sr
    
    # Step 3: Extract features for each window
    all_features = []
    window_index = 0
    current_position = 0  # Track position in samples
    
    while current_position < len(y):
        # Calculate base window boundaries
        window_start_time = current_position / sr
        window_end_time = min((current_position + window_samples) / sr, total_duration)
        
        # Get segments with flexible boundaries
        window_segments, actual_end_time = get_window_segments(
            full_results['asr'], 
            full_results['diarization'],
            window_start_time,
            window_end_time
        )
        
        # Extract the actual window audio (may extend beyond base window)
        actual_end_samples = int(actual_end_time * sr)
        actual_end_samples = min(actual_end_samples, len(y))  # Don't exceed audio length
        
        if actual_end_samples - current_position > 0:
            print(f"  Extracting features for window {window_index + 1} ({window_start_time:.1f}s - {actual_end_time:.1f}s)...")
            
            # Extract features
            features = extract_features_for_window(y, window_segments, window_index, window_len, sr)
            features['window_index'] = window_index
            features['window_start'] = window_start_time
            features['window_end'] = actual_end_time
            features['window_duration'] = actual_end_time - window_start_time
            features['base_window_len'] = window_len
            
            all_features.append(features)
            window_index += 1
        
        # Move to next window
        current_position = actual_end_samples
    
    return all_features


def compute_robot_winning_rate(video_path: str, window_start: float, window_end: float) -> dict:
    """
    Call the robot speed service to compute a winning rate
    for the given time window in the video.
    """
    payload = {
        "video_path": video_path,
        "window_start": window_start,
        "window_end": window_end,
    }
    response = requests.post(
        f"{SERVICES['robot_speed']}/winning_rate",
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()

def cleanup_services():
    """Shutdown all uvicorn services."""
    print("\nShutting down services...")
    try:
        subprocess.run(['pkill', '-f', 'uvicorn'], check=False)
        print("Services stopped.")
    except Exception as e:
        print(f"Error stopping services: {e}")


def signal_handler(sig, frame):
    """Handle SIGINT (Ctrl+C) and SIGTERM signals."""
    signal_name = 'SIGINT' if sig == signal.SIGINT else 'SIGTERM'
    print(f"\n{'='*60}")
    print(f"Received {signal_name} - shutting down pipeline...")
    print(f"{'='*60}")
    cleanup_services()
    sys.exit(130 if sig == signal.SIGINT else 143)