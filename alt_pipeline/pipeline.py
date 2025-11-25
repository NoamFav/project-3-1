import argparse
import signal
import os
import subprocess
from pathlib import Path
import json

from utils import (
    cleanup_services, signal_handler,
    extract_audio_features, find_session_directories,
    extract_audio_segment, extract_robot_data_features, compute_robot_winning_rate
)


def main():
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Extract args
    parser = argparse.ArgumentParser(description='Run feature extraction pipeline')
    parser.add_argument('--input', type=str, required=True, help="Folder path with raw audios")
    parser.add_argument('--output', type=str, required=True, help="Folder path for output files")
    parser.add_argument('--window-len', type=int, required=True, help="Length of audio windows in seconds")
    parser.add_argument('--default-video-name', type=str, required=True, help="Specifies which video from the Video directory to choose")
    parser.add_argument('--asr-diar-file', type=str, default=None, help="Path to JSON file with pre-computed ASR and diarization results (for debugging only)")
    parser.add_argument('--not-extract-audio', action='store_true', help="Whether to extract audios from the videos (for debugging only)")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Start services
    print("Starting services...")
    subprocess.run(['bash', 'alt_pipeline/start_services.sh'], check=True)
    
    session_dirs = find_session_directories(args.input)
    if len(session_dirs) == 0:
        print(f"No sessions found in {args.input}!")
        cleanup_services()
        exit(1)
        
    print(f"\nFound {len(session_dirs)} audio files to process")
    print(f"\n Preprocessing the data...")
    
    for dir in session_dirs:
        audio_dir = f"{dir}/Audio"
        
        if args.not_extract_audio is False:
            extract_audio_segment(f"{dir}/Videos/{args.default_video_name}.mp4", f"{audio_dir}/raw/{args.default_video_name}.wav", 0)
        
        subprocess.run(
            ['bash', 'src/data_pipeline/convert_audio.sh', '-o', f"{audio_dir}/processed", f"{audio_dir}/raw/{args.default_video_name}.wav"],
            check=True
        )
    
    print(f"Window length: {args.window_len} seconds\n")
    
    # Process audios sequentially
    processed_cnt = 0
    for dir in session_dirs:
        
        # Extract robot data features
        robot_data_features = extract_robot_data_features(dir)
        
        # Extract audio features
        audio_path = f"{dir}/Audio/processed/{args.default_video_name}.wav"
        video_path = f"{dir}/Videos/{args.default_video_name}.mp4"
        audio_features = extract_audio_features(audio_path, args.window_len, args.output, args.asr_diar_file)
        robot_speed_features = []

        for w in audio_features:
            res = compute_robot_winning_rate(video_path, w["window_start"], w["window_end"])
            robot_speed_features.append({
                "window_index": w["window_index"],
                "window_start": w["window_start"],
                "window_end": w["window_end"],
                "avg_speed_cm_s": res.get("avg_speed_cm_s"),
                "num_detections": res.get("num_detections"),
                "winning_rate": res.get("winning_rate")
            })

        
        # Save results
        dir_name = Path(dir).stem
        output_file = os.path.join(args.output, f"{dir_name}_features.json")
        
        with open(output_file, 'w') as f:
            json.dump({
                'session': dir_name,
                'base_window_length': args.window_len,
                'num_windows': len(audio_features),
                'audio_features': audio_features,
                'robot_data_features': robot_data_features,
                'robot_speed_features': robot_speed_features
            }, f, indent=2)
        
        print(f"✓ Completed {dir_name} -> {output_file}")
        processed_cnt += 1
    
    print(f"\n{'='*60}")
    print(f"Pipeline completed!")
    print(f"Successfully processed: {processed_cnt}/{len(session_dirs)} files")
    print(f"Output directory: {args.output}")
    print(f"{'='*60}")
    
    # Clean shutdown
    cleanup_services()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{'='*60}")
        print(f"Pipeline interrupted by user (Ctrl+C)")
        print(f"{'='*60}")
        cleanup_services()
        exit(130)
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"FATAL ERROR: {e}")
        print(f"{'='*60}")
        cleanup_services()
        exit(1)