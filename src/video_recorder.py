"""
Video Recorder - Captures screen content with high performance
"""
import cv2
import mss
import numpy as np
import threading
import time
from typing import Tuple, Optional


class VideoRecorder:
    """Records screen content using mss for fast capture and OpenCV for encoding"""

    def __init__(
        self,
        output_file: str,
        fps: int = 30,
        resolution: Optional[Tuple[int, int]] = None,
        monitor_number: int = 1
    ):
        """
        Initialize video recorder

        Args:
            output_file: Path to output video file
            fps: Frames per second for recording
            resolution: Optional (width, height) tuple for output resolution
            monitor_number: Monitor number to capture (1 = primary)
        """
        self.output_file = output_file
        self.fps = fps
        self.resolution = resolution
        self.monitor_number = monitor_number

        self.is_recording = False
        self.record_thread = None
        self.video_writer = None
        self.sct = mss.mss()

        # Get monitor information
        self.monitor = self.sct.monitors[monitor_number]

        # Calculate actual resolution
        if resolution:
            self.width, self.height = resolution
        else:
            self.width = self.monitor["width"]
            self.height = self.monitor["height"]

        # Frame timing
        self.frame_duration = 1.0 / fps
        self.frames_recorded = 0
        self.start_time = None

    def start(self) -> bool:
        """
        Start recording video

        Returns:
            True if started successfully
        """
        if self.is_recording:
            return False

        try:
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.output_file,
                fourcc,
                self.fps,
                (self.width, self.height)
            )

            if not self.video_writer.isOpened():
                print("Failed to open video writer")
                return False

            self.is_recording = True
            self.frames_recorded = 0
            self.start_time = time.time()

            # Start recording thread
            self.record_thread = threading.Thread(target=self._record_loop, daemon=True)
            self.record_thread.start()

            return True

        except Exception as e:
            print(f"Error starting video recorder: {e}")
            return False

    def stop(self) -> None:
        """Stop recording and finalize video file"""
        if not self.is_recording:
            return

        self.is_recording = False

        # Wait for recording thread to finish
        if self.record_thread:
            self.record_thread.join(timeout=5.0)

        # Release video writer
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        elapsed_time = time.time() - self.start_time if self.start_time else 0
        print(f"Video recording stopped. Recorded {self.frames_recorded} frames in {elapsed_time:.2f}s")

    def _record_loop(self) -> None:
        """Main recording loop (runs in separate thread)"""
        try:
            next_frame_time = time.time()

            while self.is_recording:
                current_time = time.time()

                # Maintain target framerate
                if current_time >= next_frame_time:
                    # Capture screen
                    frame = self._capture_frame()

                    if frame is not None:
                        # Write frame to video
                        self.video_writer.write(frame)
                        self.frames_recorded += 1

                    # Calculate next frame time
                    next_frame_time += self.frame_duration

                    # If we're falling behind, skip to current time
                    if next_frame_time < current_time:
                        next_frame_time = current_time + self.frame_duration
                else:
                    # Sleep for a short time to avoid busy waiting
                    time.sleep(0.001)

        except Exception as e:
            print(f"Error in video recording loop: {e}")
            self.is_recording = False

    def _capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from the screen

        Returns:
            Frame as numpy array in BGR format, or None on error
        """
        try:
            # Capture screen
            screenshot = self.sct.grab(self.monitor)

            # Convert to numpy array
            frame = np.array(screenshot)

            # Convert BGRA to BGR (remove alpha channel)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Resize if necessary
            if self.resolution and (frame.shape[1] != self.width or frame.shape[0] != self.height):
                frame = cv2.resize(frame, (self.width, self.height))

            return frame

        except Exception as e:
            print(f"Error capturing frame: {e}")
            return None

    def get_stats(self) -> dict:
        """
        Get recording statistics

        Returns:
            Dictionary with recording stats
        """
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        actual_fps = self.frames_recorded / elapsed_time if elapsed_time > 0 else 0

        return {
            "frames_recorded": self.frames_recorded,
            "elapsed_time": elapsed_time,
            "actual_fps": actual_fps,
            "target_fps": self.fps,
            "resolution": f"{self.width}x{self.height}"
        }

    def cleanup(self) -> None:
        """Clean up resources"""
        if self.video_writer:
            self.video_writer.release()
        if self.sct:
            self.sct.close()

    @staticmethod
    def list_monitors() -> list:
        """
        List available monitors

        Returns:
            List of monitor information dictionaries
        """
        with mss.mss() as sct:
            monitors = []
            for i, monitor in enumerate(sct.monitors[1:], 1):  # Skip monitor 0 (all monitors)
                monitors.append({
                    "number": i,
                    "width": monitor["width"],
                    "height": monitor["height"],
                    "left": monitor["left"],
                    "top": monitor["top"]
                })
            return monitors
