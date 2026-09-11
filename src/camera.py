"""
camera.py — Webcam access and frame capture.

Provides a clean interface for opening, reading, and releasing the webcam.
Handles errors gracefully (camera unavailable, permission denied, etc.).
"""

import cv2
import sys


class Camera:
    """Manages webcam access and frame capture."""

    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720):
        """
        Initialize the camera.

        Args:
            camera_index: Index of the camera device (0 = default webcam).
            width: Desired capture width.
            height: Desired capture height.
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None

    def open(self) -> bool:
        """
        Open the webcam.

        Returns:
            True if opened successfully, False otherwise.
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                # Fallback: try without DirectShow
                self.cap = cv2.VideoCapture(self.camera_index)

            if not self.cap.isOpened():
                print(f"[ERROR] Cannot open camera (index={self.camera_index}).")
                print("        Make sure your webcam is connected and not in use by another app.")
                return False

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            # Verify we can actually read a frame
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("[ERROR] Camera opened but cannot read frames.")
                print("        Check webcam permissions in Windows Settings > Privacy > Camera.")
                self.release()
                return False

            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[INFO] Camera opened successfully: {actual_w}x{actual_h}")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to open camera: {e}")
            return False

    def read(self):
        """
        Read a frame from the webcam.

        Returns:
            tuple: (success: bool, frame: numpy.ndarray or None)
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None

        return True, frame

    def release(self):
        """Release the webcam."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def is_opened(self) -> bool:
        """Check if camera is currently opened."""
        return self.cap is not None and self.cap.isOpened()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def test_camera():
    """Quick test: open camera, show a few frames, then close."""
    print("Testing camera access...")
    cam = Camera()
    if not cam.open():
        print("Camera test FAILED.")
        sys.exit(1)

    print("Camera test PASSED. Showing 3 seconds of feed...")
    import time
    start = time.time()
    while time.time() - start < 3:
        ret, frame = cam.read()
        if ret:
            cv2.imshow("Camera Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cam.release()
    cv2.destroyAllWindows()
    print("Camera test complete.")


if __name__ == "__main__":
    test_camera()
