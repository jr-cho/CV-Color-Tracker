import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk


class BlueTracker:
    def __init__(self, root):
        self.root = root
        self.root.title("Blue Tracker")

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")

        self.lower_blue = np.array([100, 150, 50])
        self.upper_blue = np.array([140, 255, 255])

        self.video_label = tk.Label(root)
        self.video_label.pack()

        self.root.bind("<KeyPress-q>", lambda event: self.cleanup())

        self.root.focus_set()

        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to read from camera")
            return
        frame = cv2.resize(frame, (640, 480))

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)

        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        blue_detected = False
        largest_center = None

        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 500:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(largest)
                cv2.rectangle(
                    masked_frame, (x, y), (x + w, y + h), (0, 255, 0), 3
                )  # Green rectangle
                blue_detected = True
                largest_center = (x + w // 2, y + h // 2)

                cv2.circle(
                    masked_frame, largest_center, 5, (0, 0, 255), -1
                )  # Red center point

        frame_rgb = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        if blue_detected:
            print(f"Blue detected! Center at {largest_center}")
        else:
            print("Blue not detected")

        self.root.after(10, self.update_frame)

    def cleanup(self):
        """Clean up resources and close application"""
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    try:
        app = BlueTracker(root)
        root.mainloop()
    except RuntimeError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("Application interrupted by user")
    finally:
        cv2.destroyAllWindows()
