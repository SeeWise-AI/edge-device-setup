"""display.py
"""


import time

import cv2


def open_window(window_name, title, width=None, height=None):
    """Open the display window."""
    print("Opening the window.")
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowTitle(window_name, title)
    if width and height:
        cv2.resizeWindow(window_name, width, height)


def show_help_text(img, help_text):
    """Draw help text on image."""
    cv2.putText(img, help_text, (11, 20), cv2.FONT_HERSHEY_PLAIN, 1.0,
                (32, 32, 32), 4, cv2.LINE_AA)
    cv2.putText(img, help_text, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.0,
                (240, 240, 240), 1, cv2.LINE_AA)
    return img


def show_fps(img, fps):
    """Draw FPS number at top-left corner of image."""
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2  # Increase this value to make the text bigger
    font_thickness = 3  # Increase this value to make the text bolder
    fps_text = 'FPS: {:.2f}'.format(fps)
    text_size, baseline = cv2.getTextSize(fps_text, font_face, font_scale, font_thickness)
    text_origin = (10, 10 + text_size[1])
    cv2.putText(img, fps_text, text_origin, font_face, font_scale, (0, 255, 0), font_thickness)
    return img


def set_display(window_name, full_scrn):
    """Set disply window to either full screen or normal."""
    if full_scrn:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
    else:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_NORMAL)


class FpsCalculator():
    """Helper class for calculating frames-per-second (FPS)."""

    def __init__(self, decay_factor=0.95):
        self.fps = 0.0
        self.tic = time.time()
        self.decay_factor = decay_factor

    def update(self):
        toc = time.time()
        curr_fps = 1.0 / (toc - self.tic)
        self.fps = curr_fps if self.fps == 0.0 else self.fps
        self.fps = self.fps * self.decay_factor + \
                   curr_fps * (1 - self.decay_factor)
        self.tic = toc
        return self.fps

    def reset(self):
        self.fps = 0.0


class ScreenToggler():
    """Helper class for toggling between non-fullscreen and fullscreen."""

    def __init__(self):
        self.full_scrn = False

    def toggle(self):
        self.full_scrn = not self.full_scrn
        set_display(WINDOW_NAME, self.full_scrn)
