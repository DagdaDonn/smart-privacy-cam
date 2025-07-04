"""
Author: Ethan O'Brien
Date: 4th July 2025
License: Open license, free to be redistributed

AudioController (audio.py)
-------------------------
1. Initialize AudioController with manual_override flag (default False)
2. Provide methods to mute and unmute the microphone by simulating F4 keypress (with FnLock enabled)
3. Track mute state internally for GUI indicator
4. Provide method to set manual override (for GUI integration)
5. Provide method to check mute state
6. (Stub) Provide method for VAD integration
"""

import pyautogui

class AudioController:
    """
    AudioController.__init__
    -----------------------
    1. Set manual_override attribute
    2. Initialize internal mute state
    """
    def __init__(self, manual_override=False):
        self.manual_override = manual_override
        self._is_muted = False

    """
    AudioController.mute_mic
    -----------------------
    1. IF manual_override is False THEN
        a. Simulate F4 keypress to mute microphone (requires FnLock enabled)
        b. Set internal mute state to True
    """
    def mute_mic(self):
        if not self.manual_override:
            pyautogui.press('f4')
            self._is_muted = True

    """
    AudioController.unmute_mic
    -------------------------
    1. IF manual_override is False THEN
        a. Simulate F4 keypress to unmute microphone (requires FnLock enabled)
        b. Set internal mute state to False
    """
    def unmute_mic(self):
        if not self.manual_override:
            pyautogui.press('f4')
            self._is_muted = False

    """
    AudioController.set_manual_override
    ----------------------------------
    1. Set manual_override attribute to given value
    """
    def set_manual_override(self, override: bool):
        self.manual_override = override

    """
    AudioController.is_muted
    -----------------------
    1. Return current internal mute state
    """
    def is_muted(self):
        return self._is_muted

    """
    AudioController.detect_speech_while_muted (Stub)
    -----------------------------------------------
    1. Placeholder for VAD integration
    """
    def detect_speech_while_muted(self):
        pass  # To be implemented with VAD 