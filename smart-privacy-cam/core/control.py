"""
Author: Ethan O'Brien
Date: 4th July 2025
License: Open license, free to be redistributed

GammaController (control.py)
----------------------------
1. Initialize GammaController with target_brightness and PID parameters
2. Use SimplePID from utils.pid to compute gamma correction
3. Provide update method to compute new gamma value
4. Provide method to set target brightness
"""

from utils.pid import SimplePID

class GammaController:
    """
    GammaController.__init__
    ------------------------
    1. Initialize SimplePID with PID parameters and target brightness as setpoint
    2. Set initial gamma value to 1.0
    """
    def __init__(self, target_brightness=120, kp=0.01, ki=0.001, kd=0.005):
        self.pid = SimplePID(kp, ki, kd, setpoint=target_brightness)
        self.gamma = 1.0

    """
    GammaController.update
    ---------------------
    1. Use PID controller to compute new gamma value based on current brightness
    2. Update and return gamma value
    """
    def update(self, current_brightness):
        self.gamma = self.pid.update(current_brightness)
        return self.gamma

    """
    GammaController.set_target_brightness
    -------------------------------------
    1. Update the setpoint of the PID controller to the new target brightness
    """
    def set_target_brightness(self, target):
        self.pid.setpoint = target

    """
    GammaController.set_gamma
    ------------------------
    1. Set gamma value directly (for manual control)
    """
    def set_gamma(self, gamma):
        self.gamma = max(0.1, min(3.0, gamma))  # Clamp between 0.1 and 3.0 