"""
Author: Ethan O'Brien
Date: 4th July 2025
License: Open license, free to be redistributed

SimplePID (pid.py)
------------------
1. Initialize SimplePID with kp, ki, kd, setpoint, and output limits
2. Provide update method to compute PID output given a measurement
3. Store and update internal state (last error, integral)
"""

class SimplePID:
    """
    SimplePID.__init__
    ------------------
    1. Set proportional (kp), integral (ki), and derivative (kd) gains
    2. Set setpoint and output limits
    3. Initialize last error and integral accumulator to zero
    """
    def __init__(self, kp, ki, kd, setpoint=0, output_limits=(0.5, 3.0)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self._last_error = 0
        self._integral = 0

    """
    SimplePID.update
    ----------------
    1. Compute error as setpoint minus measurement
    2. Add error to integral accumulator
    3. Compute derivative as difference from last error
    4. Calculate output as weighted sum of error, integral, and derivative
    5. Update last error
    6. Clamp output to output limits
    7. Return output
    """
    def update(self, measurement):
        error = self.setpoint - measurement
        self._integral += error
        derivative = error - self._last_error
        output = (
            self.kp * error +
            self.ki * self._integral +
            self.kd * derivative
        )
        self._last_error = error
        min_out, max_out = self.output_limits
        return max(min_out, min(max_out, output)) 