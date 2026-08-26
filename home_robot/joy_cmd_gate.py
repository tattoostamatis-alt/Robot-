"""State machine that suppresses repeated idle joystick Twist messages."""


class JoyCommandGate:
    """Forward motion continuously, but forward only the first zero on release."""

    def __init__(self, epsilon=1e-4):
        self.epsilon = float(epsilon)
        self.active = False

    def should_forward(self, values):
        moving = any(abs(float(value)) > self.epsilon for value in values)
        if moving:
            self.active = True
            return True
        if self.active:
            self.active = False
            return True
        return False

    def stop_if_active(self):
        if not self.active:
            return False
        self.active = False
        return True
