import numpy as np


# Object-specific manipulation segments
SEGMENTS = [
    ("Cereal", 0, 668, 315),
    ("Milk", 669, 1521, 1139),
    ("Can", 1522, 3811, 2232),
    ("Bread", 3812, 4441, 4130),
    (None, 4442, 4523, None),
]

# New initial object positions
SHIFTS = {
    "Cereal": np.array([0.05, 0, 0]),
    "Milk": np.array([-0.015, 0, 0]),
    "Can": np.array([0, 0.015, 0]),
    "Bread": np.array([0, -0.015, 0]),
}
