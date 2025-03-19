from enum import Enum, unique


@unique
class GMDataEnum(Enum):
    """
    Type of Ground Motion Data
    Attributes:
        ACC: Acceleration
        VEL: Velocity
        DISP: Displacement
    """
    ACC = 0
    VEL = 1
    DISP = 2
