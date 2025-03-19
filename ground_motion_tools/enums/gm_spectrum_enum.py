from enum import Enum, unique


@unique
class GMSpectrumEnum(Enum):
    """
    Type of Ground Motion Spectrum.
    Each Ground Motion have five different Spectrum types, see Details: http://www.jdcui.com/?p=713.
    Attributes:
        ACC: Acceleration response spectrum
        VEL: Velocity response spectrum
        DISP: Displacement response spectrum
        PSE_ACC: Pseudo acceleration response spectrum
        PSE_VEL: Pseudo velocity response spectrum
    """
    ACC = 0
    VEL = 1
    DISP = 2
    PSE_ACC = 3
    PSE_VEL = 4
