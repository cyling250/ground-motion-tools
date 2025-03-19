from enum import Enum, unique


@unique
class GMIMEnum(Enum):
    """
    Type of Ground Motions Intensity Measures.

     ['PGA', 'PGV', 'PGD', 'RMSA', 'RMSV', 'RMSD', 'I_A', 'I_C', 'SED', 'CAV',
     'ASI', 'VSI', 'HI', 'SMA', 'SMV', 'Ia', 'Id', 'Iv',
    'If', 'Sa(T1)', 'Sv(T1)', 'Sd(T1)', 'T70', 'T90', 'FFT']

    Attributes:
        PGA: Peek Ground Acceleration.
        TODO Add More
    """
    PGA = 0
    PGV = 1
    PGD = 2
    RMSA = 3
    RMSV = 4
    RMSD = 5
    I_SUFFIX_A = 6
    I_SUFFIX_C = 7
    SED = 8
    CAV = 9
    SA_T1 = 10
    SV_T1 = 11
    SD_T1 = 12
    ASI = 13
    VSI = 14
    HI = 15
    SMA = 16
    SMV = 17
    I_A = 18
    I_D = 19
    I_V = 20
    I_F = 21
    MIV = 22
    DI = 23
    T70 = 24
    T90 = 25
