from enum import IntEnum


class SDESolverTypes(IntEnum):
    ItoEulerMaruyama = 0
    ItoMilstein = 1
    ItoSRK1W1 = 2
    ItoSRK2W1 = 3
    ItoSRK2Wm = 4
    ItoSRK1Wm = 5
    ItoSRA3Wm = 6
