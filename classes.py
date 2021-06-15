from enum import Enum


class ProficiencyLabel(Enum):
    """
        Scanned region
    """
    Expert = 0
    Intermediate = 1
    Novice = 2


class Scan(Enum):
    """
        Scanned region
    """
    LUQ = 'Scan01'
    RUQ = 'Scan02'
    PERICARD = 'Scan03'
    PELVIC = 'Scan04'
    ALL = 'ALL'


TRANSFORM_KEY: str = 'transforms'
TIME_KEY: str = 'time'
PATH_LENGTH_KEY: str = 'path_len'
ANGULAR_SPEED: str = 'ang_speed'
LINEAR_SPEED: str = 'lin_speed'


class TransformationRecord:
    def __init__(self, trans_mat, time_stamp, linear_speed=0, angular_speed=0, path_length=0):
        self.trans_mat = trans_mat
        self.time_stamp = time_stamp
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.path_length = path_length


class RegionScan:
    def __init__(self, reg: Scan):
        self._region = reg
        self.path_len = 0.0
        self.linear_speed = 0.0
        self.angular_speed = 0.0
        self.time = 0.0
        self.transformations = []

    def get_regon(self) -> Scan:
        return self._region


class ParticipantScan:
    def __init__(self, part_name):
        self.store = dict()
        self.name = part_name
        self.time = 0.0
        self.path_length = 0.0
        self.angular_speed = 0.0
        self.linear_speed = 0.0

        self.store[Scan.ALL] = RegionScan(Scan.ALL)
        self.store[Scan.LUQ] = RegionScan(Scan.LUQ)
        self.store[Scan.RUQ] = RegionScan(Scan.RUQ)
        self.store[Scan.PELVIC] = RegionScan(Scan.PELVIC)
        self.store[Scan.PERICARD] = RegionScan(Scan.PERICARD)

    def get_transforms(self, reg: Scan) -> list:
        return self.store[reg].transformations

    def get_time(self) -> float:
        return self.time

    def get_reg_time(self, reg: Scan) -> float:
        return self.store[reg].time

    def get_region(self, reg: Scan) -> RegionScan:
        return self.store[reg]

    def add_transform(self, reg, transform_rec: TransformationRecord):
        self.store[reg].transformations.append(transform_rec)

    def get_name(self):
        return self.name

    def set_reg_time(self, reg: Scan, t: float) -> bool:
        if reg not in self.store:
            return False

        self.store[reg][TIME_KEY] = t
        return True

    def set_reg_lin_speed(self, reg: Scan, lin_s: float) -> bool:
        if reg not in self.store:
            return False

        self.store[reg].linear_speed = lin_s
        return True

    def add_reg_time(self, reg: Scan, t: float):
        if reg not in self.store:
            return False

        self.store[reg].time = self.store[reg].time + t
        return True

    def set_time(self, t: float) -> bool:
        self.time = t

    def add_time(self, t: float):
        self.time = self.time + t

    def set_reg_angular_speed(self, reg: Scan, ang_s: float):
        self.store[reg].angular_speed = ang_s

    def set_angular_speed(self, ang_s: float):
        self.angular_speed = ang_s


class ParticipantsData:
    def __init__(self):
        self.store = dict()

    def add_participant(self, part: ParticipantScan) -> bool:
        if part.get_name() not in self.store:
            self.store[part.get_name()] = part
            return True

        return False

    def __getitem__(self, item: str) -> ParticipantScan:
        return self.store[item]

    def __contains__(self, part_name: str) -> bool:
        return part_name in self.store

    def __iter__(self) -> ParticipantScan:
        for p in self.store:
            yield self.store[p]


class FoldSplit:
    def __init__(self, folds: list):
        self.folds = folds

    def __iter__(self):
        for i in range(len(self.folds)):
            yield self.folds[i]
