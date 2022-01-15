from enum import Enum
from typing import Final, Union, List


class WorkPackage(Enum):
    WP2: Final = 0
    # Benchmark with three specified blocked leafes
    WP3_1: Final = 1
    # Benchmark with four specified blocked leafes
    WP3_2: Final = 2
    # Benchmark with subsets of three blocked leafes
    WP3_3_1: Final = 3
    # Benchmark with subsets of four blocked leafes
    WP3_3_2: Final = 4
    # Benchmark like WP2 with smallest spikes and our calculation
    WP4_1: Final = 5
    # Benchmark like WP2 with smallest spikes and erdbeermet calculation
    WP4_2: Final = 6

    def __str__(self) -> str:
        map = {WorkPackage.WP2: 'WP2',
               WorkPackage.WP3_1: 'WP3_1',
               WorkPackage.WP3_2: 'WP3_2',
               WorkPackage.WP3_3_1: 'WP3_3_1',
               WorkPackage.WP3_3_2: 'WP3_3_2',
               WorkPackage.WP4_1: 'WP4_1',
               WorkPackage.WP4_2: 'WP4_2'}
        return map[self]

    def get_forbidden_leaves(self) -> Union[List[int], int, None]:
        '''Get the forbidden_leaves value for this workpackage'''
        if self == WorkPackage.WP3_1:
            return [0, 1, 2]
        elif self == WorkPackage.WP3_2:
            return [0, 1, 2, 3]
        elif self == WorkPackage.WP3_3_1:
            return 3
        elif self == WorkPackage.WP3_3_2:
            return 4
        else:
            return None

    @staticmethod
    def from_cli_arg(arg: str) -> List['WorkPackage']:
        map = {'2': WorkPackage.WP2,
               '31': WorkPackage.WP3_1,
               '32': WorkPackage.WP3_2,
               '331': WorkPackage.WP3_3_1,
               '332': WorkPackage.WP3_3_2,
               '41': WorkPackage.WP4_1,
               '42': WorkPackage.WP4_2}
        if arg == 'all':
            # Return a list of all workpackages
            return list(map.values())
        else:
            # Return a list with a single element,
            # the workpackage that needs execution
            return [map[arg]]
        pass
