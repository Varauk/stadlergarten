from timeit import default_timer as timer
from typing import Optional


class Output:
    classified_as_r_map: bool
    classified_as_matching_four_leaves: bool
    divergence_without_order: float
    divergence_with_order: float
    timer_start: float
    timer_end: Optional[float]
    plot_matrix: bool
    plot_steps: bool
    plot_box_plots: bool

    def __init__(self) -> None:
        self.classified_as_r_map = False
        self.classified_as_matching_four_leaves = False
        self.divergence_without_order = 0.0
        self.divergence_with_order = 0.0
        self.timer_start = timer()
        self.timer_end = None
        self.plot_matrix = False
        self.plot_steps = False
        self.plot_box_plots = False

    def __str__(self) -> str:
        string = 'Statistics for this runtrough:\n'

        if self.classified_as_r_map:
            string += '* Was correctly recognized as an R-Map.\n'
        if self.classified_as_matching_four_leaves:
            string += '* The final 4-leaf-map match the original 4-leaf-map!\n'

        string += ('* Proportion of diverging ' +
                   f'steps with order was {self.divergence_with_order :.2%}.\n')

        string += ('* Proportion of diverging steps ' +
                   f'without order was {self.divergence_without_order :.2%}.\n')

        total_time = (self.timer_end or float('inf')) - self.timer_start
        if total_time != 0.0:
            string += f'* The task took {total_time :.2f} seconds.\n'

        # TODO: Plotters
        if self.plot_matrix:
            pass
        if self.plot_steps:
            pass
        if self.plot_box_plots:
            pass

        return string

    def set_failed_state(self) -> None:
        self.classified_as_matching_four_leaves = False
        self.divergence_with_order = 1.0
        self.divergence_without_order = 1.0
        self.stop_timer()

    def stop_timer(self) -> None:
        self.timer_end = timer()

    @DeprecationWarning
    def print(self) -> None:
        print(self)
