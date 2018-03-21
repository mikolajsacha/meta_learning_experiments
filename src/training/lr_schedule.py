"""
Learning rate schedule for meta-learner optimization
"""
from typing import List, Tuple, Callable


def get_lr_scheduler(lr_schedule: List[Tuple[int, float]]) -> Callable[[int], float]:
    def get_meta_lr(epoch: int) -> float:
        for i, (min_epoch, rate) in enumerate(lr_schedule[1:]):
            if epoch < min_epoch:
                return lr_schedule[i][1]
        return lr_schedule[-1][1]
    return get_meta_lr


def create_cyclical_schedule(slope_length: int, n_cycles: int, min_lr: float, max_lr: float) -> List[Tuple[int, float]]:
    increasing_lrs = [(i, min_lr + i/slope_length * (max_lr - min_lr)) for i in range(slope_length)]
    decreasing_lrs = [(i + slope_length, max_lr - i/slope_length * (max_lr - min_lr)) for i in range(slope_length)]

    base_lr_schedule = [(i, round(v, 5)) for i, v in increasing_lrs + decreasing_lrs]
    lr_schedule = []
    for j in range(n_cycles):
        k = j*slope_length*2
        lr_schedule += [(k+i, v) for i, v in base_lr_schedule]
    lr_schedule.append((slope_length*n_cycles*2, min_lr))
    return lr_schedule


