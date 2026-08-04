"""Predeclared recurrence ablations; this module does not claim compatibility."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RecurrenceArm:
    name: str
    high_cycles: int
    low_cycles: int
    is_pretrained_schedule: bool

    @property
    def stack_invocations(self) -> int:
        return self.high_cycles * (self.low_cycles + 1)


def recurrence_arms(low_cycles: int = 3) -> tuple[RecurrenceArm, ...]:
    return tuple(
        RecurrenceArm(f"H{cycles}L{low_cycles}", cycles, low_cycles, cycles == 2)
        for cycles in (1, 2, 3, 4)
    )
