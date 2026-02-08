"""Time budget manager for MLE-bench agent.

Tracks elapsed time and manages phase allocation across a 24-hour run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class Phase(Enum):
    """Agent phases within the time budget."""
    PARSING = "parsing"           # Task understanding
    BASELINE = "baseline"         # First valid submission
    SEARCH = "search"             # Main search/improvement loop
    REFINEMENT = "refinement"     # Final polish
    BUFFER = "buffer"             # Safety buffer -- stop here


# Default phase allocations as fractions of total time
DEFAULT_ALLOCATIONS = {
    Phase.PARSING: 0.02,      # 2% -- ~29 minutes for 24h
    Phase.BASELINE: 0.08,     # 8% -- ~115 minutes
    Phase.SEARCH: 0.70,       # 70% -- ~16.8 hours
    Phase.REFINEMENT: 0.15,   # 15% -- ~3.6 hours
    Phase.BUFFER: 0.05,       # 5% -- ~72 minutes safety buffer
}


@dataclass
class BudgetManager:
    """Manages time budget across an MLE-bench run.

    Usage:
        budget = BudgetManager(total_seconds=86400)
        budget.start()

        # Check if we should still be in a phase
        if budget.in_phase(Phase.SEARCH):
            ...

        # Check remaining time
        remaining = budget.remaining()

        # Get timeout for current step (leave room for future steps)
        step_timeout = budget.step_timeout(max_timeout=32400)
    """

    total_seconds: float = 86400.0  # 24 hours
    allocations: dict[Phase, float] = field(default_factory=lambda: dict(DEFAULT_ALLOCATIONS))

    # Internal state
    _start_time: Optional[float] = None
    _phase_start_times: dict[Phase, float] = field(default_factory=dict)
    _current_phase: Phase = Phase.PARSING
    _iteration_count: int = 0
    _has_valid_submission: bool = False

    def start(self) -> None:
        """Start the budget timer."""
        self._start_time = time.time()
        self._phase_start_times[Phase.PARSING] = self._start_time
        self._current_phase = Phase.PARSING
        logger.info(f"Budget started: {self.total_seconds}s total")

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds since start."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    @property
    def remaining(self) -> float:
        """Remaining time in seconds."""
        return max(0.0, self.total_seconds - self.elapsed)

    @property
    def fraction_elapsed(self) -> float:
        """Fraction of total time elapsed (0.0 to 1.0)."""
        if self.total_seconds <= 0:
            return 1.0
        return min(1.0, self.elapsed / self.total_seconds)

    @property
    def current_phase(self) -> Phase:
        """Current phase based on elapsed time."""
        return self._current_phase

    def phase_budget(self, phase: Phase) -> float:
        """Total seconds allocated to a phase."""
        return self.total_seconds * self.allocations.get(phase, 0.0)

    def phase_deadline(self, phase: Phase) -> float:
        """Cumulative deadline (seconds from start) for a phase."""
        cumulative = 0.0
        for p in Phase:
            cumulative += self.allocations.get(p, 0.0)
            if p == phase:
                break
        return self.total_seconds * cumulative

    def in_phase(self, phase: Phase) -> bool:
        """Check if we should still be in the given phase."""
        elapsed = self.elapsed
        # Calculate cumulative fraction for this phase's end
        return elapsed < self.phase_deadline(phase)

    def should_stop(self) -> bool:
        """Check if we should stop entirely (exceeded budget minus buffer)."""
        buffer_start = self.total_seconds * (1.0 - self.allocations.get(Phase.BUFFER, 0.05))
        return self.elapsed >= buffer_start

    def advance_phase(self, next_phase: Phase) -> None:
        """Manually advance to a new phase."""
        self._current_phase = next_phase
        self._phase_start_times[next_phase] = time.time()
        logger.info(
            f"Phase -> {next_phase.value} | "
            f"elapsed={self.elapsed:.0f}s | "
            f"remaining={self.remaining:.0f}s"
        )

    def step_timeout(self, max_timeout: float = 32400) -> float:
        """Calculate a reasonable timeout for the current step.

        Ensures we don't use all remaining time on one step.
        Leaves room for at least 2 more iterations + buffer.

        Args:
            max_timeout: Maximum timeout per step (default 9h).

        Returns:
            Timeout in seconds for the current step.
        """
        remaining = self.remaining
        buffer = self.phase_budget(Phase.BUFFER)

        # Available time = remaining - buffer - some reserve for next steps
        available = remaining - buffer
        if available <= 0:
            return 60.0  # Minimum 1 minute

        # Don't use more than 50% of available time on one step
        step_time = min(available * 0.5, max_timeout)

        # But at least 60 seconds
        return max(60.0, step_time)

    def record_iteration(self) -> None:
        """Record that an iteration was completed."""
        self._iteration_count += 1

    def record_valid_submission(self) -> None:
        """Record that a valid submission was produced."""
        self._has_valid_submission = True

    @property
    def has_valid_submission(self) -> bool:
        """Whether we have produced at least one valid submission."""
        return self._has_valid_submission

    @property
    def iteration_count(self) -> int:
        """Number of completed iterations."""
        return self._iteration_count

    def status(self) -> str:
        """Get a human-readable status string."""
        return (
            f"Budget: {self.elapsed:.0f}s / {self.total_seconds:.0f}s "
            f"({self.fraction_elapsed:.1%}) | "
            f"Phase: {self._current_phase.value} | "
            f"Iterations: {self._iteration_count} | "
            f"Valid submission: {self._has_valid_submission}"
        )
