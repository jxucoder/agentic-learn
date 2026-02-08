"""Hierarchical memory for sustained 24-hour MLE-bench runs.

Three tiers prevent context window overflow while preserving strategic knowledge:
- Working Memory: current code, recent errors, last few execution results (full detail)
- Episode Memory: condensed summary per approach branch (approach, score, learnings)
- Semantic Memory: distilled facts about the task and what works
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class WorkingMemoryEntry:
    """A single entry in working memory (full detail)."""
    iteration: int
    approach: str
    code: str
    score: Optional[float]
    success: bool
    stdout: str
    stderr: str
    error_message: str = ""


@dataclass
class EpisodeEntry:
    """Condensed summary of an approach attempt."""
    approach: str
    best_score: Optional[float]
    attempts: int
    last_score: Optional[float]
    key_learnings: list[str] = field(default_factory=list)
    error_summary: str = ""
    is_promising: bool = False


@dataclass
class SemanticEntry:
    """A distilled fact about the task or solution space."""
    fact: str
    confidence: float = 1.0  # 0-1
    source: str = ""         # Which episode produced this


class HierarchicalMemory:
    """Three-tier memory system for long-running agent sessions.

    Usage:
        memory = HierarchicalMemory(max_working=5)

        # After each execution
        memory.add_working(entry)

        # Periodically compress
        memory.compress()

        # Get context for LLM
        context = memory.to_context(max_tokens=4000)
    """

    def __init__(
        self,
        max_working: int = 5,
        max_episodes: int = 20,
        max_semantic: int = 30,
    ):
        self.max_working = max_working
        self.max_episodes = max_episodes
        self.max_semantic = max_semantic

        self.working: list[WorkingMemoryEntry] = []
        self.episodes: dict[str, EpisodeEntry] = {}  # Keyed by approach
        self.semantic: list[SemanticEntry] = []

    # -----------------------------------------------------------------
    # Working Memory
    # -----------------------------------------------------------------

    def add_working(self, entry: WorkingMemoryEntry) -> None:
        """Add an entry to working memory, evicting oldest if full."""
        self.working.append(entry)

        # Evict oldest to episode memory if over capacity
        while len(self.working) > self.max_working:
            evicted = self.working.pop(0)
            self._promote_to_episode(evicted)

    def _promote_to_episode(self, entry: WorkingMemoryEntry) -> None:
        """Promote a working memory entry to episode memory."""
        approach = entry.approach
        if approach not in self.episodes:
            self.episodes[approach] = EpisodeEntry(
                approach=approach,
                best_score=entry.score,
                attempts=0,
                last_score=entry.score,
            )

        ep = self.episodes[approach]
        ep.attempts += 1
        ep.last_score = entry.score

        if entry.score is not None:
            if ep.best_score is None or entry.score > ep.best_score:
                ep.best_score = entry.score
                ep.is_promising = True

        if entry.error_message:
            ep.error_summary = entry.error_message[:200]

        # Extract learning from the attempt
        if entry.success and entry.score is not None:
            ep.key_learnings.append(f"Score {entry.score:.4f} achieved")
        elif entry.error_message:
            ep.key_learnings.append(f"Failed: {entry.error_message[:100]}")

        # Keep learnings manageable
        if len(ep.key_learnings) > 5:
            ep.key_learnings = ep.key_learnings[-5:]

    # -----------------------------------------------------------------
    # Compression
    # -----------------------------------------------------------------

    def compress(self) -> None:
        """Compress memories: promote working -> episode, distill episode -> semantic."""
        # Promote all working entries older than the most recent
        while len(self.working) > self.max_working:
            evicted = self.working.pop(0)
            self._promote_to_episode(evicted)

        # Distill episodes into semantic facts
        for approach, ep in self.episodes.items():
            if ep.attempts >= 3 and ep.best_score is not None:
                fact = f"{approach}: best score {ep.best_score:.4f} in {ep.attempts} attempts"
                if not any(s.fact.startswith(approach) for s in self.semantic):
                    self.semantic.append(SemanticEntry(
                        fact=fact,
                        confidence=min(1.0, ep.attempts / 5),
                        source=approach,
                    ))

        # Trim semantic memory
        if len(self.semantic) > self.max_semantic:
            # Keep highest confidence
            self.semantic.sort(key=lambda s: s.confidence, reverse=True)
            self.semantic = self.semantic[:self.max_semantic]

        # Trim episodes
        if len(self.episodes) > self.max_episodes:
            # Keep most promising + most recent
            sorted_eps = sorted(
                self.episodes.items(),
                key=lambda x: (x[1].is_promising, x[1].best_score or 0),
                reverse=True,
            )
            self.episodes = dict(sorted_eps[:self.max_episodes])

    # -----------------------------------------------------------------
    # Add semantic facts directly
    # -----------------------------------------------------------------

    def add_semantic(self, fact: str, confidence: float = 1.0, source: str = "") -> None:
        """Add a semantic fact directly."""
        self.semantic.append(SemanticEntry(fact=fact, confidence=confidence, source=source))
        if len(self.semantic) > self.max_semantic:
            self.semantic.sort(key=lambda s: s.confidence, reverse=True)
            self.semantic = self.semantic[:self.max_semantic]

    # -----------------------------------------------------------------
    # Context Generation
    # -----------------------------------------------------------------

    def to_context(self, max_chars: int = 6000) -> str:
        """Generate LLM-consumable context from all memory tiers.

        Prioritizes: semantic (most compressed) > episodes > working (most detailed).
        """
        parts: list[str] = []
        remaining = max_chars

        # Semantic memory (most compressed, always included)
        if self.semantic:
            facts = "\n".join(f"- {s.fact}" for s in self.semantic[:15])
            section = f"## Key Learnings\n{facts}\n"
            parts.append(section)
            remaining -= len(section)

        # Episode memory (condensed per approach)
        if self.episodes and remaining > 500:
            ep_lines = []
            sorted_eps = sorted(
                self.episodes.values(),
                key=lambda e: e.best_score or 0,
                reverse=True,
            )
            for ep in sorted_eps[:10]:
                score_str = f"{ep.best_score:.4f}" if ep.best_score is not None else "N/A"
                status = "promising" if ep.is_promising else "explored"
                ep_lines.append(f"- {ep.approach}: best={score_str}, attempts={ep.attempts} ({status})")
                for learning in ep.key_learnings[-2:]:
                    ep_lines.append(f"  - {learning}")

            section = f"\n## Approaches Tried\n" + "\n".join(ep_lines) + "\n"
            if len(section) < remaining:
                parts.append(section)
                remaining -= len(section)

        # Working memory (recent, full detail -- truncated)
        if self.working and remaining > 500:
            wm_lines = []
            for entry in self.working[-3:]:
                score_str = f"{entry.score:.4f}" if entry.score is not None else "N/A"
                status = "OK" if entry.success else "FAIL"
                wm_lines.append(f"### Iteration {entry.iteration} ({status}, score={score_str})")
                if entry.error_message:
                    wm_lines.append(f"Error: {entry.error_message[:200]}")
                if entry.stdout:
                    wm_lines.append(f"Output (last 500 chars):\n{entry.stdout[-500:]}")

            section = "\n## Recent Results\n" + "\n".join(wm_lines) + "\n"
            if len(section) < remaining:
                parts.append(section)

        return "\n".join(parts)

    # -----------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize memory state."""
        return {
            "working_count": len(self.working),
            "episode_count": len(self.episodes),
            "semantic_count": len(self.semantic),
            "episodes": {k: {"approach": v.approach, "best_score": v.best_score,
                             "attempts": v.attempts} for k, v in self.episodes.items()},
            "semantic": [{"fact": s.fact, "confidence": s.confidence} for s in self.semantic],
        }

    def summary(self) -> str:
        """Brief summary."""
        return (
            f"Memory: working={len(self.working)}, "
            f"episodes={len(self.episodes)}, "
            f"semantic={len(self.semantic)}"
        )
