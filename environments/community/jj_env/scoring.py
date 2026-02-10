"""
Scoring — Verification and Scoring Logic
=========================================

Compares final repo state against expected goal state.
Score breakdown:
  - Primary (0.0-0.7): file content match via difflib.SequenceMatcher
  - Bonus (0.0-0.1): no remaining conflicts
  - Bonus (0.0-0.1): commit graph matches expected pattern
  - Bonus (0.0-0.1): efficiency (fewer commands = higher)
  - Penalty: degenerate detection (empty repo, all deleted) → -0.5
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from .jj_executor import RepoState

logger = logging.getLogger(__name__)


@dataclass
class ScoreBreakdown:
    """Detailed score breakdown for debugging."""

    file_match_score: float = 0.0
    conflict_bonus: float = 0.0
    graph_bonus: float = 0.0
    efficiency_bonus: float = 0.0
    degenerate_penalty: float = 0.0
    write_budget_penalty: float = 0.0
    total: float = 0.0
    details: str = ""


class JJScorer:
    """Score a repository's final state against an expected goal."""

    def score(
        self,
        final_state: RepoState,
        goal_file_contents: Dict[str, str],
        goal_log_pattern: Optional[str],
        commands_used: List[str],
        expected_min_commands: int,
        original_file_count: int,
        total_lines_changed: int = 0,
        max_write_lines: Optional[int] = None,
        allow_write_file: bool = True,
        write_file_attempted: bool = False,
    ) -> ScoreBreakdown:
        breakdown = ScoreBreakdown()

        # Check for degenerate solutions
        degenerate = self._check_degenerate(
            final_state, goal_file_contents, commands_used, original_file_count
        )
        if degenerate:
            breakdown.degenerate_penalty = -0.5
            breakdown.total = -0.5
            breakdown.details = f"Degenerate: {degenerate}"
            return breakdown

        # Primary: file content match (0.0 - 0.7)
        breakdown.file_match_score = self._score_file_contents(
            final_state.file_contents, goal_file_contents
        )

        # Bonus: no conflicts (0.0 - 0.1)
        breakdown.conflict_bonus = 0.1 if not final_state.has_conflicts else 0.0

        # Bonus: commit graph pattern (0.0 - 0.1)
        if goal_log_pattern:
            breakdown.graph_bonus = self._score_graph_pattern(
                final_state.log_output, goal_log_pattern
            )
        else:
            # No graph requirement — give full bonus if files match well
            breakdown.graph_bonus = 0.1 if breakdown.file_match_score > 0.5 else 0.0

        # Bonus: efficiency (0.0 - 0.1)
        breakdown.efficiency_bonus = self._score_efficiency(
            len(commands_used), expected_min_commands
        )

        # Write budget enforcement
        breakdown.write_budget_penalty = self._score_write_budget(
            total_lines_changed=total_lines_changed,
            max_write_lines=max_write_lines,
            allow_write_file=allow_write_file,
            write_file_attempted=write_file_attempted,
        )

        breakdown.total = (
            breakdown.file_match_score
            + breakdown.conflict_bonus
            + breakdown.graph_bonus
            + breakdown.efficiency_bonus
            + breakdown.degenerate_penalty
            + breakdown.write_budget_penalty
        )
        breakdown.total = max(-1.0, min(1.0, breakdown.total))

        breakdown.details = (
            f"files={breakdown.file_match_score:.2f} "
            f"conflicts={breakdown.conflict_bonus:.2f} "
            f"graph={breakdown.graph_bonus:.2f} "
            f"efficiency={breakdown.efficiency_bonus:.2f}"
            + (f" write_penalty={breakdown.write_budget_penalty:.2f}" if breakdown.write_budget_penalty != 0 else "")
        )

        return breakdown

    @staticmethod
    def _filter_git_internals(files: Dict[str, str]) -> Dict[str, str]:
        """Filter out .git/ internal files from file contents dict."""
        return {k: v for k, v in files.items() if not k.startswith(".git/")}

    def _score_file_contents(
        self,
        actual: Dict[str, str],
        expected: Dict[str, str],
    ) -> float:
        """
        Score file content match using SequenceMatcher.
        Returns 0.0 - 0.7.
        """
        actual = self._filter_git_internals(actual)
        if not expected:
            return 0.7 if not actual else 0.0

        total_ratio = 0.0
        matched_files = 0

        for filename, expected_content in expected.items():
            actual_content = actual.get(filename, "")
            ratio = difflib.SequenceMatcher(
                None, actual_content, expected_content
            ).ratio()
            total_ratio += ratio
            matched_files += 1

        # Penalize extra unexpected files (mildly)
        extra_files = set(actual.keys()) - set(expected.keys())
        extra_penalty = min(len(extra_files) * 0.05, 0.1)

        # Penalize missing expected files
        missing_files = set(expected.keys()) - set(actual.keys())
        missing_penalty = len(missing_files) * 0.15

        avg_ratio = total_ratio / max(matched_files, 1)
        score = avg_ratio * 0.7 - extra_penalty - missing_penalty

        return max(0.0, min(0.7, score))

    def _score_graph_pattern(
        self,
        log_output: str,
        pattern: str,
    ) -> float:
        """
        Score commit graph against expected regex pattern.
        Returns 0.0 or 0.1.
        """
        try:
            if re.search(pattern, log_output, re.MULTILINE | re.DOTALL):
                return 0.1
        except re.error:
            logger.warning(f"Invalid graph pattern regex: {pattern}")
        return 0.0

    def _score_efficiency(
        self,
        commands_used: int,
        expected_min: int,
    ) -> float:
        """
        Score command efficiency. Full bonus if at or near minimum.
        Returns 0.0 - 0.1.
        """
        if commands_used == 0:
            return 0.0
        if expected_min <= 0:
            return 0.05

        ratio = expected_min / max(commands_used, 1)
        # Clamp: 1.0 means perfectly efficient, >1.0 means faster than expected
        ratio = min(ratio, 1.5)
        return round(ratio * 0.1, 3)

    def _score_write_budget(
        self,
        total_lines_changed: int,
        max_write_lines: Optional[int],
        allow_write_file: bool,
        write_file_attempted: bool,
    ) -> float:
        """
        Penalize write_file misuse.
        - Flat -0.3 if write_file used when not allowed
        - Proportional penalty if lines changed exceed budget (50% tolerance)
        """
        # Unauthorized write_file usage
        if not allow_write_file and write_file_attempted:
            return -0.3

        # Write budget overshoot
        if max_write_lines is not None and max_write_lines > 0 and total_lines_changed > 0:
            tolerance = max_write_lines * 1.5
            if total_lines_changed > tolerance:
                overshoot = total_lines_changed / max_write_lines
                return -0.1 * min(overshoot - 1.0, 2.0)

        return 0.0

    def _check_degenerate(
        self,
        final_state: RepoState,
        goal_file_contents: Dict[str, str],
        commands_used: List[str],
        original_file_count: int,
    ) -> Optional[str]:
        """
        Detect degenerate solutions. Returns description if degenerate, None if ok.
        """
        # No commands issued
        if len(commands_used) == 0:
            return "No commands issued"

        # Empty repo
        if final_state.is_empty and goal_file_contents:
            return "Repo is empty but goal expects files"

        # File count dropped below 50% of original
        if original_file_count > 0:
            current_count = len(self._filter_git_internals(final_state.file_contents))
            if current_count < original_file_count * 0.5 and len(goal_file_contents) >= original_file_count * 0.5:
                return f"File count dropped from {original_file_count} to {current_count}"

        return None
