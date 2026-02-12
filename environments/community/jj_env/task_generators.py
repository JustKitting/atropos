"""
Task Generators — Composable Algebraic DSL
==========================================

Generates broken repo states via a composable pipeline of ScrambleOps:
  1. Start with template files as the known goal state
  2. Thread a RepoContext through a sequence of typed operations
  3. Each op emits scramble_ops (server commands) and updates abstract state
  4. Difficulty = composition depth (easy=1, medium=2-3, hard=3-5, expert=6-10)

Operations compose freely — preconditions gate what can follow what,
and cross-category tasks emerge naturally (e.g. conflict + rebase).

Critical constraint: write_file is only exposed to the model when the
pipeline contains an op with requires_write_file=True (IntroduceConflict).
"""

from __future__ import annotations

import difflib
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from random import Random
from typing import Dict, List, Optional, Tuple, Type

logger = logging.getLogger(__name__)

from .code_templates import (
    TEMPLATES,
    FunctionInfo,
    ReturnInfo,
    find_config_values,
    find_functions,
    find_return_statements,
)
from .file_sources import FileSource, SyntheticFileSource


# ═══════════════════════════════════════════════════════════════════════
# TaskSpec — Final output (extended with new fields)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class TaskSpec:
    """Specification for a single training task."""

    task_id: str
    category: str  # kept for backward compat; primary category or comma-joined tags
    difficulty: str
    goal_description: str
    scramble_ops: List[Dict]
    goal_file_contents: Dict[str, str]
    goal_log_pattern: Optional[str] = None
    expected_min_commands: int = 1
    original_file_count: int = 1
    # New fields for composable DSL
    allow_write_file: bool = False
    max_write_lines: Optional[int] = None
    tags: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# CommitInfo — Abstract commit description for RepoContext
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class CommitInfo:
    """Tracks a commit in the abstract repo state."""

    message: str
    files_modified: List[str] = field(default_factory=list)
    bookmark: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════
# RepoContext — State threaded through composition
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class RepoContext:
    """
    Tracks evolving abstract state during pipeline construction.
    Each ScrambleOp mutates this in-place.
    """

    files: Dict[str, str]  # path → current content
    commits: List[CommitInfo] = field(default_factory=list)
    bookmarks: Dict[str, int] = field(default_factory=dict)  # name → commit index
    scramble_ops: List[Dict] = field(default_factory=list)
    goal_file_contents: Dict[str, str] = field(default_factory=dict)
    goal_fragments: List[str] = field(default_factory=list)
    goal_log_pattern: Optional[str] = None
    expected_min_commands: int = 0
    _bookmark_counter: int = 0
    _conflict_line_budget: int = 0  # total conflict marker lines for write budget

    @classmethod
    def from_files(cls, base_files: Dict[str, str]) -> "RepoContext":
        """Create a fresh context from template files."""
        ctx = cls(
            files=dict(base_files),
            goal_file_contents=dict(base_files),
        )
        return ctx

    def emit_write(self, path: str, content: str) -> None:
        """Append a write_file scramble op."""
        self.scramble_ops.append({"type": "write_file", "path": path, "content": content})

    def emit_jj(self, *args: str) -> None:
        """Append a jj_command scramble op."""
        self.scramble_ops.append({"type": "jj_command", "args": list(args)})

    def emit_delete(self, path: str) -> None:
        """Append a delete_file scramble op and remove from abstract state."""
        self.scramble_ops.append({"type": "delete_file", "path": path})
        # Keep abstract state consistent — deleted files must not be targeted.
        self.files.pop(path, None)

    def add_goal(self, fragment: str) -> None:
        """Append a goal description fragment."""
        self.goal_fragments.append(fragment)

    def fresh_bookmark(self, prefix: str = "branch") -> str:
        """Generate a unique bookmark name."""
        self._bookmark_counter += 1
        return f"{prefix}-{self._bookmark_counter}"

    def add_commit(self, message: str, files: Optional[List[str]] = None) -> int:
        """Record a commit in abstract state. Returns commit index."""
        idx = len(self.commits)
        self.commits.append(CommitInfo(message=message, files_modified=files or []))
        return idx

    def validate(self) -> Tuple[List[str], List[str]]:
        """
        Check state invariants.
        Returns (errors, warnings):
          - errors: hard violations that mean the op produced invalid state
          - warnings: soft issues (pipeline may still be valid but future ops limited)

        Hard errors:
        - Goal expects content for a file that was deleted from abstract state
        Soft warnings:
        - All files are sentinel-marked (limits what subsequent ops can target)
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Hard: goal references a deleted file
        for path in self.goal_file_contents:
            if path not in self.files:
                if self.goal_file_contents[path]:
                    errors.append(
                        f"Goal expects content for '{path}' but it's absent from abstract state"
                    )

        # Soft: all files are sentinels (limits subsequent ops, but current op may be fine)
        real_files = [
            p for p, c in self.files.items()
            if not c.startswith("<<CONFLICTED:") and "accidentally overwritten" not in c
        ]
        if self.files and not real_files:
            warnings.append("All files are sentinel-marked; no real content remains")

        return errors, warnings

    def to_task_spec(
        self,
        seed: int,
        difficulty: str,
        allow_write_file: bool,
        tags: List[str],
    ) -> TaskSpec:
        """Finalize into a TaskSpec."""
        goal_desc = " ".join(self.goal_fragments)
        tag_str = ",".join(tags) if tags else "mixed"

        max_write_lines = self._conflict_line_budget if self._conflict_line_budget > 0 else None

        return TaskSpec(
            task_id=f"{tag_str}_{difficulty}_{seed}",
            category=tag_str,
            difficulty=difficulty,
            goal_description=goal_desc,
            scramble_ops=self.scramble_ops,
            goal_file_contents=self.goal_file_contents,
            goal_log_pattern=self.goal_log_pattern,
            expected_min_commands=max(self.expected_min_commands, 1),
            original_file_count=len(self.goal_file_contents),
            allow_write_file=allow_write_file,
            max_write_lines=max_write_lines,
            tags=tags,
        )


# ═══════════════════════════════════════════════════════════════════════
# TextEditStrategy — Parameterized file mutations
# ═══════════════════════════════════════════════════════════════════════


class TextEditStrategy(ABC):
    """Parameterized way to produce two different edits of the same region."""

    @abstractmethod
    def apply(self, content: str, rng: Random) -> Optional[Tuple[str, str]]:
        """
        Return (variant_a, variant_b) — two different edits of content.
        Returns None if no valid edit target found.
        """

    @abstractmethod
    def describe(self, variant: str) -> str:
        """Human-readable description of what a variant does."""


class ModifyReturnValue(TextEditStrategy):
    """Change what a function returns."""

    def apply(self, content: str, rng: Random) -> Optional[Tuple[str, str]]:
        returns = find_return_statements(content)
        # Filter to simple returns (not raise, not multi-line)
        simple = [r for r in returns if "raise" not in r.line_text and r.function_name != "<module>"]
        if not simple:
            return None

        target = rng.choice(simple)
        lines = content.split("\n")
        orig_line = lines[target.line_number]
        indent = orig_line[: len(orig_line) - len(orig_line.lstrip())]

        # Extract the return expression
        expr = target.line_text[len("return "):]

        variant_a_line = f"{indent}return round({expr}, 3)"
        variant_b_line = f"{indent}return abs({expr})"

        lines_a = list(lines)
        lines_a[target.line_number] = variant_a_line
        lines_b = list(lines)
        lines_b[target.line_number] = variant_b_line

        return ("\n".join(lines_a), "\n".join(lines_b))

    def describe(self, variant: str) -> str:
        if variant == "a":
            return "rounds the return value to 3 decimal places"
        return "takes the absolute value of the return"


class AddValidation(TextEditStrategy):
    """Add input validation at the start of a function."""

    def apply(self, content: str, rng: Random) -> Optional[Tuple[str, str]]:
        funcs = find_functions(content)
        # Filter to functions with at least a few body lines
        eligible = [f for f in funcs if f.end_line - f.start_line >= 2]
        if not eligible:
            return None

        target = rng.choice(eligible)
        lines = content.split("\n")
        body_line = lines[target.body_start] if target.body_start < len(lines) else ""
        indent = body_line[: len(body_line) - len(body_line.lstrip())]
        if not indent:
            indent = "    "

        # Skip if there's already a docstring right after def
        insert_at = target.body_start
        if insert_at < len(lines) and lines[insert_at].strip().startswith('"""'):
            # Skip past the docstring
            if lines[insert_at].strip().endswith('"""') and lines[insert_at].strip() != '"""':
                insert_at += 1
            else:
                for k in range(insert_at + 1, len(lines)):
                    if '"""' in lines[k]:
                        insert_at = k + 1
                        break

        validation_a = f"{indent}if not isinstance(a, (int, float)):\n{indent}    raise TypeError(\"Expected numeric input\")"
        validation_b = f"{indent}if a is None or b is None:\n{indent}    raise ValueError(\"Arguments must not be None\")"

        lines_a = list(lines)
        lines_a.insert(insert_at, validation_a)
        lines_b = list(lines)
        lines_b.insert(insert_at, validation_b)

        return ("\n".join(lines_a), "\n".join(lines_b))

    def describe(self, variant: str) -> str:
        if variant == "a":
            return "adds type checking validation"
        return "adds None-checking validation"


class RenameSymbol(TextEditStrategy):
    """Rename a function."""

    def apply(self, content: str, rng: Random) -> Optional[Tuple[str, str]]:
        funcs = find_functions(content)
        if not funcs:
            return None

        target = rng.choice(funcs)
        old_name = target.name
        new_name = f"compute_{old_name}"

        # Variant A: rename the function definition and all call sites
        # Use a sentinel to avoid double-replacing inside new_name
        sentinel = "\x00RENAMED\x00"
        variant_a = content.replace(f"def {old_name}(", f"def {sentinel}(")
        variant_a = variant_a.replace(f"{old_name}(", f"{new_name}(")
        variant_a = variant_a.replace(f"def {sentinel}(", f"def {new_name}(")

        # Variant B: add an alias wrapper at the end
        variant_b = content + f"\n\ndef {new_name}(*args, **kwargs):\n    return {old_name}(*args, **kwargs)\n"

        return (variant_a, variant_b)

    def describe(self, variant: str) -> str:
        if variant == "a":
            return "renames the function"
        return "adds a wrapper function with new name"


class ChangeConfig(TextEditStrategy):
    """Modify a config value."""

    def apply(self, content: str, rng: Random) -> Optional[Tuple[str, str]]:
        entries = find_config_values(content)
        # Filter to numeric or simple string values
        eligible = [e for e in entries if e.value.replace(".", "").isdigit()]
        if not eligible:
            return None

        target = rng.choice(eligible)
        lines = content.split("\n")
        orig_line = lines[target.line_number]
        key_part = orig_line[: orig_line.index(":") + 1]

        try:
            num_val = int(target.value)
        except ValueError:
            num_val = int(float(target.value))

        val_a = num_val * 2
        val_b = max(num_val // 2, 1)

        lines_a = list(lines)
        lines_a[target.line_number] = f"{key_part} {val_a}"
        lines_b = list(lines)
        lines_b[target.line_number] = f"{key_part} {val_b}"

        return ("\n".join(lines_a), "\n".join(lines_b))

    def describe(self, variant: str) -> str:
        if variant == "a":
            return "doubles the config value"
        return "halves the config value"


class InsertFunction(TextEditStrategy):
    """Add a new helper function at the end of the file."""

    HELPERS = [
        (
            '\ndef clamp(value, lo, hi):\n    """Clamp value between lo and hi."""\n    return max(lo, min(hi, value))\n',
            '\ndef clamp(value, lo, hi):\n    """Clamp value between lo and hi."""\n    if value < lo:\n        return lo\n    if value > hi:\n        return hi\n    return value\n',
        ),
        (
            '\ndef safe_divide(a, b, default=0):\n    """Divide a by b, returning default on zero division."""\n    return a / b if b != 0 else default\n',
            '\ndef safe_divide(a, b, default=0):\n    """Divide a by b, returning default on zero division."""\n    try:\n        return a / b\n    except ZeroDivisionError:\n        return default\n',
        ),
        (
            '\ndef flatten(lst):\n    """Flatten a list of lists."""\n    return [item for sub in lst for item in sub]\n',
            '\ndef flatten(lst):\n    """Flatten a list of lists."""\n    result = []\n    for sub in lst:\n        result.extend(sub)\n    return result\n',
        ),
    ]

    def apply(self, content: str, rng: Random) -> Optional[Tuple[str, str]]:
        helper_a, helper_b = rng.choice(self.HELPERS)
        return (content + helper_a, content + helper_b)

    def describe(self, variant: str) -> str:
        if variant == "a":
            return "adds helper (comprehension style)"
        return "adds helper (explicit loop style)"


# All available strategies
ALL_STRATEGIES: List[TextEditStrategy] = [
    ModifyReturnValue(),
    AddValidation(),
    RenameSymbol(),
    ChangeConfig(),
    InsertFunction(),
]


# ═══════════════════════════════════════════════════════════════════════
# ScrambleOp — The algebraic operations
# ═══════════════════════════════════════════════════════════════════════


class ScrambleOp(ABC):
    """
    A typed transformation: RepoContext → RepoContext (with side effects).
    Each op declares whether it requires write_file for the model.
    """

    requires_write_file: bool = False

    @abstractmethod
    def precondition(self, ctx: RepoContext) -> bool:
        """Can this op be applied to the current context?"""

    @abstractmethod
    def apply(self, ctx: RepoContext, rng: Random) -> None:
        """Mutate ctx: emit scramble ops, update goals, update abstract state."""

    @abstractmethod
    def expected_commands(self) -> int:
        """Minimum jj commands model needs to undo this."""

    @abstractmethod
    def tag(self) -> str:
        """Category tag for this operation."""


# ── IntroduceConflict ────────────────────────────────────────────────


class IntroduceConflict(ScrambleOp):
    """
    Branch A edits a region, branch B edits same region, merge → conflict.
    Model must write resolved file + squash. Only op that enables write_file.
    """

    requires_write_file = True

    def precondition(self, ctx: RepoContext) -> bool:
        # Need at least one Python file with ≥10 lines that isn't already conflicted
        for path, content in ctx.files.items():
            if (
                path.endswith(".py")
                and not content.startswith("<<CONFLICTED:")
                and len(content.split("\n")) >= 10
            ):
                return True
        return False

    def apply(self, ctx: RepoContext, rng: Random) -> None:
        # Pick a Python file with enough content that isn't already conflicted
        eligible = [
            (p, c)
            for p, c in ctx.files.items()
            if p.endswith(".py")
            and not c.startswith("<<CONFLICTED:")
            and len(c.split("\n")) >= 10
        ]
        path, base_content = rng.choice(eligible)

        # Pick a text edit strategy
        strategy = None
        for s in rng.sample(ALL_STRATEGIES, len(ALL_STRATEGIES)):
            result = s.apply(base_content, rng)
            if result is not None:
                strategy = s
                variant_a, variant_b = result
                break

        if strategy is None:
            # Fallback: simple line modification
            lines = base_content.split("\n")
            mid = len(lines) // 2
            orig = lines[mid]
            lines_a = list(lines)
            lines_a[mid] = orig + "  # variant A"
            lines_b = list(lines)
            lines_b[mid] = orig + "  # variant B"
            variant_a = "\n".join(lines_a)
            variant_b = "\n".join(lines_b)
            strategy_desc_a = "adds comment marker A"
            strategy_desc_b = "adds comment marker B"
        else:
            strategy_desc_a = strategy.describe("a")
            strategy_desc_b = strategy.describe("b")

        # Goal: variant A is the correct resolution
        goal_content = variant_a

        # Compute conflict line budget: count differing lines between variant_a and variant_b
        diff_lines = _count_differing_lines(variant_a, variant_b)
        # Conflict markers add ~7 lines per region; resolved file = variant_a lines
        # Budget = differing lines + markers overhead
        ctx._conflict_line_budget += diff_lines + 10

        # Emit scramble ops to create the conflict
        bookmark_a = ctx.fresh_bookmark("conflict-a")
        bookmark_b = ctx.fresh_bookmark("conflict-b")

        # Write base content
        ctx.emit_write(path, base_content)
        ctx.emit_jj("describe", "-m", f"Base version of {path}")
        base_idx = ctx.add_commit(f"Base version of {path}", [path])

        # Branch A
        ctx.emit_jj("new", "@")
        ctx.emit_write(path, variant_a)
        ctx.emit_jj("describe", "-m", f"Change: {strategy_desc_a}")
        ctx.emit_jj("bookmark", "create", bookmark_a)
        ctx.add_commit(f"Change: {strategy_desc_a}", [path])

        # Branch B (from base)
        ctx.emit_jj("new", "@-")
        ctx.emit_write(path, variant_b)
        ctx.emit_jj("describe", "-m", f"Change: {strategy_desc_b}")
        ctx.add_commit(f"Change: {strategy_desc_b}", [path])

        # Merge → conflict
        ctx.emit_jj("new", bookmark_a, "@")
        ctx.emit_jj("describe", "-m", "Merge branches")
        ctx.add_commit("Merge branches", [path])

        # Update context state
        ctx.files[path] = f"<<CONFLICTED:{path}>>"  # mark as conflicted
        ctx.goal_file_contents[path] = goal_content

        ctx.add_goal(
            f"Resolve the merge conflict in {path}. "
            f"Keep the version that {strategy_desc_a}, "
            f"not the one that {strategy_desc_b}."
        )
        ctx.expected_min_commands += 2  # write_file + squash

    def expected_commands(self) -> int:
        return 2

    def tag(self) -> str:
        return "conflict"


# ── MisplaceCommit ───────────────────────────────────────────────────


class MisplaceCommit(ScrambleOp):
    """
    Put a commit on wrong parent. Model must jj rebase.
    """

    def precondition(self, ctx: RepoContext) -> bool:
        return len(ctx.commits) >= 2

    def apply(self, ctx: RepoContext, rng: Random) -> None:
        # Pick a file and create a feature commit on the wrong parent
        eligible = [(p, c) for p, c in ctx.files.items() if p.endswith(".py")]
        if not eligible:
            eligible = list(ctx.files.items())
        path, content = rng.choice(eligible)

        # Add a feature to the file
        if path.endswith(".py"):
            feature_name = rng.choice(["sqrt", "cube", "negate", "double"])
            return_expr = {"sqrt": "x ** 0.5", "cube": "x ** 3", "negate": "-x", "double": "2 * x"}[feature_name]
            extra = '\n\ndef {}(x):\n    """Compute {}."""\n    return {}\n'.format(feature_name, feature_name, return_expr)
            new_content = content + extra
        else:
            new_content = content + "\n# feature addition\n"

        # Create a "correct base" commit first
        feat_label = feature_name if path.endswith(".py") else "misc"
        readme_content = f"# Project\n\nFeature: {feat_label}\n"
        ctx.emit_write("README.md", readme_content)
        ctx.emit_jj("describe", "-m", "Add README")
        correct_parent_idx = ctx.add_commit("Add README", ["README.md"])

        # Feature on wrong parent (root, not README)
        ctx.emit_jj("new", "@-")  # go back to grandparent
        ctx.emit_write(path, new_content)
        feature_msg = f"Add {feature_name}" if path.endswith(".py") else "Add feature"
        ctx.emit_jj("describe", "-m", feature_msg)
        ctx.add_commit(feature_msg, [path])

        # Update state
        ctx.files[path] = new_content
        ctx.files["README.md"] = readme_content
        ctx.goal_file_contents[path] = new_content
        ctx.goal_file_contents["README.md"] = readme_content
        ctx.goal_log_pattern = r"README.*" + feature_msg.replace(" ", ".")

        ctx.add_goal(
            f"The '{feature_msg}' commit is on the wrong parent. "
            f"Rebase it on top of the 'Add README' commit so history is linear."
        )
        ctx.expected_min_commands += 1

    def expected_commands(self) -> int:
        return 1

    def tag(self) -> str:
        return "rebase"


# ── FragmentHistory ──────────────────────────────────────────────────


class FragmentHistory(ScrambleOp):
    """
    Split a logical change into N wip commits with bad messages.
    Model must squash and describe.
    """

    def precondition(self, ctx: RepoContext) -> bool:
        return len(ctx.files) >= 1

    def apply(self, ctx: RepoContext, rng: Random) -> None:
        path = rng.choice(list(ctx.files.keys()))
        content = ctx.files[path]

        n_fragments = rng.randint(2, 4)
        wip_messages = ["wip", "wip2", "tmp", "fixup", "more stuff", "cleanup"]
        func_name = None  # set below for .py files

        if path.endswith(".py"):
            # Add a function incrementally across commits
            func_name = rng.choice(["absolute", "sign", "factorial", "identity"])
            return_exprs = {
                "absolute": "abs(x)",
                "sign": "(x > 0) - (x < 0)",
                "factorial": "1 if x <= 1 else x * {}(x - 1)".format(func_name),
                "identity": "x",
            }
            return_expr = return_exprs[func_name]

            extras = [
                "\n\ndef {}(x):\n    pass\n".format(func_name),
                '\n\ndef {}(x):\n    """Compute {}."""\n    pass\n'.format(func_name, func_name),
                '\n\ndef {}(x):\n    """Compute {}."""\n    return {}\n'.format(func_name, func_name, return_expr),
            ]
            final_extra = '\n\ndef {}(x):\n    """Compute {}."""\n    return {}\n'.format(func_name, func_name, return_expr)
            final_content = content + final_extra
        else:
            final_content = content + "\n# completed\n"

        # First fragment
        ctx.emit_write(path, content)
        msg0 = rng.choice(wip_messages)
        ctx.emit_jj("describe", "-m", msg0)
        ctx.add_commit(msg0, [path])

        # Subsequent fragments
        for i in range(1, min(n_fragments, len(extras) if path.endswith(".py") else n_fragments)):
            ctx.emit_jj("new", "@")
            if path.endswith(".py") and i - 1 < len(extras):
                ctx.emit_write(path, content + extras[min(i - 1, len(extras) - 1)])
            msg = rng.choice(wip_messages)
            ctx.emit_jj("describe", "-m", msg)
            ctx.add_commit(msg, [path])

        # Final fragment with complete content
        ctx.emit_jj("new", "@")
        ctx.emit_write(path, final_content)
        msg_final = rng.choice(wip_messages)
        ctx.emit_jj("describe", "-m", msg_final)
        ctx.add_commit(msg_final, [path])

        ctx.files[path] = final_content
        ctx.goal_file_contents[path] = final_content

        target_msg = f"Add {func_name} function" if func_name else f"Update {path}"
        pattern_alt = func_name if func_name else path.replace(".", "[.]")
        ctx.goal_log_pattern = f"(?i)({pattern_alt}|{path.replace('.', '[.]')})"

        ctx.add_goal(
            f"There are {n_fragments + 1} WIP commits with unhelpful messages. "
            f"Squash them into a single commit with a descriptive message like '{target_msg}'."
        )
        ctx.expected_min_commands += n_fragments  # squash × (N-1) + describe

    def expected_commands(self) -> int:
        return 3

    def tag(self) -> str:
        return "squash"


# ── AbandonCommit ────────────────────────────────────────────────────


class AbandonCommit(ScrambleOp):
    """
    Abandon a commit. Model must jj undo.
    """

    def precondition(self, ctx: RepoContext) -> bool:
        # Need at least one file with real content
        return any(
            not c.startswith("<<CONFLICTED:") and "accidentally overwritten" not in c
            for c in ctx.files.values()
        )

    def apply(self, ctx: RepoContext, rng: Random) -> None:
        # Pick a file with real content
        eligible = [
            p for p, c in ctx.files.items()
            if not c.startswith("<<CONFLICTED:") and "accidentally overwritten" not in c
        ]
        path = rng.choice(eligible)
        content = ctx.files[path]

        ctx.emit_write(path, content)
        commit_msg = f"Add {path}"
        ctx.emit_jj("describe", "-m", commit_msg)
        ctx.add_commit(commit_msg, [path])

        # Create empty working copy
        ctx.emit_jj("new", "@")
        ctx.emit_jj("describe", "-m", "Working copy")
        ctx.add_commit("Working copy", [])

        # Abandon the good commit
        ctx.emit_jj("abandon", "@-")

        ctx.goal_file_contents[path] = content

        ctx.add_goal(
            f"The '{commit_msg}' commit was accidentally abandoned. "
            "Use 'jj undo' to reverse the abandon operation and restore it."
        )
        ctx.expected_min_commands += 1

    def expected_commands(self) -> int:
        return 1

    def tag(self) -> str:
        return "restore"


# ── OverwriteFile ────────────────────────────────────────────────────


class OverwriteFile(ScrambleOp):
    """
    Replace a file with garbage. Model must jj restore.
    """

    _BAD_CONTENT = "# This file was accidentally overwritten\npass\n"

    def precondition(self, ctx: RepoContext) -> bool:
        # Need at least one file with real content (not already garbage/conflicted)
        return any(
            not c.startswith("<<CONFLICTED:") and c != self._BAD_CONTENT
            for c in ctx.files.values()
        )

    def apply(self, ctx: RepoContext, rng: Random) -> None:
        # Pick a file that still has real content
        eligible = [
            p for p, c in ctx.files.items()
            if not c.startswith("<<CONFLICTED:") and c != self._BAD_CONTENT
        ]
        path = rng.choice(eligible)
        good_content = ctx.files[path]

        bad_content = self._BAD_CONTENT

        # Write good version first
        ctx.emit_write(path, good_content)
        ctx.emit_jj("describe", "-m", f"Good version of {path}")
        ctx.add_commit(f"Good version of {path}", [path])

        # Overwrite with garbage
        ctx.emit_jj("new", "@")
        ctx.emit_write(path, bad_content)
        ctx.emit_jj("describe", "-m", f"Accidentally overwrote {path}")
        ctx.add_commit(f"Accidentally overwrote {path}", [path])

        # Don't mutate ctx.files — keep good content so subsequent ops see real data.
        # The scramble ops on the server will create the junk state; the goal stays good.
        ctx.goal_file_contents[path] = good_content

        ctx.add_goal(
            f"{path} was accidentally overwritten with junk content. "
            f"Use 'jj restore' to restore it from the parent commit."
        )
        ctx.expected_min_commands += 1

    def expected_commands(self) -> int:
        return 1

    def tag(self) -> str:
        return "restore"


# ── CorruptMessages ──────────────────────────────────────────────────


class CorruptMessages(ScrambleOp):
    """
    Replace commit descriptions with 'wip'. Model must jj describe each.
    """

    def precondition(self, ctx: RepoContext) -> bool:
        # Need at least 1 commit with a real message
        return any(
            c.message not in ("wip", "tmp", "fixup", "", "Working copy")
            for c in ctx.commits
        )

    def apply(self, ctx: RepoContext, rng: Random) -> None:
        # We can't directly address commits by index in jj,
        # so we create commits with bad messages
        path = rng.choice(list(ctx.files.keys()))
        content = ctx.files[path]

        good_messages = [
            f"Implement {path.replace('.py', '')} module",
            f"Add core functionality to {path}",
            f"Set up {path} with proper structure",
        ]
        good_msg = rng.choice(good_messages)

        ctx.emit_write(path, content)
        ctx.emit_jj("describe", "-m", "wip")
        ctx.add_commit("wip", [path])

        ctx.goal_file_contents[path] = content
        ctx.goal_log_pattern = f"(?i)({path.replace('.', '[.]')}|implement|module|function)"

        ctx.add_goal(
            f"The commit for {path} has an unhelpful 'wip' message. "
            f"Use 'jj describe' to give it a proper descriptive message like '{good_msg}'."
        )
        ctx.expected_min_commands += 1

    def expected_commands(self) -> int:
        return 1

    def tag(self) -> str:
        return "describe"


# ═══════════════════════════════════════════════════════════════════════
# Pipeline & Sampler
# ═══════════════════════════════════════════════════════════════════════

# All op types
ALL_OPS: List[Type[ScrambleOp]] = [
    IntroduceConflict,
    MisplaceCommit,
    FragmentHistory,
    AbandonCommit,
    OverwriteFile,
    CorruptMessages,
]


@dataclass
class ScramblePipeline:
    """Ordered sequence of operations applied to a RepoContext."""

    ops: List[ScrambleOp]

    @property
    def allow_write_file(self) -> bool:
        """Only allow write_file if any op requires it."""
        return any(op.requires_write_file for op in self.ops)

    @property
    def tags(self) -> List[str]:
        """Collect unique tags from all ops."""
        seen = set()
        tags = []
        for op in self.ops:
            t = op.tag()
            if t not in seen:
                seen.add(t)
                tags.append(t)
        return tags

    def execute(self, base_files: Dict[str, str], seed: int, difficulty: str) -> TaskSpec:
        """Execute pipeline against base files and produce a TaskSpec."""
        ctx = RepoContext.from_files(base_files)
        rng = Random(seed)

        for op in self.ops:
            op.apply(ctx, Random(rng.randint(0, 2**31)))
            # State consistency check after each op
            errors, warnings = ctx.validate()
            op_name = type(op).__name__
            for e in errors:
                logger.error("State error after %s: %s", op_name, e)
            for w in warnings:
                logger.debug("State note after %s: %s", op_name, w)

        return ctx.to_task_spec(
            seed=seed,
            difficulty=difficulty,
            allow_write_file=self.allow_write_file,
            tags=self.tags,
        )


class PipelineSampler:
    """Samples random composable pipelines at target difficulty."""

    def __init__(
        self,
        ops: Optional[List[Type[ScrambleOp]]] = None,
        file_source: Optional[FileSource] = None,
    ):
        self.all_ops = ops or ALL_OPS
        self.file_source = file_source or SyntheticFileSource()

    def sample(
        self,
        seed: int,
        difficulty: str = "depth_1",
        category_hints: Optional[List[str]] = None,
        depth: Optional[int] = None,
    ) -> Tuple[ScramblePipeline, Dict[str, str]]:
        """
        Sample a pipeline and base files.
        Returns (pipeline, base_files).

        If depth is provided directly, use it. Otherwise fall back to
        legacy difficulty string mapping.
        category_hints biases op selection but doesn't filter exclusively.
        """
        rng = Random(seed)

        if depth is None:
            depth_map = {"easy": 1, "medium": rng.randint(2, 3), "hard": rng.randint(3, 5), "expert": rng.randint(6, 10)}
            depth = depth_map.get(difficulty, 1)

        # Sample base files from the file source — more files for complex scenarios
        n_files = 3 if depth >= 6 else (2 if (depth >= 2 and rng.random() < 0.5) else 1)
        base_files = self.file_source.sample(seed, n_files)

        # Build pipeline greedily
        ctx = RepoContext.from_files(base_files)
        ops: List[ScrambleOp] = []

        # Map category hints to preferred op types
        hint_to_ops: Dict[str, List[Type[ScrambleOp]]] = {
            "conflict_resolution": [IntroduceConflict],
            "conflict": [IntroduceConflict],
            "squash": [FragmentHistory],
            "rebase": [MisplaceCommit],
            "restore": [AbandonCommit, OverwriteFile],
            "describe": [CorruptMessages],
        }

        for i in range(depth):
            # Determine eligible ops
            eligible = [op_cls for op_cls in self.all_ops if op_cls().precondition(ctx)]
            if not eligible:
                break

            # Bias toward category hints for the first op
            if i == 0 and category_hints:
                preferred = []
                for hint in category_hints:
                    preferred.extend(hint_to_ops.get(hint, []))
                preferred_eligible = [op for op in eligible if op in preferred]
                if preferred_eligible:
                    eligible = preferred_eligible

            chosen_cls = rng.choice(eligible)
            op = chosen_cls()
            ops.append(op)

            # Simulate apply to update ctx for next op's precondition
            op.apply(ctx, Random(rng.randint(0, 2**31)))

            # Validate state after each op during sampling
            errors, warnings = ctx.validate()
            if errors:
                op_name = type(op).__name__
                for e in errors:
                    logger.warning("Sampler error after %s: %s", op_name, e)
                # Hard error: drop the offending op and stop
                ops.pop()
                break
            # Soft warnings don't stop the pipeline — they just limit what comes next
            # (preconditions on subsequent ops will naturally gate)

        return ScramblePipeline(ops), base_files


# ═══════════════════════════════════════════════════════════════════════
# TaskGenerator — Same public API, delegates to PipelineSampler
# ═══════════════════════════════════════════════════════════════════════


class TaskGenerator:
    """Main task generator that dispatches to composable pipeline sampler."""

    def __init__(
        self,
        categories: Optional[List[str]] = None,
        difficulties: Optional[List[str]] = None,
        file_source: Optional[FileSource] = None,
        # Depth-based curriculum params
        min_depth: int = 1,
        max_depth: int = 2,
    ):
        self.categories = categories or ["conflict_resolution", "squash", "rebase", "restore"]
        self.difficulties = difficulties or ["easy"]
        self.sampler = PipelineSampler(file_source=file_source)
        self._counter = 0
        self.min_depth = min_depth
        self.max_depth = max_depth

    def generate_task(
        self,
        seed: Optional[int] = None,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        depth: Optional[int] = None,
    ) -> TaskSpec:
        """
        Generate a task spec.

        If depth is provided, uses it directly (new curriculum system).
        Otherwise falls back to legacy difficulty string.
        """
        rng = random.Random(seed if seed is not None else self._counter)
        self._counter += 1

        # New depth-based system: sample a depth from the current range
        if depth is None:
            depth = rng.randint(self.min_depth, self.max_depth)

        # Difficulty label for logging/tracking
        if difficulty is None:
            difficulty = f"depth_{depth}"

        # Category becomes a hint for the sampler
        hints = None
        if category is not None:
            hints = [category]
        else:
            hints = [rng.choice(self.categories)]

        task_seed = seed if seed is not None else rng.randint(0, 2**31)
        pipeline, base_files = self.sampler.sample(
            seed=task_seed, difficulty=difficulty, category_hints=hints,
            depth=depth,
        )
        return pipeline.execute(base_files, task_seed, difficulty)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _count_differing_lines(a: str, b: str) -> int:
    """Count how many lines differ between two strings."""
    lines_a = a.split("\n")
    lines_b = b.split("\n")
    matcher = difflib.SequenceMatcher(None, lines_a, lines_b)
    changed = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            changed += max(i2 - i1, j2 - j1)
    return changed


def count_lines_changed(before: str, after: str) -> int:
    """
    Count lines that changed between before and after content.
    Used by jj_env.py to track write_file line budget.
    """
    return _count_differing_lines(before, after)
