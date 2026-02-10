"""
Jujutsu (jj) VCS Operation RL Environment
==========================================

Trains LLMs to perform version control operations using Jujutsu (jj)
via a "reverse scramble" approach: start with a known goal state,
programmatically scramble it, and the model must use jj commands to fix it.

Uses interleaved tool execution: the model generates inside a <think> block,
emits <tool_call> tags to run jj commands, observes results, and continues
until it closes </think> and writes <done/>.

Requires the jj execution server (Docker container):
    docker build -t jj-exec-server environments/community/jj_env/
    docker run -p 5003:5003 jj-exec-server
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from typing import Dict, List, Optional, Tuple

try:
    import wandb
except ImportError:
    wandb = None

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    ScoredDataGroup,
)
from atroposlib.type_definitions import Item, Message
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

from .file_sources import HFCodeFileSource, SyntheticFileSource
from .jj_executor import JJExecutor
from .scoring import JJScorer
from .task_generators import TaskGenerator, TaskSpec, count_lines_changed

logger = logging.getLogger(__name__)

# Generation limits
MAX_REPLY_TOKENS = 4096
MAX_GEN_PER_TURN = 1024
MAX_ROLLOUT_TURNS = 15
MAX_COMMANDS_PER_ROLLOUT = 15

_JJ_TOOL = '{"name": "jj", "description": "Execute a jj command (without the \'jj\' prefix)", "parameters": {"command": "the jj subcommand and arguments, e.g. \'squash -m Fix merge\'"}}'
_WRITE_FILE_TOOL = '{"name": "write_file", "description": "Write content to a file in the repo (for conflict resolution only)", "parameters": {"path": "file path relative to repo root", "content": "the full file content to write"}}'


def build_system_prompt(allow_write_file: bool = True) -> str:
    """
    Build the system prompt with tool availability gated by task requirements.
    When allow_write_file=False, the write_file tool is omitted entirely.
    """
    tools = f"[{_JJ_TOOL}]"
    if allow_write_file:
        tools += f"\n[{_WRITE_FILE_TOOL}]"

    return f"""\
You are an expert at Jujutsu (jj), a modern version control system. You will be shown a \
repository in a broken or messy state. Your job is to use jj commands to fix it to match \
the described goal state.

Available tools:
{tools}

Key jj concepts:
- Commits are immutable and auto-rebase when parents change
- First-class conflict handling (conflict markers appear in files)
- @ = working copy commit, @- = parent of working copy
- jj new = create new commit on top of current
- jj squash = fold working copy into parent commit
- jj rebase -r <rev> -d <dest> = move a commit to a new parent
- jj rebase -s <source> -d <dest> = move a commit and its descendants
- jj restore --from <rev> <path> = restore file from another revision
- jj undo = undo the last jj operation
- jj describe -m "message" = set commit description
- jj abandon <rev> = abandon a commit
- jj log = show commit history
- jj status = show working copy status
- jj diff = show changes in working copy

Issue commands one at a time inside <tool_call> tags within your <think> block.
Observe each command's output before deciding the next step.
When you have finished fixing the repository, close </think> and write <done/>.

Example interaction:
<think>
Let me check the current state of the repo.
<tool_call>{{"name": "jj", "arguments": {{"command": "log"}}}}</tool_call>
<tool_response>{{"stdout": "@ qpvuntsm user@example.com ...", "stderr": "", "returncode": 0}}</tool_response>
I see the issue. Let me fix it.
<tool_call>{{"name": "jj", "arguments": {{"command": "squash"}}}}</tool_call>
<tool_response>{{"stdout": "Working copy now at: ...", "stderr": "", "returncode": 0}}</tool_response>
The repo is fixed now.
</think>
<done/>
"""


def _build_user_prompt(
    task_spec: TaskSpec,
    repo_info: Dict[str, str],
) -> str:
    """Build the user prompt showing the broken repo state and goal."""
    return f"""\
The repository is in a broken/messy state. Here is the current state:

## jj log
```
{repo_info['log']}
```

## jj status
```
{repo_info['status']}
```

## jj diff
```
{repo_info['diff']}
```

## Current file contents
```
{repo_info['files']}
```

## Goal
{task_spec.goal_description}

Fix the repository to match the goal. Use <tool_call> tags inside your <think> block to issue jj commands one at a time. When done, close </think> and write <done/>."""


class JJEnvConfig(BaseEnvConfig):
    """Configuration for the JJ VCS environment."""

    task_categories: List[str] = ["conflict_resolution", "squash", "rebase", "restore"]
    difficulty_levels: List[str] = ["easy"]
    jj_server_url: str = "http://localhost:5003"
    command_timeout_seconds: int = 30
    max_commands_per_rollout: int = MAX_COMMANDS_PER_ROLLOUT
    max_rollout_turns: int = MAX_ROLLOUT_TURNS
    max_gen_per_turn: int = MAX_GEN_PER_TURN
    use_curriculum: bool = True
    curriculum_threshold: float = 0.3
    # File source: "hf" for HuggingFace real code, "synthetic" for templates
    file_source: str = "hf"
    hf_dataset_name: str = "bigcode/starcoderdata"
    hf_pool_size: int = 5000


class JJEnv(BaseEnv):
    """
    RL environment for training LLMs on Jujutsu VCS operations.

    Uses interleaved tool execution: model thinks, calls jj commands,
    observes output, and continues until done.

    Requires the jj Docker server to be running (see Dockerfile).
    """

    name = "jj_vcs"
    env_config_cls = JJEnvConfig

    def __init__(
        self,
        config: JJEnvConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = True,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: JJEnvConfig = config
        self.executor = JJExecutor(
            server_url=config.jj_server_url,
            command_timeout=config.command_timeout_seconds,
        )
        self.scorer = JJScorer()
        # Task generator is created in setup() after file source is loaded
        self.task_generator: Optional[TaskGenerator] = None
        self.percent_correct_buffer: List[float] = []
        self.eval_metrics: List[Tuple[str, float]] = []
        self.iter = 0
        self.rng = random.Random()
        self.curriculum_scores: List[float] = []
        self.max_token_len = 16384

    @classmethod
    def config_init(cls):
        cfg = JJEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=8,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=2000,
            batch_size=256,
            steps_per_eval=50,
            max_token_length=16384,
            inference_weight=1.0,
            wandb_name="jj_vcs",
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
        )
        servers = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9004/v1",
                api_key="x",
                num_max_requests_at_once=32,
                num_requests_for_eval=64,
            )
        ]
        return cfg, servers

    async def setup(self):
        """Verify the jj Docker server is reachable and load file source."""
        healthy = await self.executor.health_check()
        if not healthy:
            raise RuntimeError(
                f"JJ execution server not available at {self.config.jj_server_url}. "
                "Please start the Docker container:\n"
                "  docker build -t jj-exec-server environments/community/jj_env/\n"
                "  docker run -p 5003:5003 jj-exec-server"
            )

        # Load file source
        file_source = None
        if self.config.file_source == "hf":
            try:
                hf_source = HFCodeFileSource(
                    dataset_name=self.config.hf_dataset_name,
                    pool_size=self.config.hf_pool_size,
                )
                hf_source.load()
                file_source = hf_source
                logger.info("Using HF code file source (%d files)", hf_source.pool_size)
            except Exception as e:
                logger.warning(
                    "Failed to load HF file source (%s), falling back to synthetic", e
                )
                file_source = SyntheticFileSource()
        else:
            file_source = SyntheticFileSource()
            logger.info("Using synthetic file source")

        self.task_generator = TaskGenerator(
            categories=self.config.task_categories,
            difficulties=self.config.difficulty_levels,
            file_source=file_source,
        )

        logger.info(
            "JJEnv setup: categories=%s, difficulties=%s, file_source=%s",
            self.config.task_categories,
            self.config.difficulty_levels,
            type(file_source).__name__,
        )

    async def get_next_item(self) -> Item:
        """Generate a new task and return it as item data."""
        self.iter += 1
        seed = self.rng.randint(0, 2**31)
        task_spec = self.task_generator.generate_task(seed=seed)

        return {
            "task_id": task_spec.task_id,
            "category": task_spec.category,
            "difficulty": task_spec.difficulty,
            "goal_description": task_spec.goal_description,
            "scramble_ops": task_spec.scramble_ops,
            "goal_file_contents": task_spec.goal_file_contents,
            "goal_log_pattern": task_spec.goal_log_pattern,
            "expected_min_commands": task_spec.expected_min_commands,
            "original_file_count": task_spec.original_file_count,
            "allow_write_file": task_spec.allow_write_file,
            "max_write_lines": task_spec.max_write_lines,
            "tags": task_spec.tags,
        }

    @staticmethod
    def _extract_tool_call(text: str) -> Optional[Dict]:
        """
        Extract the last <tool_call> JSON from text.
        Handles both complete and incomplete (missing closing tag) calls.
        """
        last_pos = text.rfind("<tool_call>")
        if last_pos == -1:
            return None

        json_start = last_pos + len("<tool_call>")
        json_text = text[json_start:].strip()
        json_text = json_text.replace("</tool_call>", "").strip()

        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            # Try brace-counting extraction
            brace_count = 0
            json_end = 0
            for i, char in enumerate(json_text):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            if json_end > 0:
                try:
                    return json.loads(json_text[:json_end])
                except json.JSONDecodeError:
                    pass
        return None

    @staticmethod
    def _has_unresponded_tool_call(text: str) -> bool:
        """Check if there's a <tool_call> without a matching <tool_response> after it."""
        pos = text.rfind("<tool_call>")
        if pos == -1:
            return False
        return "</tool_response>" not in text[pos:]

    async def _exec_tool_call(
        self,
        call_json: Dict,
        repo_id: str,
        allow_write_file: bool = True,
        write_tracking: Optional[Dict] = None,
    ) -> Dict:
        """
        Execute a tool call against the jj server and return result dict.

        When allow_write_file=False, write_file calls are rejected (defense in depth).
        write_tracking dict accumulates: {"attempted": bool, "lines_changed": int}
        """
        name = call_json.get("name", "")
        args = call_json.get("arguments", {})

        if name == "jj":
            command = args.get("command", "")
            result = await self.executor.execute_model_command(repo_id, command)
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        elif name == "write_file":
            if write_tracking is not None:
                write_tracking["attempted"] = True

            if not allow_write_file:
                return {
                    "stdout": "",
                    "stderr": "write_file is not available for this task. Use jj commands only.",
                    "returncode": 1,
                }

            path = args.get("path", "")
            new_content = args.get("content", "")

            # Capture pre-write state to compute lines changed
            if write_tracking is not None:
                try:
                    pre_state = await self.executor.capture_state(repo_id)
                    old_content = pre_state.file_contents.get(path, "")
                    lines_diff = count_lines_changed(old_content, new_content)
                    write_tracking["lines_changed"] = write_tracking.get("lines_changed", 0) + lines_diff
                except Exception:
                    pass  # Don't fail the write if tracking fails

            result = await self.executor.write_file_in_repo(repo_id, path, new_content)
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        else:
            return {
                "stdout": "",
                "stderr": f"Unknown tool: {name}. Use 'jj'{' or write_file' if allow_write_file else ''}.",
                "returncode": 1,
            }

    def _task_spec_from_item(self, task_data: dict) -> TaskSpec:
        """Reconstruct a TaskSpec from item data."""
        return TaskSpec(
            task_id=task_data["task_id"],
            category=task_data["category"],
            difficulty=task_data["difficulty"],
            goal_description=task_data["goal_description"],
            scramble_ops=task_data["scramble_ops"],
            goal_file_contents=task_data["goal_file_contents"],
            goal_log_pattern=task_data.get("goal_log_pattern"),
            expected_min_commands=task_data["expected_min_commands"],
            original_file_count=task_data["original_file_count"],
            allow_write_file=task_data.get("allow_write_file", True),
            max_write_lines=task_data.get("max_write_lines"),
            tags=task_data.get("tags", []),
        )

    async def collect_trajectories(self, item) -> Tuple[Optional[ScoredDataGroup], List]:
        """
        Run interleaved tool execution for a group of rollouts.

        For each rollout:
        1. Create a repo on the jj server
        2. Set up the scrambled state
        3. Loop: generate → parse tool_call → execute → inject response
        4. Capture final state and score
        5. Clean up
        """
        task_data = item
        task_spec = self._task_spec_from_item(task_data)
        num_rollouts = self.config.group_size

        scored: ScoredDataGroup = {
            "tokens": [],
            "masks": [],
            "scores": [],
            "advantages": None,
            "ref_logprobs": None,
            "messages": None,
            "group_overrides": {},
            "overrides": None,
            "images": None,
        }

        # Create repos and set up scrambled state for each rollout
        repo_ids: List[Optional[str]] = [None] * num_rollouts
        prompt_msgs_list: List[Optional[List[Dict]]] = [None] * num_rollouts

        # Build system prompt with write_file gating
        system_prompt = build_system_prompt(allow_write_file=task_spec.allow_write_file)

        for i in range(num_rollouts):
            try:
                repo_id = await self.executor.create_repo()
                repo_ids[i] = repo_id
                await self.executor.setup_scrambled_repo(repo_id, task_spec.scramble_ops)
                repo_info = await self.executor.get_initial_prompt_info(repo_id)
                prompt_msgs_list[i] = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": _build_user_prompt(task_spec, repo_info)},
                ]
            except Exception as e:
                logger.error(f"Failed to setup repo for rollout {i}: {e}")
                if repo_ids[i]:
                    await self.executor.cleanup_repo(repo_ids[i])
                    repo_ids[i] = None

        # Check if all setups failed
        if not any(rid is not None for rid in repo_ids):
            logger.error("All scrambled repo setups failed")
            return None, []

        # Fill failed slots with first successful prompt (they'll get -1 score)
        first_good_msgs = next(m for m in prompt_msgs_list if m is not None)
        for i in range(num_rollouts):
            if prompt_msgs_list[i] is None:
                prompt_msgs_list[i] = first_good_msgs

        # Initialize per-rollout state
        assistant_contents = [""] * num_rollouts
        done = [repo_ids[i] is None for i in range(num_rollouts)]
        commands_used: List[List[str]] = [[] for _ in range(num_rollouts)]
        write_tracking: List[Dict] = [
            {"attempted": False, "lines_changed": 0} for _ in range(num_rollouts)
        ]

        # ── Interleaved generation loop ──────────────────────────────────
        max_turns = self.config.max_rollout_turns

        for turn_idx in range(max_turns):
            if all(done):
                break

            active_prompts = []
            active_indices = []

            for i in range(num_rollouts):
                if not done[i]:
                    prompt_txt = self.tokenizer.apply_chat_template(
                        prompt_msgs_list[i],
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    prompt_txt += assistant_contents[i]
                    active_prompts.append(prompt_txt)
                    active_indices.append(i)

            if not active_prompts:
                break

            # Get completions
            if turn_idx == 0 and len(set(active_prompts)) == 1:
                # First turn: all prompts identical → batch with n
                resp = await self.server.completion(
                    prompt=active_prompts[0],
                    n=len(active_prompts),
                    max_tokens=self.config.max_gen_per_turn,
                    temperature=0.8,
                    stop="</tool_call>",
                )
                replies = [c.text for c in resp.choices]
            else:
                # Heterogeneous → parallel individual requests
                async def _call_single(prompt_str: str) -> str:
                    try:
                        comp = await self.server.completion(
                            prompt=prompt_str,
                            n=1,
                            max_tokens=self.config.max_gen_per_turn,
                            temperature=0.8,
                            stop="</tool_call>",
                        )
                        return comp.choices[0].text
                    except Exception as e:
                        logger.error(f"Completion error: {e}")
                        return ""

                tasks = [_call_single(p) for p in active_prompts]
                replies = await asyncio.gather(*tasks)

            # Process each rollout's reply
            for prompt_idx, rollout_idx in enumerate(active_indices):
                if done[rollout_idx]:
                    continue

                reply = replies[prompt_idx] or ""
                assistant_contents[rollout_idx] += reply
                raw = assistant_contents[rollout_idx]

                # Check if model is done
                if "</think>" in raw or "<done/>" in raw:
                    done[rollout_idx] = True
                    continue

                # Check for tool call
                if self._has_unresponded_tool_call(raw):
                    call_json = self._extract_tool_call(raw)
                    if call_json is None:
                        done[rollout_idx] = True
                        continue

                    # Check command limit
                    if len(commands_used[rollout_idx]) >= self.config.max_commands_per_rollout:
                        assistant_contents[rollout_idx] += "</tool_call>\n"
                        assistant_contents[rollout_idx] += (
                            '<tool_response>{"stderr": "Command limit reached", "returncode": 1}</tool_response>\n'
                        )
                        done[rollout_idx] = True
                        continue

                    # Execute tool call against the jj server
                    try:
                        result = await self._exec_tool_call(
                            call_json,
                            repo_ids[rollout_idx],
                            allow_write_file=task_spec.allow_write_file,
                            write_tracking=write_tracking[rollout_idx],
                        )

                        # Track command
                        cmd_name = call_json.get("name", "")
                        cmd_args = call_json.get("arguments", {})
                        if cmd_name == "jj":
                            commands_used[rollout_idx].append(
                                cmd_args.get("command", "")
                            )
                        else:
                            commands_used[rollout_idx].append(
                                f"{cmd_name}:{cmd_args}"
                            )

                        # Clean up partial closing tag and append proper response
                        content = assistant_contents[rollout_idx]
                        content = re.sub(
                            r"</tool_call.*?$", "", content, flags=re.MULTILINE
                        )
                        assistant_contents[rollout_idx] = content
                        assistant_contents[rollout_idx] += "</tool_call>\n"
                        assistant_contents[rollout_idx] += (
                            f"<tool_response>{json.dumps(result)}</tool_response>\n"
                        )
                    except Exception as e:
                        logger.error(
                            f"Tool exec failed for rollout {rollout_idx}: {e}"
                        )
                        done[rollout_idx] = True
                        continue
                else:
                    # No tool call and not done — last turn means done
                    if turn_idx + 1 >= max_turns:
                        done[rollout_idx] = True

        # ── Score each rollout ───────────────────────────────────────────
        for rollout_idx in range(num_rollouts):
            try:
                raw = assistant_contents[rollout_idx]

                if repo_ids[rollout_idx] is None:
                    # Failed setup
                    scored["tokens"].append([])
                    scored["masks"].append([])
                    scored["scores"].append(-1.0)
                    self.percent_correct_buffer.append(0.0)
                    continue

                # Capture final state from server
                final_state = await self.executor.capture_state(
                    repo_ids[rollout_idx]
                )

                breakdown = self.scorer.score(
                    final_state=final_state,
                    goal_file_contents=task_spec.goal_file_contents,
                    goal_log_pattern=task_spec.goal_log_pattern,
                    commands_used=commands_used[rollout_idx],
                    expected_min_commands=task_spec.expected_min_commands,
                    original_file_count=task_spec.original_file_count,
                    total_lines_changed=write_tracking[rollout_idx]["lines_changed"],
                    max_write_lines=task_spec.max_write_lines,
                    allow_write_file=task_spec.allow_write_file,
                    write_file_attempted=write_tracking[rollout_idx]["attempted"],
                )

                # Format penalty: must have </think> or <done/>
                if "</think>" not in raw and "<done/>" not in raw:
                    reward = max(breakdown.total - 0.2, -1.0)
                else:
                    reward = breakdown.total

                logger.info(
                    f"[Rollout {rollout_idx}] {task_spec.task_id} "
                    f"score={reward:.3f} ({breakdown.details})"
                )

                # Tokenize
                final_assistant_msg = {"role": "assistant", "content": raw}
                full_ctx: List[Message] = prompt_msgs_list[rollout_idx] + [
                    final_assistant_msg
                ]

                toks = self.tokenizer.encode(raw)
                if len(toks) > MAX_REPLY_TOKENS:
                    toks = toks[:MAX_REPLY_TOKENS]
                    raw = self.tokenizer.decode(toks)
                    final_assistant_msg = {"role": "assistant", "content": raw}
                    full_ctx = prompt_msgs_list[rollout_idx] + [final_assistant_msg]

                tok = tokenize_for_trainer(self.tokenizer, full_ctx)
                scored["tokens"].append(tok["tokens"])
                scored["masks"].append(tok["masks"])
                scored["scores"].append(reward)
                self.percent_correct_buffer.append(max(reward, 0))

            except Exception as e:
                logger.error(f"Error scoring rollout {rollout_idx}: {e}")
                scored["tokens"].append([])
                scored["masks"].append([])
                scored["scores"].append(-1.0)
                self.percent_correct_buffer.append(0.0)

        # ── Clean up all repos on the server ─────────────────────────────
        for repo_id in repo_ids:
            if repo_id is not None:
                await self.executor.cleanup_repo(repo_id)

        # ── Curriculum advancement ───────────────────────────────────────
        if self.config.use_curriculum and scored["scores"]:
            avg_score = sum(scored["scores"]) / len(scored["scores"])
            self.curriculum_scores.append(avg_score)
            if len(self.curriculum_scores) >= 20:
                recent_avg = sum(self.curriculum_scores[-20:]) / 20
                if recent_avg > self.config.curriculum_threshold:
                    all_diffs = ["easy", "medium", "hard"]
                    max_idx = max(
                        all_diffs.index(d) for d in self.config.difficulty_levels
                    )
                    if max_idx < len(all_diffs) - 1:
                        next_diff = all_diffs[max_idx + 1]
                        if next_diff not in self.config.difficulty_levels:
                            self.config.difficulty_levels.append(next_diff)
                            self.task_generator.difficulties = (
                                self.config.difficulty_levels
                            )
                            logger.info(
                                f"Curriculum: advancing to include '{next_diff}' "
                                f"(recent avg: {recent_avg:.3f})"
                            )
                            self.curriculum_scores.clear()

        # Check if all scores are the same (bad for training signal)
        if scored["scores"] and all(
            s == scored["scores"][0] for s in scored["scores"]
        ):
            logger.warning(
                f"All {len(scored['scores'])} rollouts have same score "
                f"{scored['scores'][0]:.3f} — returning None"
            )
            return None, []

        return scored, []

    async def evaluate(self, *args, **kwargs):
        """
        Evaluate: generate tasks and check if model produces tool calls.
        """
        total, correct = 0, 0

        for cat in self.config.task_categories[:2]:
            for diff in self.config.difficulty_levels[:1]:
                seed = self.rng.randint(0, 2**31)
                task_spec = self.task_generator.generate_task(
                    seed=seed, category=cat, difficulty=diff
                )

                repo_id = None
                try:
                    repo_id = await self.executor.create_repo()
                    await self.executor.setup_scrambled_repo(
                        repo_id, task_spec.scramble_ops
                    )
                    repo_info = await self.executor.get_initial_prompt_info(repo_id)

                    prompt_msgs = [
                        {"role": "system", "content": build_system_prompt(task_spec.allow_write_file)},
                        {
                            "role": "user",
                            "content": _build_user_prompt(task_spec, repo_info),
                        },
                    ]

                    prompt_txt = self.tokenizer.apply_chat_template(
                        prompt_msgs,
                        add_generation_prompt=True,
                        tokenize=False,
                    )

                    comp = await self.server.completion(
                        prompt=prompt_txt,
                        n=1,
                        max_tokens=512,
                        temperature=0.0,
                        split="eval",
                    )
                    reply = comp.choices[0].text
                    total += 1

                    if "<tool_call>" in reply:
                        correct += 1

                except Exception as e:
                    logger.error(f"Eval error for {cat}/{diff}: {e}")
                    total += 1
                finally:
                    if repo_id:
                        await self.executor.cleanup_repo(repo_id)

        if total > 0:
            accuracy = correct / total
            self.eval_metrics.append(("eval/tool_call_rate", accuracy))
            logger.info(
                f"Eval: tool_call_rate={accuracy:.3f} ({correct}/{total})"
            )

    async def wandb_log(self, metrics: Dict = None):
        metrics = metrics or {}
        if self.percent_correct_buffer:
            metrics["train/avg_score"] = sum(self.percent_correct_buffer) / len(
                self.percent_correct_buffer
            )
        self.percent_correct_buffer = []
        for k, v in self.eval_metrics:
            metrics[k] = v
        self.eval_metrics = []
        metrics["train/num_difficulties"] = len(self.config.difficulty_levels)
        metrics["train/difficulties"] = ",".join(self.config.difficulty_levels)
        await super().wandb_log(metrics)


if __name__ == "__main__":
    JJEnv.cli()
