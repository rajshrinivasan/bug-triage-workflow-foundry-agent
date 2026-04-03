"""
Workflow engine — defines the WorkflowStep, BranchCondition, and WorkflowEngine
primitives used to declare and execute a conditional DAG of LLM steps.

Design:
- Each WorkflowStep is a named LLM call with a system prompt and input builder.
- A BranchCondition inspects step output and returns the next step name.
- WorkflowEngine runs steps in order, evaluating branch conditions after
  each step to decide which step comes next.
- Step outputs are accumulated in WorkflowContext and passed forward.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class WorkflowContext:
    """Accumulated state passed through the workflow."""
    raw_input: str
    steps: dict[str, str] = field(default_factory=dict)    # step_name → output text
    metadata: dict[str, dict] = field(default_factory=dict) # step_name → parsed JSON

    def get_json(self, step_name: str) -> dict:
        """Return parsed JSON metadata for a step, or empty dict."""
        return self.metadata.get(step_name, {})


@dataclass
class WorkflowStep:
    """
    A single LLM step in the workflow.

    name:           unique identifier
    system_prompt:  instruction text for the LLM
    input_builder:  callable(WorkflowContext) -> str fed to the LLM as user message
    parse_json:     if True, attempt to parse output as JSON and store in context.metadata
    """
    name: str
    system_prompt: str
    input_builder: Callable[[WorkflowContext], str]
    parse_json: bool = False


@dataclass
class BranchCondition:
    """
    After `after_step` completes, call `condition(ctx)` to get the next step name.
    If condition returns None, the workflow ends.
    """
    after_step: str
    condition: Callable[[WorkflowContext], str | None]


class WorkflowEngine:
    """
    Executes a conditional DAG of WorkflowSteps.

    Usage:
        engine = WorkflowEngine(llm_call_fn)
        engine.add_step(step1)
        engine.add_step(step2)
        engine.add_branch(BranchCondition(after_step="classify", condition=my_router))
        engine.set_entry("intake")
        ctx = engine.run(raw_input)
    """

    def __init__(self, llm_call: Callable[[str, str], str]):
        """
        llm_call: fn(system_prompt, user_message) -> str
        """
        self._llm_call = llm_call
        self._steps: dict[str, WorkflowStep] = {}
        self._branches: dict[str, BranchCondition] = {}
        self._entry: str | None = None
        self._default_next: dict[str, str] = {}  # step_name → next step name (linear)

    def add_step(self, step: WorkflowStep, next_step: str | None = None):
        self._steps[step.name] = step
        if next_step:
            self._default_next[step.name] = next_step

    def add_branch(self, branch: BranchCondition):
        self._branches[branch.after_step] = branch

    def set_entry(self, step_name: str):
        self._entry = step_name

    def run(self, raw_input: str, verbose: bool = True) -> WorkflowContext:
        ctx = WorkflowContext(raw_input=raw_input)
        current = self._entry

        while current:
            step = self._steps[current]
            if verbose:
                print(f"\n  [step: {step.name}]")

            user_message = step.input_builder(ctx)
            output = self._llm_call(step.system_prompt, user_message)
            ctx.steps[step.name] = output

            if step.parse_json:
                try:
                    # Strip markdown code fences if present
                    clean = output.strip()
                    if clean.startswith("```"):
                        clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
                    ctx.metadata[step.name] = json.loads(clean)
                except json.JSONDecodeError:
                    ctx.metadata[step.name] = {}

            # Determine next step: branch condition takes priority over default
            if current in self._branches:
                current = self._branches[current].condition(ctx)
            elif current in self._default_next:
                current = self._default_next[current]
            else:
                current = None

        return ctx
