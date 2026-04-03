"""
Workflow engine — defines the WorkflowStep and WorkflowEngine primitives
used to declare and execute a conditional DAG of LLM steps.

Design:
- Each WorkflowStep is a named LLM call with a system prompt and input builder.
- A branch condition is a callable that inspects step output and returns the next
  step name. Registered via add_branch(after_step, condition).
- WorkflowEngine runs steps in order, evaluating branch conditions after
  each step to decide which step comes next.
- Step outputs are accumulated in WorkflowContext and passed forward.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Callable

log = logging.getLogger(__name__)


@dataclass
class WorkflowContext:
    """Accumulated state passed through the workflow."""
    raw_input: str
    steps: dict[str, str] = field(default_factory=dict)     # step_name → output text
    metadata: dict[str, dict] = field(default_factory=dict)  # step_name → parsed JSON

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


class WorkflowEngine:
    """
    Executes a conditional DAG of WorkflowSteps.

    Usage:
        engine = WorkflowEngine(llm_call_fn)
        engine.add_step(step1, next_step="step2")
        engine.add_step(step2)
        engine.add_branch(after_step="step2", condition=my_router)
        engine.set_entry("step1")
        engine.validate()
        ctx = engine.run(raw_input)
    """

    def __init__(self, llm_call: Callable[[str, str], str]):
        """llm_call: fn(system_prompt, user_message) -> str"""
        self._llm_call = llm_call
        self._steps: dict[str, WorkflowStep] = {}
        self._branches: dict[str, Callable[[WorkflowContext], str | None]] = {}
        self._entry: str | None = None
        self._default_next: dict[str, str] = {}

    def add_step(self, step: WorkflowStep, next_step: str | None = None):
        self._steps[step.name] = step
        if next_step:
            self._default_next[step.name] = next_step

    def add_branch(self, after_step: str, condition: Callable[[WorkflowContext], str | None]):
        self._branches[after_step] = condition

    def set_entry(self, step_name: str):
        self._entry = step_name

    def validate(self):
        """Check graph consistency. Raises ValueError on wiring errors."""
        if not self._entry:
            raise ValueError("No entry step set. Call set_entry() before validate().")
        if self._entry not in self._steps:
            raise ValueError(f"Entry step '{self._entry}' is not registered.")
        for step_name, next_name in self._default_next.items():
            if next_name not in self._steps:
                raise ValueError(
                    f"Step '{step_name}' declares next_step='{next_name}', "
                    f"but '{next_name}' is not registered."
                )
        for after in self._branches:
            if after not in self._steps:
                raise ValueError(
                    f"Branch declared after_step='{after}', "
                    f"but '{after}' is not registered."
                )

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove markdown code fences from LLM output before JSON parsing."""
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        return text

    def run(self, raw_input: str) -> WorkflowContext:
        ctx = WorkflowContext(raw_input=raw_input)
        current = self._entry

        while current:
            step = self._steps[current]
            log.debug("[step: %s]", step.name)

            user_message = step.input_builder(ctx)
            output = self._llm_call(step.system_prompt, user_message)
            ctx.steps[step.name] = output

            if step.parse_json:
                try:
                    ctx.metadata[step.name] = json.loads(self._strip_fences(output))
                except json.JSONDecodeError:
                    ctx.metadata[step.name] = {}

            # Branch condition takes priority over default next
            if current in self._branches:
                current = self._branches[current](ctx)
            elif current in self._default_next:
                current = self._default_next[current]
            else:
                current = None

        return ctx
