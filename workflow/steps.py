"""
Workflow step definitions for the Bug Triage Workflow.

Each step is a WorkflowStep with a system prompt loaded from prompts/
and an input_builder that assembles the user message from WorkflowContext.
"""

from pathlib import Path
from .engine import WorkflowStep, WorkflowContext

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text()


def _bug_and_classification(ctx: WorkflowContext) -> str:
    return (
        f"Bug report:\n{ctx.steps.get('intake', ctx.raw_input)}\n\n"
        f"Classification:\n{ctx.steps.get('classify', '')}"
    )


# ── Step definitions ─────────────────────────────────────────────────────────

intake_step = WorkflowStep(
    name="intake",
    system_prompt=_load("intake_prompt.txt"),
    input_builder=lambda ctx: f"Bug report:\n\n{ctx.raw_input}",
    parse_json=True,
)

classifier_step = WorkflowStep(
    name="classify",
    system_prompt=_load("classifier_prompt.txt"),
    input_builder=lambda ctx: (
        f"Structured bug report:\n{ctx.steps.get('intake', ctx.raw_input)}"
    ),
    parse_json=True,
)

critical_escalation_step = WorkflowStep(
    name="critical_escalation",
    system_prompt=_load("critical_escalation_prompt.txt"),
    input_builder=_bug_and_classification,
)

sprint_triage_step = WorkflowStep(
    name="sprint_triage",
    system_prompt=_load("sprint_triage_prompt.txt"),
    input_builder=_bug_and_classification,
)

backlog_step = WorkflowStep(
    name="backlog",
    system_prompt=_load("backlog_prompt.txt"),
    input_builder=_bug_and_classification,
)

summary_step = WorkflowStep(
    name="summary",
    system_prompt=_load("summary_prompt.txt"),
    input_builder=lambda ctx: "\n\n".join(
        f"=== {name.upper()} ===\n{output}"
        for name, output in ctx.steps.items()
    ),
)
