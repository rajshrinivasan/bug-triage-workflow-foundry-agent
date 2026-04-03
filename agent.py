"""
Bug Triage Workflow
Pattern: Conditional DAG Workflow

The workflow is declared as a graph of LLM steps with explicit branch
conditions. After each branching step, a Python condition function inspects
the step's JSON output and returns the next step name.

DAG:
    intake
      │
      ▼
    classify ──────────────────────────────────────────────┐
      │                                                     │
      ├── severity == "critical"  → critical_escalation     │
      ├── severity == "high"      → sprint_triage           │ all paths
      └── severity in med/low     → backlog                 │ converge
                                        │                   │
                                        └───────────────────┘
                                                │
                                                ▼
                                            summary
"""

import json
import logging
import os
import sys
from pathlib import Path

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI
from dotenv import load_dotenv

from workflow.engine import WorkflowEngine, WorkflowContext
from workflow.steps import (
    intake_step,
    classifier_step,
    critical_escalation_step,
    sprint_triage_step,
    backlog_step,
    summary_step,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(Path(__file__).parent / ".env")

PROJECT_ENDPOINT = os.environ["PROJECT_ENDPOINT"]
MODEL_DEPLOYMENT_NAME = os.environ["MODEL_DEPLOYMENT_NAME"]
API_VERSION = os.environ["API_VERSION"]

log = logging.getLogger(__name__)


# ── LLM call function ────────────────────────────────────────────────────────

def make_llm_call(token_provider):
    # PROJECT_ENDPOINT may include /api/projects/... (Foundry project path) which is
    # not a valid base for OpenAI inference — use only the resource root.
    inference_endpoint = PROJECT_ENDPOINT.split("/api/projects")[0].rstrip("/")
    client = AzureOpenAI(
        azure_endpoint=inference_endpoint,
        azure_ad_token_provider=token_provider,
        api_version=API_VERSION,
    )

    def llm_call(system_prompt: str, user_message: str) -> str:
        response = client.chat.completions.create(
            model=MODEL_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content

    return llm_call


# ── Branch condition: routes after classify step ─────────────────────────────

def route_by_severity(ctx: WorkflowContext) -> str:
    severity = ctx.get_json("classify").get("severity", "").lower()

    if severity == "critical":
        log.debug("[branch] severity=%r → critical_escalation", severity)
        return "critical_escalation"
    elif severity == "high":
        log.debug("[branch] severity=%r → sprint_triage", severity)
        return "sprint_triage"
    else:
        log.debug("[branch] severity=%r → backlog", severity)
        return "backlog"


# ── Workflow assembly ────────────────────────────────────────────────────────

def build_workflow(llm_call) -> WorkflowEngine:
    engine = WorkflowEngine(llm_call)

    engine.add_step(intake_step,              next_step="classify")
    engine.add_step(classifier_step)          # branch decides next
    engine.add_step(critical_escalation_step, next_step="summary")
    engine.add_step(sprint_triage_step,       next_step="summary")
    engine.add_step(backlog_step,             next_step="summary")
    engine.add_step(summary_step)             # terminal

    engine.add_branch(after_step="classify", condition=route_by_severity)

    engine.set_entry("intake")
    engine.validate()
    return engine


# ── Sample bugs ──────────────────────────────────────────────────────────────

_BUGS_FILE = Path(__file__).parent / "sample_bugs.json"
SAMPLE_BUGS: list[dict] = json.loads(_BUGS_FILE.read_text(encoding="utf-8"))


# ── Input selection ──────────────────────────────────────────────────────────

def select_bug() -> str:
    """Present the bug menu and return the raw report text."""
    print("Select a bug to triage:")
    for i, bug in enumerate(SAMPLE_BUGS):
        print(f"  {i + 1}. [{bug['id']}] {bug['label']}")
    print("  4. Enter custom bug report")
    print()

    choice = input("Choice (1-4): ").strip()

    if choice in ("1", "2", "3"):
        bug = SAMPLE_BUGS[int(choice) - 1]
        print(f"\nProcessing: {bug['label']}")
        return bug["report"].strip()

    if choice == "4":
        print("\nPaste your bug report (press Enter twice when done):")
        lines = []
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        return "\n".join(lines).strip()

    print("Invalid choice.")
    return ""


# ── Output formatting ────────────────────────────────────────────────────────

def print_results(ctx: WorkflowContext) -> None:
    """Pretty-print classification, branch output, and summary from a completed run."""
    print("\n" + "=" * 60)
    print("TRIAGE COMPLETE")
    print("=" * 60)

    classification = ctx.get_json("classify")
    if classification:
        print(f"\nSeverity : {classification.get('severity', '?').upper()}")
        print(f"Category : {classification.get('category', '?')}")
        print(f"Reasoning: {classification.get('reasoning', '')}")

    branch_steps = ["critical_escalation", "sprint_triage", "backlog"]
    for step_name in branch_steps:
        if step_name in ctx.steps:
            print(f"\n{'─' * 60}")
            print(f"{step_name.replace('_', ' ').upper()}")
            print("─" * 60)
            print(ctx.steps[step_name])
            break

    if "summary" in ctx.steps:
        print(f"\n{'─' * 60}")
        print("TRIAGE SUMMARY")
        print("─" * 60)
        print(ctx.steps["summary"])


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.WARNING, format="  %(message)s", stream=sys.stdout)
    logging.getLogger("workflow").setLevel(logging.DEBUG)
    logging.getLogger(__name__).setLevel(logging.DEBUG)

    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        credential, "https://ai.azure.com/.default"
    )
    llm_call = make_llm_call(token_provider)
    engine = build_workflow(llm_call)

    print("Bug Triage Workflow")
    print("=" * 60)
    print()

    report = select_bug()
    if not report:
        return

    print("\nRunning triage workflow...")
    ctx = engine.run(report)
    print_results(ctx)


if __name__ == "__main__":
    main()
