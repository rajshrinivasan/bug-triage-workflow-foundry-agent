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

import os
import sys
import json
from pathlib import Path

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI
from dotenv import load_dotenv

from workflow.engine import WorkflowEngine, BranchCondition, WorkflowContext
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


# ── LLM call function ────────────────────────────────────────────────────────

def make_llm_call(token_provider):
    # PROJECT_ENDPOINT may include /api/projects/... (Foundry project path) which is
    # not a valid base for OpenAI inference — use only the resource root.
    inference_endpoint = PROJECT_ENDPOINT.split("/api/projects")[0].rstrip("/")
    client = AzureOpenAI(
        azure_endpoint=inference_endpoint,
        azure_ad_token_provider=token_provider,
        api_version="2024-10-21",
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
    classification = ctx.get_json("classify")
    severity = classification.get("severity", "").lower()

    if severity == "critical":
        print(f"  [branch] severity={severity!r} → critical_escalation")
        return "critical_escalation"
    elif severity == "high":
        print(f"  [branch] severity={severity!r} → sprint_triage")
        return "sprint_triage"
    else:
        print(f"  [branch] severity={severity!r} → backlog")
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

    engine.add_branch(BranchCondition(
        after_step="classify",
        condition=route_by_severity,
    ))

    engine.set_entry("intake")
    return engine


# ── Sample bugs ──────────────────────────────────────────────────────────────

SAMPLE_BUGS = [
    {
        "id": "BUG-001",
        "label": "Critical — authentication outage",
        "report": """
Users cannot log in to the application. The login page returns a 500 error
after submitting credentials. This started approximately 20 minutes ago.
All users are affected — nobody can access the platform. Our on-call engineer
checked and the auth service is returning database connection errors.
Environment: Production.
Reporter: Platform Operations team.
""",
    },
    {
        "id": "BUG-002",
        "label": "High — payment processing failure",
        "report": """
Subscription renewals are failing for users on annual plans.
The payment processor returns error code 4012 (card declined) even for cards
that were previously charged successfully. Monthly plan renewals appear unaffected.
Approximately 200 users are impacted. No workaround available.
Steps to reproduce: wait for an annual plan renewal to trigger, or manually
trigger a renewal from the admin panel for an annual plan user.
Reporter: Customer Success, escalated from 3 support tickets.
""",
    },
    {
        "id": "BUG-003",
        "label": "Low — cosmetic UI misalignment",
        "report": """
On the contacts list page, the column headers are misaligned by a few pixels
in Safari 17 on macOS. The columns themselves display correctly; only the header
row is shifted right. Chrome and Firefox are not affected.
This is a cosmetic issue — all functionality works correctly.
Steps to reproduce: open contacts list in Safari 17 on macOS Sonoma.
Reporter: QA team during regression testing.
""",
    },
]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        credential, "https://ai.azure.com/.default"
    )
    llm_call = make_llm_call(token_provider)
    engine = build_workflow(llm_call)

    print("Bug Triage Workflow")
    print("=" * 60)
    print("\nSelect a bug to triage:")
    for i, bug in enumerate(SAMPLE_BUGS):
        print(f"  {i + 1}. [{bug['id']}] {bug['label']}")
    print("  4. Enter custom bug report")
    print()

    choice = input("Choice (1-4): ").strip()

    if choice in ("1", "2", "3"):
        bug = SAMPLE_BUGS[int(choice) - 1]
        print(f"\nProcessing: {bug['label']}")
        raw_input = bug["report"].strip()
    elif choice == "4":
        print("\nPaste your bug report (press Enter twice when done):")
        lines = []
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        raw_input = "\n".join(lines).strip()
    else:
        print("Invalid choice.")
        return

    print("\nRunning triage workflow...")
    ctx = engine.run(raw_input, verbose=True)

    print("\n" + "=" * 60)
    print("TRIAGE COMPLETE")
    print("=" * 60)

    # Print classification
    classification = ctx.get_json("classify")
    if classification:
        print(f"\nSeverity : {classification.get('severity', '?').upper()}")
        print(f"Category : {classification.get('category', '?')}")
        print(f"Reasoning: {classification.get('reasoning', '')}")

    # Print branch output
    branch_steps = ["critical_escalation", "sprint_triage", "backlog"]
    for step_name in branch_steps:
        if step_name in ctx.steps:
            print(f"\n{'─' * 60}")
            print(f"{step_name.replace('_', ' ').upper()}")
            print("─" * 60)
            print(ctx.steps[step_name])
            break

    # Print summary
    if "summary" in ctx.steps:
        print(f"\n{'─' * 60}")
        print("TRIAGE SUMMARY")
        print("─" * 60)
        print(ctx.steps["summary"])


if __name__ == "__main__":
    main()
