# Bug Triage Workflow

## Pattern
**Conditional DAG Workflow** — LLM steps are declared as a graph with explicit branch conditions. After the classifier step, a Python function inspects the step's JSON output and returns the next step name. All branch paths converge at a final summary step.

## Architecture

```
intake
  │
  ▼
classify  ──────────────────────────────────────────────┐
  │                                                      │
  ├── severity == "critical" → critical_escalation       │  all paths
  ├── severity == "high"     → sprint_triage             │  feed into
  └── severity in med/low    → backlog                   │  summary
                                    │                    │
                                    └────────────────────┘
                                            │
                                            ▼
                                         summary
```

Each node is a `WorkflowStep` — a named LLM call with its own system prompt and an `input_builder` that assembles the user message from accumulated context. Branch decisions are made in Python, not by the LLM.

## Files

```
13-bug-triage-workflow/
├── workflow/
│   ├── __init__.py
│   ├── engine.py          # WorkflowEngine, WorkflowStep, WorkflowContext
│   └── steps.py           # Step instances with prompts and input builders
├── prompts/
│   ├── intake_prompt.txt
│   ├── classifier_prompt.txt
│   ├── critical_escalation_prompt.txt
│   ├── sprint_triage_prompt.txt
│   ├── backlog_prompt.txt
│   └── summary_prompt.txt
├── agent.py               # Workflow assembly, branch condition, I/O
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
az login
```

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

```
PROJECT_ENDPOINT=https://<resource>.services.ai.azure.com/api/projects/<project>
MODEL_DEPLOYMENT_NAME=gpt-4o
```

> The `PROJECT_ENDPOINT` is the Azure AI Foundry project URL. The inference client automatically derives the correct resource base from it.

## Running

```bash
python agent.py
```

Select one of the three pre-seeded bugs (covering all three severity paths) or enter a custom report.

## Sample output

### Critical path (BUG-001 — authentication outage)

```
Bug Triage Workflow
============================================================

Select a bug to triage:
  1. [BUG-001] Critical — authentication outage
  2. [BUG-002] High — payment processing failure
  3. [BUG-003] Low — cosmetic UI misalignment
  4. Enter custom bug report

Choice (1-4):
Processing: Critical — authentication outage

Running triage workflow...
  [step: intake]
  [step: classify]
  [branch] severity='critical' → critical_escalation
  [step: critical_escalation]
  [step: summary]

============================================================
TRIAGE COMPLETE
============================================================

Severity : CRITICAL
Category : authentication
Reasoning: The bug prevents all users from logging into the platform, effectively
           blocking core functionality and causing system downtime.

────────────────────────────────────────────────────────────
CRITICAL ESCALATION
────────────────────────────────────────────────────────────
INCIDENT SUMMARY
The login page is returning a 500 error due to database connection failures in the
authentication service, preventing all users from accessing the platform.

IMMEDIATE ACTIONS
1. Page critical personnel: on-call engineer, engineering lead, product manager,
   communications lead.
2. Mitigation: rollback last stable auth service version; enable circuit breaker;
   disable recent feature flags via admin panel.
3. Customer communication: YES — trigger immediately (total login outage).
...

────────────────────────────────────────────────────────────
TRIAGE SUMMARY
────────────────────────────────────────────────────────────
BUG TITLE: Login page returns 500 error
SEVERITY & CATEGORY: Critical — Authentication
TRIAGE PATH: Critical Escalation → complete login outage, all users blocked
ESTIMATED RESOLUTION: 4–8 hours
```

### High path (BUG-002 — payment processing failure)

```
  [step: intake]
  [step: classify]
  [branch] severity='high' → sprint_triage
  [step: sprint_triage]
  [step: summary]

Severity : HIGH
Category : payments
→ Sprint-ready ticket created; backend engineer (payments) assigned
```

### Low path (BUG-003 — cosmetic UI misalignment)

```
  [step: intake]
  [step: classify]
  [branch] severity='low' → backlog
  [step: backlog]
  [step: summary]

Severity : LOW
Category : ui
→ Backlog ticket created; deferred to future maintenance cycle
```

## Key implementation details

- **`WorkflowEngine`**: takes a single `llm_call(system_prompt, user_message) -> str` function. LLM provider is injected — the engine itself has no Azure dependency.
- **`WorkflowContext`**: accumulates step outputs by name. Each step's `input_builder` receives the full context, so later steps can reference earlier outputs.
- **`parse_json=True`**: steps that return JSON (intake, classify) have their output parsed and stored in `ctx.metadata`. The branch condition reads `ctx.get_json("classify")["severity"]`.
- **`engine.validate()`**: called at build time to verify all `next_step` references and branch `after_step` names resolve to registered steps. Wiring errors surface before any LLM call is made.
- **Branch registration**: `engine.add_branch(after_step="classify", condition=fn)` — branch conditions are plain callables, no wrapper class needed.
- **Branch safety**: if JSON parsing fails (model returns malformed JSON), `get_json()` returns `{}` and the branch defaults to `"backlog"`.
- **Logging**: the engine emits step trace via `logging.getLogger("workflow.engine")`. Branch trace comes from `logging.getLogger(__name__)` in `agent.py`. Azure SDK noise is suppressed by scoping both loggers explicitly — no debug spam from the identity library.
- **No agent framework dependency**: uses the OpenAI client directly against the Azure AI Foundry endpoint. Workflow logic is pure Python in `workflow/engine.py`.

## Extending the DAG

Adding a new step is three lines:

```python
# 1. Define the step
review_step = WorkflowStep(
    name="peer_review",
    system_prompt=_load("peer_review_prompt.txt"),
    input_builder=lambda ctx: ctx.steps.get("sprint_triage", ""),
)

# 2. Register it
engine.add_step(review_step, next_step="summary")

# 3. Update the step that should flow into it
engine.add_step(sprint_triage_step, next_step="peer_review")  # was next_step="summary"
```

`engine.validate()` will catch any broken references immediately at startup.
