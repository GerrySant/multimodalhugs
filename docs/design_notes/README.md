# Design notes

This directory holds the project's design memory: one note per design decision, one handoff
note per long-running effort, and one compatibility record per dependency major-version
upgrade. `AGENTS.md` §8 says when each is written; this file says what each contains.

Notes are committed, never gitignored, so that the history is reviewable and the next session
or the next upgrade can learn from this one. A note is short: if it needs more than two screens,
the detail belongs in `docs/` with a link from the note.

## Decision note

Written for a design decision: why X over Y, what was rejected, what it costs. Name the file
after the decision (`<decision>.md`, for example `flat_processor_slots.md`).

```
# <Decision title>

Status: proposed | accepted | superseded by <file>
Date: YYYY-MM-DD

## Context
The problem, the constraints, and what forced a decision now.

## Options considered
One paragraph per option, including the one chosen, with its trade-offs.

## Decision
What was chosen and why, in a few sentences.

## Consequences
What becomes easier, what becomes harder, what must change elsewhere, what stays
backward compatible, and the migration plan when there is one.
```

## Handoff note

Kept for an effort that spans sessions (a dependency major-version migration, a large
refactor). It is the first thing a new session reads and the last thing a session updates;
it records state, which is why state never goes into `AGENTS.md`. Delete it when the effort
lands; what is worth keeping has by then moved into the CHANGELOG and into decision notes.
Name it `handoff_<effort>.md`.

```
# Handoff — <effort>

Branch: <name>
Goal: one or two sentences, with the acceptance criterion (for example "full suite green on
<library> <version>").

## Done
Bullet per completed step, with the commit hash.

## Remaining
Ordered list of what is left, most blocking first, with the failing tests or symptoms.

## How to resume
The commands to set up and run (install, test command, scripts), and anything a fresh
session would otherwise have to rediscover.

## Open questions for the maintainer
Decisions that cannot be made from the code.
```

## Compatibility record

Written during a dependency major-version upgrade, one file per library and major version,
named `compat_<library>_<major>.md`. Each incompatibility is recorded before its fix is
committed, one commit per entry.

```
## <Short name of the incompatibility>

Symbol: the class, method, attribute or parameter that changed, and how
(removed / renamed / new return type / new default).
Files affected: paths in this repository.
What changed upstream: one or two sentences, with the upstream reference when known.
Fix: what was done here and why it is the HF-aligned choice rather than a shim.
```