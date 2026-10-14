# CLI Contract

**Status**: Planned commands; executable does not exist yet. Controls do not download models.
Use the [HTTP contract](http.md) and [resource model](../data-model.md) as shared authority.

## Commands

```text
mrun --version
mrun model verify --path PATH --sha256 HASH
mrun serve --model PATH --qualification REPORT [OPTIONS]
mrun status [--url http://127.0.0.1:8012] [--json]
mrun cache clear [--url http://127.0.0.1:8012]
mrun model unload [--url http://127.0.0.1:8012]
```

`model verify` checks file length/hash, GGUF metadata, tokenizer/template identity and declared
architecture against the primary artifact manifest; it does not imply backend/quality qualification.
`serve` verifies before mapping/loading; no arbitrary unqualified artifact is accepted in this feature.
Load lazily on the first admitted generation; status/discovery do not force a model load.
Control commands use the runtime HTTP endpoints and return only after acknowledged safe completion.
No interactive confirmations, background daemon installation, tool execution or shell interpolation.

## Serve Options

| Option | Default and validation |
| --- | --- |
| `--model PATH` | Required local verified file; never log its absolute path |
| `--qualification REPORT` | Required local report matching binary/backend/model/profile; capabilities must pass |
| `--bind ADDRESS` | 127.0.0.1:8012; only literal IPv4/IPv6 loopback |
| `--reuse on\|off` | Default on only after full reuse eligibility, otherwise off with explicit status; explicit on without eligibility is rejected |
| `--budget-bytes N` | Automatic resource policy; positive integer bytes |
| `--allow-budget-increase` | false; required to exceed automatic recommendation; sanitized warning |
| `--context-tokens N` | 16384; only lower values >=256; no automatic model-max expansion |
| `--output-tokens N` | 2048; only lower positive values; must be <=context |
| `--cache-entries N` | 4; 0..4; zero disables retention |
| `--cache-cells N` | 16384; 0..16384; unique retained cells, not physical allocation size |
| `--idle-seconds N` | 0 means warm indefinitely; positive enables explicit idle expiry |
| `--origin URL` | None; repeated exact loopback browser origins only; no wildcard |
| `--log-level LEVEL` | info; metadata only even at debug/trace |

Queue size/deadlines, chunk size, physical KV pool, byte limits and KV precision are frozen profile
settings from the data model, not casual flags. Qualification harness profiles may vary them before
freezing a run manifest; expose selected values in status. Raising context/output/cache maxima,
changing precision, backend or template requires separate profile review and qualification.

No config file discovery in private repos; explicit flags only. No metrics exporter or content log
switch in this feature. Missing artifacts/backend resources fail without inference-time fetch.
`--version` reports build version/backend commit without machine paths or secrets.

Qualification reports are explicit trusted-operator inputs, not model downloads or private conversation
state. Parse a bounded versioned report, verify referenced public evidence hashes and exact identities,
and reject missing/failed/inconclusive capability gates. A declared pass without required records is
invalid. Reuse additionally requires state/numerical, IC-001..020 protocol and SC-007 task-quality
passes for the same candidate/profile; performance failures remain
visible and prohibit performance-qualified status even if the basic runtime can serve safely.
The report is local evidence, not a cryptographically certified claim; the operator remains trusted.

Use the [QualificationReport](../data-model.md#qualificationreport) version/limits and lifecycle.
An adapter-spike report cannot authorize `serve`. Candidate evidence binds the actual runtime executable
SHA-256 as well as the separately identified harness, backend/model/build/profile and sealed fixtures.
Reject replaced/rebuilt executables or stale affected evidence before load/admission; rerun candidate
qualification rather than changing a report's subject hash. A matching unchanged binary may reuse its
verified report. Unsupported evidence/reuse eligibility fails with exit 3 and sanitized diagnostics.

The cache-off MVP must already implement OS/device observations, automatic budget/delta policy,
full preallocation leases, prompt/output reservations, pressure handling and safe cleanup. These are
serving prerequisites, not optional later memory optimizations. No production flag bypasses them or
the reuse gate. Qualification-only candidate execution is governed by the fixture-scoped
[harness contract](qualification.md#staged-evidence-and-candidate-runs), never by a `serve` override.

## Shutdown and Exit

SIGINT/SIGTERM closes admission, cancels queued/active requests, synchronizes submitted work,
releases native resources, then exits. Do not promise bounded GPU destruction after a driver hang;
report failed cleanup externally and do not label it safely released. A subsequent process starts
without retained conversation state. Ready/loading/draining/failure transitions are visible in status.

Exit codes: 0 success; 2 invalid CLI/configuration; 3 invalid/unavailable artifact or native backend;
4 control connection/service failure; 5 failed validation/evidence gate when used by the future
qualification harness. Errors omit content, secrets and absolute paths; numeric limits are safe.