# Verification performed on this rebuild

This isn't a claim of "fully tested" — it's a precise record of what was
actually checked, and how, so you know exactly how much to trust before
deploying.

## Static checks
- Every `.py` file compiles (`python -m py_compile`) across the entire
  `src/` tree.
- `uv lock` resolved the full dependency graph against PyPI with zero
  conflicts (152 packages).
- `uv sync --all-extras` installed the complete dependency set into a
  real virtualenv with zero errors.
- Verified — by actually installing and importing, not from memory —
  that `tavily.AsyncTavilyClient`, `ollama.AsyncClient`, and LangChain's
  `BaseChatModel.ainvoke` genuinely exist before the async rewrite
  depended on them.

## Live execution checks
Run against the real FastAPI app (via `httpx.ASGITransport`, with the
app's actual `lifespan` triggered — not mocked):

| Check | Result |
|---|---|
| `GET /health` | 200 |
| `POST /api/v1/auth/register` | 201, valid JWT returned |
| `POST /api/v1/auth/login` | 200, valid JWT returned |
| `GET /api/v1/conversations` with valid token | 200, `[]` (correctly empty, correctly scoped) |
| `GET /api/v1/conversations` with no token | 401 (auth genuinely enforced, not just present in code) |
| `GET /metrics` | 200, real Prometheus text output |
| Graceful shutdown | logged waiting for 0 in-flight requests, disposed the DB engine cleanly |
| Route registration | all 16 expected routes present on the built app, including the new `/report/submit` and `/report/jobs/{id}` |

18 new unit/integration tests (auth flow, circuit breaker state machine,
cache behavior) — all passing.

## What was NOT verified
- No live Groq/Tavily/Ollama API keys were used — provider calls
  themselves (as opposed to the app wiring around them) are unverified
  against real upstream services.
- No live Redis/Postgres/Celery worker was actually started — Docker
  Compose config is written and YAML-valid, but not run end-to-end here.
- The original 175-test suite (written for the sync version) was run and
  confirmed to fail wholesale (142/176) on the async conversion — this is
  expected and mechanical (sync test functions calling now-async code),
  not evidence of an app bug, but it means **the pre-existing test
  suite's coverage is not currently backing this rebuild**. Treat that
  suite as a known follow-up, not as passing regression coverage.

## Bug found and fixed in a follow-up pass

A live test (register → mocked research call) revealed that structured
logging's `run_id` correlation — the actual point of the feature — only
worked for the `init_run` node's own log line. Every subsequent node
(`scope_question`, `search_web`, etc.) logged `run_id=-` instead of the
real ID.

**Root cause:** `bind_run_id()` was called inside the `init_run` node
itself. Python's `contextvars` are captured by value when `asyncio`
creates a new `Task`. LangGraph runs each node as effectively its own
task; setting the var *inside* one node's task only affects that task's
own context, not sibling tasks the graph creates afterward for later
nodes.

**Fix:** moved `run_id` generation into `make_initial_state()` (the
application layer, before `graph.ainvoke()` is called) and call
`bind_run_id()` there — the true ancestor of every task the graph will
spawn for that run. `init_run` now just uses the passed-in `run_id`
(with a fallback `uuid4()` for safety).

**Verified fixed** by re-running the same live test: every node's log
line for a given run now carries the same correct `run_id`, and two
concurrent-ish runs get distinct, non-cross-contaminated IDs.

## Bug found and fixed while packaging

The first packaging pass accidentally zipped a 654MB `.venv/` (created
while verifying `uv sync`). Caught by checking actual zip contents
rather than trusting the exclude flags — same lesson as before: always
verify the artifact, not just the command that should have produced it.

## Bug found and fixed in a second follow-up pass: dead metrics

Grepping for every metric defined in `core/metrics.py` against where each
was actually used elsewhere in the codebase found that only 2 of 8
(`http_requests_total`, `http_request_duration_seconds`) were ever
incremented. The other 6 — `graph_run_duration_seconds`,
`graph_run_total`, `provider_call_duration_seconds`,
`circuit_breaker_state`, `cache_lookups_total`,
`search_reflection_triggered_total` — were declared and wired into the
Grafana dashboard's panels, but nothing in the application ever called
`.observe()` or `.inc()` on them. Those panels would have rendered
permanently empty.

**Fix:** instrumented all 6:
- `graph_run_duration_seconds` / `graph_run_total` — wrapped around
  `graph.ainvoke()` in all three use cases, labeled by mode and status.
- `provider_call_duration_seconds` / `circuit_breaker_state` — wired into
  `CircuitBreaker.call()` itself, so every provider automatically gets
  both without each provider needing its own instrumentation.
- `cache_lookups_total` — wired into both cache backends' `get()`.
- `search_reflection_triggered_total` — wired into `reflect_and_refine`.

**Verified with two live tests, not just code inspection:**
1. A full research call through the API (mocked providers) followed by
   an identical second call — confirmed the second call logged "Research
   cache hit", made zero additional provider calls, and `/metrics`
   showed `cache_lookups_total{result="miss"} 1` +
   `cache_lookups_total{result="hit"} 1`, `graph_run_total{status="completed"} 1`
   (only counted once, correctly skipping the cached second run).
2. A direct circuit breaker test (one success, one induced failure) —
   confirmed `provider_call_duration_seconds_count` reached 2 and
   `circuit_breaker_open_total` reached 1 immediately after.

Also worth noting: the first version of this second test used only 1
mocked source, which correctly triggered the reflection loop's routing
logic (since it requires 2+ sources) — the reflection loop itself works;
the test just needed a second mock LLM response to complete cleanly,
which is a test-authoring detail, not an application bug.

## Bug found and fixed in a third pass: unhandled broker failure on /report/submit

Testing `/report/submit` with no Redis running raised an unhandled
`ConnectionError` all the way to the caller instead of a clean error.
First attempted fix silently failed to apply — a `str.replace()`-based
edit targeted text that a prior edit had already changed (adding
`_current_user.id` to the `.delay()` call), so the match no longer
existed and the edit was a silent no-op. Caught by re-viewing the file's
actual content rather than trusting that the edit call succeeded.

**Fix, once applied for real:** wrapped `generate_report_task.delay(...)`
in `try`/`except`, converting a broker connection failure into a clean
`503` with a message pointing the caller at the synchronous `/report`
endpoint as a fallback.

**A test-methodology lesson along the way:** the fix initially still
appeared broken under a direct `httpx.ASGITransport` test — investigation
traced this to Starlette's `ServerErrorMiddleware`, which sends the
error response *and then deliberately re-raises* the exception (so a
real ASGI server like uvicorn can log the traceback even though the
response was already flushed to the client). `httpx.ASGITransport`
doesn't shield against that re-raise the way a real server does, so it
looked like the exception was escaping uncaught when it wasn't.
Confirmed by switching to `starlette.testclient.TestClient(app,
raise_server_exceptions=False)`, which reveals what a real client
actually receives — a deliberately-broken test route confirmed the
existing generic exception handler was working correctly all along
(clean `500` JSON), and the fixed `/report/submit` now correctly returns
`503` when Redis is unreachable.

**Also confirmed independently, and worth knowing:** `REDIS_URL` is read
by both the Celery broker config and the rate limiter's storage backend.
If Redis is configured but unreachable, *any* rate-limited endpoint
fails at the `@limiter.limit(...)` decorator itself, before the route
body even runs — this is a slowapi/Redis characteristic, not something
this codebase's own error handling can intercept from inside a route.
Worth knowing if you deploy with `REDIS_URL` set: an unreachable Redis
affects rate limiting on every decorated endpoint, not just report
submission.

If you're deploying this, the highest-value next steps in order: (1) get
real provider credentials and run one live research/chat/report call end
to end, (2) bring up the Docker Compose stack for real, (3) update the
pre-existing test suite to async.
