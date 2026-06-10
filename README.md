# Jeff

Personal AI assistant that lives on the [Ensemble](https://github.com/boxsie/ensemble) decentralized P2P network. Jeff registers as a service against an Ensemble daemon, replies to chats from allowlisted contacts using a local Ollama LLM, and remembers prior conversation in Postgres + pgvector.

## Architecture

```
[ Ensemble TUI peer ] --chat-->  [ Ensemble daemon ]  <--gRPC bidi-->  [ Jeff ]  --HTTP-->  [ Ollama ]
                                                                          |
                                                                          +--SQL/pgvector-->  [ Postgres ]
```

Jeff is the first inhabitant of the Ensemble multi-service platform — it models the third-party-app pattern future services will copy. Lives in its own repo on purpose; the only Ensemble-side coupling is the [`ensemble-client`](https://github.com/boxsie/ensemble/tree/main/clients/python) Python package.

## Run locally

You need three things up:

1. **Ensemble daemon** (and a TUI peer to chat from). See the [ensemble README](https://github.com/boxsie/ensemble).
2. **Ollama** with the chat + embed models pulled.
3. **Postgres with pgvector**.

```bash
# 1. Pull the models
ollama pull gemma3:12b-it-qat
ollama pull bge-m3

# 2. Local Postgres with pgvector for dev
docker run --rm -d --name jeff-pg -p 5432:5432 \
  -e POSTGRES_PASSWORD=dev -e POSTGRES_DB=jeff \
  pgvector/pgvector:pg16

# 3. Install + run jeff
git clone git@github.com:boxsie/jeff.git
cd jeff
python -m venv .venv && source .venv/bin/activate
pip install -e .

chmod 600 /tmp/admin.seed   # W3 #78131ae8: jeff refuses to load a seed
                            # file readable by group or world; chmod 600
                            # (or set ENSEMBLE_SEED_SKIP_MODE_CHECK=1 if
                            # the file lives on a namespace-isolated
                            # tmpfs like a k8s Secret mount).

JEFF_DB_URL=postgresql://postgres:dev@localhost:5432/jeff \
JEFF_ALLOWLIST=<your-tui-address> \
ENSEMBLE_SOCKET=/tmp/d-jeff/sock \
ENSEMBLE_AUTH_SEED=/tmp/admin.seed \
  python -m jeff
```

Jeff prints its registered Ensemble address + onion on startup. Add that address as a contact in your TUI, then say hi.

## Configuration (env vars)

| Variable | Default | Purpose |
| --- | --- | --- |
| `JEFF_DB_URL` | _required_ | Full libpq URL, e.g. `postgresql://jeff:pwd@host:5432/jeff` |
| `JEFF_NAME` | `jeff` | Service name registered with the daemon |
| `JEFF_DESCRIPTION` | `Personal AI assistant` | Service description |
| `JEFF_ALLOWLIST` | _empty_ | Comma-separated peer addresses allowed to chat. Empty = ignore everyone (safer than "anyone"). |
| `ENSEMBLE_SOCKET` | `/run/ensemble/sock` | Path to the daemon's gRPC unix socket |
| `ENSEMBLE_AUTH_SEED` | _none_ | Path to the admin-key seed file for gRPC auth (optional for local sockets without auth) |
| `JEFF_LLM_PROVIDER` | `ollama` | Chat provider: `ollama` (local) or `grok` (xAI cloud). Unknown values fail at startup. |
| `JEFF_CHAT_MODEL` | _provider default_ | Provider-agnostic chat model override. Wins over `OLLAMA_CHAT_MODEL`. Defaults: `gemma3:12b-it-qat` (ollama), `grok-4` (grok). |
| `XAI_API_KEY` | _none_ | xAI API key. **Required** when `JEFF_LLM_PROVIDER=grok` (startup fails fast if missing). Never logged. |
| `XAI_BASE_URL` | `https://api.x.ai/v1` | xAI API base URL (OpenAI-compatible `/chat/completions`). |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama HTTP endpoint (always used for embeddings, regardless of chat provider) |
| `OLLAMA_CHAT_MODEL` | `gemma3:12b-it-qat` | Chat model under the `ollama` provider (legacy; `JEFF_CHAT_MODEL` overrides it) |
| `OLLAMA_EMBED_MODEL` | `bge-m3` | Embedding model (embeddings stay local). `bge-m3` separates relevant from unrelated far better than `nomic-embed-text`. Changing this changes the vector dimension — run `python -m jeff reset-memory --yes` after switching. |
| `OLLAMA_EMBED_DIM` | `1024` | Embedding vector dimensionality (must match the model: `bge-m3`=1024, `nomic-embed-text`=768). A mismatch with stored data fails loudly at startup. |
| `MEMORY_RECALL_K` | `5` | How many semantically-similar past messages to retrieve per turn |
| `MEMORY_RECENT_TURNS` | `10` | How many most-recent messages to include per turn |
| `MEMORY_RECALL_DISTANCE` | `0.55` | Cosine-distance ceiling for recall (0..2; lower = stricter). Tuned for `bge-m3`; use ~0.4 for `nomic-embed-text`. Inspect live distances with the `/debug recall <query>` chat command. |
| `JEFF_TOOLS_ENABLED` | `true` | Master switch for tool use. When off (or the registry is empty) the turn loop is byte-identical to the no-tools single-shot path. |
| `JEFF_MAX_TOOL_ITERS` | `5` | Max provider↔tool round-trips per turn before a graceful "tool-use limit" reply. |
| `JEFF_TOOL_TIMEOUT_S` | `30` | Per-tool execution timeout (seconds). A timed-out tool returns a safe error to the model; the turn survives. |
| `JEFF_SEARCH_ENABLED` | `false` | Enable the SearXNG-backed `web_search` / `image_search` tools (also requires `JEFF_TOOLS_ENABLED`). |
| `JEFF_SEARXNG_URL` | `http://localhost:8888` | SearXNG JSON-API base URL. The real in-cluster URL is **not committed** — it arrives via the serves ConfigMap. Startup fails fast if search is enabled but this is empty. |
| `JEFF_SEARXNG_AUTH` | _none_ | Optional full `Authorization` header value if the SearXNG instance requires auth (e.g. `Basic …` / `Bearer …`). Never logged. |
| `JEFF_SEARCH_SAFESEARCH` | `0` | Safe-search level stamped on every query: `0` off, `1` moderate, `2` strict. Off by default. A bad value fails fast at load. |
| `JEFF_COMMANDS_ENABLED` | `true` | Whether Jeff declares its slash-commands (`/clear`, `/forget`, `/stats`) to the daemon at registration. When off, Jeff declares none and the daemon routes no command invocations to it. |
| `JEFF_CURIOSITY_ENABLED` | `false` | Curiosity drive: after a turn, a fire-and-forget pass distils open questions Jeff genuinely wants to ask, surfaces them in the prompt, and marks them answered when you reply. Off ⇒ no extra store/LLM calls (byte-identical to today). Adds the `/mind` command. |
| `JEFF_CURIOSITY_EVERY_TURNS` | `1` | Run the curiosity detection pass every N turns per peer (throttle the extra LLM call). |
| `JEFF_CURIOSITY_MAX_OPEN` | `6` | Cap on how many open questions are injected into the prompt's "You're curious about" block. |
| `JEFF_CURIOSITY_MAX_NEW` | `3` | Cap on new questions stored per detection pass. |
| `JEFF_MOOD_ENABLED` | `false` | Mood drive: Jeff carries a short-lived, self-chosen affective state set via the `set_mood` tool, which colours how it talks (an additive "How you're feeling right now" block) until it fades. Jeff authors its own mood definitions with `define_mood` — there is no seed palette, so the feature is inert until the first definition. Off ⇒ no extra store/tools (byte-identical to today). Adds the `/mood` command and a mood section to `/mind`. Needs `JEFF_TOOLS_ENABLED`. |
| `JEFF_MOOD_DEFAULT_HOURS` | `6` | How many hours a mood lasts when Jeff doesn't specify a duration. |
| `JEFF_MOOD_MAX_HOURS` | `48` | Hard ceiling on any single mood's duration (the tool clamps to this). |
| `JEFF_MOOD_MAX_CHARS` | `2000` | Cap on a stored mood definition's length (tool-enforced, with a higher hard byte cap in the store as defence-in-depth). |
| `JEFF_REMEMBER_ENABLED` | `false` | Explicit/pinned memory: lets Jeff deliberately keep a note via the `remember` tool, and you pin one via `/remember <text>` — both write to one shared store and ride in the prompt as an additive "Things to remember" block (always injected, never decays). Distinct from the automatic per-turn memory and the auto-distilled reflection facts. Off ⇒ no extra store/tool/command (byte-identical to today). Adds the `/remember` command and a section to `/mind`. Needs `JEFF_TOOLS_ENABLED` for the tool. |
| `JEFF_REMEMBER_MAX_ITEMS` | `20` | Cap on how many pins are injected into the prompt and listed in `/mind` (most recent first). |
| `JEFF_REMEMBER_MAX_CHARS` | `2000` | Cap on a single pinned note's length (tool/command-enforced, with a higher hard byte cap in the store as defence-in-depth). |
| `JEFF_APPRAISAL_ENABLED` | `false` | Appraisal/reward drive: after a turn, a fire-and-forget pass rates the exchange against four standing drives (connection, novelty, competence, self-expression) and nudges their levels, which decay toward a baseline over time and ride in the prompt as an additive "Your drives right now" block. An affective state machine, not RL — wipe the store and it's gone. Off ⇒ no extra store/LLM calls (byte-identical to today). Adds a drives section to `/mind`. |
| `JEFF_APPRAISAL_EVERY_TURNS` | `1` | Run the appraisal pass every N turns per peer (throttle the extra LLM call). |
| `JEFF_DRIVE_DECAY_HALF_LIFE_HOURS` | `24` | Half-life (hours) for a drive level relaxing back toward its baseline. Decay is computed lazily at read time, no background loop. |
| `JEFF_DRIVES_MAX_CHARS` | `2000` | Cap on the rendered "Your drives right now" prompt block. |
| `JEFF_PROACTIVE_ENABLED` | `false` | Proactive messaging: a heartbeat that lets Jeff reach out **unprompted** when it genuinely has something to say. Most ticks are silent and cost nothing — the model is only consulted (as a silence-default gatekeeper) when the `connection` drive has decayed below the threshold *and* there's a concrete candidate (an open curiosity). **Needs `JEFF_APPRAISAL_ENABLED` (the pressure signal) and `JEFF_CURIOSITY_ENABLED` (the candidates) on** — inert otherwise. Reaches out over Ensemble only (not Signal, yet). Off ⇒ no extra store/loop (byte-identical to today). Adds `/mute`+`/unmute` and a proactive section to `/mind`. |
| `JEFF_PROACTIVE_INTERVAL_S` | `300` | How often the loop wakes to *check* (not how often it messages — the gates decide that). |
| `JEFF_PROACTIVE_CONNECTION_THRESHOLD` | `0.35` | Reach-out pressure bar: the loop only considers messaging when the `connection` drive has decayed below this (it rests low at 0.2 and is bumped up by warm exchanges, so this is "it's been a while"). |
| `JEFF_PROACTIVE_MIN_GAP_S` | `1800` | Hard minimum gap between unprompted messages — an anti-machine-gun fuse, not a politeness cooldown. |
| `JEFF_PROACTIVE_PRESENCE_TTL_S` | `3600` | How long after the operator's last inbound event Jeff still treats them as reachable. Generous so a reconnect-after-silence is the natural moment to reach out (the SDK has no transport-presence signal; recent activity is the proxy). |
| `JEFF_SIGNAL_ENABLED` | `false` | Signal front door: a second inbound channel where you text Jeff's dedicated Signal number and it replies on the same thread, reusing the whole turn pipeline (memory, tools, drives). Off ⇒ no Signal client/loop (byte-identical to today). Needs a registered number + a running [signal-cli-rest-api](https://github.com/bbernhard/signal-cli-rest-api) in `json-rpc` mode. |
| `JEFF_SIGNAL_API_URL` | `http://localhost:8080` | Base URL of the signal-cli-rest-api instance. Startup fails fast if Signal is enabled but this is empty. |
| `JEFF_SIGNAL_NUMBER` | _empty_ | Jeff's own registered Signal number (E.164). **Required** when `JEFF_SIGNAL_ENABLED=true` (startup fails fast if missing). |
| `JEFF_SIGNAL_ALLOWLIST` | _empty_ | Comma-separated operator phone numbers (E.164) allowed to talk to Jeff over Signal. Empty = answer nobody (default-deny). Signal authenticates the sender; this authorises it. |
| `JEFF_SIGNAL_POLL_INTERVAL` | `1.0` | Seconds between signal-cli receive polls. |

## Commands

Slash-commands are **owned by the Ensemble daemon**, not parsed out of chat text.
At registration Jeff *declares* the commands it handles; the daemon parses
`/command` input, routes invocations to Jeff as control events, and merges Jeff's
reply with its own built-ins. A command and its reply are control traffic: they
**never touch memory** (so they don't pollute recall or the recent window) and
**never reach the model**. A handler that fails returns a safe apology, never
exception text.

The daemon uses **augment dispatch** — a command both it and Jeff handle runs
*both* legs. So `/clear` is one keystroke with two effects: the daemon's built-in
clears your local transcript while Jeff's leg resets its working memory window.

| Command | Effect |
| --- | --- |
| `/clear` | **Session reset** — drops the active thread from Jeff's recent window while keeping long-term semantic memory (older facts can still resurface via recall). Augments the daemon's built-in transcript clear. |
| `/forget` | **Hard wipe** — permanently deletes every stored message for you. Irreversible, so it's confirm-gated: send `/forget yes` to actually wipe. Jeff-only; no daemon counterpart. |
| `/stats` | Stored-message counts (you + all peers), process uptime, active provider/model, and system-prompt source. No secrets. |
| `/debug` | Deterministic introspection of Jeff's working context: effective system prompt, session cutoff, recent window, and what recall would surface (with cosine distances). `/debug prompt` and `/debug recall <query>` for the detail views. |
| `/mind` | What Jeff is curious about, its persona/mood/drives, and (when proactive is on) its reach-out state. Declared when any of curiosity / reflection / mood / remember / appraisal / proactive is on. |
| `/mute` | Stop Jeff reaching out **unprompted** for a while (`/mute`, or `/mute 2h` / `30m` / `1d`). Reactive replies are unaffected. Only declared when `JEFF_PROACTIVE_ENABLED` is on. |
| `/unmute` | Lift a `/mute` so Jeff can reach out again. Only declared when `JEFF_PROACTIVE_ENABLED` is on. |

`/help` and `/whoami` are the daemon's built-ins (it aggregates every service's
commands and reports node identity) — Jeff no longer declares them.

## Tools

When `JEFF_TOOLS_ENABLED` is on, the LLM can call registered tools mid-turn: the
turn handler runs an execute-and-loop (provider → tool calls → results fed back
→ repeat, bounded by `JEFF_MAX_TOOL_ITERS`). Only the final assistant message is
sent to the peer and stored in memory — intermediate tool chatter is working
state, not a conversational turn. Every tool failure (unknown tool, bad args,
raise, timeout) becomes a short safe `error: …` string for the model; a tool
fault never crashes the turn, and no exception text reaches the model.

At startup a **capabilities addendum** is appended to the active system prompt
(file / env / built-in default) describing the registered tools and how to use
them, plus a note that the chat client renders Markdown (so Jeff writes
`[label](url)` links and cites search results). The addendum tracks the actual
registry — it never advertises a tool that isn't enabled — and the search
guidance only appears when a search tool is registered.

Tools available today:

- **`get_time`** — current UTC time (zero-dependency built-in).
- **`web_search` / `image_search`** — query the self-hosted [SearXNG](https://docs.searxng.org/) metasearch proxy (enable with `JEFF_SEARCH_ENABLED`). Jeff only ever talks to SearXNG, and returns **links + text only**: it does not fetch the result pages or image bytes (auto-fetching would re-leak interest to third parties and is an SSRF vector). The model surfaces citations the operator clicks.

## Memory

Jeff stores every user/assistant message it sees, with a vector embedding, in a single `messages` table. Each turn:

1. Embed the incoming message.
2. Recall the top-k semantically similar prior messages from the same peer.
3. Pull the most-recent N messages from the same peer.
4. Build the chat history: `[system] + recall + recent + user_turn` (deduplicated).
5. Send to Ollama, reply back to the peer, store both sides in memory.

The schema is created idempotently on first connect, so an empty database is fine.

## Tests

```bash
pip install -e .[dev]
pytest tests/ -v
```

The memory tests spin up an ephemeral pgvector Postgres via `testcontainers` (Docker required); they're skipped automatically if Docker isn't available. The other tests have no external dependencies.

## Constraints to know about (v0)

- **`keypair_seed` is silently ignored by the daemon** (current daemon limitation). Register once, capture `handle.address`, persist the resulting `E…` address as Jeff's identity. Daemon's keystore is keyed by service name and returns the same keypair on reconnect.
- **Outbound chat travels under the daemon's NODE identity**, not Jeff's registered service identity. Peers see "from: node E…", not "from: jeff E…", until the per-service outbound dispatch follow-up lands.
- **Daemon event queue is 256-deep, drops oldest** under sustained backpressure (no on-wire signal). Jeff handles this by dispatching each turn as `asyncio.create_task`, so the events iterator keeps draining while the LLM call is in flight.
- **Allowlist is enforced in Jeff, not the daemon.** The daemon's signaling layer in v0 only honors the `contacts` ACL tier; `allowlist` is advisory there. Jeff drops chats from non-allowlisted peers explicitly (logged as "ignoring chat from non-allowlisted peer=…").
- **TLS-insecure is a no-op in the Python client.** For self-signed CAs, set `GRPC_DEFAULT_SSL_ROOTS_FILE_PATH` or install the CA into the system trust store.

## Out of scope (today)

Push notifications, streaming token-by-token replies, multi-modal (vision/image
input), file transfer, and proactive messaging — tracked in the capabilities
phase. Tool use itself has landed (function-calling foundation + SearXNG
web/image search); kubectl/shell/file-read tools are deliberately **not** built.
