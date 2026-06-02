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
ollama pull nomic-embed-text

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
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model (embeddings stay local) |
| `OLLAMA_EMBED_DIM` | `768` | Embedding vector dimensionality (must match the model) |
| `MEMORY_RECALL_K` | `5` | How many semantically-similar past messages to retrieve per turn |
| `MEMORY_RECENT_TURNS` | `10` | How many most-recent messages to include per turn |
| `JEFF_TOOLS_ENABLED` | `true` | Master switch for tool use. When off (or the registry is empty) the turn loop is byte-identical to the no-tools single-shot path. |
| `JEFF_MAX_TOOL_ITERS` | `5` | Max provider↔tool round-trips per turn before a graceful "tool-use limit" reply. |
| `JEFF_TOOL_TIMEOUT_S` | `30` | Per-tool execution timeout (seconds). A timed-out tool returns a safe error to the model; the turn survives. |
| `JEFF_SEARCH_ENABLED` | `false` | Enable the SearXNG-backed `web_search` / `image_search` tools (also requires `JEFF_TOOLS_ENABLED`). |
| `JEFF_SEARXNG_URL` | `http://localhost:8888` | SearXNG JSON-API base URL. The real in-cluster URL is **not committed** — it arrives via the serves ConfigMap. Startup fails fast if search is enabled but this is empty. |
| `JEFF_SEARXNG_AUTH` | _none_ | Optional full `Authorization` header value if the SearXNG instance requires auth (e.g. `Basic …` / `Bearer …`). Never logged. |

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
