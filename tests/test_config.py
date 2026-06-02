import pytest

from jeff.config import _SYSTEM_PROMPT_MAX_CHARS, Config, ConfigError
from jeff.prompt import SYSTEM_PROMPT


def test_from_env_requires_db_url():
    with pytest.raises(ConfigError):
        Config.from_env({})


def test_from_env_defaults():
    cfg = Config.from_env({"JEFF_DB_URL": "postgresql://x"})
    assert cfg.name == "jeff"
    assert cfg.allowlist == []
    assert cfg.chat_model == "gemma3:12b-it-qat"
    assert cfg.embed_model == "nomic-embed-text"
    assert cfg.embed_dim == 768
    assert cfg.recall_k == 5
    assert cfg.recent_turns == 10
    assert cfg.socket == "/run/ensemble/sock"
    assert cfg.auth_seed_path is None


def test_from_env_csv_allowlist():
    cfg = Config.from_env(
        {
            "JEFF_DB_URL": "postgresql://x",
            "JEFF_ALLOWLIST": "EaaaA, EbbbB ,EcccC",
        }
    )
    assert cfg.allowlist == ["EaaaA", "EbbbB", "EcccC"]


def test_embed_dim_bounds():
    # W3 #20123205: env-supplied OLLAMA_EMBED_DIM must be bounded so a
    # hostile or typo value can't reach the schema DDL (was format()-ed
    # into vector(N) literally). Below the minimum and above the max
    # both fail at config load.
    with pytest.raises(ConfigError):
        Config.from_env({"JEFF_DB_URL": "postgresql://x", "OLLAMA_EMBED_DIM": "0"})
    with pytest.raises(ConfigError):
        Config.from_env({"JEFF_DB_URL": "postgresql://x", "OLLAMA_EMBED_DIM": "-1"})
    with pytest.raises(ConfigError):
        Config.from_env({"JEFF_DB_URL": "postgresql://x", "OLLAMA_EMBED_DIM": "99999999"})
    with pytest.raises(ConfigError):
        Config.from_env({"JEFF_DB_URL": "postgresql://x", "OLLAMA_EMBED_DIM": "8193"})
    # In-bounds values still parse.
    cfg = Config.from_env({"JEFF_DB_URL": "postgresql://x", "OLLAMA_EMBED_DIM": "8192"})
    assert cfg.embed_dim == 8192
    cfg = Config.from_env({"JEFF_DB_URL": "postgresql://x", "OLLAMA_EMBED_DIM": "1"})
    assert cfg.embed_dim == 1


def test_embed_dim_rejects_non_integer():
    with pytest.raises(ConfigError):
        Config.from_env({"JEFF_DB_URL": "postgresql://x", "OLLAMA_EMBED_DIM": "abc"})


def test_default_provider_is_ollama():
    cfg = Config.from_env({"JEFF_DB_URL": "postgresql://x"})
    assert cfg.llm_provider == "ollama"
    assert cfg.chat_model == "gemma3:12b-it-qat"
    assert cfg.xai_api_key is None
    assert cfg.xai_base_url == "https://api.x.ai/v1"


def test_unknown_provider_rejected():
    with pytest.raises(ConfigError, match="JEFF_LLM_PROVIDER"):
        Config.from_env({"JEFF_DB_URL": "postgresql://x", "JEFF_LLM_PROVIDER": "openai"})


def test_grok_requires_api_key():
    with pytest.raises(ConfigError, match="XAI_API_KEY"):
        Config.from_env({"JEFF_DB_URL": "postgresql://x", "JEFF_LLM_PROVIDER": "grok"})


def test_grok_provider_with_key():
    cfg = Config.from_env(
        {
            "JEFF_DB_URL": "postgresql://x",
            "JEFF_LLM_PROVIDER": "grok",
            "XAI_API_KEY": "xai-secret",
        }
    )
    assert cfg.llm_provider == "grok"
    assert cfg.xai_api_key == "xai-secret"
    # Default grok model when neither override is set.
    assert cfg.chat_model == "grok-4"


def test_generic_chat_model_overrides_provider_default():
    # JEFF_CHAT_MODEL wins under either provider.
    cfg = Config.from_env(
        {
            "JEFF_DB_URL": "postgresql://x",
            "JEFF_LLM_PROVIDER": "grok",
            "XAI_API_KEY": "k",
            "JEFF_CHAT_MODEL": "grok-4.3",
        }
    )
    assert cfg.chat_model == "grok-4.3"
    cfg = Config.from_env(
        {"JEFF_DB_URL": "postgresql://x", "JEFF_CHAT_MODEL": "llama3"}
    )
    assert cfg.llm_provider == "ollama"
    assert cfg.chat_model == "llama3"


def test_ollama_chat_model_still_honoured():
    cfg = Config.from_env(
        {"JEFF_DB_URL": "postgresql://x", "OLLAMA_CHAT_MODEL": "phi3:mini"}
    )
    assert cfg.chat_model == "phi3:mini"


def test_provider_name_normalised():
    cfg = Config.from_env(
        {"JEFF_DB_URL": "postgresql://x", "JEFF_LLM_PROVIDER": " Grok ", "XAI_API_KEY": "k"}
    )
    assert cfg.llm_provider == "grok"


def test_system_prompt_defaults_to_builtin():
    cfg = Config.from_env({"JEFF_DB_URL": "postgresql://x"})
    assert cfg.system_prompt == SYSTEM_PROMPT
    assert cfg.system_prompt_source == "default"


def test_system_prompt_inline_env_overrides_verbatim():
    cfg = Config.from_env(
        {"JEFF_DB_URL": "postgresql://x", "JEFF_SYSTEM_PROMPT": "You are Bob."}
    )
    # Verbatim — nothing appended (no forced guardrail; single-user assistant).
    assert cfg.system_prompt == "You are Bob."
    assert cfg.system_prompt_source == "env"


def test_system_prompt_inline_env_is_stripped():
    cfg = Config.from_env(
        {"JEFF_DB_URL": "postgresql://x", "JEFF_SYSTEM_PROMPT": "  spaced  \n"}
    )
    assert cfg.system_prompt == "spaced"


def test_system_prompt_blank_env_falls_back_to_default():
    # An empty/whitespace env value is treated as "unset" (k8s may inject "").
    cfg = Config.from_env(
        {"JEFF_DB_URL": "postgresql://x", "JEFF_SYSTEM_PROMPT": "   "}
    )
    assert cfg.system_prompt == SYSTEM_PROMPT
    assert cfg.system_prompt_source == "default"


def test_system_prompt_file_used_and_stripped(tmp_path):
    p = tmp_path / "prompt.txt"
    # Trailing newline (e.g. from `op read`) must be stripped.
    p.write_text("You are a file-defined assistant.\n", encoding="utf-8")
    cfg = Config.from_env(
        {"JEFF_DB_URL": "postgresql://x", "JEFF_SYSTEM_PROMPT_FILE": str(p)}
    )
    assert cfg.system_prompt == "You are a file-defined assistant."
    assert cfg.system_prompt_source == "file"


def test_system_prompt_file_beats_inline_env(tmp_path):
    p = tmp_path / "prompt.txt"
    p.write_text("from file", encoding="utf-8")
    cfg = Config.from_env(
        {
            "JEFF_DB_URL": "postgresql://x",
            "JEFF_SYSTEM_PROMPT_FILE": str(p),
            "JEFF_SYSTEM_PROMPT": "from env",
        }
    )
    assert cfg.system_prompt == "from file"
    assert cfg.system_prompt_source == "file"


def test_system_prompt_missing_file_raises(tmp_path):
    missing = tmp_path / "nope.txt"
    with pytest.raises(ConfigError, match="JEFF_SYSTEM_PROMPT_FILE"):
        Config.from_env(
            {"JEFF_DB_URL": "postgresql://x", "JEFF_SYSTEM_PROMPT_FILE": str(missing)}
        )


def test_system_prompt_empty_file_raises(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("   \n", encoding="utf-8")
    with pytest.raises(ConfigError, match="empty"):
        Config.from_env(
            {"JEFF_DB_URL": "postgresql://x", "JEFF_SYSTEM_PROMPT_FILE": str(p)}
        )


def test_system_prompt_over_length_rejected():
    with pytest.raises(ConfigError, match="too long"):
        Config.from_env(
            {
                "JEFF_DB_URL": "postgresql://x",
                "JEFF_SYSTEM_PROMPT": "x" * (_SYSTEM_PROMPT_MAX_CHARS + 1),
            }
        )


def test_from_env_overrides():
    cfg = Config.from_env(
        {
            "JEFF_DB_URL": "postgresql://x",
            "JEFF_NAME": "jeff-test",
            "MEMORY_RECALL_K": "3",
            "MEMORY_RECENT_TURNS": "4",
            "OLLAMA_EMBED_DIM": "1024",
        }
    )
    assert cfg.name == "jeff-test"
    assert cfg.recall_k == 3
    assert cfg.recent_turns == 4
    assert cfg.embed_dim == 1024


def test_tools_defaults():
    cfg = Config.from_env({"JEFF_DB_URL": "postgresql://x"})
    assert cfg.tools_enabled is True
    assert cfg.max_tool_iters == 5
    assert cfg.tool_timeout_s == 30.0


def test_tools_can_be_disabled():
    cfg = Config.from_env({"JEFF_DB_URL": "postgresql://x", "JEFF_TOOLS_ENABLED": "false"})
    assert cfg.tools_enabled is False


def test_search_defaults_off_with_localhost_url():
    cfg = Config.from_env({"JEFF_DB_URL": "postgresql://x"})
    assert cfg.search_enabled is False
    assert cfg.searxng_url == "http://localhost:8888"
    assert cfg.searxng_auth is None


def test_search_url_and_auth_parsed():
    cfg = Config.from_env(
        {
            "JEFF_DB_URL": "postgresql://x",
            "JEFF_SEARCH_ENABLED": "true",
            "JEFF_SEARXNG_URL": "http://searxng-service.searxng.svc:8080",
            "JEFF_SEARXNG_AUTH": "Bearer tok",
        }
    )
    assert cfg.search_enabled is True
    assert cfg.searxng_url == "http://searxng-service.searxng.svc:8080"
    assert cfg.searxng_auth == "Bearer tok"


def test_search_enabled_without_url_fails_fast():
    with pytest.raises(ConfigError, match="JEFF_SEARXNG_URL"):
        Config.from_env(
            {
                "JEFF_DB_URL": "postgresql://x",
                "JEFF_SEARCH_ENABLED": "true",
                "JEFF_SEARXNG_URL": "",
            }
        )
