import pytest

from jeff.config import Config, ConfigError


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
