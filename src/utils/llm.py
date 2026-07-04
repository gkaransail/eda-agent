import os
from pathlib import Path

import yaml


_HIERARCHY_PATH = Path(__file__).parents[2] / "llm_hierarchy.yaml"


def _build_provider(config: dict):
    provider = config["provider"]
    model = config["model"]
    api_key_env = config.get("api_key_env")
    api_key = os.getenv(api_key_env) if api_key_env else None

    # skip if key required but missing
    if api_key_env and not api_key:
        return None

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(api_key=api_key, model=model)

    if provider == "together":
        from langchain_together import ChatTogether
        return ChatTogether(api_key=api_key, model=model)

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            model=model,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=model,
        )

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEndpoint
        return HuggingFaceEndpoint(repo_id=model, huggingfacehub_api_token=api_key)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(api_key=api_key, model=model)

    raise ValueError(f"Unknown provider: '{provider}'")


def get_llm():
    if not _HIERARCHY_PATH.exists():
        raise FileNotFoundError(f"LLM hierarchy config not found: {_HIERARCHY_PATH}")

    with open(_HIERARCHY_PATH) as f:
        hierarchy = yaml.safe_load(f)

    providers_config = hierarchy.get("providers", [])
    if not providers_config:
        raise ValueError("llm_hierarchy.yaml has no providers defined.")

    # build only providers whose API keys are present
    available = []
    skipped = []
    for cfg in providers_config:
        llm = _build_provider(cfg)
        if llm is not None:
            available.append((cfg["name"], llm))
        else:
            skipped.append(cfg["name"])

    if skipped:
        print(f"[LLM] Skipped (missing API key): {', '.join(skipped)}")

    if not available:
        raise RuntimeError(
            "No LLM providers available. Add at least one API key to .env "
            "or enable Ollama locally."
        )

    names, llms = zip(*available)
    primary = llms[0]
    fallbacks = list(llms[1:])

    print(f"[LLM] Primary: {names[0]}")
    if fallbacks:
        print(f"[LLM] Fallback chain: {' → '.join(names[1:])}")

    if fallbacks:
        return primary.with_fallbacks(fallbacks)
    return primary
