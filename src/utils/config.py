"""
Configuration loader for the feature store.

Loads YAML config files with environment variable interpolation.
Environment variables in the format ${VAR_NAME:-default} are replaced
at load time, allowing the same config files to work across environments.
"""

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load .env file from project root if it exists
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"
if _ENV_FILE.exists():
    load_dotenv(_ENV_FILE)


def _resolve_env_vars(value: str) -> str:
    """Replace ${VAR:-default} patterns with environment variable values.

    Supports two forms:
        ${VAR}          — raises KeyError if VAR is not set
        ${VAR:-default} — falls back to 'default' if VAR is not set
    """
    pattern = r"\$\{([^}:]+)(?::-(.*?))?\}"

    def _replace(match: re.Match) -> str:
        var_name = match.group(1)
        default = match.group(2)
        env_value = os.environ.get(var_name)
        if env_value is not None:
            return env_value
        if default is not None:
            return default
        raise KeyError(
            f"Environment variable '{var_name}' is not set and no default provided"
        )

    return re.sub(pattern, _replace, value)


def _walk_and_resolve(obj: Any) -> Any:
    """Recursively walk a parsed YAML structure and resolve env vars in strings."""
    if isinstance(obj, str):
        return _resolve_env_vars(obj)
    elif isinstance(obj, dict):
        return {k: _walk_and_resolve(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_walk_and_resolve(item) for item in obj]
    return obj


def load_config(config_name: str) -> dict:
    """Load a YAML config file from the configs/ directory.

    Args:
        config_name: Name of the config file (without .yaml extension).
                     E.g., 'kafka', 'redis', 'snowflake', 'iceberg'.

    Returns:
        Parsed and environment-resolved configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If a required environment variable is not set.
    """
    config_dir = _PROJECT_ROOT / "configs"
    config_path = config_dir / f"{config_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            f"Available configs: {[f.stem for f in config_dir.glob('*.yaml')]}"
        )

    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)

    return _walk_and_resolve(raw_config)


def get_kafka_config() -> dict:
    """Convenience loader for Kafka configuration."""
    return load_config("kafka")


def get_redis_config() -> dict:
    """Convenience loader for Redis configuration."""
    return load_config("redis")


def get_snowflake_config() -> dict:
    """Convenience loader for Snowflake configuration."""
    return load_config("snowflake")


def get_iceberg_config() -> dict:
    """Convenience loader for Iceberg configuration."""
    return load_config("iceberg")
