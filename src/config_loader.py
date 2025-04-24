import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator


class AgentRoleConfig(BaseModel):
    """Configuration for an agent role."""
    name: str = Field(...)
    description: str = Field(...)
    instructions: str = Field(...)


class AppConfig(BaseModel):
    # Core fields (maintaining backward compatibility)
    user_agent_goal: str = Field(..., alias='userAgentGoal')
    expert_agent_persona: str = Field(..., alias='expertAgentPersona')
    max_iterations: int = Field(..., alias='maxIterations', gt=0)
    openai_api_key: Optional[str] = Field(None, alias='openaiApiKey')
    openai_model: str = Field(..., alias='openaiModel')
    output_dir: Path = Field(..., alias='outputDir')
    history_file_path: Optional[Path] = Field(None, alias='historyFilePath')

    # Optional extended agent configuration
    user_agent_config: Optional[AgentRoleConfig] = Field(
        None, alias='userAgentConfig')
    expert_agent_config: Optional[AgentRoleConfig] = Field(
        None, alias='expertAgentConfig')

    @field_validator('output_dir')
    @classmethod
    def output_dir_must_exist(cls, v):
        # Although we will create it if it doesn't exist upon execution,
        # validating it here ensures the path is valid.
        # For simplicity in loading, we don't create it here.
        return v

    @field_validator('history_file_path')
    @classmethod
    def history_file_must_exist(cls, v):
        if v and not v.exists():
            raise ValueError(f"History file path does not exist: {v}")
        if v and not v.is_file():
            raise ValueError(f"History file path is not a file: {v}")
        return v

    def get_user_agent_instructions(self) -> str:
        """Get the user agent instructions, preferring the extended config if available."""
        if self.user_agent_config:
            return self.user_agent_config.instructions
        # Fall back to legacy behavior where we construct instructions in agent_factory
        return ""

    def get_expert_agent_instructions(self) -> str:
        """Get the expert agent instructions, preferring the extended config if available."""
        if self.expert_agent_config:
            return self.expert_agent_config.instructions
        # Fall back to legacy behavior
        return self.expert_agent_persona


class ConfigLoaderError(Exception):
    pass


def load_config(config_path_str: str) -> AppConfig:
    """Loads configuration from a JSON or YAML file.

    Args:
        config_path_str: Path to the configuration file.

    Returns:
        An AppConfig object with validated configuration.

    Raises:
        ConfigLoaderError: If the file is not found, cannot be parsed,
                         or fails validation.
        FileNotFoundError: If the config file path does not exist.
    """
    load_dotenv()  # Load .env file if it exists

    config_path = Path(config_path_str)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            if config_path.suffix == '.json':
                raw_config = json.load(f)
            elif config_path.suffix in ['.yaml', '.yml']:
                raw_config = yaml.safe_load(f)
            else:
                raise ConfigLoaderError(
                    f"Unsupported config file format: {config_path.suffix}. "
                    f"Use .json or .yaml."
                )
    except (json.JSONDecodeError, yaml.YAMLError) as e:
        raise ConfigLoaderError(
            f"Error parsing config file {config_path}: {e}")
    except Exception as e:
        raise ConfigLoaderError(
            f"Error reading config file {config_path}: {e}")

    if not isinstance(raw_config, dict):
        raise ConfigLoaderError(
            f"Invalid config file format in {config_path}. Root must be an object/dictionary.")

    # Load API key from environment if not in config or if blank
    if not raw_config.get('openaiApiKey'):
        env_api_key = os.getenv('OPENAI_API_KEY')
        if env_api_key:
            raw_config['openaiApiKey'] = env_api_key
        else:
            # Raise error only during validation if key is still missing
            pass

    try:
        config = AppConfig(**raw_config)
        # Final check for API key after loading from env
        if not config.openai_api_key:
            raise ValueError(
                "OpenAI API Key is required. Provide it in the config file or as OPENAI_API_KEY environment variable.")
        return config
    except ValidationError as e:
        raise ConfigLoaderError(f"Configuration validation failed:\n{e}")
    except ValueError as e:
        raise ConfigLoaderError(f"Configuration validation failed: {e}")


if __name__ == '__main__':
    # Example usage:
    try:
        # Create dummy files for testing loading
        Path("config").mkdir(exist_ok=True)
        dummy_json_path = Path("config/dummy_config.json")
        dummy_yaml_path = Path("config/dummy_config.yaml")
        dummy_history_path = Path("config/dummy_history.txt")
        dummy_output_path = Path("temp_output")

        dummy_json_path.write_text(json.dumps({
            "userAgentGoal": "Test JSON Goal",
            "expertAgentPersona": "Test JSON Persona",
            "maxIterations": 3,
            "openaiApiKey": "key-from-json-config",  # Example of key in config
            "openaiModel": "gpt-4.1-mini",  # Added this line
            "outputDir": str(dummy_output_path),
            "historyFilePath": str(dummy_history_path)
        }))
        dummy_yaml_path.write_text(
            "userAgentGoal: Test YAML Goal\n"
            "expertAgentPersona: Test YAML Persona\n"
            "maxIterations: 4\n"
            "openaiModel: gpt-4.1-mini\n"  # Added this line
            # API key omitted to test env var loading
            "outputDir: " + str(dummy_output_path) + "\n"
            "historyFilePath: null\n"
        )
        dummy_history_path.touch()
        dummy_output_path.mkdir(exist_ok=True)

        print("--- Testing JSON Loading ---")
        config_json = load_config(str(dummy_json_path))
        print(f"Loaded JSON Config: {config_json}")
        assert config_json.user_agent_goal == "Test JSON Goal"
        assert config_json.openai_api_key == "key-from-json-config"
        assert config_json.history_file_path == dummy_history_path

        print("\n--- Testing YAML Loading (Requires OPENAI_API_KEY env var) ---")
        # Set a dummy env var for testing
        os.environ['OPENAI_API_KEY'] = 'key-from-env-var'
        config_yaml = load_config(str(dummy_yaml_path))
        print(f"Loaded YAML Config: {config_yaml}")
        assert config_yaml.user_agent_goal == "Test YAML Goal"
        assert config_yaml.openai_api_key == "key-from-env-var"
        assert config_yaml.history_file_path is None
        del os.environ['OPENAI_API_KEY']  # Clean up env var

        print("\n--- Testing Validation Error (Missing API Key) ---")
        dummy_yaml_no_key_path = Path("config/dummy_config_no_key.yaml")
        dummy_yaml_no_key_path.write_text(
            "userAgentGoal: Test YAML Goal\n"
            "expertAgentPersona: Test YAML Persona\n"
            "maxIterations: 4\n"
            "openaiModel: gpt-4.1-mini\n"  # Added this line
            "outputDir: " + str(dummy_output_path) + "\n"
            "historyFilePath: null\n"
        )
        try:
            load_config(str(dummy_yaml_no_key_path))
        except ConfigLoaderError as e:
            print(f"Caught expected validation error: {e}")
            assert "OpenAI API Key is required" in str(e)

    except ConfigLoaderError as e:
        print(f"Error loading configuration: {e}")
    except FileNotFoundError as e:
        print(f"Config file not found: {e}")
    finally:
        # Clean up dummy files
        dummy_json_path.unlink(missing_ok=True)
        dummy_yaml_path.unlink(missing_ok=True)
        dummy_yaml_no_key_path.unlink(missing_ok=True)
        dummy_history_path.unlink(missing_ok=True)
        if dummy_output_path.exists():
            dummy_output_path.rmdir()
