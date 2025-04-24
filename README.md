# Expert Agent Conversation Engine

This project implements a multi-agent system designed to simulate conversations between a configurable User Agent and an Expert Agent. The goal is to facilitate knowledge discovery and mining on specific topics by automating the dialogue process.

It utilizes the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) to manage agent interactions and LLM calls.

## Features

- **Configurable Agents:** Define distinct roles, goals, and detailed instructions for both the User Agent (asking questions) and the Expert Agent (providing answers) via YAML/JSON configuration files.
- **Turn-Based Simulation:** Runs a conversation loop for a specified number of iterations.
- **Transcript Saving:** Automatically saves the full conversation transcript to a plain text file in the specified output directory, including timestamps in the filename.
- **Takeaway Generation:** Uses the Expert Agent (or potentially a dedicated summarizer in the future) to distill key takeaways from the completed conversation.
- **Takeaway Saving:** Saves the generated takeaways to a separate text file, linked to the original transcript filename.
- **CLI Interface:** Provides a command-line tool (`src/cli.py`) to run conversations using a specified configuration file and optionally override the number of turns.
- **OpenAI Agents SDK Integration:** Leverages the `agents` library for agent definition and execution (`Runner.run`).

## Requirements

- Python 3.9+
- [Poetry](https://python-poetry.org/) for dependency management.
- OpenAI API Key
- Required Python packages (defined in `pyproject.toml`):
  - `openai-agents`
  - `pydantic`
  - `pyyaml`
  - `python-dotenv`

## Configuration

Conversations are configured using YAML or JSON files (e.g., `config/config.example.yaml`, `config/water_kefir_config.yaml`).

**Key Configuration Fields:**

- `userAgentGoal` (str): The primary objective the User Agent is trying to achieve through the conversation.
- `expertAgentPersona` (str): A basic description of the expert agent (used if `expertAgentConfig` is not provided).
- `maxIterations` (int): The maximum number of conversation turns (User question + Expert answer = 1 turn).
- `openaiModel` (str): The OpenAI model to use for the agents (e.g., `gpt-4.1-mini`, `gpt-4-turbo`).
- `outputDir` (str): Path to the directory where transcripts and takeaways will be saved.
- `historyFilePath` (str, optional): Path to a file containing initial conversation history to load (if any).
- `userAgentConfig` (object, optional): Detailed configuration for the User Agent:
  - `name` (str): Agent name.
  - `description` (str): Brief description.
  - `instructions` (str): Detailed prompt/instructions for the User Agent.
- `expertAgentConfig` (object, optional): Detailed configuration for the Expert Agent:
  - `name` (str): Agent name.
  - `description` (str): Brief description.
  - `instructions` (str): Detailed prompt/instructions for the Expert Agent.

**API Key:**

- The OpenAI API key is required.
- It's recommended to set it as an environment variable: `OPENAI_API_KEY`.
- You can create a `.env` file in the project root:
  ```
  OPENAI_API_KEY=your_actual_api_key_here
  ```
- The application will load the key from the `.env` file or the environment.

## Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2.  **Install Poetry (if you don't have it):**
    - Follow the instructions on the [official Poetry website](https://python-poetry.org/docs/#installation).
3.  **Install dependencies using Poetry:**
    - Poetry will automatically create a virtual environment if one doesn't exist.
    ```bash
    poetry install
    ```
4.  **Set up OpenAI API Key:**
    - Create a `.env` file in the project root.
    - Add the line `OPENAI_API_KEY=your_actual_api_key_here` with your key.
5.  **Activate the virtual environment (optional but recommended):**
    ```bash
    poetry shell
    ```

## Usage

Run conversations using the Command Line Interface (CLI).
Make sure your Poetry environment is active (run `poetry shell`) or prepend commands with `poetry run`.

```bash
python -m src.cli <path_to_config_file.yaml> [--max-iterations N]
# Or using poetry run:
# poetry run python -m src.cli <path_to_config_file.yaml> [--max-iterations N]
```

**Examples:**

- Run using the example quantum computing config for 10 turns (as defined in the file):
  ```bash
  python -m src.cli config/config.example.yaml
  ```
- Run using the water kefir config, overriding to 5 turns:
  ```bash
  python -m src.cli config/water_kefir_config.yaml --max-iterations 5
  ```

**Output:**

- Conversation progress will be logged to the console.
- The full transcript will be saved as a `.txt` file in the directory specified by `outputDir` in your config.
- The generated takeaways will be saved as a separate `_takeaways.txt` file in the same directory.

## Project Structure

```
├── config/               # Configuration files
│   ├── config.example.yaml
│   └── water_kefir_config.yaml
├── output/               # Default output directory for transcripts/takeaways
├── src/                  # Source code
│   ├── __init__.py
│   ├── agent_factory.py  # Creates User and Expert agents
│   ├── cli.py            # Command Line Interface entry point
│   ├── config_loader.py  # Loads and validates configuration
│   ├── conversation.py   # Manages conversation history
│   ├── context.py        # Defines conversation context
│   ├── engine.py         # Orchestrates the conversation loop
│   └── output_manager.py # Handles saving transcripts and takeaways
├── .env                  # Environment variables (contains OPENAI_API_KEY, gitignored)
├── .gitignore
├── pyproject.toml        # Defines project metadata and dependencies (for Poetry)
├── poetry.lock           # Exact versions of dependencies
├── README.md             # This file
```
