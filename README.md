# Obsidian Notes Organiser

Python script that organises your Obsidian notes (also supports text files) into folders using Generative AI-powered content analysis. It supports multiple LLM providers:

- OpenAI (GPT-3.5-turbo)
- Anthropic (Claude 3 Haiku)
- Google (Gemini 1.5 Flash)

## Requirements

- Python 3.x
- API keys for your chosen LLM provider(s):
  - OpenAI API key
  - Anthropic API key
  - Google API key

## Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Add your API keys to the `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   GOOGLE_API_KEY=your_google_api_key
   ```

## Usage

### Command Line Arguments

```bash
python main.py [-h] [-d DIRECTORY] [-p {openai,claude,gemini}] [--dry-run]

options:
  -h, --help            Show this help message and exit
  -d DIRECTORY, --directory DIRECTORY
                        Path to your Obsidian notes directory
  -p {openai,claude,gemini}, --provider {openai,claude,gemini}
                        LLM provider to use (default: openai)
  --dry-run            Show what would be done without actually moving files
```

### Examples

1. Basic usage (will prompt for directory and provider):
   ```bash
   python main.py
   ```

2. Specify directory and provider:
   ```bash
   python main.py -d /path/to/notes -p gemini
   ```

3. Dry run with OpenAI:
   ```bash
   python main.py -d /path/to/notes --dry-run
   ```

The script will analyse your notes and organise them into folders based on their content. Each note will be moved to a category folder suggested by the chosen LLM.

## Features

- Multi-LLM support (OpenAI, Claude, Gemini)
- Command-line interface with optional arguments
- Automatic folder creation based on content analysis
- Support for both Markdown (.md) and text (.txt) files
- Error handling for file operations
- UTF-8 encoding support
- Progress feedback during organization
- Dry run mode for testing
