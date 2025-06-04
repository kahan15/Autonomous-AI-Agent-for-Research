# Autonomous AI Research Agent

An advanced, autonomous AI agent system for research and decision support, built in Python. The agent autonomously conducts research, analyzes data, and provides actionable recommendations using a systematic Plan-Execute-Reflect workflow. It integrates with web sources and APIs, validates and synthesizes information, and generates comprehensive reports.

## Features

- **Autonomous research and data gathering** from diverse web sources (academic, tech blogs, news, documentation)
- **Plan-Execute-Reflect workflow** for robust, iterative research
- **Source validation and credibility scoring**
- **Content extraction and analysis** (including key findings and trends)
- **Comprehensive report generation** (Markdown and JSON)
- **Configurable via YAML and environment variables**
- **Logging and reproducibility**
- **Test script for validation**

## Project Structure

```
.
├── src/
│   └── research_agent/         # Core agent implementation
│       ├── base.py             # Abstract agent and workflow
│       ├── web_research_agent.py # Web research agent implementation
│       └── __init__.py         # Package exports
├── data/                      # Data storage (sources, extracted content, logs)
│   ├── sources.json
│   ├── extracted_content.json
│   ├── analysis.json
│   └── logs/
├── reports/                   # Generated research reports (Markdown, JSON)
├── config/
│   └── config.yaml            # Main configuration file
├── test_research.py           # Test script for the agent
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
└── .env                       # Environment variables (not committed)
```

## Setup

1. **Clone the repository**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure environment variables:**
   - Create a `.env` file in the root directory.
   - Add your Serper API key:
     ```
     SEARCH_API_KEY=your_serper_api_key
     ```
4. **Configure the agent:**
   - Edit `config/config.yaml` to adjust sources, workflow, security, and output options.

## Usage

### Command Line
Run a research task from the command line:
```bash
python src/main.py --task "What are the latest developments in large language models in 2024?" --config config/config.yaml
```
- `--task` (required): The research question or topic.
- `--config` (optional): Path to the YAML config file (default: `config/config.yaml`).

### Output
- Reports are saved in the `reports/` directory as both Markdown (`research_report.md`) and JSON (`research_report.json`).
- Data, logs, and intermediate results are stored in `data/`.

## Plan-Execute-Reflect Workflow
The agent follows a robust, iterative workflow:
1. **Plan:** Generate a stepwise research plan for the task.
2. **Execute:** For each step, gather sources, validate, extract, and analyze content.
3. **Reflect:** Assess results, log reflections, and adapt the plan if needed.
4. **Report:** Synthesize findings and generate a comprehensive report.

## Configuration
- **YAML:** `config/config.yaml` controls sources, validation, workflow, security, and output.
- **Environment Variables:** `.env` file (e.g., `SEARCH_API_KEY` for Serper API access).
- **Logging:** Logs are saved in `data/logs/` and to the console.

## Dependencies
See `requirements.txt` for all dependencies. Key packages:
- `requests`, `beautifulsoup4`, `feedparser`, `python-dotenv`, `pyyaml`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `langchain`, `biopython`, `validators`, and more.

## Testing
Run the included test script to validate the agent:
```bash
python test_research.py
```
- Ensures environment and API key are set up.
- Runs a sample research task and prints results and log file location.

## License

MIT License 
