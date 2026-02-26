# Introduction
This example demonstrates how to build a simple Multi-Agent System using Agent-Kernel, designed to help users understand the execution flow and facilitate future extensions.

To simplify the process, we have provided basic implementations for five core plugins: Perceive, Plan, Invoke, Communication, and Space. The remaining plugins are structured as placeholders (using pass) to allow for easy customization and expansion by the user.

# Quick Start
1. Set your api key in **`examples/scimas_mvp/configs/models_config.yaml`**

    ```yaml
    - name: OpenAIProvider
      model: your-model-name
      api_key: your-api-key
      base_url: your-base-url
      capabilities: ["your-capabilities"] # e.g., capabilities: ["chat"]
    ```
    
2. Install the required dependencies:
    ```bash
    pip install "agentkernel-standalone[all]"
    ```
    
3. Run
    ```bash
    cd Agent-Kernel
    python -m examples.scimas_mvp.run_simulation
    ```

4. Inspect research logs
    - Evidence cards per tick/agent: `examples/scimas_mvp/logs/app/research/evidence_cards.jsonl`
    - Paper submissions and evaluations: `examples/scimas_mvp/logs/app/research/papers.jsonl`
    - Full action trace: `examples/scimas_mvp/logs/app/action/trace.jsonl`
    - Taskboard events (create/claim/complete): `examples/scimas_mvp/logs/app/environment/taskboard.jsonl`
    - Team metrics across episodes: `examples/scimas_mvp/logs/app/simulation/team_metrics.jsonl`
    - Policy evolution records: `examples/scimas_mvp/logs/app/simulation/evolution.jsonl`
    - Trend dashboard (auto-generated after run): `examples/scimas_mvp/logs/app/simulation/trend_dashboard.html`

5. Analyze cross-episode evolution
    ```bash
    python -m examples.scimas_mvp.analysis.analyze_metrics
    ```
    - Summary JSON: `examples/scimas_mvp/logs/app/simulation/analysis_summary.json`
    - Summary text: `examples/scimas_mvp/logs/app/simulation/analysis_summary.txt`
    - Episode time series CSV: `examples/scimas_mvp/logs/app/simulation/analysis_timeseries.csv`
    - Trend chart: `examples/scimas_mvp/logs/app/simulation/analysis_trend.html`

        
            
