# Introduction
This example demonstrates how to build a simple Multi-Agent System using Agent-Kernel, designed to help users understand the execution flow and facilitate future extensions.

To simplify the process, we have provided basic implementations for five core plugins: Perceive, Plan, Invoke, Communication, and Space. The remaining plugins are structured as placeholders (using pass) to allow for easy customization and expansion by the user.

# Quick Start
1. Set your api key in **`examples/scimas_mve/configs/models_config.yaml`**

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
    python -m examples.scimas_mve.run_simulation
    ```

4. Inspect research logs
    - Evidence cards per tick/agent: `examples/scimas_mve/logs/app/research/evidence_cards.jsonl`
    - Paper submissions and evaluations: `examples/scimas_mve/logs/app/research/papers.jsonl`
    - Full action trace: `examples/scimas_mve/logs/app/action/trace.jsonl`
    - Taskboard events (create/claim/complete): `examples/scimas_mve/logs/app/environment/taskboard.jsonl`
    - Team metrics across episodes: `examples/scimas_mve/logs/app/simulation/team_metrics.jsonl`
    - Policy evolution records: `examples/scimas_mve/logs/app/simulation/evolution.jsonl`
    - Trend dashboard (auto-generated after run): `examples/scimas_mve/logs/app/simulation/trend_dashboard.html`

5. Analyze cross-episode evolution
    ```bash
    python -m examples.scimas_mve.analysis.analyze_metrics
    ```
    - Summary JSON: `examples/scimas_mve/logs/app/simulation/analysis_summary.json`
    - Summary text: `examples/scimas_mve/logs/app/simulation/analysis_summary.txt`
    - Episode time series CSV: `examples/scimas_mve/logs/app/simulation/analysis_timeseries.csv`
    - Trend chart: `examples/scimas_mve/logs/app/simulation/analysis_trend.html`

# Docker Runtime (AIRS Full Dependencies)

Build the dedicated AIRS runtime image once:

```bash
docker build --no-cache \
  -t scimas-airs-runtime:py311-cu124-v2 \
  -f examples/scimas_mve/docker/Dockerfile.airs-runtime \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
  --build-arg TORCH_VERSION=2.5.1 \
  .
```

Quick verify (must print `True` for CUDA availability):

```bash
docker run --rm --gpus all scimas-airs-runtime:py311-cu124-v2 \
  python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

`environment_config.yaml` is configured to use this image:
`code_docker_image: "scimas-airs-runtime:py311-cu124-v2"`.

        
            
