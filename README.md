What we could do that remains:
```
├── apps/
│   ├── worker/
│   │   ├── main.py
│   │   ├── server.py
│   │   └── service.yaml
│   ├── master/
│   │   ├── main.py
│   │   ├── coordinator.py
│   │   └── scheduler.py
│   └── single_node/
│       └── main.py
├── checkpoints/
│   ├── baseline/
│   ├── early_exit/
│   └── gating/
├── data/
│   ├── raw/
│   ├── processed/
│   └── cache/
├── notebooks/
│   ├── 01_model_dev.ipynb
│   ├── 02_training_debug.ipynb
│   ├── 03_single_node_debug.ipynb
│   ├── 04_distributed_debug.ipynb
│   └── 05_results_analysis.ipynb
├── docs/
│   ├── experiment_plan.md
│   ├── metrics_definition.md
│   ├── topology.md
│   ├── deployment.md
│   ├── reproducibility.md
│   └── thesis_figures/
├── tests/
│   ├── test_models.py
│   ├── test_partitioning.py
│   ├── test_entropy_policy.py
│   ├── test_serialization.py
│   ├── test_metrics.py
│   └── test_end_to_end.py
└── services/
    ├── systemd/
    │   ├── thesis-worker.service
    │   └── thesis-master.service
    └── docker/
        └── Dockerfile
```

Steps after cloining:
- git clone ...
- python -m venv venv
- source venv/bin/activate
- pip install -r requirements.txt

for torch and torchvision use the commands:
```bash
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://www.piwheels.org/simple \
  torch
```

and 
```bash
pip install torchvision --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://www.piwheels.org/simple
```