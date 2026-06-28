"""agent_system — clean-architecture pipeline that turns endoscopy frames into reliability-weighted
clinical-concept labels for training the RACE foundation model.

Layering (dependencies point inward):

    cli ─► application ─► domain
            │      ▲
            ▼      │ (ports)
        infrastructure / outputs

- domain         : entities + the clinical-concept vocabulary (no I/O, no frameworks)
- ports          : abstract interfaces the application depends on (VLMClient, VoteCache)
- application    : use-cases — anchor selection, extraction, aggregation, trust, pipeline
- infrastructure : concrete adapters — VLM proxy client, filesystem cache, dataset loader, logging
- outputs        : artifact writers — raw store (logs + raw votes) and training store (images + labels)

Run `python -m agent_system.cli --help`.
"""
__version__ = "0.1.0"
