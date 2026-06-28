# RACE Phase-1 — Label Generation & Confirmation Pipeline

End-to-end flow from raw frames to a **best-sure** foundation-label manifest, mapped to the code.

---

## 1. Label generation (multi-expert concept extraction)

```mermaid
flowchart TD
    subgraph IN[Inputs]
        TR["train.csv · 2476 labelled"]
        VA["val.csv · 619 labelled"]
        UN["unlabeled_data/ · 9937 frames"]
        SC["concept_schema.py<br/>~30 atomic concepts<br/>(json_spec + concept_guide)"]
        AN["data.pick_anchors(seed=0)<br/>2 neo + 2 ndbe reference frames"]
    end

    TR & VA & UN --> EXT
    SC --> EXT
    AN --> EXT

    subgraph GEN["run_extract.py  →  extract.py"]
        EXT["extract_frame(path, anchors)"]
        EXP["per expert in {gemini, claude}"]
        VOTE["votes_per_expert = 3<br/>5 LENSES × temp 0.5–0.7<br/>llm.invoke_vision (proxy)"]
        EXT --> EXP --> VOTE
    end

    VOTE --> CACHE["cache/concepts/&lt;sha1(path|expert|anchors)&gt;.json<br/>raw votes — resumable, 1 file per (img,expert,anchor)"]
    CACHE --> IDX["cache/index_{train,val,unlabeled}.json<br/>(path, label, cache paths)"]

    CACHE --> AGG["aggregate.py · aggregate_frame()<br/>c = mean value [0,1]<br/>r = inter-vote agreement<br/>m = assessable fraction"]
    IDX --> AGG
    AGG --> REL["reliability.py · study()<br/>agreement × MI × AUROC<br/>→ reliability_report.txt"]

    classDef done fill:#d6f5d6,stroke:#2d8a2d;
    classDef todo fill:#fde2e2,stroke:#c0392b;
    class TR,VA,UN,CACHE,IDX,AGG,REL done;
```

> **State today:** `gemini` expert complete on all splits (2476 / 619 / 9937, 0 empty caches).
> `claude` expert blocked — cloud key invalid (resumable once `CF_CLOUD_KEY` is set).

---

## 2. Label confirmation & audit (is it usable as foundation supervision?)

```mermaid
flowchart TD
    AGG["aggregated votes<br/>(c, r, m) per frame"] --> L1

    subgraph L1["Layer 1 — Statistical · confirm.py (offline)"]
        G1["held-out val AUROC<br/>+ 2000× bootstrap CI"]
        G2["within-center AUROC<br/>+ concept→center leak"]
        G3["agreement ≥ 0.6<br/>assessable ≥ 0.5"]
        G1 & G2 & G3 --> VRD["per-concept<br/>PASS / REVIEW / FAIL"]
    end

    VRD --> L2
    subgraph L2["Layer 2 — Agentic faithfulness · audit.py (proxy)"]
        ST["stratify: baseline vs risk frames"]
        RG["independent re-grade<br/>(strict rubric)"]
        MR["LLM meta-review<br/>GO / GO_WITH_CAVEATS / NO_GO"]
        ST --> RG --> MR
    end

    VRD --> L3
    subgraph L3["Layer 3 — Fair recheck · recheck_failcases.py --protocol anchored"]
        RE["re-extract worst cases<br/>DIFFERENT anchor set (seeds 1,2)<br/>= real protocol, 1 variable changed"]
        CL["classify: STABLE_WRONG /<br/>AMBIGUOUS / NOISE"]
        XA["cross-anchor agreement<br/>x = 1 − |c(A) − c(B)|"]
        RE --> CL
        RE --> XA
    end

    VRD --> TRUST
    XA --> TRUST
    AGG --> TRUST
    subgraph L4["Layer 4 — Trust & curation · trust.py + run_curate.py"]
        TRUST["trust = gate × m × (r × x)<br/>confident-but-fragile → down"]
        SUP{"trust ≥ 0.6 ?"}
        TRUST --> SUP
        SUP -->|yes| KEEP["SUPERVISE<br/>(~1.1% cross-anchor flip)"]
        SUP -->|no| MASK["MASK<br/>(~17.6% flip — quarantined)"]
    end

    KEEP --> DEC{"frame decision"}
    MASK --> DEC
    DEC --> OUT["labels_split.jsonl<br/>foundation manifest"]

    classDef done fill:#d6f5d6,stroke:#2d8a2d;
    class VRD,XA,KEEP,MASK,OUT done;
```

---

## 3. Trust signal — why cross-anchor, not just vote-agreement

```mermaid
flowchart LR
    R["r · within-vote agreement<br/>(3 votes, SAME anchors)"] -->|"ρ≈0.47, overconfident:<br/>agrees even when wrong"| C
    X["x · cross-anchor agreement<br/>(re-extract, DIFFERENT anchors)"] -->|"the independence signal<br/>r lacks"| C
    G["gate (PASS/REVIEW/FAIL)"] --> C
    M["m · assessability"] --> C
    C["trust = g × m × (r × x)"] --> V["validated on val:<br/>kept 1.1% flip vs masked 17.6%"]
```

---

## 4. Per-frame foundation decision

```mermaid
flowchart TD
    F["frame curation<br/>(frame_trust, suspicion)"] --> Q1{"frame_trust &lt; 0.5<br/>or 0 supervised?"}
    Q1 -->|yes| AB["ABSTAIN<br/>(PU-safe — not a hard negative)"]
    Q1 -->|no| Q2{"label?"}
    Q2 -->|"neo (1)"| P["POSITIVE"]
    Q2 -->|"ndbe (0)"| N["TRUE_NEGATIVE"]
    Q2 -->|"unlabeled (−1)"| Q3{"suspicion ≥ 0.55?"}
    Q3 -->|yes| HN["HARD_NEG_CANDIDATE<br/>looks neoplastic @1% prevalence<br/>→ FPR lever (DESIGN §5.1)"]
    Q3 -->|no| CN["CONFIDENT_NEGATIVE"]

    classDef hot fill:#ffe9cc,stroke:#d35400;
    class HN hot;
```
