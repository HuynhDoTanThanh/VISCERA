# agent_system — Architecture

Clean / hexagonal architecture for generating reliability-weighted clinical-concept labels for the
RACE foundation model. Dependencies point **inward**: `application` and `domain` know nothing about
the VLM gateway, the filesystem, or the CLI — those are adapters behind `ports`.

---

## 1. Layered overview (dependency rule)

```mermaid
flowchart TB
    subgraph CLI["interface · cli.py (composition root)"]
        C["argparse · build Settings · wire adapters · run"]
    end

    subgraph APP["application — use-cases"]
        AS["AnchorSelector"]
        EX["Extractor"]
        AG["Aggregator"]
        TS["TrustScorer"]
        CU["Curator"]
        PL["LabelPipeline"]
    end

    subgraph DOM["domain — pure core (no I/O)"]
        EN["entities<br/>Frame · Vote · ConceptCell<br/>CuratedFrame · Decision"]
        CO["concepts<br/>vocabulary · ROBUST_CORE<br/>normalize_vote"]
    end

    subgraph PORTS["ports — abstractions"]
        PV["VLMClient"]
        PC["VoteCache"]
    end

    subgraph INFRA["infrastructure — adapters"]
        VC["ProxyVLMClient"]
        FC["FileVoteCache"]
        DL["DatasetLoader"]
        LG["run logging"]
    end

    subgraph OUT["outputs — writers"]
        RS["RawStore"]
        TStore["TrainingStore"]
    end

    C --> PL
    C --> DL
    C --> RS
    C --> TStore
    PL --> EX --> AG --> TS --> CU
    AS --> EN
    EX --> PV
    EX --> PC
    APP --> DOM
    VC -. implements .-> PV
    FC -. implements .-> PC
    APP --> PORTS
    INFRA --> DOM
    OUT --> DOM

    classDef core fill:#eef7ff,stroke:#2b6cb0;
    classDef app fill:#eafaf1,stroke:#2d8a5f;
    classDef infra fill:#fff5e6,stroke:#c77d12;
    class EN,CO core;
    class AS,EX,AG,TS,CU,PL app;
    class VC,FC,DL,LG,RS,TStore infra;
```

> **Why:** the only code that talks to a model is `ProxyVLMClient`; the only code that touches disk
> for votes is `FileVoteCache`. Swapping the gateway or the cache is one new adapter — use-cases and
> domain stay untouched and unit-testable.

---

## 2. Module / class map

```mermaid
classDiagram
    class Settings {
      +experts: tuple
      +votes_per_expert: int
      +trust_supervise: float
      +raw_store: Path
      +training_store: Path
      +ensure_dirs()
    }
    class VLMClient {
      <<interface>>
      +infer_concepts(img, anchors, lens, temp, seed) dict
      +verify_concepts(img, draft) dict
    }
    class VoteCache {
      <<interface>>
      +get(key) list
      +put(key, payload)
    }
    class ProxyVLMClient {
      -spec: ModelSpec
      +infer_concepts(...)
      +verify_concepts(...)
    }
    class FileVoteCache {
      +key(path, expert, sig) str
      +get(key) / put(key, payload)
    }
    class Extractor {
      +extract(frame, anchors) Votes
    }
    class Aggregator {
      +aggregate(frame, votes) FrameAggregate
    }
    class TrustScorer {
      +score(agg, x) TrustedCells
    }
    class Curator {
      +curate(agg, cells, verified) CuratedFrame
    }
    class LabelPipeline {
      +run(frames, anchors, on_result) CuratedFrames
    }

    VLMClient <|.. ProxyVLMClient
    VoteCache <|.. FileVoteCache
    Extractor --> VLMClient
    Extractor --> VoteCache
    LabelPipeline --> Extractor
    LabelPipeline --> Aggregator
    LabelPipeline --> TrustScorer
    LabelPipeline --> Curator
    Aggregator --> FrameAggregate
    Curator --> CuratedFrame
```

---

## 3. Per-frame request flow (sequence)

```mermaid
sequenceDiagram
    autonumber
    participant CLI as cli.py
    participant PL as LabelPipeline
    participant EX as Extractor
    participant CA as FileVoteCache
    participant VL as ProxyVLMClient (panel)
    participant AG as Aggregator
    participant TS as TrustScorer
    participant CU as Curator
    participant TW as TrainingStore

    CLI->>PL: run(frames, anchors)
    loop each frame (concurrent)
        PL->>EX: extract(frame, anchors)
        loop each expert × 5 lenses
            EX->>CA: get(frame, expert, anchorSig)
            alt cache miss
                EX->>VL: infer_concepts(img, anchors, lens, temp)
                VL-->>EX: concept JSON
                EX->>CA: put(votes) into raw_labels
            else cache hit
                CA-->>EX: cached votes (resumable)
            end
        end
        EX-->>PL: votes
        PL->>AG: aggregate(frame, votes)
        AG-->>PL: (value, reliability, mask) per concept
        PL->>TS: score(agg, cross_anchor?)
        TS-->>PL: trust + supervise/mask per concept
        PL->>CU: curate(agg, cells)
        CU-->>PL: CuratedFrame (decision, frame_trust)
        PL->>TW: write row -> labels_<split>.jsonl
    end
```

---

## 3b. Multi-stage extraction agent (the reading of one frame × expert)

`Extractor` delegates one (frame, expert) reading to an `ExtractionStrategy`. The default
`MultiStageStrategy` (v3) decomposes the weak "all-35-concepts-in-one-prompt" call into focused,
skill-hardened prompts. Confident, cross-checked values survive; low-confidence or unconfirmed
ones become `not_assessable` (a mask) so reliability/trust down-weights them.

```mermaid
flowchart TD
    IMG["frame + anchors<br/>(incl. hard-negative)"] --> A
    A["A · Context & Quality<br/>modality-first · quality gate"] --> B
    B["B · Region grounding<br/>focal bbox → crop+upscale (multi-scale)"] --> C1
    A -. modality .-> C2
    B -. zoom crop .-> C
    subgraph C["C · Chunked reading (contrastive + rubric-anchored + evidence + confidence)"]
        C1["surface + colour"]
        C2["vascular<br/>(modality-conditioned abstention)"]
        C3["demarcation + lesion presence"]
    end
    C1 --> C2 --> C3 --> CM
    CM["C2 · Lesion morphology<br/>(only if lesion/border — Paris/size/border, on crop)"] --> D
    D["D · Refine<br/>cross-concept consistency rules"] --> E
    E["E · Confirm (skeptical, default-absent)<br/>disagree ⇒ mask"] --> F
    F["finalize: conf < per-concept floor ⇒ mask<br/>schema-conformant vote + meta"]

    classDef s fill:#eef7ff,stroke:#2b6cb0;
    class A,B,C1,C2,C3,CM,D,E,F s;
```

**Hard-class skills (v3):** ordinal **rubric anchoring** (each 0–4 level defined), **contrastive**
rating vs the NDBE reference, **modality-conditioned** vascular assessment with calibrated
abstention, **multi-scale crop** of the focal region for resolution-limited features, a
**conditional morphology** sub-stage (Paris/size/border only when a lesion/border exists), and
**per-concept confidence floors** (vascular lower so it isn't over-masked).

Strategy is swappable (`--strategy single|multistage`); cache is namespaced by strategy so the two
never collide. Per-stage outputs, confidences, region, and masked concepts are stored in the raw
label `meta` for full traceability.

## 4. Data transformation

```mermaid
flowchart LR
    F["Frame<br/>(image, label, center)"] --> EX
    A["Anchors<br/>2 neo + 3 ndbe<br/>(incl. 1 hard-neg)"] --> EX
    EX["Extractor<br/>panel × 5 lenses"] --> V["Vote[]<br/>raw concept JSON<br/>× experts × votes"]
    V --> AG["Aggregator"]
    AG --> CC["ConceptCell<br/>value c · reliability r · mask m"]
    CC --> TS["TrustScorer<br/>g × m × (r × x)"]
    XA["cross-anchor x<br/>(optional, best-sure)"] --> TS
    TS --> TC["TrustedCell<br/>value · trust · supervise?"]
    TC --> CU["Curator"]
    CU --> CF["CuratedFrame<br/>decision · frame_trust · suspicion"]

    classDef hot fill:#ffe9cc,stroke:#d35400;
    class CF hot;
```

---

## 5. Output stores (artifacts)

```mermaid
flowchart TB
    RUN["pipeline run"] --> RAW
    RUN --> TRN

    subgraph RAW["raw_store/ — logs + raw labels"]
        L["logs/&lt;run&gt;.log"]
        RL["raw_labels/&lt;sha1&gt;.json<br/>raw votes per (frame, expert, anchors)<br/>resumable"]
        MAN["runs/&lt;run&gt;/manifest.json<br/>config + stats"]
    end

    subgraph TRN["training_store/ — images + labels for foundation"]
        JL["labels/labels_&lt;split&gt;.jsonl<br/>image + supervised concepts (value, trust)<br/>+ decision + frame_trust"]
        IMG["images/&lt;split&gt;/*.png<br/>symlink to source (--link-images)"]
        DC["dataset_card_&lt;split&gt;.md<br/>provenance · counts · schema"]
    end

    classDef store fill:#eef7ff,stroke:#2b6cb0;
    class L,RL,MAN,JL,IMG,DC store;
```

---

## 6. Frame decision policy

```mermaid
flowchart TD
    S["CuratedFrame<br/>frame_trust · suspicion"] --> Q1{"frame_trust &lt; 0.5<br/>or 0 core supervised?"}
    Q1 -->|yes| AB["ABSTAIN<br/>PU-safe · not mined"]
    Q1 -->|no| Q2{"label?"}
    Q2 -->|"neo (1)"| P["POSITIVE"]
    Q2 -->|"ndbe (0)"| N["TRUE_NEGATIVE"]
    Q2 -->|"unlabeled (−1)"| Q3{"suspicion ≥ 0.55?"}
    Q3 -->|yes| HN["HARD_NEG_CANDIDATE<br/>FPR lever (DESIGN §5.1)"]
    Q3 -->|no| CN["CONFIDENT_NEGATIVE"]

    classDef hot fill:#ffe9cc,stroke:#d35400;
    class HN hot;
```
