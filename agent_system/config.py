"""Central, typed configuration. One frozen Settings object is threaded through the app; nothing
else reads the environment or hardcodes a path. Override any field via env (AS_* ) or constructor.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent              # .../concept_foundation/agent_system
PROJECT_ROOT = PKG_ROOT.parent                          # .../concept_foundation


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


@dataclass(frozen=True)
class ModelSpec:
    """One VLM 'expert'. `key` namespaces its cache; `model` is the proxy/gateway model id.
    `votes` overrides the per-expert vote count for single-shot (0 = use settings.votes_per_expert).
    `fallback_model`: if set, a call that fails on `model` (errors or unparseable after retries) is
    re-attempted on this model id (same gateway). Empty = no fallback."""
    key: str
    model: str
    is_cloud: bool = False
    votes: int = 0
    fallback_model: str = ""


@dataclass(frozen=True)
class Settings:
    # --- endpoints ---------------------------------------------------------------------
    proxy_url: str = field(default_factory=lambda: _env("AS_PROXY_URL", "http://localhost:20128/v1"))
    proxy_key: str = field(default_factory=lambda: _env("AS_PROXY_KEY", "dummy-key"))
    cloud_url: str = field(default_factory=lambda: _env("AS_CLOUD_URL", "https://1gw.gwai.cloud"))
    cloud_key: str = field(default_factory=lambda: _env("CF_CLOUD_KEY", ""))
    timeout_s: int = 120
    max_tokens: int = 8192            # model emits a thinking block first — budget for it + JSON

    # --- expert panel (the labeler ensemble) -------------------------------------------
    # gemini-pro-agent is the strongest labeler; flash3 adds a second, de-correlated family
    # so reliability `r` measures cross-MODEL agreement, not one model agreeing with itself.
    # Asymmetric votes (calibrated 2026-06-14): flash3 carries the neoplasia-suspicion signal, so it
    # gets 3 votes; pro-agent contributes 1 (its extra votes added noise). pro1+flash3×3 = 4 calls/img
    # matches the old 6-call panel (susp-AUROC 0.913 vs 0.907; core-AUROC 0.814 vs 0.834).
    experts: tuple[ModelSpec, ...] = (
        # proagent: prefer the gc/ preview pro; fall back to the ag/ production pro if it fails.
        ModelSpec("proagent", "ag/gemini-pro-agent", votes=1,
                  fallback_model="ag/gemini-pro-agent"),
        ModelSpec("flash3", "ag/gemini-3-flash-agent", votes=3),
    )

    # --- extraction protocol -----------------------------------------------------------
    extraction_strategy: str = "single"        # "single" (1 call/vote, quality prompt) | "multistage"
    single_quality: bool = True       # single-shot: use the rubric-anchored + contrastive prompt
    votes_per_expert: int = 3         # sweet spot: v=3/model ≈ v=5 within noise, −40% calls
    #                                   (vote sweep: disc-AUROC 1→0.73, 2→0.78, 3→0.78, 5→0.80)
    vote_retries: int = 1             # extra attempts on a parse failure
    base_temp: float = 0.4            # value votes; diversity comes from lenses / stages
    n_anchors_neo: int = 2
    n_anchors_ndbe: int = 3           # incl. hard-negative look-alikes (boundary calibration)
    self_verify: bool = False         # (single-shot only) legacy second-pass

    # --- multi-stage agent ---------------------------------------------------------------
    ms_votes_per_expert: int = 2      # each reading is a full pipeline → few repeats needed
    ms_region: bool = True            # Stage B: localize the suspicious region (recall lever)
    ms_refine: bool = True            # Stage D: cross-concept consistency repair
    ms_confirm: bool = True           # Stage E: skeptical confirmation (disagree → mask)
    ms_min_conf: float = 0.35         # default per-concept confidence floor; below → mask
    # per-concept floor overrides — vascular is resolution-limited, so accept lower confidence
    # rather than abstaining on everything (Tier-1 calibration of the hard classes)
    ms_floor_overrides: tuple = (("vascular_irregularity", 0.25), ("dilated_vessels", 0.25),
                                 ("focal_abnormal_vessels", 0.25), ("border_sharpness", 0.25))
    ms_crop: bool = True              # Stage B → crop+upscale the focal region (multi-scale)
    ms_crop_size: int = 512
    extraction_version: str = "v4"    # bump to invalidate multistage cache after a logic change

    def floor_for(self, concept: str) -> float:
        return dict(self.ms_floor_overrides).get(concept, self.ms_min_conf)

    # --- trust gates (calibrated on the val cross-anchor pool) -------------------------
    enable_all_concepts: bool = True  # supervise ALL ~35 classes (role-weighted), not just core
    trust_supervise: float = 0.55     # ≥ → use the concept value as a training target
    # trust = gate × mask × reliability × unverified_penalty (no-cross-anchor path). With the
    # 2-model panel, reliability IS cross-model agreement, so we do NOT discount it (=1.0).
    # cross-anchor (when run) multiplies in as a stricter ≤1 factor that only lowers fragile labels.
    unverified_penalty: float = 1.00
    suspicion_hi: float = 0.55        # trust-weighted core profile above this == looks neoplastic
    frame_trust_min: float = 0.50     # below this a frame ABSTAINs (PU-safe)

    # --- data --------------------------------------------------------------------------
    dataset_root: Path = PROJECT_ROOT / "dataset"
    unlabeled_path: Path | None = None     # override the unlabeled dir (--unlabeled-dir)
    dataset_name: str = ""                 # if set (--name), nest output under training_store/<name>/
    max_workers: int = 256            # async call concurrency: fixed cap, or MAX limit when adaptive
    adaptive_concurrency: bool = False  # --workers auto: AIMD auto-tune to the gateway sweet spot

    # --- artifact stores (the two output folders) --------------------------------------
    raw_store: Path = PKG_ROOT / "artifacts" / "raw_store"            # logs + raw votes
    training_store: Path = PKG_ROOT / "artifacts" / "training_store"  # images + curated labels

    @property
    def train_csv(self) -> Path:
        return self.dataset_root / "train.csv"

    @property
    def val_csv(self) -> Path:
        return self.dataset_root / "val.csv"

    @property
    def unlabeled_dir(self) -> Path:
        return self.unlabeled_path or (self.dataset_root / "unlabeled_data")

    # raw-store sub-paths
    @property
    def log_dir(self) -> Path:
        return self.raw_store / "logs"

    @property
    def raw_label_dir(self) -> Path:
        return self.raw_store / "raw_labels"      # one JSON per (frame, expert, anchors) — resumable

    @property
    def run_dir(self) -> Path:
        return self.raw_store / "runs"

    def ensure_dirs(self) -> None:
        for d in (self.log_dir, self.raw_label_dir, self.run_dir,
                  self.training_store / "images", self.training_store / "labels"):
            d.mkdir(parents=True, exist_ok=True)

    def abspath(self, rel: str) -> str:
        p = Path(rel)
        return str(p if p.is_absolute() else PROJECT_ROOT / p)
