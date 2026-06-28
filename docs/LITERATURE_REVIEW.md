# Literature Review: VLM-Concept Foundation for Medical Image Analysis
### RACE — Reliability-Aware Concept Ensemble
**Document type:** Research foundation & related work synthesis
**Companion:** `DESIGN.md` (system design), `../../docs/02_FEASIBILITY_RESULTS.md` (motivation)

---

## 0. Positioning in one paragraph

RACE sits at the intersection of five literature threads — **Concept Bottleneck Models**, **VLM-as-annotator paradigms**, **multi-annotator aggregation**, **SSL domain adaptation**, and **tail-focused objectives** — none of which individually addresses the full combination. The system's core novelty is **per-image, per-concept reliability weighting derived from multi-expert VLM agreement**, used as a training signal for a downstream discriminative model. This is the finest-grained reliability signal in the CBM literature: existing work quantifies uncertainty at the image level (ProbCBM), the model level (SCoOP), or through jointly trained ensemble heads (Credal CBMs) — but none uses independently pre-trained VLMs' natural disagreement as a training-time weighting signal at per-concept granularity.

---

## 1. Concept Bottleneck Models (CBMs)

### 1.1 Original CBM

**Koh et al. (2020), ICML.** "Concept Bottleneck Models."
- Architecture: concept predictor g: X → R^k maps input to concept scores; label predictor f: R^k → Y makes the final decision through the bottleneck y_hat = f(g(x)).
- Four training regimes: *independent* (g task-blind, f on true concepts), *sequential* (f on g(x)), *joint* (end-to-end with λ·L_concept + L_task), *standard* (no concepts).
- Joint achieves highest task accuracy; independent supports the most effective test-time concept interventions.
- **Limitation:** Requires dense per-image concept annotations; accuracy below black-box baselines when concept sets are incomplete; concepts assumed binary/independent.
- **Relation to RACE:** Original CBM uses a single supervised concept predictor. RACE generalizes to an ensemble of K frozen VLMs, each producing concept vectors — analogous to multi-head concept prediction with measured, not assumed, uncertainty.

### 1.2 Information leakage

**Mahinpei et al. (2021), ICML Workshop.** "Promises and Pitfalls of Black-Box Concept Learning Models."
- Soft concept representations encode unintended information about the task label beyond named concepts — even under task-blind sequential training.
- Hard (binary thresholded) concepts prevent leakage but lose uncertainty information.
- **Implication for RACE:** Leakage is less acute when concepts come from frozen VLMs not jointly trained with the classifier — a key architectural distinction from neural CBMs.

### 1.3 Concept Embedding Models (CEMs)

**Espinosa Zarlenga et al. (2022), NeurIPS.** "Concept Embedding Models."
- Replace scalar concept scores with high-dimensional embeddings: c_hat_i = p_i · e_i^+ + (1-p_i) · e_i^-, where positive/negative embeddings are learned.
- Matches or exceeds black-box accuracy without sacrificing interpretability.
- Introduces RandInt (training-time randomized intervention) and Concept Alignment Score (CAS).
- **Relation to RACE:** Multi-expert VLM outputs naturally produce a CEM-like multi-dimensional representation per concept. Reliability weighting gates each expert's contribution — a principled form of concept embedding.

### 1.4 Post-hoc CBMs (PCBM)

**Yuksekgonul et al. (2023), ICLR Spotlight.** "Post-hoc Concept Bottleneck Models."
- Converts any pre-trained model into a CBM post-hoc using Concept Activation Vectors (CAVs) or CLIP embeddings.
- PCBM-h adds a residual predictor to recover accuracy when concepts are insufficient.
- **Relation to RACE:** PCBM demonstrates CLIP-grounded concepts from language descriptions can score concepts per-image without explicit labels — foundational for label-free VLM-driven approaches.

### 1.5 Probabilistic CBMs

**Kim et al. (2023), ICML.** "Probabilistic Concept Bottleneck Models."
- Each concept represented as a Gaussian N(μ_i, Σ_i), modeling concept-level uncertainty.
- Class uncertainty is **decomposable by concept** — identifies which concepts drive uncertainty.
- **Relation to RACE:** ProbCBM learns uncertainty from a single model via a probabilistic objective. RACE derives reliability from **natural diversity of independently trained VLMs** — a model-free, training-free signal. Inter-expert disagreement is arguably more robust than intra-model learned variance.

### 1.6 Energy-Based CBMs

**Xu et al. (2024), ICLR.** "Energy-Based Concept Bottleneck Models."
- Joint energy function over (input, concept, label) tuples. Propagates concept corrections to related concepts (unlike standard CBMs where corrections are independent).
- Models inter-concept dependencies but not per-image reliability of the concept predictor itself.

### 1.7 Credal CBMs

**CREDENCE (2025), arXiv:2602.11219.**
- Represents concept predictions as credal sets (probability intervals).
- Epistemic uncertainty from ensemble disagreement across jointly trained heads.
- **Closest existing work** to using multi-expert disagreement as a concept reliability signal. Key difference: CREDENCE uses jointly trained heads; RACE uses independently pre-trained VLMs — a model-free, zero-shot reliability signal.

### 1.8 Label-Free CBMs

**Oikarinen et al. (2023), ICLR.** "Label-free Concept Bottleneck Models."
- GPT-3 generates concept sets; CLIP scores concept-image alignment. First CBM scaled to ImageNet (72% top-1).
- Concept accuracy depends on CLIP alignment quality — noisy for domain-specific concepts.

**Yang et al. (2023), CVPR.** "Language in a Bottle (LaBo)."
- GPT-3 generates candidate concepts; submodular optimization selects discriminative+diverse subset.
- +11.7% over black-box linear probes at 1-shot across 11 datasets.

**Relation to RACE:** Both replace human annotators with a single VLM (CLIP). RACE generalizes to multiple VLMs whose ensemble is more comprehensive. The gap these papers expose — imperfect concept faithfulness — is precisely what reliability weighting addresses.

### 1.9 VLM-Guided CBMs

**VH-CBM (2025), arXiv:2605.16405.** Acknowledges VLM annotations are not entirely reliable; uses Gaussian Process propagation to improve concept accuracy.

**VLG-CBM (2024), arXiv:2408.01432.** Uses grounded object detectors for visually grounded concept annotations. Introduces "Number of Effective Concepts" (NEC) to control information leakage.

**SCoOP (2025), arXiv:2603.23853.** Uncertainty-weighted opinion pooling across multiple VLMs: w_k = 1/H_k (inverse entropy). AUROC 0.866 for hallucination detection vs. 0.732–0.757 baselines. Operates at task level — RACE applies analogous weighting at the **concept level**.

### 1.10 CBMs in Medical Imaging

- **Koh et al. (2020):** OAI x-ray grading with clinical concepts (bone spurs, joint space narrowing).
- **Yan et al. (2023), arXiv:2310.03182:** Label-free CBM for medical imaging using GPT-4 concepts + MedCLIP. Evaluated on 8 datasets including CheXpert, skin lesion, fundus.
- **XpertCausal (2025), arXiv:2605.07785:** Causal CBM for chest X-ray with radiologist-curated concept-pathology matrix.
- **CCBM (2024), arXiv:2410.15446:** Concept complement model for dermatology, ultrasound, CT. Per-concept adapters with cross-attention scoring.
- **Endoscopy-specific CBMs:** An open research area. Existing GI work uses clinical descriptors (NICE, Paris) in non-CBM frameworks. RACE would be among the first CBMs applied to Barrett's endoscopy.

---

## 2. VLM-as-Annotator Paradigm

### 2.1 LLMs/VLMs as data annotators

**Gilardi et al. (2023), PNAS.** "ChatGPT outperforms crowd workers for text-annotation tasks."
- Zero-shot ChatGPT outperforms MTurk by ~25pp on 5 NLP annotation tasks at 30× lower cost.
- Performance depends on prompt quality; some tasks lag behind trained specialists — motivates reliability estimation.

**Goel et al. (2023), ML4H.** "LLMs Accelerate Annotation for Medical Information Extraction."
- Human-LLM hybrid pipeline for clinical NLP. Demonstrates LLM annotation as scalable complement to expert annotation, not substitute.

**DataComp (Gadre et al., 2023, NeurIPS):** Model-based data quality scoring (CLIP score) outperforms raw scale — smaller filtered subsets consistently beat larger unfiltered pools.

**DCLM (Li et al., 2024, NeurIPS):** Model-based quality filtering (fastText classifier) dominates all other filtering strategies for LLM pretraining data.

### 2.2 Multi-LLM committees

**Plaza-del-Arco et al. (2024), NLPerspectives.** "Wisdom of Instruction-Tuned Language Model Crowds."
- 4 LLMs aggregated via MACE outperform any individual by 4.2 F1 points.
- MACE competence scores correlate at Spearman ρ = 0.93 with true model accuracy.
- No model dominates across all tasks — specialization justifies aggregation.
- **Closest direct precedent** for the Gemini + Claude committee system.

### 2.3 Multi-annotator aggregation theory

**Dawid & Skene (1979), JRSS-C.** EM-based latent class model for per-annotator error rates without gold labels. Originally developed for *medical record* data — directly relevant clinical origin.

**GLAD (Whitehill et al., 2009, NeurIPS).** Extends Dawid-Skene with per-item difficulty: labels modeled as Bernoulli(σ(α·β)) where α = annotator expertise, β = item difficulty. If VLMs consistently disagree on "vascular_irregularity" but agree on "modality," GLAD attributes this to concept difficulty, not annotator unreliability.

**Spectral+EM (Zhang et al., 2016, JMLR).** Spectral initialization of Dawid-Skene EM achieves minimax-optimal convergence rate. Important when scaling to many concepts or VLMs.

**MACE (Hovy et al., 2013, NAACL).** Bayesian per-annotator competence estimation via variational inference — the algorithmic backbone used in Plaza-del-Arco's LLM aggregation work.

### 2.4 Noisy label learning

**Co-teaching (Han et al., 2018, NeurIPS).** Two peer networks select small-loss instances to train each other. Structurally analogous to using Gemini and Claude whose disagreement identifies uncertain labels.

**MentorNet (Jiang et al., 2018, ICML).** Teacher network assigns per-sample weights. Per-sample weighting by estimated label reliability is precisely RACE's mechanism — but MentorNet estimates reliability from training dynamics, while RACE estimates from multi-VLM agreement.

**DivideMix (Li et al., 2020, ICLR).** GMM on per-sample loss divides clean/noisy samples; treats noisy as unlabeled in MixMatch SSL. De facto standard for noisy-label learning.

**Label Smoothing (Szegedy et al., 2016, CVPR).** Replace one-hot with q'(k) = (1-ε)·δ + ε·u. In RACE, VLM reliability naturally parameterizes smoothing: low-agreement → more smoothing.

**Instance-dependent noise (ECCV 2024, IJCV 2024).** Per-instance noise transition matrices. Theoretical precedent for per-instance reliability weights from VLM agreement.

### 2.5 VLM clinical concept extraction

**GPT-4V medical evaluations (Yan et al. 2023; Wu et al. 2023; Lecler et al. 2024):**
- Strong at modality/anatomy recognition (~100%); weak at pathology diagnosis (29–35%).
- Hallucination rates >40% in pathology — quantifies the noise floor that reliability weighting must overcome.
- Report generation (structured free text) is the strongest medical VLM capability — motivates extracting structured concepts rather than direct classification.

**Lu et al. (2024), arXiv:2403.07407.** GPT-4V with kNN 10-shot in-context learning matches fine-tuned specialists on colorectal tissue subtyping. Zero-shot fails; few-shot with relevant context succeeds. Directly applicable to improving Gemini/Claude extraction reliability via anchor selection.

### 2.6 Knowledge distillation from VLMs

**Hinton et al. (2015).** "Distilling the Knowledge in a Neural Network." Soft targets carry richer information than one-hot labels. RACE uses structured concept extractions (not soft logits) as teacher signal — a structured-output distillation problem.

**VLM-KD (2024), arXiv:2408.16930.** VLM-generated text descriptions as contrastive distillation supervision. +3.5% on ImageNet-LT. Most direct precedent for RACE's core mechanism: VLMs generate structured descriptions → supervise a discriminative model. RACE adds multi-source reliability weighting on top.

---

## 3. SSL Domain Adaptation for Medical Imaging

### 3.1 DINOv2

**Oquab et al. (2024), TMLR.** "DINOv2: Learning Robust Visual Features without Supervision."
- Combines self-distillation (DINO) with masked image modeling (iBOT). Trained on LVD-142M curated dataset.
- De-facto standard pretraining recipe for all leading medical foundation models (UNI, Virchow, Phikon-v2).
- Frozen features provide strong linear probe baselines — the regime of interest for RACE.

**Roth et al. (2024), arXiv:2401.04720.** "Low-resource finetuning of foundation models beats SOTA in histopathology."
- DINOv2 ViT-S fine-tuned for 2 hours on a single GPU matches domain-specific encoders.
- **Existence proof** for ~10k frame domain adaptation.

**MedDINOv3 (2025), arXiv:2509.02379.** 3-stage adaptation recipe: ImageNet init → global-only pretraining → full DINOv2 recipe. Multi-scale token extraction (CLS + intermediate blocks) enriches features.

### 3.2 Medical foundation models

| Model | Arch | SSL | Training Data | License |
|-------|------|-----|---------------|---------|
| **UNI** (Chen et al., 2024, Nature Medicine) | ViT-L/16 | DINOv2 | 100M tiles, 100K H&E WSIs | CC-BY-NC-ND-4.0 |
| **UNI2** (Jan 2025) | ViT-H/14 | DINOv2 | 200M tiles, 350K WSIs | CC-BY-NC-ND-4.0 |
| **Virchow** (Vorontsov et al., 2024, Nature Medicine) | ViT-H/14 | DINOv2 | 1.5M H&E WSIs (MSKCC) | Apache 2.0 |
| **Phikon-v2** (Filiot et al., 2024) | ViT-L/16 | DINOv2 | 460M tiles, 55K public slides | Public |
| **BiomedCLIP** (Zhang et al., 2024, NEJM AI) | ViT-B/16 | CLIP | 15M biomedical image-text pairs | MIT |
| **PLIP** (Huang et al., 2023, Nature Medicine) | ViT-B/32 | CLIP | 208K pathology image-text | MIT |
| **CONCH** (Lu et al., 2024, Nature Medicine) | ViT-B/16 | CoCa | Pathology image-text | CC-BY-NC-ND-4.0 |
| **Endo-FM** (Wang et al., 2023, MICCAI) | ViT-B/16 | DINO | 33K video clips, 5M frames | Open |

**Endo-FM** is the most directly relevant: endoscopy-specific, trained on GI video. Primary recommendation for endoscopy feature extraction with ~10k frames.

**Pathology models (UNI, Virchow):** Optimized for H&E microscopy, fundamentally different acquisition from endoscopy. Useful as comparison baselines but expect domain-continued DINOv2 to outperform.

### 3.3 SSL continued pretraining protocol for ~10k frames

1. Initialize from DINOv2 ViT-B/14 (or Endo-FM as stronger starting point)
2. Stage 1 (10 epochs): global crops only (224×224), lr=1e-5 — CLS token alignment
3. Stage 2 (20 epochs): full DINOv2 recipe with 2 global + 8 local crops, teacher EMA 0.994–0.9999
4. Extract: CLS token (768-dim) + mean of final-block patch tokens (768-dim)
5. Expected compute: ~12 hours on single A100 for ViT-B

### 3.4 Concept-guided fusion strategies

**Concatenation:** [z ∥ c ∥ r] — simplest; effective when concepts are orthogonal to visual features.

**Gated fusion:** e_fused = [e_img, sigmoid(W_gate · c) ⊙ e_concept] — concepts gate visual features. Low parameter count, recommended for small labeled sets.

**When concepts help:** (a) encode attributes not captured by SSL (clinical judgment features), (b) low-labeled-data regimes (structural inductive bias), (c) interpretability required.

**When pure embeddings suffice:** (a) large labeled data, (b) concept annotations systematically biased, (c) task is pure pattern recognition.

**Practical recommendation:** Keep concept projection dimension small (64–256); use concept dropout (p=0.3) during training for robustness to missing/noisy concept scores.

### 3.5 OOD detection methods

**Mahalanobis distance (Lee et al., 2018, NeurIPS):** Class-conditional Gaussians; multi-layer ensemble for robustness. Best-paper extension for medical imaging (Anthony & Kamnitsas, 2023, UNSURE): optimal layer is OOD-pattern-specific — no single layer is universally best.

**kNN OOD (Sun et al., 2022, ICML):** k-th nearest neighbor distance on normalized features. Reduces FPR@TPR95 by 24.77% vs Mahalanobis. No distributional assumptions. **Recommended** for frozen ViT encoders — kNN on PCA-reduced features achieves top AUROC (Woodland et al., 2024, MELBA).

**Energy-based OOD (Liu et al., 2020, NeurIPS):** E(x) = -T·log Σ exp(f_c(x)/T). Requires a classification head; not directly applicable to purely frozen encoders.

**Recommended for RACE:** kNN-based OOD with PCA-reduced frozen encoder embeddings (FAISS index, k=5, cosine distance).

---

## 4. Tail-Focused Objectives for PPV@90Recall

### 4.1 Constrained precision@recall optimization

**Narasimhan et al. (2015–2024), ICML/AISTATS/NeurIPS/JMLR.** Definitive framework for optimizing functions of the confusion matrix:
- Frank-Wolfe/bisection algorithms for cost-sensitive learning under recall constraints.
- Three-player game (θ-player + ξ-player + λ-player) with O(1/√T) convergence.
- **Kumar et al. (2021), ICML: Implicit Rate-Constrained Optimization** — expresses threshold τ as implicit function of parameters via IFT, converting constrained to unconstrained optimization. "Particularly effective when targeting extreme values of FPR or recall." **Most directly applicable** to maximizing PPV at Recall ≥ 0.90.

### 4.2 Partial AUC optimization

**Narasimhan & Agarwal (2013), ICML.** Structural SVM for partial AUC in any FPR interval.

**Yang et al. (2021), ICML.** Two-way pAUC (TPAUC): restricts to TPR ≥ α AND FPR ≤ β — most directly relevant variant for PPV@recall.

**Zhu et al. (2022), ICML.** Partial AUC as DRO problem. Implemented in **LibAUC** as `pAUCLoss`. With DualSampler oversampling positives at 1:1, gradient quality improves for 127-positive datasets.

### 4.3 Focal loss and variants

**Focal Loss (Lin et al., 2017, ICCV).** FL(p_t) = -α_t(1-p_t)^γ log(p_t). Designed for 1:1000 imbalance. Useful baseline for 1:18.5 ratio but does not directly optimize PPV@recall.

**Class-Balanced Loss (Cui et al., 2019, CVPR).** Reweights by inverse effective number: (1-β)/(1-β^n_j). With β=0.999: positive weight ~8× negative weight.

**Asymmetric Loss (Ridnik et al., 2021, ICCV).** Decoupled γ+ and γ- with probability shifting. Best native imbalance loss for this regime — preserves full positive gradient while aggressively down-weighting easy negatives.

**LDAM + DRW (Cao et al., 2019, NeurIPS).** Class-specific margins Δ_j = C/n_j^{1/4}. Two-stage: LDAM first, then class-reweighted LDAM. ~2× margin asymmetry for 18.5:1 ratio.

### 4.4 Prevalence-aware calibration

The critical math: at 90% recall and 1% prevalence, achieving PPV ≥ 30% requires specificity ≥ 99.7%.

```
PPV = (Sensitivity × Prevalence) / (Sensitivity × Prevalence + (1-Specificity) × (1-Prevalence))
0.30 = (0.90 × 0.01) / (0.90 × 0.01 + FPR × 0.99) ⟹ FPR ≤ 0.003 ⟹ Specificity ≥ 99.7%
```

**Prior recalibration (Saerens et al., 2002):**
P_dep(Y=1|x) = [P_train(Y=1|x) · r] / [P_train(Y=1|x) · r + (1-P_train(Y=1|x))]
where r = (π_dep/(1-π_dep)) / (π₀/(1-π₀)) ≈ 0.188 for 5.1% → 1%.

**Affine recalibration (Godau et al., 2025, arXiv:2303.12540).** Prevalence-weighted (T, b) learning outperforms temperature scaling alone across 30 medical tasks. **Recommended** over plain Platt scaling.

**Platt scaling:** With only 127 positives, Platt preferred over isotonic regression (which needs >1000 samples). But standard Platt does not correct for prevalence shift.

### 4.5 Multi-task learning as regularization

With 127 positives and the 10 EPP (events per predictor) guideline, the final decision layer should have ≤12 effective parameters — favoring linear probing over frozen backbone.

**Concept reconstruction as auxiliary task:** Predicting core concepts as auxiliary outputs provides dense gradient signal, regularizes toward clinical directions, and enables per-image explanations.

**Recommended training schedule:**
1. Pretrain 5 epochs with CB-Focal or ASL (all samples)
2. Switch to stratified OHEM (min 40% positives in selected batch)
3. Fine-tune with TPAUC or implicit rate-constrained optimization
4. Apply prevalence-adjusted affine recalibration

---

## 5. Synthesis: RACE Novelty Map

| Literature Thread | What exists | What RACE adds |
|---|---|---|
| **CBMs** | Single-model concept prediction; learned uncertainty (ProbCBM); jointly trained ensemble heads (Credal) | Multi-VLM committee with **model-free** per-image, per-concept reliability from independent frozen models |
| **VLM-as-annotator** | Single VLM labeling (GPT-4V evals); multi-LLM text aggregation (Plaza-del-Arco) | First multi-VLM system for **structured clinical concept extraction from medical images** |
| **Multi-annotator** | Dawid-Skene EM; GLAD with item difficulty; MACE Bayesian competence | Cross-model agreement as **unsupervised reliability signal** without gold labels, at concept-level granularity |
| **Noisy labels** | Co-teaching (training dynamics); MentorNet (learned curriculum); DivideMix (GMM split) | Per-sample reliability from **pre-training-time VLM agreement**, not training dynamics |
| **Knowledge distillation** | VLM-KD (text descriptions → contrastive loss); Hinton (soft logits) | **Reliability-weighted structured concept distillation** in clinical domain |
| **SSL + concepts** | DINOv2 domain adaptation; fusion strategies; OOD detection | Concept-grounded multitask regularization for **127-positive** regime |
| **Tail objectives** | Narasimhan constrained optimization; pAUC; focal variants | Applied to reliability-weighted concept features under **extreme prevalence shift** (5% → 1%) |

### Key novelty claims

1. **Per-image, per-concept reliability from multi-expert VLM agreement** — finest-grained reliability signal in CBM literature. Model-free, training-free, from natural diversity of independent VLMs.

2. **First multi-VLM structured clinical concept extraction for medical imaging** — extending Plaza-del-Arco's "wisdom of LLM crowds" from text annotation to clinical image concept extraction.

3. **Reliability-weighted concept bottleneck as structured distillation** — VLM-KD's distillation architecture + MentorNet/DivideMix's per-sample weighting philosophy, applied through the CBM formalism.

4. **Concept-grounded hard-negative mining from unlabeled corpus** — using VLM-extracted neoplasia-like concept profiles to identify look-alikes in unlabeled data.

5. **Prevalence-shaped objective on concept+embedding fusion** — directly optimizing PPV@90Recall through implicit rate-constrained optimization on the fused [z ∥ c ∥ r] representation.

---

## 6. Headline paper claim

> *"A reliability-filtered, multi-expert VLM-extracted clinical-concept vocabulary — fused with a domain-adapted foundation encoder and trained under a prevalence-shaped objective — turns VLMs that cannot individually exceed 0.06 PPV@90R into the supervision for a model that does, while remaining fully interpretable and offline-deployable."*

---

## 7. Prioritized reading list (top 20 papers)

### Must-cite (directly positioned against)
1. Koh et al. (2020), ICML — Original CBM
2. Kim et al. (2023), ICML — ProbCBM (concept uncertainty)
3. Oikarinen et al. (2023), ICLR — Label-free CBM
4. Espinosa Zarlenga et al. (2022), NeurIPS — CEM
5. Dawid & Skene (1979), JRSS-C — Multi-annotator aggregation
6. Plaza-del-Arco et al. (2024), NLPerspectives — Wisdom of LLM crowds
7. Hinton et al. (2015) — Knowledge distillation
8. VLM-KD (2024), arXiv:2408.16930 — VLM text distillation

### Should-cite (methodological foundation)
9. Whitehill et al. (2009), NeurIPS — GLAD
10. Li et al. (2020), ICLR — DivideMix
11. Kumar et al. (2021), ICML — Implicit rate-constrained optimization
12. Zhu et al. (2022), ICML — Partial AUC with DRO
13. Oquab et al. (2024), TMLR — DINOv2
14. Wang et al. (2023), MICCAI — Endo-FM
15. Sun et al. (2022), ICML — kNN OOD
16. Cui et al. (2019), CVPR — Class-balanced loss
17. Ridnik et al. (2021), ICCV — Asymmetric loss

### Good-to-cite (context)
18. Yuksekgonul et al. (2023), ICLR — PCBM
19. Gilardi et al. (2023), PNAS — ChatGPT as annotator
20. Vorontsov et al. (2024), Nature Medicine — Virchow

---

## 8. Design implications from the literature

### Phase 1 (extraction) — informed by §2
- **Anchor selection:** kNN few-shot (Lu et al., 2024) > random few-shot > zero-shot. Use semantic embeddings, not color histograms.
- **Aggregation:** Apply MACE or Dawid-Skene EM for per-concept, per-model competence estimation. GLAD adds per-concept difficulty modeling.
- **Reliability signal:** Dispersion-based r (current implementation) is a simple first-pass. Consider upgrading to MACE competence scores if >2 VLM families are added.

### Phase 2 (representation) — informed by §3
- **Encoder:** Start from Endo-FM or DINOv2 ViT-B/14. Domain-adapt on ~10k unlabeled frames using 2-stage continued pretraining (~12h on A100).
- **Fusion:** Gated concatenation [z ∥ sigmoid(W·c) ⊙ proj(c)] with concept dropout (p=0.3). Keep concept projection dim ≤128.
- **OOD:** kNN on PCA-reduced encoder embeddings (FAISS, k=5, cosine).

### Phase 3 (prediction) — informed by §4
- **Loss schedule:** CB-Focal/ASL pretrain → stratified OHEM → TPAUC or implicit rate-constrained fine-tune.
- **Calibration:** Affine recalibration (T, b) with prevalence-adjusted weights. NOT plain temperature scaling.
- **Multitask:** Concept reconstruction heads as auxiliary outputs. ≤12 effective parameters in final decision layer (EPP guideline).
- **Hard-negative mining:** Concept-profile similarity on unlabeled corpus to surface neoplasia look-alikes.

### Ablation table (required for paper)
| Config | What it tests |
|---|---|
| z-only (encoder, no concepts) | Do concepts add value over pure SSL? |
| c-only (concepts, no encoder) | Do concepts alone carry enough signal? |
| z + c (no reliability weighting) | Does reliability weighting matter? |
| z + c + r (full fusion) | Full RACE |
| + hard-neg mining | Impact of concept-guided mining |
| + tail objective | Impact of PPV@90R-shaped loss |
| + OOD gating | Impact of negative-manifold density |
| + multitask | Impact of concept reconstruction regularization |

---

## Appendix: Full citation index

### Concept Bottleneck Models
- Koh, P.W., et al. (2020). Concept Bottleneck Models. *ICML*, pp. 5338–5348.
- Mahinpei, A., et al. (2021). Promises and Pitfalls of Black-Box Concept Learning Models. *ICML Workshop*. arXiv:2106.13314.
- Espinosa Zarlenga, M., et al. (2022). Concept Embedding Models. *NeurIPS*, vol. 35. arXiv:2209.09056.
- Yuksekgonul, M., et al. (2023). Post-hoc Concept Bottleneck Models. *ICLR 2023* (Spotlight). arXiv:2205.15480.
- Havasi, M., et al. (2022). Addressing Leakage in Concept Bottleneck Models. *NeurIPS*.
- Kim, E., et al. (2023). Probabilistic Concept Bottleneck Models. *ICML*, pp. 16521–16540. arXiv:2306.01574.
- Oikarinen, T., et al. (2023). Label-free Concept Bottleneck Models. *ICLR 2023*. arXiv:2304.06129.
- Yang, Y., et al. (2023). Language in a Bottle. *CVPR*, pp. 19187–19197. arXiv:2211.11158.
- Xu, B., et al. (2024). Energy-Based Concept Bottleneck Models. *ICLR 2024*. arXiv:2401.14142.
- CREDENCE (2025). Credal Concept Bottleneck Models. arXiv:2602.11219.
- VH-CBM (2025). arXiv:2605.16405.
- VLG-CBM (2024). arXiv:2408.01432.
- SCoOP (2025). arXiv:2603.23853.

### Medical CBMs
- Yan, S., et al. (2023). Robust and Interpretable Medical Image Classifiers via CBMs. arXiv:2310.03182.
- XpertCausal (2025). arXiv:2605.07785.
- CCBM (2024). arXiv:2410.15446.
- Pang, W., et al. (2024). Concept Alignment. *MICCAI 2024*.

### VLM-as-annotator
- Gilardi, F., et al. (2023). ChatGPT outperforms crowd workers. *PNAS*, 120(30), e2305016120.
- Goel, S., et al. (2023). LLMs Accelerate Annotation for Medical Information Extraction. *ML4H*. PMLR 225:82–100.
- Gadre, S.Y., et al. (2023). DataComp. *NeurIPS 2023*. arXiv:2304.14108.
- Li, J., et al. (2024). DataComp-LM. *NeurIPS 2024*. arXiv:2406.11794.
- Plaza-del-Arco, F.M., et al. (2024). Wisdom of Instruction-Tuned LLM Crowds. *NLPerspectives*. arXiv:2307.12973.

### Multi-annotator aggregation
- Dawid, A.P. & Skene, A.M. (1979). Maximum Likelihood Estimation of Observer Error-Rates. *JRSS-C*, 28(1), 20–28.
- Whitehill, J., et al. (2009). Whose Vote Should Count More (GLAD). *NeurIPS 2009*.
- Zhang, Y., et al. (2016). Spectral Methods Meet EM. *JMLR*, 17, 1–44.
- Hovy, D., et al. (2013). Learning Whom to Trust with MACE. *NAACL-HLT*, pp. 1120–1130.

### Noisy label learning
- Han, B., et al. (2018). Co-teaching. *NeurIPS*, pp. 8536–8546. arXiv:1804.06872.
- Jiang, L., et al. (2018). MentorNet. *ICML*, pp. 2304–2313. arXiv:1712.05055.
- Li, J., et al. (2020). DivideMix. *ICLR 2020*. arXiv:2002.07394.
- Szegedy, C., et al. (2016). Rethinking the Inception Architecture (label smoothing). *CVPR*, pp. 2818–2826.
- Lukasik, M., et al. (2020). Does Label Smoothing Mitigate Label Noise? *ICML*. PMLR 119.

### Knowledge distillation
- Hinton, G., et al. (2015). Distilling the Knowledge in a Neural Network. arXiv:1503.02531.
- VLM-KD (2024). arXiv:2408.16930.
- Müller, R., et al. (2019). When Does Label Smoothing Help? *NeurIPS*, 32.

### VLM medical evaluations
- Yan, Z., et al. (2023). Multimodal ChatGPT for Medical Applications. arXiv:2310.19061.
- Wu, S., et al. (2023). Holistic Evaluation of GPT-4V for Biomedical Imaging. arXiv:2312.05256.
- Lecler, A., et al. (2024). GPT-4 Multimodal Performance in Radiology. *European Radiology*.
- Lu, M.Y., et al. (2024). In-context learning for cancer pathology. arXiv:2403.07407.

### SSL & Foundation models
- Oquab, M., et al. (2024). DINOv2. *TMLR*. arXiv:2304.07193.
- Roth, B., et al. (2024). Low-resource DINOv2 finetuning. arXiv:2401.04720.
- MedDINOv3 (2025). arXiv:2509.02379.
- Chen, R.J., et al. (2024). UNI. *Nature Medicine*. doi:10.1038/s41591-024-02857-3.
- Vorontsov, E., et al. (2024). Virchow. *Nature Medicine*, 30(10), 2924–2935.
- Filiot, A., et al. (2024). Phikon-v2. arXiv:2409.09173.
- Huang, Z., et al. (2023). PLIP. *Nature Medicine*, 29, 2307–2316.
- Zhang, S., et al. (2024). BiomedCLIP. *NEJM AI*.
- Lu, M.Y., et al. (2024). CONCH. *Nature Medicine*, 30(3), 863–874.
- Wang, Z., et al. (2023). Endo-FM. *MICCAI 2023*. arXiv:2306.16741.
- Caron, M., et al. (2021). DINO. *ICCV 2021*.
- Zhou, J., et al. (2022). iBOT. *ICLR 2022*.

### OOD detection
- Lee, K., et al. (2018). Mahalanobis OOD. *NeurIPS 2018*.
- Anthony, H. & Kamnitsas, K. (2023). Mahalanobis for Medical OOD. *UNSURE 2023* (Best Paper).
- Sun, Y., et al. (2022). kNN OOD. *ICML 2022*. PMLR 162:20827–20840.
- Woodland, M.C.K., et al. (2024). Dimensionality Reduction + kNN OOD. *MELBA*.
- Liu, W., et al. (2020). Energy-based OOD. *NeurIPS 2020*. arXiv:2010.03759.

### Loss functions & calibration
- Kumar, A., et al. (2021). Implicit Rate-Constrained Optimization. *ICML*. 
- Narasimhan, H., et al. (2024). Consistent Multiclass Algorithms. *JMLR*, 25.
- Narasimhan, H., et al. (2019). Optimizing Generalized Rate Metrics with Three Players. *NeurIPS*.
- Zhu, D., et al. (2022). Partial AUC with DRO. *ICML*. PMLR 162.
- Yang, Z., et al. (2021). Two-way Partial AUC. *ICML*. PMLR 139.
- Lin, T.Y., et al. (2017). Focal Loss. *ICCV*, pp. 2980–2988. arXiv:1708.02002.
- Cui, Y., et al. (2019). Class-Balanced Loss. *CVPR*.
- Ridnik, T., et al. (2021). Asymmetric Loss. *ICCV*.
- Cao, K., et al. (2019). LDAM. *NeurIPS*. arXiv:1906.07413.
- Godau, P., et al. (2025). Navigating Prevalence Shifts. arXiv:2303.12540.
- Riley, R.D., et al. (2019). Minimum Sample Size for Binary Prediction. *Statistics in Medicine*.
