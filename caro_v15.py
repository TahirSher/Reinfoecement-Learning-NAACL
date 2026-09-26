#!/usr/bin/env python3
"""CARO v15 -- Counterfactual Affect Reward Optimisation on EmoWOZ, end to end.

Stages: sft -> simulator -> validate -> corpus -> reward -> train -> eval -> report  (or `all`),
then analysis (ablations, sensitivity, length analysis, external evaluator, claims, LaTeX), and
the human study (human-export -> annotate -> human-analyze).

Problem.  RLHF needs explicit preference labels.  In deployed customer-service dialogue the
customer almost never leaves them, but the customer's NEXT turn carries an implicit, affective
signal of how the agent's turn landed.  CARO turns that implicit signal into a reward: a customer
simulator scores counterfactual agent responses by the affect of the customer reply they are
expected to elicit, a reward model is fitted to those counterfactual outcomes, and the agent is
optimised against it with GRPO.  The target is emotional appropriateness WITHOUT loss of task
information, so information retention is measured and tested for non-inferiority.

What changed from v14 (all prompted by the failed v14 validate run; stages 3+ must be re-run):
  V1  Root cause of the failure.  With --outcome-mode auto and no outcome_mode.json in the run
      directory, v14 SILENTLY used the Monte Carlo ("sample") estimator.  Re-running that estimator on
      the same responses flipped 34.6% of within-context pairs (the replication null), so the flip
      ceiling of 10% could not be met whatever the length behaviour: the TOST passed, tau_corrected
      was 0, and pure length flipped FEWER pairs (17.0%) than noise did.  The validate stage now selects
      the estimator itself (TRAIN-split data only) and nothing falls back silently.
  V2  Estimator selection requires determinism.  The deterministic panel expectation is selected
      whenever its human anchor CI excludes zero; v14 maximised the anchor alone, which cannot see that
      a noisy estimator is unusable for within-context ranking.  This CHANGES the pre-registered
      selection rule; report the change and the reason (V1) in the paper.
  V3  Affect-neutral filler banks.  The courtesy bank used to train the simulator and to fit h(L)
      measured sentiment +0.42: fitting a "pure-length" correction on it conflates length with affect,
      and training the simulator to IGNORE courtesy tails removes part of the signal CARO rewards.
      New runs use three disjoint neutral banks (train / calib / heldout, one role each); simulators
      trained on the old bank are recognised (simulator_meta.json) and flagged for retraining.
  V4  Stability-constrained panel temperature.  T maximises the calibration anchor subject to a
      FIT-half flip rate <= 0.8 x the ceiling (after h(L)), inside both the cross-fitted projector
      probe and the final freeze.  Only FIT-half data enter; the gate is untouched.
  V5  One simulator load in validate (v14 loaded the 3B model twice to build the panel); the adapter is
      loaded into the live LoRA modules instead of stacking a second PeftModel.
  V6  Attribution: a sampled estimator whose replication null exceeds the length contrast is
      reported as Monte Carlo noise, not as "surface brittleness".
None of these changes relaxes a threshold.  If the gate still fails with the deterministic estimator,
the failure is real and must be reported.

What changed from v13 (each item names the reviewer finding it answers; the stage that has to be
re-run is in brackets):
  R1  Baselines.  sentiment_only is a first-class reward function (no monkey-patching) and is in
      the default arm list; an offline DPO arm is trained on best/worst-outcome pairs from the
      SAME corpus split the reward model saw (Rafailov et al., 2023); two eval-only controls are
      added: best-of-N reranking of SFT samples by the reward model (sft_bon_rm), and a
      length-matched SFT control that, per context, picks the SFT sample whose word count is
      closest to the CARO response (sft_lenmatch).  [train, eval]
  R2  Evaluator circularity.  eval-external re-scores every arm with an OUT-OF-FAMILY customer
      (a different base LLM, no simulator adapter) whose replies are labelled by an independent
      emotion classifier that no training stage used.  It is still automatic; the human study is
      the real fix.  [eval-external]
  R3  Within-context validity.  human-export writes a blinded, randomised pairwise packet that
      includes simulator-validity items (two responses to the SAME context) and attention checks;
      human-analyze estimates within-context simulator/human agreement, arm win rates, Krippendorff
      alpha and non-inferiority on information.  [human-*]
  R4  Reward gold anchor.  Reported with a clustered CI everywhere, and DEV selection now also
      requires that the anchor is not degraded relative to the plain ranker by more than a
      pre-registered tolerance.  [reward]
  R5  Reward sensitivity.  The grid always contains the DEV-selected cell, uses TEST accuracy (v13
      mixed calibration accuracy into it), and reports which criterion fails in each cell.
  R6  Simulator ablation.  The validation stage stores its exact gate sample; the ablation
      re-scores THAT sample (a replica that must reproduce the gate) and an independent larger
      sample with the same variants-per-context, both with tau and flip CIs.  An optional arm
      retrains the simulator WITHOUT the invariance regulariser, which is the control needed to
      attribute invariance to h(L) or to the regulariser.  [validate, ablate-simulator]
  R7  Reward ablations.  The "no abstention" row (identical by construction) is gone; abstention
      is ablated where it acts, in RL (arm caro_no_abstain).  "No pessimism" replaces it.  Each
      variant is refitted over several seeds, and "no effect" is claimed only through a TOST.
  R8  ICC / replicate reliability are reported as NOT APPLICABLE for the deterministic panel
      estimator (v13 printed 1.0).  A non-trivial responsiveness statistic replaces them: the
      share of outcome variance that lies within context.
  R9  p-values.  The sign-flip test is exact when the cluster count allows, otherwise uses 20000
      draws; a p at the Monte Carlo floor is written as "< 1/(B+1)".  The reward permutation test
      scores once, not once per permutation, and uses 2000 permutations.
  R10 Naming.  RatioScorer -> ResidualMLPScorer; "platt" -> linear recalibration (OLS), with a guard
      against a negative slope, which would silently invert the reward.
  R11 Config / CLI drift.  Every CLI default is read from Config, so the two cannot disagree (v13:
      Config.panel_size=24 but the CLI default was 48, which is what produced the effective panel
      size of 47.1).  The run manifest records the realised panel size.
  R12 Length.  length-analysis: paired contrasts at equal length (regression intercept), within
      the matched stratum |delta words| <= 2, and against the design-based sft_lenmatch control.
Defects found in the v13 code itself and fixed here:
  D1  During GRPO the reward-model features were computed by the policy being TRAINED, whereas the
      reward model was fitted on features of the frozen SFT proposal.  The optimised reward was
      therefore not the validated reward.  Features are now computed under the frozen SFT weights.
      [train, eval, report; the v13 CARO numbers do not stand]
  D2  Pseudo-replication.  Contrasts clustered on "seed|dialogue", so one dialogue counted as three
      independent clusters, and stage_all COPIED the SFT eval rows across seeds.  Clusters are now
      dialogues, and SFT is evaluated under every eval seed.  [report]
  D3  The sign-flip statistic (unweighted mean of cluster means) was not the estimator whose CI was
      reported (row mean).  The test now flips cluster SUMS, which targets the row mean.
  D4  The objective's information clause had no measurement: info_recall (recall of the gold turn's
      task-critical tokens) is recorded for every response and tested for non-inferiority.
  D5  The GRPO ratio is identically 1 (one update per batch), so the clipped objective reduces to
      REINFORCE with a k3 KL penalty; the docstring now says so, and so should the paper.

Earlier history:

What changed from v8, and why (details in the docstrings of the named objects):
  * InterventionalLengthCalibration replaces the observational within-context length spline.
    The v8 spline adjusted for length, a descendant of content ("bad control"), and thereby paid
    reward for content-free padding; that is exactly what the v8 gate caught (delta +0.0105 after
    the spline vs +0.0013 before it).  h(L) is now identified from paired do(length) contrasts by
    first differences, so every context-level confounder cancels.
  * Gates perturb with a HELD-OUT filler bank that no training or fitting step has seen; v8
    re-used the bank the simulator had been explicitly regularised on (circular).
  * The corpus no longer applies a second observational residualisation on top of the first.
  * Reward training uses counterfactual logit pairing on padded copies, its weight chosen on
    DEV, and the reward gate adds a held-out padding TOST: the RM cannot re-acquire length as a
    proxy through its features, which is the channel GRPO would actually exploit.
  * Cluster-bootstrap statistics that group within clusters keep duplicated draws distinct.
  * Pre-registered thresholds are restored (TOST margin 0.15 sd, flip ceiling 0.10); v8's CLI
    had drifted to 0.20 / 0.12 while its Config and docstrings said 0.15 / 0.10.
  * Cost: exact memo of panel log-probabilities, free out-of-fold projector probes, cached
    calibration, no replication sweep for the deterministic estimator, and a self-verifying
    tail-logit scorer.

What changed from v9 (v9 stage-1..4 artefacts stay valid; only the reward stage must re-run):
  * The reward validity gate no longer uses an ABSOLUTE within-context length threshold.  That
    threshold contradicted the v9 argument against observational length adjustment: the labels
    themselves carry a content-driven length association, so a reward that reproduces it is
    correct, not broken.  The gate now tests the reward's length dependence AGAINST THE LABELS'
    (excess-over-labels), keeps the interventional held-out-padding TOST as the hard
    reward-hacking gate, and retains a loose absolute backstop for degenerate rewards.
  * Reward training gains a within-group length-calibration penalty that pulls the reward's
    standardised length slope towards the LABEL slope (not towards zero).  Its weight is
    selected on DEV together with the pairing weight.
  * The fast scorer is verified in OUTCOME units against the reference path, with the reference
    path's own batch-composition noise as the baseline.  v9's nats-scale tolerance made a
    quantisation-noise disagreement look like a bug and silently fell back to the slow scorer
    for a whole corpus build (~14 h).
  * The gold-response anchor of the reward is reported and warned on.

What changed from v10 (only the RL stage; stages 1-5 artefacts stay valid):
  * The KL reference is the SFT policy, not the base model.  v10 computed the reference with the
    LoRA adapter DISABLED, so the penalty measured the constant SFT-vs-base divergence (~1.1
    nats) instead of the RL drift, the adaptive coefficient saturated at its ceiling, and the
    objective was dominated by a term pulling the policy back to the pre-SFT base model.  That is
    why reward fell (0.070 -> 0.008) and hygiene decayed while KL rose.  The reference is now a
    frozen snapshot of the trainable weights taken at the start of GRPO.
  * GRPO is token-level (DeepSeekMath, Shao et al. 2024): per-token importance ratios, clipped
    surrogate and k3 KL averaged over generated tokens.  v10 used the sequence MEAN log-prob, so
    the "ratio" was a geometric mean and the clip range and KL target had no token-level meaning.
  * Hygiene failures no longer enter the group mean and sd of the reward.  Substituting -1 for a
    malformed sample inflated the within-group sd and crushed the affect signal into numerical
    noise; malformed samples now carry a fixed negative advantage instead.
  * KL divergence is monitored with an abort, and the requires_grad->float warnings are gone.

What changed from v11 (RL stage only; stages 1-5 artefacts stay valid):
  * Advantages are no longer divided by the GROUP standard deviation.  Dividing by a per-group sd
    rescales every group to unit spread, so a group whose six samples differ by reward noise
    produces advantages as large as a group with a real quality difference; with a within-group
    reward sd of ~0.04 that is almost pure noise amplification, and it showed up as |adv| rising
    0.48 -> 0.82 while the reward spread FELL 0.060 -> 0.041 (Liu et al., 2025, make the same
    argument for Dr. GRPO).  Advantages are now centred within the group and scaled by a single
    running estimate of the within-group reward sd, so weak groups contribute weak gradients.
  * Groups whose reward spread is inside the reward model's own ensemble uncertainty contribute
    no gradient at all (signal-to-noise gate), instead of being amplified to unit scale.
  * The token-level loss is averaged per sequence and then over sequences, not over all tokens in
    the batch.  Batch-token normalisation weights a long sample's gradient in proportion to its
    length, which is a direct length-inflation pressure.
  * The KL coefficient has a floor, so the anchor cannot vanish once drift is small.
  * Length and hygiene drift against the SFT policy are tracked and abort the run, the same way
    KL drift does.

What v13 adds (analysis only; no estimator or training change):
  * ablate-reward  -- reward-stage component ablations (pairing, length calibration, abstention,
    ensemble, and the length-proxy control) on the frozen corpus, CPU only.
  * ablate-simulator -- the paper's central causal claim, priced at ONE GPU sweep: the same
    sampled responses are scored under {no correction, v8 observational spline, v12
    interventional h(L)} x {train filler bank, held-out bank}, so the reward-hacking channel and
    the circularity of a train-bank gate are both quantified rather than asserted.
  * sensitivity -- conclusions under a grid of analysis choices: TOST margin, tau ceiling, flip
    ceiling, panel temperature, reward regularisation weights, RL signal-to-noise gate.
  * report -- pooled paired cluster-bootstrap contrasts (v12 averaged per-seed p-values, which is
    not a valid combination), sign-flip permutation p, Holm across arms x metrics, Cliff's delta
    and d_z effect sizes, TOST equivalence for null results, minimum detectable effect, and a
    seed/context variance decomposition.
  * paper -- runs the above and writes LaTeX tables plus a reproducibility checklist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import time
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-12
Z975 = 1.959963984540054

ZENODO_FILES = {
    "emowoz-multiwoz.json": "https://zenodo.org/records/6506504/files/emowoz-multiwoz.json?download=1",
    "emowoz-dialmage.json": "https://zenodo.org/records/6506504/files/emowoz-dialmage.json?download=1",
    "data-split.json": "https://zenodo.org/records/6506504/files/data-split.json?download=1",
}

EMOTION_NAMES = {
    0: "neutral", 1: "fearful", 2: "dissatisfied", 3: "apologetic",
    4: "abusive", 5: "excited", 6: "satisfied",
}

EMOTION_VALENCE = {0: 0.0, 1: -1.0, 2: -1.0, 3: -1.0, 4: -1.0, 5: 1.0, 6: 1.0}

EMOTION_ORDINAL = {4: -2.0, 2: -1.5, 1: -1.0, 3: -0.5, 0: 0.0, 5: 1.0, 6: 1.5}

SPLIT_ALIASES = {"train": "train", "dev": "valid", "valid": "valid", "validation": "valid", "test": "test"}

NEUTRAL_TAILS = (
    "Let me know if that works for you.",
    "I hope that helps.",
    "Please let me know if you need anything else.",
    "Just let me know how you would like to proceed.",
    "I am happy to help further if needed.",
    "Do let me know if you have any other questions.",
    "I will be glad to assist with anything else.",
    "Feel free to tell me if you would prefer something different.",
)

NEUTRAL_HEADS = (
    "Of course.", "Certainly.", "Sure thing.", "Absolutely.",
    "No problem at all.", "Thank you for waiting.",
)

# v9: a HELD-OUT filler bank.  The "train" bank above is the one the simulator was explicitly
# regularised on during stage 2 (invariance penalty on the squared mean-logprob difference), and
# the one on which the interventional length calibration and the logit projector are fitted.  A
# gate that re-uses it only certifies invariance to phrasings the model was TRAINED to ignore,
# which is circular.  The gate therefore perturbs with this disjoint bank, which never appears in
# any training or fitting step.  Phrases are informationally empty, affect-neutral statements of
# record, and deliberately avoid courtesy/thanks/apology lexemes that sentiment models score.
HELDOUT_TAILS = (
    "That is the information I have on this.",
    "Those are the details on my side.",
    "This is what the system currently shows.",
    "I have noted this on the record.",
    "That covers the points you raised.",
    "This is the current status as it stands.",
    "I have checked this against the listing.",
    "Those are the relevant details for now.",
)
HELDOUT_HEADS = ("Okay.", "Right.", "I see.", "Understood.", "Noted.", "Alright.")

# v15: the v8-v14 "train" bank above is NOT affect-neutral: the validate log measures its mean sentiment at
# +0.42 (held-out bank +0.08), because it is made of courtesy phrases ("I hope that helps", "happy to
# help").  Two consequences.  (i) Fitting h(L) and the projector on it confounds length with affect: the
# fitted "pure-length" curve also absorbs the effect of added courtesy.  (ii) Training the simulator to be
# INVARIANT to courtesy tails teaches it to ignore exactly the emotional wording this work is about.  The
# bank is kept, renamed courtesy_v13, only so that simulators trained on it can still be analysed; new
# runs use three disjoint affect-neutral banks with one role each:
#   train    simulator invariance regularisation (stage 2)
#   calib    every FITTED correction: h(L), the logit projector, the panel temperature constraint,
#            orbit averaging and the reward's counterfactual logit pairing
#   heldout  every GATE (simulator validation, reward padding TOST); never seen by any fit
TRAIN_TAILS = (
    "That is what the booking system lists.",
    "Those details are from the current timetable.",
    "This matches the entry in the database.",
    "The listing was last updated this week.",
    "I have entered that into the system.",
    "The record shows the same details.",
    "That is the entry on file for this.",
    "Those figures come from the provider.",
)
TRAIN_HEADS = ("So.", "Now.", "Well.", "Let me see.", "One moment.", "Checking.")
CALIB_TAILS = (
    "These details are taken from the listing.",
    "That is how it appears on the schedule.",
    "The system has this recorded.",
    "This is what the provider has listed.",
    "The information above is from the database.",
    "That entry is in the current records.",
    "These are the details held on file.",
    "The schedule shows it this way.",
)
CALIB_HEADS = ("Ok.", "Yes.", "So then.", "Right then.", "Looking now.", "Here it is.")

FILLER_BANKS = {"courtesy_v13": (NEUTRAL_TAILS, NEUTRAL_HEADS), "train": (TRAIN_TAILS, TRAIN_HEADS),
                "calib": (CALIB_TAILS, CALIB_HEADS), "heldout": (HELDOUT_TAILS, HELDOUT_HEADS)}
FIT_BANK = "calib"
_all_fill = [x for b in FILLER_BANKS.values() for part in b for x in part]
assert len(_all_fill) == len(set(_all_fill)), "filler banks must be disjoint"
VERSION = "v15"
COMPATIBLE_VERSIONS = ("v9", "v10", "v11", "v12", "v13", "v14", "v15")   # stage-3/4/5 artefacts from v9 onwards remain valid

# Arms that are optimised (need a train stage) versus arms that only re-use the SFT adapter.
TRAINED_ARMS = ("sentiment_only", "dpo", "caro", "caro_no_abstain")
EVAL_ONLY_ARMS = ("sft", "sft_bon_rm", "sft_lenmatch")
KNOWN_ARMS = TRAINED_ARMS + EVAL_ONLY_ARMS

# Rude/irrelevant catch responses for the human-study attention checks.  They are never shown as a
# system output and never enter any training or scoring step.
CATCH_RESPONSES = (
    "That is not my problem, figure it out yourself.",
    "I do not care what you want. Stop asking me.",
    "Whatever. Ask someone else.",
)

_STOPWORDS = frozenset(
    "a an the and or but if then so to of in on at for from by with about as is are was were be been being "
    "it its this that these those there here i you we they he she me my your our their his her them us "
    "do does did doing have has had can could would should will shall may might must not no yes please "
    "thank thanks sure okay ok would like just also any some all what which who whom when where why how "
    "am pm let know need help anything something nothing everything else another other "
    # courtesy and affect lexemes are NOT information: counting them would confound information
    # retention with the very affect the policy is optimised for
    "great glad happy sorry unfortunately welcome afraid apologise apologize apologies certainly "
    "absolutely course problem wonderful perfect lovely hope enjoy hello goodbye good nice have "
    "day evening morning afternoon".split())

EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")
ROLE_LEAK_RE = re.compile(r"(?im)^\s*(customer|user|agent|system|assistant)\s*:")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def make_logger(out_dir: Path, tag: str) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(f"caro.{tag}")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    lg.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    lg.addHandler(sh)
    fh = logging.FileHandler(out_dir / f"{tag}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    lg.addHandler(fh)
    return lg


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha_of(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=_json_default).encode()).hexdigest()[:16]


def rankdata(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    n = a.size
    if n == 0:
        return np.zeros(0, float)
    order = np.argsort(a, kind="mergesort")
    s = a[order]
    r = np.empty(n, float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 3:
        return float("nan")
    rx = rankdata(x[ok])
    ry = rankdata(y[ok])
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    d = math.sqrt(float(rx @ rx) * float(ry @ ry))
    return float(rx @ ry / d) if d > EPS else float("nan")


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 3:
        return float("nan")
    a = x[ok] - x[ok].mean()
    b = y[ok] - y[ok].mean()
    d = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b / d) if d > EPS else float("nan")


def fisher_ci(rho: float, n: int) -> Tuple[float, float]:
    if not math.isfinite(rho) or n < 6 or abs(rho) >= 1.0:
        return (float("nan"), float("nan"))
    z = 0.5 * math.log((1 + rho) / (1 - rho))
    se = 1.0 / math.sqrt(n - 3)
    return (float(math.tanh(z - Z975 * se)), float(math.tanh(z + Z975 * se)))


def holm(pvals: Sequence[float]) -> List[float]:
    p = list(pvals)
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    out = [0.0] * m
    prev = 0.0
    for k, i in enumerate(order):
        v = min(1.0, (m - k) * p[i])
        prev = max(prev, v)
        out[i] = prev
    return out


def loglen(texts: Sequence[str]) -> np.ndarray:
    return np.asarray([math.log1p(len(str(t).split())) for t in texts], float)


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def salient_tokens(text: str) -> set:
    """Task-critical tokens of an agent turn: anything with a digit (times, prices, reference
    numbers, phone numbers), capitalised words that do not start a sentence (venue, street and
    place names), and remaining content words of >= 4 letters that are not stop words.  This is a
    deliberately transparent proxy for "the information the turn conveys"; it is not an NLI model
    and it is reported as such."""
    toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-:/.]*[A-Za-z0-9]|[A-Za-z0-9]", norm_text(text))
    out = set()
    prev_end = True
    for raw in toks:
        w = raw.lower()
        if any(ch.isdigit() for ch in raw):
            out.add(w)
        elif raw[0].isupper() and not prev_end and w not in _STOPWORDS:
            out.add(w)
        elif len(w) >= 4 and w not in _STOPWORDS:
            out.add(w)
        prev_end = raw.endswith((".", "!", "?"))
    return out


def info_recall(response: str, reference: str) -> float:
    """Fraction of the reference turn's salient tokens that the response also carries.  NaN when
    the reference has none (e.g. a pure greeting), so such turns drop out of the mean instead of
    counting as perfect retention."""
    ref = salient_tokens(reference)
    if not ref:
        return float("nan")
    return float(len(ref & salient_tokens(response)) / len(ref))


def format_p(p: float, n_draws: Optional[int] = None, exact: bool = False) -> str:
    """Report a resampling p-value honestly.  A Monte Carlo p equal to its floor 1/(B+1) only says
    that no draw was as extreme as the data; it is written as an inequality, not as a value."""
    if p is None or not math.isfinite(p):
        return "n/a"
    if n_draws and not exact and p <= 1.0 / (n_draws + 1) + 1e-15:
        return f"< {1.0 / (n_draws + 1):.2g}"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


@dataclass
class NaturalCubicBasis:
    knots: np.ndarray
    center: float = 0.0
    scale: float = 1.0

    @staticmethod
    def from_data(x: np.ndarray, n_knots: int = 8, span: float = 0.98) -> "NaturalCubicBasis":
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if x.size < 8:
            raise ValueError("NaturalCubicBasis.from_data: too few points")
        n_knots = int(max(3, min(n_knots, max(3, x.size // 10))))
        lo = (1.0 - span) / 2.0
        q = np.quantile(x, np.linspace(lo, 1.0 - lo, n_knots))
        q = np.maximum.accumulate(q + 1e-9 * np.arange(q.size))
        if q[-1] - q[0] < 1e-8:
            raise ValueError("NaturalCubicBasis.from_data: degenerate range")
        return NaturalCubicBasis(q, float(np.mean(x)), float(np.std(x)) or 1.0)

    @property
    def n_cols(self) -> int:
        return 1 + max(0, len(self.knots) - 2)

    def _d(self, x: np.ndarray, k: int) -> np.ndarray:
        xi = self.knots
        K = len(xi)
        num = np.clip(x - xi[k], 0.0, None) ** 3 - np.clip(x - xi[K - 1], 0.0, None) ** 3
        return num / max(xi[K - 1] - xi[k], EPS)

    def design(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        K = len(self.knots)
        cols = [(x - self.center) / self.scale]
        if K >= 3:
            dl = self._d(x, K - 2)
            for k in range(K - 2):
                cols.append((self._d(x, k) - dl) / (self.scale ** 3))
        return np.stack(cols, 1)

    def state(self) -> Dict[str, Any]:
        return {"knots": self.knots.tolist(), "center": self.center, "scale": self.scale}

    @staticmethod
    def load(d: Dict[str, Any]) -> "NaturalCubicBasis":
        return NaturalCubicBasis(np.asarray(d["knots"], float), float(d["center"]), float(d["scale"]))


def group_demean(v: np.ndarray, gid: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    _, inv = np.unique(np.asarray(gid), return_inverse=True)
    m = inv.max() + 1
    cnt = np.bincount(inv, minlength=m).astype(float)
    if v.ndim == 1:
        s = np.bincount(inv, weights=v, minlength=m)
        return v - (s / np.maximum(cnt, 1.0))[inv]
    out = np.empty_like(v)
    for j in range(v.shape[1]):
        s = np.bincount(inv, weights=v[:, j], minlength=m)
        out[:, j] = v[:, j] - (s / np.maximum(cnt, 1.0))[inv]
    return out


def within_group_spearman(y: np.ndarray, x: np.ndarray, gid: Sequence[Any], min_size: int = 3) -> Dict[str, float]:
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    _, inv = np.unique(np.asarray(list(gid)), return_inverse=True)
    vals = []
    for g in range(inv.max() + 1 if inv.size else 0):
        m = inv == g
        if int(m.sum()) < min_size or np.ptp(x[m]) <= 0 or np.ptp(y[m]) <= 0:
            continue
        r = spearman(y[m], x[m])
        if math.isfinite(r):
            vals.append(r)
    if not vals:
        return {"mean": float("nan"), "sd": float("nan"), "se": float("nan"), "n_groups": 0}
    v = np.asarray(vals, float)
    sd = float(v.std(ddof=1)) if v.size > 1 else float("nan")
    return {"mean": float(v.mean()), "sd": sd,
            "se": float(sd / math.sqrt(v.size)) if v.size > 1 else float("nan"),
            "n_groups": int(v.size)}


def cluster_bootstrap_ci(stat_fn: Callable[..., float], cluster: Sequence[Any],
                         n_boot: int = 1000, seed: int = 0, conf: float = 0.95,
                         with_copy: bool = False) -> Tuple[float, float, float]:
    """Percentile cluster bootstrap.

    with_copy=True passes stat_fn(idx, copy) where copy[j] is the draw slot of element j.  Any
    statistic that groups by a within-cluster id (within-context variance, within-context rank
    correlation) MUST use it: when a cluster is drawn twice its copies otherwise share group ids,
    so two independent draws of a context are silently merged into one over-sized group whose
    duplicated values bias within-group variances and rank correlations (v8 bug, fixed here)."""
    cl = np.asarray(list(cluster))
    uniq, inv = np.unique(cl, return_inverse=True)
    idx_by = [np.flatnonzero(inv == g) for g in range(uniq.size)]
    full = np.arange(cl.size)
    point = float(stat_fn(full, np.zeros(cl.size, int)) if with_copy else stat_fn(full))
    rng = np.random.default_rng(seed)
    draws = np.full(n_boot, np.nan)
    for b in range(n_boot):
        pick = rng.integers(0, uniq.size, uniq.size)
        idx = np.concatenate([idx_by[j] for j in pick])
        try:
            if with_copy:
                copy = np.concatenate([np.full(idx_by[j].size, slot, int) for slot, j in enumerate(pick)])
                draws[b] = stat_fn(idx, copy)
            else:
                draws[b] = stat_fn(idx)
        except Exception:
            pass
    d = draws[np.isfinite(draws)]
    if d.size < 50:
        return (point, float("nan"), float("nan"))
    a = (1.0 - conf) / 2.0
    return (point, float(np.quantile(d, a)), float(np.quantile(d, 1 - a)))


def boot_groups(gid_sub: np.ndarray, copy: np.ndarray) -> np.ndarray:
    """Group ids that keep every bootstrap copy of a context distinct."""
    _, g = np.unique(np.asarray(gid_sub), return_inverse=True)
    return g.astype(np.int64) * (int(np.max(copy)) + 1 if np.size(copy) else 1) + np.asarray(copy, np.int64)


def within_group_var(y: np.ndarray, gid: np.ndarray) -> float:
    """Mean unbiased within-group variance over groups of size >= 2."""
    y = np.asarray(y, float)
    _, inv = np.unique(np.asarray(gid), return_inverse=True)
    m = inv.max() + 1 if inv.size else 0
    cnt = np.bincount(inv, minlength=m).astype(float)
    s1 = np.bincount(inv, weights=y, minlength=m)
    s2 = np.bincount(inv, weights=y * y, minlength=m)
    ok = cnt >= 2
    if not ok.any():
        return float(np.var(y, ddof=1)) if y.size > 1 else float("nan")
    v = (s2[ok] - s1[ok] ** 2 / cnt[ok]) / (cnt[ok] - 1.0)
    return float(np.mean(np.maximum(v, 0.0)))


def within_group_sd(y: np.ndarray, gid: np.ndarray) -> float:
    """Mean within-group sample sd (pre-registered scale of the TOST margin)."""
    y = np.asarray(y, float)
    _, inv = np.unique(np.asarray(gid), return_inverse=True)
    vals = [float(np.std(y[inv == g], ddof=1)) for g in range(inv.max() + 1 if inv.size else 0)
            if int((inv == g).sum()) > 1]
    return float(np.mean(vals)) if vals else float("nan")


class InterventionalLengthCalibration:
    """Pure-length response curve identified by paired content-free interventions.

    Why v8 failed.  v8 removed length from the outcome with an OBSERVATIONAL within-context
    spline: it regressed the outcome of naturally sampled variants on their log-length.  Natural
    variants that are longer also SAY different things, so that slope estimates the association
    of length with content, not the causal effect of length.  In the causal graph
        content -> length,  content -> outcome,  length -> outcome (pure-length path)
    length is a descendant of the treatment of interest (content), and adjusting for it is the
    textbook "bad control" (Cinelli, Forney & Pearl, 2022; Angrist & Pischke, 2009, sec. 3.2.3).
    The run log shows the consequence exactly: on the FIT half the raw estimator was already
    invariant to padding (delta=+0.0013, TOST PASS), and the gate on the TEST half only failed
    AFTER the observational spline was applied (delta=+0.0105, about 0.5 within-context sd).  The
    spline had learnt "longer natural responses score lower" (within rho -0.155) and therefore
    ADDED reward for every extra word, including content-free padding -- a reward-hacking channel
    that GRPO would have exploited.  The gate was right; the correction was wrong.

    Estimator.  Let h(L) be the average causal response of the outcome to log-length L when the
    content is held fixed.  For a paired intervention (x, r) -> (x, pad(r)) the context and the
    content are identical, so every context-level confounder cancels in the first difference
        d_i = O(x_i, pad(r_i)) - O(x_i, r_i) = h(L1_i) - h(L0_i) + u_i ,   E[u_i | L0_i, L1_i] = 0,
    which is the first-difference (fixed-effects) identification argument of panel econometrics
    (Wooldridge, 2010, ch. 10) applied to a do(length) contrast.  h is a natural cubic spline in
    log-length; only its differences are identified, so it carries no intercept.  The non-linear
    coefficients are ridge-penalised with lambda chosen by generalised cross-validation (Craven &
    Wahba, 1979).  Variation in the base length L0 identifies the curvature from a single
    padding rung, and the fitting sample mixes two rungs to make that identification robust.

    Scope.  apply() subtracts h(L) from every outcome.  This removes the PURE-LENGTH component of
    any response's score (a controlled direct effect, Pearl 2001) and leaves every content effect
    untouched, including content effects that happen to correlate with length.  It is fitted on
    the TRAIN filler bank only and is gated on the disjoint HELD-OUT bank, so a pass certifies
    generalisation to padding phrasings the calibration never saw."""

    def __init__(self, n_knots: int = 6, span: float = 0.98):
        self.n_knots = int(n_knots)
        self.span = float(span)
        self.basis: Optional[NaturalCubicBasis] = None
        self.beta: Optional[np.ndarray] = None
        self.lam: float = float("nan")
        self.diag: Dict[str, Any] = {}

    @property
    def fitted(self) -> bool:
        return self.basis is not None and self.beta is not None

    @staticmethod
    def _solve(X: np.ndarray, d: np.ndarray, lam: float) -> Tuple[np.ndarray, float]:
        p = X.shape[1]
        P = np.eye(p)
        P[0, 0] = 1e-6          # the linear term is (almost) unpenalised
        G = X.T @ X
        A = G + lam * P + 1e-10 * np.eye(p)
        beta = np.linalg.solve(A, X.T @ d)
        edf = float(np.trace(np.linalg.solve(A, G)))
        return beta, edf

    def fit(self, base_texts: Sequence[str], pert_texts: Sequence[str], d: np.ndarray,
            cluster: Optional[Sequence[Any]] = None, logger: Optional[logging.Logger] = None,
            grid: Sequence[float] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4),
            n_folds: int = 5, seed: int = 0) -> "InterventionalLengthCalibration":
        d = np.asarray(d, float)
        L0 = loglen(base_texts)
        L1 = loglen(pert_texts)
        if not (L0.size == L1.size == d.size):
            raise ValueError("InterventionalLengthCalibration.fit: ragged inputs")
        ok = np.isfinite(d) & np.isfinite(L0) & np.isfinite(L1) & (np.abs(L1 - L0) > 1e-9)
        if int(ok.sum()) < 40:
            raise ValueError(f"InterventionalLengthCalibration.fit: only {int(ok.sum())} usable pairs")
        L0, L1, d = L0[ok], L1[ok], d[ok]
        cl = (np.asarray(list(cluster))[ok] if cluster is not None else np.arange(d.size))
        self.basis = NaturalCubicBasis.from_data(np.concatenate([L0, L1]), self.n_knots, self.span)
        X = self.basis.design(L1) - self.basis.design(L0)
        n = d.size
        best = (float("inf"), float(grid[0]), 0.0)
        for lam in grid:
            try:
                beta, edf = self._solve(X, d, lam)
            except np.linalg.LinAlgError:
                continue
            den = n - edf
            if den <= 1e-6:
                continue
            gcv = n * float(np.sum((d - X @ beta) ** 2)) / den ** 2
            if gcv < best[0]:
                best = (gcv, float(lam), edf)
        self.lam = best[1]
        self.beta, edf = self._solve(X, d, self.lam)
        fit_d = X @ self.beta
        # Honest out-of-sample check: cluster-disjoint K-fold residual mean shift.
        uniq = np.unique(cl)
        rng = np.random.default_rng(seed)
        fold_of = {u: i % n_folds for i, u in enumerate(uniq[rng.permutation(uniq.size)])}
        fold = np.asarray([fold_of[c] for c in cl], int)
        oos = np.full(n, np.nan)
        for k in range(n_folds):
            tr, te = fold != k, fold == k
            if te.sum() == 0 or tr.sum() < 20:
                continue
            b_k, _ = self._solve(X[tr], d[tr], self.lam)
            oos[te] = d[te] - X[te] @ b_k
        sst = float(np.sum((d - d.mean()) ** 2)) or EPS
        self.diag = {
            "kind": "interventional", "n_pairs": int(n), "lam": self.lam, "edf": float(edf),
            "mean_delta_raw": float(d.mean()), "mean_delta_fitted": float(fit_d.mean()),
            "mean_delta_residual_in_sample": float((d - fit_d).mean()),
            "mean_delta_residual_cross_fitted": float(np.nanmean(oos)) if np.isfinite(oos).any() else float("nan"),
            "r2_in_sample": float(1.0 - np.sum((d - fit_d) ** 2) / sst),
            "mean_words_added": float(np.mean(np.expm1(L1) - np.expm1(L0))),
            "slope_per_loglen": float(self.beta[0] / self.basis.scale),
        }
        if logger is not None:
            logger.info("interventional length calibration | %d paired do(length) contrasts | lam=%.3g edf=%.2f | "
                        "mean shift %+.5f -> in-sample %+.5f, cross-fitted %+.5f | R2=%.3f | +%.1f words mean | "
                        "linear slope %+.5f per unit log-length", n, self.lam, edf, self.diag["mean_delta_raw"],
                        self.diag["mean_delta_residual_in_sample"], self.diag["mean_delta_residual_cross_fitted"],
                        self.diag["r2_in_sample"], self.diag["mean_words_added"], self.diag["slope_per_loglen"])
        return self

    def curve(self, texts: Sequence[str]) -> np.ndarray:
        if not self.fitted:
            return np.zeros(len(texts), float)
        return self.basis.design(loglen(texts)) @ self.beta

    def apply(self, texts: Sequence[str], y: np.ndarray) -> np.ndarray:
        return np.asarray(y, float) - self.curve(texts)

    def state(self) -> Dict[str, Any]:
        return {"kind": "interventional", "n_knots": self.n_knots, "span": self.span,
                "basis": self.basis.state() if self.basis is not None else None,
                "beta": np.asarray(self.beta).tolist() if self.beta is not None else None,
                "lam": self.lam, "diag": self.diag}

    @staticmethod
    def load(d: Dict[str, Any]) -> "InterventionalLengthCalibration":
        if d.get("kind") != "interventional":
            raise ValueError("length_control.json was written by the v8 OBSERVATIONAL length control, which is "
                             "invalid (it rewards content-free padding); re-run the validate stage")
        c = InterventionalLengthCalibration(int(d.get("n_knots", 6)), float(d.get("span", 0.98)))
        if d.get("basis") is not None and d.get("beta") is not None:
            c.basis = NaturalCubicBasis.load(d["basis"])
            c.beta = np.asarray(d["beta"], float)
        c.lam = float(d.get("lam", float("nan")))
        c.diag = d.get("diag", {})
        return c


@dataclass
class FrozenControlVariate:
    b: float = 0.0
    c_bar: float = 0.0
    fitted: bool = False

    def fit(self, y: np.ndarray, c: np.ndarray, logger: Optional[logging.Logger] = None) -> "FrozenControlVariate":
        y = np.asarray(y, float)
        c = np.asarray(c, float)
        ok = np.isfinite(y) & np.isfinite(c)
        if int(ok.sum()) < 30 or float(np.var(c[ok], ddof=1)) <= EPS:
            self.b = 0.0
            self.c_bar = float(np.mean(c[ok])) if ok.any() else 0.0
            self.fitted = True
            return self
        self.b = float(np.cov(y[ok], c[ok], ddof=1)[0, 1] / max(float(np.var(c[ok], ddof=1)), EPS))
        self.c_bar = float(np.mean(c[ok]))
        self.fitted = True
        if logger is not None:
            r = y[ok] - self.b * (c[ok] - self.c_bar)
            logger.info("control variate frozen | b=%+.4f c_bar=%+.4f | sd %.4f -> %.4f | n=%d",
                        self.b, self.c_bar, float(np.std(y[ok])), float(np.std(r)), int(ok.sum()))
        return self

    def apply(self, y: np.ndarray, c: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return np.asarray(y, float)
        return np.asarray(y, float) - self.b * (np.asarray(c, float) - self.c_bar)

    def state(self) -> Dict[str, Any]:
        return {"b": self.b, "c_bar": self.c_bar, "fitted": self.fitted}

    @staticmethod
    def load(d: Dict[str, Any]) -> "FrozenControlVariate":
        return FrozenControlVariate(float(d["b"]), float(d["c_bar"]), bool(d["fitted"]))


class LengthInvarianceAugmenter:
    """Content-free lengthening drawn from ONE filler bank ("train" or "heldout", see FILLER_BANKS)."""

    def __init__(self, n_levels: int = 2, seed: int = 0, p_head: float = 0.35, max_words: int = 110,
                 bank: str = FIT_BANK):
        if bank not in FILLER_BANKS:
            raise ValueError(f"unknown filler bank {bank!r}; choose from {sorted(FILLER_BANKS)}")
        self.bank = bank
        self.tails, self.heads = FILLER_BANKS[bank]
        self.n_levels = int(n_levels)
        self.p_head = float(p_head)
        self.max_words = int(max_words)
        self.rng = np.random.default_rng(seed)

    def _parts(self, t: str, level: int) -> Tuple[str, List[str]]:
        k = min(level, len(self.tails))
        tails = [str(x) for x in self.rng.choice(np.asarray(self.tails, dtype=object), size=k, replace=False)]
        body = t
        if self.rng.random() < self.p_head:
            body = f"{self.rng.choice(np.asarray(self.heads, dtype=object))} {body}"
        return body, tails

    def _cap(self, s: str) -> str:
        w = s.split()
        return " ".join(w[: self.max_words]) if len(w) > self.max_words else s

    def lengthen(self, text: str, level: int) -> str:
        t = norm_text(text)
        if level <= 0:
            return t
        body, tails = self._parts(t, level)
        return self._cap(" ".join([body] + tails).strip())

    def matched_pair(self, text: str, level: int = 1) -> Tuple[str, str]:
        """Two variants built from an IDENTICAL word multiset -- same word count, same
        content-free filler -- differing only in where the filler sits.  Any outcome
        difference across this pair is pure surface/position sensitivity and carries no
        length signal, so it is the negative control that attributes a failure to length
        versus generic surface brittleness.  It is reported, never subtracted from a gate."""
        t = norm_text(text)
        if level <= 0:
            return t, t
        body, tails = self._parts(t, level)
        return self._cap(" ".join([body] + tails).strip()), self._cap(" ".join(tails + [body]).strip())


@dataclass
class LogitNullspaceProjector:
    """Least-damaging erasure of the length-response subspace from the panel logits.

    Motivation.  The expected-outcome estimator is a softmax expectation over a fixed
    panel of real customer replies,

        O(x) = softmax_T(z(x)) . s ,   z_j(x) = log p(c_j | x) - center_j ,

    where x is the customer prompt carrying the agent response.  Lengthening the agent
    response by content-free filler moves z by some d(x) in R^P.  The softmax is already
    invariant to the *gauge* direction 1 (a candidate-independent shift cancels exactly),
    which is why the naive "the prefix just gets longer, it cancels" argument fails: only
    the component of d along 1 cancels.  The remaining, candidate-dependent component is
    what drives the outcome, and because it varies with x it produces variance and rank
    flips that no additive post-hoc regression on log-length can remove.  That is exactly
    the failure signature observed: the mean shift passes TOST while tau and the flip rate
    fail.

    Construction.  Estimate d(x) on paired (base, lengthened) prompts, gauge-fix both, and
    erase the leading k directions of the perturbation from z *before* the softmax.  The
    erasure is performed in coordinates whitened by the signal covariance
    Sigma = Cov_x(z(x)), which is the minimum-distortion (LEACE) concept-erasure
    projection of Belrose et al. 2023 and the linear-nullspace family of Ravfogel et al.
    2020: among all linear maps that annihilate the perturbation subspace, it is the one
    that perturbs the signal covariance least.  Erasing before the nonlinearity is what
    separates this from the v7 outcome-space spline: it removes the mean AND the
    heterogeneous, context-dependent part of the length response in one operation.

    Guarantee and its limit.  Invariance holds exactly for any perturbation whose logit
    signature lies in the fitted span, and approximately (by the reported residual energy)
    otherwise.  It is NOT a free lunch: erasing k of the P-1 gauge directions necessarily
    removes some real signal, so signal retention is reported and the pipeline still gates
    on the human-label anchor.  If the length signature is collinear with the satisfaction
    signature, the anchor test will fail after projection -- which is the correct outcome,
    not a bug to be tuned away.
    """
    U: Optional[np.ndarray] = None       # (P, k) orthonormal basis in whitened coordinates
    W: Optional[np.ndarray] = None       # (P, P) Sigma^{-1/2} on the signal subspace
    A: Optional[np.ndarray] = None       # (P, P) Sigma^{+1/2} on the signal subspace
    k: int = 0
    diag: Dict[str, Any] = field(default_factory=dict)

    @property
    def fitted(self) -> bool:
        return self.k > 0 and self.U is not None and self.W is not None and self.A is not None

    @staticmethod
    def gauge(Z: np.ndarray) -> np.ndarray:
        """Remove the softmax gauge direction: a candidate-independent shift is exactly
        cancelled by the softmax, so it carries no information and must not be counted as
        perturbation energy."""
        Z = np.asarray(Z, float)
        return Z - Z.mean(1, keepdims=True)

    @staticmethod
    def _sqrt_pair(S: np.ndarray, rel_tol: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
        w, V = np.linalg.eigh(np.asarray(S, float))
        w = np.maximum(w, 0.0)
        keep = w > rel_tol * max(float(w.max()), EPS)
        safe = np.where(keep, w, 1.0)
        wm = np.where(keep, safe ** -0.5, 0.0)
        wp = np.where(keep, safe ** 0.5, 0.0)
        return (V * wm) @ V.T, (V * wp) @ V.T

    @staticmethod
    def _build(Z0: np.ndarray, Z1: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        Z0 = LogitNullspaceProjector.gauge(Z0)
        Z1 = LogitNullspaceProjector.gauge(Z1)
        if Z0.shape != Z1.shape:
            raise ValueError("LogitNullspaceProjector: paired logit matrices must match")
        n, P = Z0.shape
        if n < 2 * P:
            raise ValueError(f"LogitNullspaceProjector: need >= {2 * P} paired contexts for a "
                             f"P={P} panel, got {n}")
        S = np.cov(Z0, rowvar=False)
        W, A = LogitNullspaceProjector._sqrt_pair(S)
        D = Z1 - Z0
        _, sv, Vt = np.linalg.svd(D @ W, full_matrices=False)
        return Z0, D, Vt, sv

    @staticmethod
    def fit(Z0: np.ndarray, Z1: np.ndarray, k: int) -> "LogitNullspaceProjector":
        Z0g, D, Vt, sv = LogitNullspaceProjector._build(Z0, Z1)
        P = Z0g.shape[1]
        S = np.cov(Z0g, rowvar=False)
        W, A = LogitNullspaceProjector._sqrt_pair(S)
        k = int(max(0, min(k, Vt.shape[0], P - 1)))
        U = Vt[:k].T if k > 0 else np.zeros((P, 0))
        pj = LogitNullspaceProjector(U=U, W=W, A=A, k=k)
        Dp = pj.apply(D) if k > 0 else LogitNullspaceProjector.gauge(D)
        Z0p = pj.apply(Z0g) if k > 0 else Z0g
        nD = float(np.linalg.norm(D)) or EPS
        nZ = float(np.linalg.norm(Z0g)) or EPS
        pj.diag = {
            "k": k, "panel": P, "n_pairs": int(Z0g.shape[0]),
            "residual_perturbation": float(np.linalg.norm(Dp) / nD),
            "signal_retention": float(np.linalg.norm(Z0p) / nZ),
            "perturbation_spectrum": (sv ** 2 / max(float(np.sum(sv ** 2)), EPS))[:12].tolist(),
        }
        return pj

    @staticmethod
    def profile(Z0: np.ndarray, Z1: np.ndarray, k_max: int = 8,
                logger: Optional[logging.Logger] = None) -> List[Dict[str, Any]]:
        rows = []
        P = np.asarray(Z0).shape[1]
        for k in range(0, int(min(k_max, P - 1)) + 1):
            rows.append(LogitNullspaceProjector.fit(Z0, Z1, k).diag)
        if logger is not None:
            logger.info("logit nullspace profile | panel P=%d over %d paired contexts", P, rows[0]["n_pairs"])
            for r in rows:
                logger.info("    k=%d | residual length-perturbation energy %.4f | signal retained %.4f",
                            r["k"], r["residual_perturbation"], r["signal_retention"])
        return rows

    def apply(self, Z: np.ndarray) -> np.ndarray:
        Zg = self.gauge(Z)
        if not self.fitted:
            return Zg
        Y = Zg @ self.W
        Y = Y - (Y @ self.U) @ self.U.T
        return self.gauge(Y @ self.A)

    def state(self) -> Dict[str, Any]:
        return {"U": np.asarray(self.U).tolist() if self.U is not None else None,
                "W": np.asarray(self.W).tolist() if self.W is not None else None,
                "A": np.asarray(self.A).tolist() if self.A is not None else None,
                "k": int(self.k), "diag": self.diag}

    @staticmethod
    def load(d: Dict[str, Any]) -> "LogitNullspaceProjector":
        def arr(x):
            return np.asarray(x, float) if x is not None else None
        return LogitNullspaceProjector(arr(d.get("U")), arr(d.get("W")), arr(d.get("A")),
                                       int(d.get("k", 0)), d.get("diag", {}))


@dataclass
class Turn:
    dialogue_id: str
    uid: str
    split: str
    source: str
    turn_index: int
    history: str
    user_text: str
    gold_response: str
    next_user_text: Optional[str]
    user_emotion: int
    next_emotion: Optional[int]

    @property
    def context(self) -> str:
        return f"{self.history}\nCustomer: {self.user_text}".strip()

    @property
    def human_valence(self) -> Optional[float]:
        if self.next_emotion is None or self.next_emotion < 0:
            return None
        return EMOTION_ORDINAL.get(int(self.next_emotion))

    @property
    def human_satisfaction(self) -> Optional[float]:
        if self.next_emotion is None or self.next_emotion < 0:
            return None
        return EMOTION_VALENCE.get(int(self.next_emotion))


def _annotator_emotion(raw: Any) -> int:
    if isinstance(raw, int):
        return int(raw)
    if isinstance(raw, list):
        if not raw:
            return -1
        pick = raw[3] if len(raw) > 3 else raw[-1]
        if isinstance(pick, dict):
            return int(pick.get("emotion", -1))
        if isinstance(pick, int):
            return int(pick)
        return -1
    if isinstance(raw, dict):
        return int(raw.get("emotion", -1))
    return -1


def download_emowoz(data_dir: Path, logger: logging.Logger) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, url in ZENODO_FILES.items():
        dst = data_dir / name
        if dst.exists() and dst.stat().st_size > 1000:
            continue
        logger.info("downloading %s", name)
        with urllib.request.urlopen(url, timeout=600) as r, open(dst, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        logger.info("  saved %s (%.1f MB)", dst, dst.stat().st_size / 1e6)


def load_emowoz(data_dir: Path, logger: logging.Logger, max_history_turns: int = 6) -> List[Turn]:
    data_dir = Path(data_dir)
    mw = data_dir / "emowoz-multiwoz.json"
    dm = data_dir / "emowoz-dialmage.json"
    sp = data_dir / "data-split.json"
    for p in (mw, dm, sp):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}; run with --download or place the Zenodo files in {data_dir}")
    dialogues: Dict[str, Any] = {}
    source_of: Dict[str, str] = {}
    for path, src in ((mw, "multiwoz"), (dm, "dialmage")):
        raw = load_json(path)
        for k, v in raw.items():
            dialogues[k] = v
            source_of[k] = src
    split_raw = load_json(sp)
    split_of: Dict[str, str] = {}
    for key, node in split_raw.items():
        canon = SPLIT_ALIASES.get(str(key).lower())
        if canon is None:
            continue
        ids: List[str] = []
        if isinstance(node, dict):
            for sub in node.values():
                ids.extend(list(sub))
        else:
            ids.extend(list(node))
        for i in dict.fromkeys(ids):
            split_of[i] = canon
    turns: List[Turn] = []
    n_missing = 0
    for did, d in dialogues.items():
        split = split_of.get(did)
        if split is None:
            n_missing += 1
            continue
        log = d.get("log") if isinstance(d, dict) else None
        if not isinstance(log, list) or len(log) < 2:
            continue
        texts = [norm_text(u.get("text", "")) for u in log]
        emos = [_annotator_emotion(u.get("emotion")) if i % 2 == 0 else -1 for i, u in enumerate(log)]
        for i in range(0, len(log) - 1, 2):
            if not texts[i] or not texts[i + 1]:
                continue
            lo = max(0, i - max_history_turns)
            hist_lines = []
            for j in range(lo, i):
                role = "Customer" if j % 2 == 0 else "Agent"
                if texts[j]:
                    hist_lines.append(f"{role}: {texts[j]}")
            nxt = texts[i + 2] if i + 2 < len(log) and texts[i + 2] else None
            nemo = emos[i + 2] if i + 2 < len(log) else None
            turns.append(Turn(
                dialogue_id=str(did), uid=f"{did}#{i}", split=split, source=source_of.get(did, "?"),
                turn_index=i, history="\n".join(hist_lines), user_text=texts[i],
                gold_response=texts[i + 1], next_user_text=nxt,
                user_emotion=int(emos[i]), next_emotion=int(nemo) if nemo is not None else None,
            ))
    by_split: Dict[str, int] = {}
    for t in turns:
        by_split[t.split] = by_split.get(t.split, 0) + 1
    n_next = sum(1 for t in turns if t.next_user_text)
    n_lab = sum(1 for t in turns if t.human_valence is not None)
    logger.info("EmoWOZ | %d turns from %d dialogues | %d (%.1f%%) have a next customer turn | "
                "%d (%.1f%%) carry a usable emotion label | per split %s | %d dialogues absent from data-split",
                len(turns), len({t.dialogue_id for t in turns}), n_next, 100 * n_next / max(1, len(turns)),
                n_lab, 100 * n_lab / max(1, len(turns)), by_split, n_missing)
    if not turns:
        raise RuntimeError("no turns parsed; check the EmoWOZ JSON schema")
    return turns


def split_turns_by_dialogue(turns: Sequence[Turn], frac: float, seed: int) -> Tuple[List[Turn], List[Turn]]:
    """Partition turns into two dialogue-disjoint halves.

    v7 drew the length-control fitting contexts and the validation contexts with two
    different sampling seeds from the SAME split, so the two sets overlapped heavily and
    the nuisance model was in part evaluated on its own training data.  Every nuisance
    component (control variate, length spline, logit projector, panel temperature) is now
    fitted on one half and gated on the other."""
    ids = sorted({t.dialogue_id for t in turns})
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    cut = max(1, int(round(frac * len(ids))))
    a = set(ids[:cut])
    return [t for t in turns if t.dialogue_id in a], [t for t in turns if t.dialogue_id not in a]


def filter_turns(turns: Sequence[Turn], split: Optional[str] = None, require_next: bool = False,
                 require_label: bool = False, limit: Optional[int] = None, seed: int = 0,
                 logger: Optional[logging.Logger] = None, what: str = "turns") -> List[Turn]:
    out = [t for t in turns
           if (split is None or t.split == split)
           and (not require_next or bool(t.next_user_text))
           and (not require_label or t.human_valence is not None)]
    if limit is not None and len(out) > limit:
        by_dialogue: Dict[str, List[Turn]] = {}
        for t in out:
            by_dialogue.setdefault(t.dialogue_id, []).append(t)
        keys = sorted(by_dialogue)
        random.Random(seed).shuffle(keys)
        sel: List[Turn] = []
        for k in keys:
            if len(sel) >= limit:
                break
            sel.extend(by_dialogue[k])
        out = sel[:limit]
    if logger is not None:
        logger.info("  selected %d %s (split=%s, require_next=%s, %d dialogues)",
                    len(out), what, split, require_next, len({t.dialogue_id for t in out}))
    return out


@dataclass
class GenConfig:
    max_new_tokens: int = 64
    min_new_tokens: int = 8
    temperature: float = 0.9
    top_p: float = 0.95
    do_sample: bool = True
    repetition_penalty: float = 1.05


@dataclass
class SFTConfig:
    lr: float = 5e-5
    weight_decay: float = 0.01
    epochs: int = 3
    batch_size: int = 8
    grad_accum: int = 2
    max_len: int = 640
    warmup_frac: float = 0.03
    evals_per_epoch: int = 4
    patience: int = 4
    max_gap: float = 0.35
    gap_aborts: bool = False
    invariance_coef: float = 0.5
    invariance_frac: float = 0.5
    dev_examples: int = 1000


AGENT_SYSTEM = ("You are a helpful customer service agent for a travel and hospitality assistant. "
                "Reply to the customer in one short, concrete, polite turn.")
CUSTOMER_SYSTEM = ("You are the CUSTOMER in a task-oriented conversation. Write only the customer's next turn, "
                   "reacting naturally to what the agent just said.")


def agent_prompt(t: Turn) -> str:
    return f"{AGENT_SYSTEM}\n{t.history}\nCustomer: {t.user_text}\nAgent:"


def customer_prompt(t: Turn, response: str) -> str:
    return f"{CUSTOMER_SYSTEM}\n{t.history}\nCustomer: {t.user_text}\nAgent: {norm_text(response)}\nCustomer:"


def trim_to_sentence(text: str) -> str:
    s = norm_text(text)
    s = ROLE_LEAK_RE.split(s)[0].strip()
    m = list(re.finditer(r"[.!?](\s|$)", s))
    if m:
        s = s[: m[-1].end()].strip()
    return s


def hygiene_ok(text: str, min_words: int = 3, max_words: int = 120) -> Tuple[bool, List[str]]:
    s = norm_text(text)
    reasons = []
    w = s.split()
    if len(w) < min_words:
        reasons.append("too_short")
    if len(w) > max_words:
        reasons.append("too_long")
    if ROLE_LEAK_RE.search(s):
        reasons.append("role_leak")
    if w:
        run = 1
        best = 1
        for a, b in zip(w, w[1:]):
            run = run + 1 if a.lower() == b.lower() else 1
            best = max(best, run)
        if best >= 5:
            reasons.append("repetition")
        tri = [tuple(x.lower() for x in w[i:i + 3]) for i in range(max(0, len(w) - 2))]
        if tri and 1.0 - len(set(tri)) / len(tri) > 0.5:
            reasons.append("trigram_loop")
    if not re.search(r"[.!?]\s*$", s):
        reasons.append("unterminated")
    return (len(reasons) == 0, reasons)


class SentimentScorer:
    def __init__(self, model_name: str, device: str, models_dir: Optional[str], logger: logging.Logger,
                 batch: int = 64, max_len: int = 128):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        path = resolve_local_model(model_name, models_dir, logger)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
        self.device = device
        self.batch = int(batch)
        self.max_len = int(max_len)
        id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self.pos = next((i for i, v in id2label.items() if v.startswith("pos")), max(id2label))
        self.neg = next((i for i, v in id2label.items() if v.startswith("neg")), min(id2label))
        logger.info("sentiment model ready | labels=%s | pos=%d neg=%d", id2label, self.pos, self.neg)

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        import torch
        texts = [norm_text(t) or "." for t in texts]
        out = np.zeros(len(texts), float)
        with torch.no_grad():
            for s in range(0, len(texts), self.batch):
                chunk = texts[s:s + self.batch]
                enc = self.tok(chunk, return_tensors="pt", padding=True, truncation=True,
                               max_length=self.max_len).to(self.device)
                p = torch.softmax(self.model(**enc).logits.float(), -1).cpu().numpy()
                out[s:s + len(chunk)] = p[:, self.pos] - p[:, self.neg]
        return out


class EmotionValenceScorer:
    """Valence from a multi-class EMOTION classifier: P(positive emotions) - P(negative emotions).

    Used only by the external evaluator (R2).  It must be a different model from the sentiment
    engine that labels the training corpus, otherwise the "independent" evaluation would share the
    labeller's idiosyncrasies.  Labels are mapped by name, so any GoEmotions- or Ekman-style head
    works; labels that are neither (neutral, surprise, confusion, ...) count as zero."""
    POS = frozenset("joy love optimism admiration gratitude approval relief amusement excitement caring pride "
                    "satisfied positive happiness".split())
    NEG = frozenset("anger disgust fear sadness annoyance disappointment disapproval embarrassment grief "
                    "nervousness remorse negative dissatisfied".split())

    def __init__(self, model_name: str, device: str, models_dir: Optional[str], logger: logging.Logger,
                 batch: int = 64, max_len: int = 128):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        path = resolve_local_model(model_name, models_dir, logger)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
        self.device = device
        self.batch = int(batch)
        self.max_len = int(max_len)
        id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        n = len(id2label)
        self.sign = np.zeros(n, float)
        for i, lab in id2label.items():
            self.sign[i] = 1.0 if lab in self.POS else (-1.0 if lab in self.NEG else 0.0)
        if not (self.sign > 0).any() or not (self.sign < 0).any():
            raise ValueError(f"emotion model {model_name} has no recognisable positive/negative labels: {id2label}")
        self.multilabel = getattr(self.model.config, "problem_type", "") == "multi_label_classification"
        logger.info("external emotion model ready | %d labels | positive=%s | negative=%s", n,
                    [id2label[i] for i in range(n) if self.sign[i] > 0], [id2label[i] for i in range(n) if self.sign[i] < 0])

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        import torch
        texts = [norm_text(t) or "." for t in texts]
        out = np.zeros(len(texts), float)
        with torch.no_grad():
            for s in range(0, len(texts), self.batch):
                chunk = texts[s:s + self.batch]
                enc = self.tok(chunk, return_tensors="pt", padding=True, truncation=True,
                               max_length=self.max_len).to(self.device)
                lg = self.model(**enc).logits.float()
                p = (torch.sigmoid(lg) if self.multilabel else torch.softmax(lg, -1)).cpu().numpy()
                v = p @ self.sign
                if self.multilabel:                  # keep the scale in [-1, 1]
                    v = v / np.maximum(p.sum(1), 1.0)
                out[s:s + len(chunk)] = v
        return out


class StubEmotion:
    """Stub external labeller: a lexicon disjoint from StubSentiment's, so the stub pipeline also
    exercises an evaluator that disagrees with the training labeller."""
    POS = ("confirmed", "available", "booked", "reference")
    NEG = ("unfortunately", "cancelled", "delay")

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros(len(texts), float)
        for i, t in enumerate(texts):
            w = [x.strip(".,!?") for x in str(t).lower().split()]
            if w:
                out[i] = math.tanh((sum(x in self.POS for x in w) - sum(x in self.NEG for x in w)) / math.sqrt(len(w)))
        return out


def resolve_local_model(name: str, models_dir: Optional[str], logger: logging.Logger) -> str:
    if models_dir:
        stem = "models--" + name.replace("/", "--")
        root = Path(models_dir) / stem / "snapshots"
        if root.exists():
            snaps = sorted([p for p in root.iterdir() if p.is_dir()])
            if snaps:
                logger.info("resolved %s -> %s", name, snaps[-1])
                return str(snaps[-1])
    return name


class Policy:
    def __init__(self, base_model: str, device: str, models_dir: Optional[str], logger: logging.Logger,
                 lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05,
                 load_4bit: bool = True, gen_batch: int = 32, max_len: int = 640, score_batch: int = 64,
                 with_lora: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.logger = logger
        self.score_batch = int(score_batch)
        self._tail_kw: Optional[str] = None
        self.fast_score = True
        self._fast_verified = False
        self.device = device
        self.gen_batch = int(gen_batch)
        self.max_len = int(max_len)
        path = resolve_local_model(base_model, models_dir, logger)
        self.tok = AutoTokenizer.from_pretrained(path, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kw: Dict[str, Any] = {"dtype": torch.bfloat16}
        if load_4bit and device.startswith("cuda"):
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
            kw["device_map"] = {"": 0}
        self.model = AutoModelForCausalLM.from_pretrained(path, **kw)
        if not (load_4bit and device.startswith("cuda")):
            self.model = self.model.to(device)
        if not with_lora:
            # Frozen generator (the external judge, R2): no adapter, and no model-specific LoRA target
            # names, which differ across families (e.g. Phi-3 fuses qkv_proj / gate_up_proj).
            self.model.eval()
            logger.info("frozen model %s loaded without LoRA | 4bit=%s", Path(path).name, load_4bit)
            return
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        if load_4bit and device.startswith("cuda"):
            try:
                self.model = prepare_model_for_kbit_training(
                    self.model, use_gradient_checkpointing=True,
                    gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=True)
        self.model = get_peft_model(self.model, LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        # Gradient checkpointing and a KV cache are incompatible during training; generation asks
        # for the cache explicitly, so the config default is off and the warning disappears.
        try:
            self.model.config.use_cache = False
        except Exception:
            pass
        tr = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in self.model.parameters())
        logger.info("policy %s | LoRA r=%d | trainable %.2fM / %.2fM (%.3f%%) | 4bit=%s",
                    Path(path).name, lora_r, tr / 1e6, tot / 1e6, 100 * tr / max(1, tot), load_4bit)

    def save_adapter(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))

    def load_adapter(self, path: Path) -> None:
        """Load saved LoRA weights INTO the existing adapter.

        v14 re-wrapped the (already LoRA-injected) base model in a second PeftModel, which is what the
        peft warning "Already found a peft_config attribute ... multiple adapters" in the logs reports.
        v15 writes the saved tensors into the live adapter and verifies that every LoRA tensor was
        matched; if anything does not match (e.g. a different rank) it falls back to the old path."""
        path = Path(path)
        try:
            from peft.utils import set_peft_model_state_dict
            st = path / "adapter_model.safetensors"
            if st.exists():
                from safetensors.torch import load_file
                sd = load_file(str(st))
            else:
                import torch
                sd = torch.load(str(path / "adapter_model.bin"), map_location="cpu")
            if not any("lora_" in k for k in sd):
                raise ValueError("no LoRA tensors in the saved adapter")
            res = set_peft_model_state_dict(self.model, sd)
            missing = [k for k in getattr(res, "missing_keys", []) if "lora_" in k]
            unexpected = list(getattr(res, "unexpected_keys", []))
            if missing or unexpected:
                raise ValueError(f"{len(missing)} LoRA tensors missing, {len(unexpected)} unexpected")
            self.logger.info("loaded adapter from %s (%d tensors into the live adapter)", path, len(sd))
        except Exception as e:
            from peft import PeftModel
            self.logger.warning("in-place adapter load failed (%s); re-wrapping the base model instead", e)
            base = self.model.get_base_model()
            self.model = PeftModel.from_pretrained(base, str(path), is_trainable=True)
            self.logger.info("loaded adapter from %s", path)

    def _encode_pairs(self, pairs: Sequence[Tuple[str, str]]):
        import torch
        ids, labels = [], []
        eos = self.tok.eos_token or ""
        for prompt, target in pairs:
            p = self.tok(prompt, add_special_tokens=False)["input_ids"]
            a = self.tok(" " + norm_text(target) + eos, add_special_tokens=False)["input_ids"]
            seq = (p + a)[-self.max_len:]
            lab = ([-100] * len(p) + list(a))[-self.max_len:]
            ids.append(seq)
            labels.append(lab)
        n = max(len(x) for x in ids)
        pad = self.tok.pad_token_id
        X = torch.full((len(ids), n), pad, dtype=torch.long)
        Y = torch.full((len(ids), n), -100, dtype=torch.long)
        M = torch.zeros((len(ids), n), dtype=torch.long)
        for i, (s, l) in enumerate(zip(ids, labels)):
            X[i, : len(s)] = torch.tensor(s)
            Y[i, : len(l)] = torch.tensor(l)
            M[i, : len(s)] = 1
        return X, Y, M

    def _nll(self, pairs: Sequence[Tuple[str, str]], batch: int) -> float:
        import torch
        self.model.eval()
        tot, ntok = 0.0, 0
        with torch.no_grad():
            for s in range(0, len(pairs), batch):
                X, Y, M = self._encode_pairs(pairs[s:s + batch])
                X, Y, M = X.to(self.device), Y.to(self.device), M.to(self.device)
                out = self.model(input_ids=X, attention_mask=M).logits[:, :-1].float()
                tgt = Y[:, 1:]
                sel = tgt != -100
                if not bool(sel.any()):
                    continue
                lp = torch.log_softmax(out, -1)
                g = lp.gather(-1, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)
                tot += float((-g[sel]).sum())
                ntok += int(sel.sum())
        return tot / max(1, ntok)

    def fit_supervised(self, train, dev: Sequence[Tuple[str, str]], cfg: SFTConfig, logger: logging.Logger,
                       best_dir: Path, tag: str, resample=None) -> Dict[str, Any]:
        import torch
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import LambdaLR
        train = list(train)
        dev = list(dev)
        steps_per_epoch = max(1, len(train) // (cfg.batch_size * cfg.grad_accum))
        total = steps_per_epoch * cfg.epochs
        warm = max(1, int(cfg.warmup_frac * total))
        every = max(1, steps_per_epoch // max(1, cfg.evals_per_epoch))
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

        def lr_at(s: int) -> float:
            if s < warm:
                return s / max(1, warm)
            prog = (s - warm) / max(1, total - warm)
            return max(0.05, 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))

        sched = LambdaLR(opt, lr_at)
        logger.info("%s | %d examples, %d steps/epoch x %d epochs = %d steps | lr=%.2e | validating every %d steps",
                    tag, len(train), steps_per_epoch, cfg.epochs, total, cfg.lr, every)
        rng = random.Random(1234)
        best = float("inf")
        best_step = -1
        stale = 0
        step = 0
        run_loss, run_n, run_inv = 0.0, 0, 0.0
        stop = False
        hist: List[Dict[str, float]] = []
        for ep in range(cfg.epochs):
            if resample is not None and ep > 0:
                train = resample(ep)
            order = list(range(len(train)))
            rng.shuffle(order)
            self.model.train()
            micro = 0
            for s in range(0, len(order) - cfg.batch_size + 1, cfg.batch_size):
                items = [train[i] for i in order[s:s + cfg.batch_size]]
                batch = [(x[0], x[1]) for x in items]
                X, Y, M = self._encode_pairs(batch)
                X, Y, M = X.to(self.device), Y.to(self.device), M.to(self.device)
                out = self.model(input_ids=X, attention_mask=M, labels=Y)
                base_loss = out.loss
                inv = torch.zeros((), device=base_loss.device)
                alts = [(k, x) for k, x in enumerate(items) if len(x) > 2 and x[2]]
                if cfg.invariance_coef > 0 and alts:
                    take = max(1, int(round(cfg.invariance_frac * len(alts))))
                    sel = alts[:take]
                    lp_main = self._seq_logprob([(items[k][0], items[k][1]) for k, _ in sel])
                    lp_alt = self._seq_logprob([(items[k][2], items[k][1]) for k, _ in sel])
                    inv = ((lp_main - lp_alt) ** 2).mean()
                loss = (base_loss + cfg.invariance_coef * inv) / cfg.grad_accum
                loss.backward()
                run_loss += float(base_loss.detach())
                run_inv += float(inv.detach())
                run_n += 1
                micro += 1
                if micro % cfg.grad_accum:
                    continue
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % every == 0 or step == total:
                    dev_nll = self._nll(dev, cfg.batch_size)
                    tr_nll = run_loss / max(1, run_n)
                    inv_mean = run_inv / max(1, run_n)
                    gap = dev_nll - tr_nll
                    run_loss, run_n, run_inv = 0.0, 0, 0.0
                    mark = ""
                    if dev_nll < best - 1e-4:
                        best, best_step, stale = dev_nll, step, 0
                        self.save_adapter(best_dir)
                        mark = "  <- best"
                    else:
                        stale += 1
                    hist.append({"step": step, "train": tr_nll, "dev": dev_nll, "gap": gap, "inv": inv_mean})
                    logger.info("%s | step %6d (epoch %.2f) | lr=%.2e | train=%.4f dev=%.4f (ppl %.3f) | "
                                "gap=%+.4f | inv=%.4f | stale=%d/%d%s", tag, step, step / steps_per_epoch,
                                sched.get_last_lr()[0], tr_nll, dev_nll, math.exp(min(20.0, dev_nll)),
                                gap, inv_mean, stale, cfg.patience, mark)
                    self.model.train()
                    if gap > cfg.max_gap:
                        if cfg.gap_aborts:
                            logger.info("%s | stopped: train/dev gap %+.4f exceeds %+.4f", tag, gap, cfg.max_gap)
                            stop = True
                            break
                        logger.warning("%s | train/dev gap %+.4f exceeds %+.4f; dev NLL is the stopping "
                                       "criterion, so training continues while dev still improves",
                                       tag, gap, cfg.max_gap)
                    if stale >= cfg.patience:
                        logger.info("%s | stopped: dev NLL has not improved for %d evaluations", tag, cfg.patience)
                        stop = True
                        break
            if stop:
                break
        if best_dir.exists():
            self.load_adapter(best_dir)
        logger.info("%s | best dev NLL=%.4f (ppl %.3f) at step %d", tag, best, math.exp(min(20.0, best)), best_step)
        return {"best_dev_nll": best, "best_step": best_step, "history": hist}

    def generate(self, prompts: Sequence[str], gen: GenConfig, seed: Optional[int] = None) -> List[str]:
        import torch
        self.model.eval()
        out: List[str] = []
        for s in range(0, len(prompts), self.gen_batch):
            chunk = list(prompts[s:s + self.gen_batch])
            if seed is not None:
                torch.manual_seed(seed + s)
            enc = self.tok(chunk, return_tensors="pt", padding=True, truncation=True,
                           max_length=self.max_len).to(self.device)
            with torch.no_grad():
                y = self.model.generate(
                    **enc, max_new_tokens=gen.max_new_tokens, min_new_tokens=gen.min_new_tokens,
                    do_sample=gen.do_sample, temperature=gen.temperature, top_p=gen.top_p,
                    repetition_penalty=gen.repetition_penalty, pad_token_id=self.tok.pad_token_id,
                    use_cache=True)
            new = y[:, enc["input_ids"].shape[1]:]
            for row in self.tok.batch_decode(new, skip_special_tokens=True):
                out.append(trim_to_sentence(row))
        return out

    def features(self, prompts: Sequence[str], responses: Sequence[str], dim: int = 256,
                 seed: int = 12345) -> np.ndarray:
        import torch
        self.model.eval()
        texts = [f"{p} {norm_text(r)}" for p, r in zip(prompts, responses)]
        reps: List[np.ndarray] = []
        with torch.no_grad():
            for s in range(0, len(texts), self.gen_batch):
                enc = self.tok(texts[s:s + self.gen_batch], return_tensors="pt", padding=True,
                               truncation=True, max_length=self.max_len).to(self.device)
                h = self.model(**enc, output_hidden_states=True).hidden_states[-1].float()
                m = enc["attention_mask"].unsqueeze(-1).float()
                reps.append(((h * m).sum(1) / m.sum(1).clamp_min(1.0)).cpu().numpy())
        H = np.concatenate(reps, 0)
        return project_features(H, dim, seed)

    def _seq_logprob(self, pairs: Sequence[Tuple[str, str]], reduce: str = "mean"):
        import torch
        X, Y, M = self._encode_pairs(pairs)
        X, Y, M = X.to(self.device), Y.to(self.device), M.to(self.device)
        logits = self.model(input_ids=X, attention_mask=M).logits[:, :-1].float()
        tgt = Y[:, 1:]
        sel = (tgt != -100).float()
        lp = torch.log_softmax(logits, -1).gather(-1, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        tot = (lp * sel).sum(1)
        return tot if reduce == "sum" else tot / sel.sum(1).clamp_min(1.0)

    def snapshot_trainable(self) -> Dict[str, Any]:
        """Detached clone of every trainable (LoRA) tensor -- the frozen RL reference policy."""
        return {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}

    @contextmanager
    def frozen_weights(self, snap: Optional[Dict[str, Any]]):
        """Temporarily swap the trainable tensors for a snapshot (reference-policy scoring)."""
        if not snap:
            yield
            return
        keep = {}
        for n, prm in self.model.named_parameters():
            if n in snap:
                keep[n] = prm.data
                prm.data = snap[n]
        try:
            yield
        finally:
            for n, prm in self.model.named_parameters():
                if n in keep:
                    prm.data = keep[n]

    def _token_logprobs(self, pairs: Sequence[Tuple[str, str]]):
        """Per-token log-probabilities of the response tokens, with the loss mask."""
        import torch.nn.functional as F
        X, Y, M = self._encode_pairs(pairs)
        X, Y, M = X.to(self.device), Y.to(self.device), M.to(self.device)
        logits = self.model(input_ids=X, attention_mask=M).logits[:, :-1]
        tgt = Y[:, 1:]
        mask = (tgt != -100)
        lp = -F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                              tgt.reshape(-1).clamp_min(0), reduction="none").view(tgt.shape)
        return lp * mask, mask.float()

    def _forward_tail(self, X, M, pos, keep: int):
        """Forward pass that materialises logits for the last `keep` positions only.  The
        full-vocabulary fp32 log-softmax over every prefix position (151k x ~400 per row) was
        the dominant memory/bandwidth cost of panel scoring in v8; only the candidate tokens,
        which sit at the END of a left-padded row, are ever read."""
        if self._tail_kw is None:
            for kw in ("logits_to_keep", "num_logits_to_keep", ""):
                try:
                    extra = {kw: keep} if kw else {}
                    lg = self.model(input_ids=X, attention_mask=M, position_ids=pos, **extra).logits
                    if lg.shape[1] < keep:
                        raise TypeError("short logits")
                    self._tail_kw = kw
                    return lg[:, -keep:]
                except TypeError:
                    continue
            raise RuntimeError("model forward rejected every logits-slicing convention")
        extra = {self._tail_kw: keep} if self._tail_kw else {}
        return self.model(input_ids=X, attention_mask=M, position_ids=pos, **extra).logits[:, -keep:]

    def candidate_logprobs_legacy(self, prefixes: Sequence[str], candidates: Sequence[str],
                                  reduce: str = "sum") -> np.ndarray:
        """v8 reference implementation (right padding, full-vocabulary logits)."""
        import torch
        self.model.eval()
        pairs = [(p, c) for p in prefixes for c in candidates]
        out: List[float] = []
        b = max(1, self.gen_batch)
        with torch.no_grad():
            for s0 in range(0, len(pairs), b):
                out.extend(self._seq_logprob(pairs[s0:s0 + b], reduce=reduce).cpu().tolist())
        return np.asarray(out, float).reshape(len(prefixes), len(candidates))

    def candidate_logprobs(self, prefixes: Sequence[str], candidates: Sequence[str],
                           reduce: str = "sum") -> np.ndarray:
        if not self.fast_score:
            return self.candidate_logprobs_legacy(prefixes, candidates, reduce)
        if not self._fast_verified:
            self._fast_verified = True
            ok, diag = self._verify_fast(prefixes, candidates, reduce)
            if not ok:
                self.fast_score = False
                return self.candidate_logprobs_legacy(prefixes, candidates, reduce)
            self.logger.info("fast panel scoring verified in outcome units | max |dO|=%.5f vs reference-path "
                             "self-disagreement %.5f (tolerance %.5f) | centred-logit max |diff|=%.4f nats | "
                             "logits kwarg=%r | score batch %d", diag["d_fast"], diag["d_self"], diag["tol"],
                             diag["d_logit_centred"], self._tail_kw, self.score_batch)
        return self._candidate_logprobs_fast(prefixes, candidates, reduce)

    @staticmethod
    def _panel_probe(Z: np.ndarray, T: float = 4.0) -> np.ndarray:
        """Softmax weights of the panel estimator, the quantity the pipeline actually consumes."""
        z = np.asarray(Z, float)
        z = z - z.mean(0, keepdims=True)          # per-candidate centring, as OutcomePanel does
        z = z / max(T, 1e-3)
        z = z - z.max(1, keepdims=True)
        w = np.exp(z)
        return w / np.maximum(w.sum(1, keepdims=True), EPS)

    def _verify_fast(self, prefixes: Sequence[str], candidates: Sequence[str],
                     reduce: str) -> Tuple[bool, Dict[str, float]]:
        """Compare the fast scorer with the v8 reference IN OUTCOME UNITS, against the reference
        path's own batch-composition noise.

        4-bit matmuls are not invariant to batch shape, so the reference path disagrees with
        ITSELF by a comparable number of nats when the batch size changes; v9 compared raw summed
        log-probabilities against a nats-scale tolerance and therefore mistook that intrinsic
        noise for a bug (and silently spent ~14 h on the slow scorer).  Only two things matter
        downstream: the per-candidate CENTRED logits, and the softmax weights they induce -- an
        additive per-row constant cancels in the softmax and a per-candidate constant cancels in
        the panel centring.  Both are checked here, the second on the scale of the outcome, whose
        own within-context sd is of order 0.05."""
        P0, C0 = list(prefixes)[:3], list(candidates)[:16]
        if not P0 or not C0:
            return True, {"d_fast": 0.0, "d_self": 0.0, "tol": 0.0, "d_logit_centred": 0.0}
        try:
            a = self._candidate_logprobs_fast(P0, C0, reduce)
        except Exception as e:
            self.logger.warning("fast panel scoring unavailable (%s); using the v8 scorer", e)
            return False, {"d_fast": float("nan"), "d_self": float("nan"), "tol": 0.0,
                           "d_logit_centred": float("nan")}
        b = self.candidate_logprobs_legacy(P0, C0, reduce)
        keep = self.gen_batch
        try:                                       # reference path against itself, different batching
            self.gen_batch = max(1, keep // 4)
            b2 = self.candidate_logprobs_legacy(P0, C0, reduce)
        finally:
            self.gen_batch = keep
        sent = np.linspace(-1.0, 1.0, len(C0))      # stand-in panel sentiments, |s| <= 1
        o_a, o_b, o_b2 = (self._panel_probe(x) @ sent for x in (a, b, b2))
        d_fast = float(np.max(np.abs(o_a - o_b)))
        d_self = float(np.max(np.abs(o_b2 - o_b)))
        cg = lambda Z: Z - Z.mean(1, keepdims=True)
        d_lc = float(np.max(np.abs(cg(a) - cg(b))))
        tol = max(0.01, 3.0 * d_self)               # 0.01 is ~0.2 of a within-context outcome sd
        if not np.all(np.isfinite(a)) or d_fast > tol:
            self.logger.warning("fast panel scoring changes the panel expectation by %.5f (reference path's own "
                                "batch noise %.5f, tolerance %.5f); falling back to the v8 scorer",
                                d_fast, d_self, tol)
            return False, {"d_fast": d_fast, "d_self": d_self, "tol": tol, "d_logit_centred": d_lc}
        return True, {"d_fast": d_fast, "d_self": d_self, "tol": tol, "d_logit_centred": d_lc}

    def _candidate_logprobs_fast(self, prefixes: Sequence[str], candidates: Sequence[str],
                                 reduce: str = "sum") -> np.ndarray:
        """log p(candidate | prefix) for every (prefix, candidate) pair.

        Numerically the same quantity as v8 (same tokenisation " " + candidate + eos, same left
        truncation to max_len, same explicit positions 0..n-1 for the real tokens), computed with
        left padding, length-bucketed batches, tokenisation done once per string, and logits kept
        for the candidate span only."""
        import torch
        self.model.eval()
        eos = self.tok.eos_token or ""
        P = [self.tok(p, add_special_tokens=False)["input_ids"] for p in prefixes]
        C = [self.tok(" " + norm_text(c) + eos, add_special_tokens=False)["input_ids"] for c in candidates]
        jobs: List[Tuple[int, int, List[int], int]] = []
        for i, p in enumerate(P):
            for j, c in enumerate(C):
                seq = (p + c)[-self.max_len:]
                jobs.append((i, j, seq, max(0, min(len(c), len(seq) - 1))))
        order = sorted(range(len(jobs)), key=lambda k: len(jobs[k][2]))
        out = np.zeros((len(P), len(C)), float)
        pad = self.tok.pad_token_id
        b = max(1, self.score_batch)
        with torch.no_grad():
            for s0 in range(0, len(order), b):
                ks = order[s0:s0 + b]
                seqs = [jobs[k][2] for k in ks]
                nts = [jobs[k][3] for k in ks]
                n = max(len(x) for x in seqs)
                T = max(1, max(nts))
                X = torch.full((len(ks), n), pad, dtype=torch.long)
                M = torch.zeros((len(ks), n), dtype=torch.long)
                for r, sq in enumerate(seqs):
                    X[r, n - len(sq):] = torch.tensor(sq)
                    M[r, n - len(sq):] = 1
                pos = (M.cumsum(1) - 1).clamp_min(0)
                X, M, pos = X.to(self.device), M.to(self.device), pos.to(self.device)
                lg = self._forward_tail(X, M, pos, T + 1)[:, :-1].float()
                tgt = X[:, n - T:]
                lp = torch.log_softmax(lg, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                nt = torch.tensor(nts, device=lp.device)
                sel = (torch.arange(T, device=lp.device)[None, :] >= (T - nt)[:, None]).float()
                tot = (lp * sel).sum(1)
                val = tot if reduce == "sum" else tot / nt.clamp_min(1).float()
                for r, v in zip(ks, val.cpu().tolist()):
                    out[jobs[r][0], jobs[r][1]] = v
        return out

    def sequence_logprobs(self, prompts: Sequence[str], responses: Sequence[str], batch: int = 4,
                          use_adapter: bool = True) -> np.ndarray:
        import torch
        self.model.eval()
        pairs = list(zip(prompts, responses))
        out: List[float] = []
        with torch.no_grad():
            if use_adapter:
                for s in range(0, len(pairs), batch):
                    out.extend(self._seq_logprob(pairs[s:s + batch]).cpu().tolist())
            else:
                with self.model.disable_adapter():
                    for s in range(0, len(pairs), batch):
                        out.extend(self._seq_logprob(pairs[s:s + batch]).cpu().tolist())
        return np.asarray(out, float)

    def grpo_step(self, prompts: Sequence[str], responses: Sequence[str], adv: np.ndarray,
                  ref_snapshot: Optional[Dict[str, Any]], clip: float, kl_coef: float,
                  opt, max_grad_norm: float, micro: int = 4) -> Tuple[float, float]:
        """One token-level GRPO update (Shao et al., 2024).

        Per generated token t of sample i:
            ratio_it = exp(logpi_it - logpi_old_it)          (== 1 here: one update per batch)
            L_it     = -min(ratio_it A_i, clip(ratio_it) A_i) + beta * k3(pi || pi_ref)_it
        and the per-token losses are averaged within each sequence, then over sequences.

        Honest reading (v14): because pi_old is the current policy and one gradient step is taken
        per sampled batch, ratio_it == 1 exactly, so the clip is never active and the objective is
        token-level REINFORCE with group-centred advantages plus a k3 KL penalty.  The clipped form
        is kept so that a multi-epoch variant is a one-line change, but the paper must not claim
        that PPO-style clipping regularised this run.  pi_ref is a frozen
        snapshot of the policy at the start of RL (the SFT policy), NOT the pre-SFT base model:
        anchoring to the base model would penalise the supervised fine-tuning itself."""
        import torch
        # eval() here disables LoRA dropout, not gradients: with dropout on, the policy and the
        # reference would be scored through different random masks and the KL estimate would
        # carry that noise as a positive bias.
        self.model.eval()
        opt.zero_grad(set_to_none=True)
        pairs = list(zip(prompts, responses))
        n = len(pairs)
        pg_tot, kl_tot, tok_tot = 0.0, 0.0, 0.0
        for s in range(0, n, micro):
            e = min(s + micro, n)
            lp, mask = self._token_logprobs(pairs[s:e])
            with torch.no_grad():
                old = lp.detach()
                if ref_snapshot:
                    with self.frozen_weights(ref_snapshot):
                        ref, _ = self._token_logprobs(pairs[s:e])
                    ref = ref.detach()
                else:
                    ref = old
            a = torch.tensor(adv[s:e], dtype=torch.float32, device=lp.device).unsqueeze(1)
            ratio = torch.exp((lp - old) * mask)
            surr = torch.min(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a)
            delta = (ref - lp) * mask
            kl = torch.exp(delta) - delta - 1.0
            # Per-sequence token mean, then mean over sequences: every sample contributes the
            # same gradient mass regardless of how long it is.  Normalising by the batch token
            # count instead makes a long sample's gradient proportional to its length, which is a
            # direct pressure towards longer generations (the v11 run drifted 36 -> 38 words).
            per_seq = ((-surr + kl_coef * kl) * mask).sum(1) / mask.sum(1).clamp_min(1.0)
            loss = per_seq.sum() / float(n)
            loss.backward()
            nt = float(mask.sum().detach())
            pg_tot += float((-(surr * mask).sum()).detach())
            kl_tot += float((kl * mask).sum().detach())
            tok_tot += nt
        params = [p for p in self.model.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        opt.step()
        opt.zero_grad(set_to_none=True)
        return (pg_tot / max(1.0, tok_tot), kl_tot / max(1.0, tok_tot))

    def _dpo_margins(self, batch: Sequence[Tuple[str, str, str]], ref_snapshot: Optional[Dict[str, Any]]):
        """Implicit-reward margins (log pi - log pi_ref)(chosen) - (log pi - log pi_ref)(rejected),
        with SUMMED response log-probabilities as in Rafailov et al. (2023)."""
        import torch
        ch = [(p, c) for p, c, _ in batch]
        rj = [(p, r) for p, _, r in batch]
        lc = self._seq_logprob(ch, reduce="sum")
        lr = self._seq_logprob(rj, reduce="sum")
        with torch.no_grad():
            with self.frozen_weights(ref_snapshot):
                rc = self._seq_logprob(ch, reduce="sum")
                rr = self._seq_logprob(rj, reduce="sum")
        return (lc - rc) - (lr - rr)

    def fit_dpo(self, train: Sequence[Tuple[str, str, str]], dev: Sequence[Tuple[str, str, str]],
                cfg: "DPOConfig", logger: logging.Logger, best_dir: Path, tag: str) -> Dict[str, Any]:
        """Offline DPO from the SFT policy, with the SFT policy (frozen LoRA snapshot) as reference.

        Early stopping on DEV preference accuracy (fraction of DEV pairs with a positive implicit
        margin); the DEV pairs come from dialogues disjoint from the training pairs."""
        import torch
        import torch.nn.functional as F
        from torch.optim import AdamW
        ref = self.snapshot_trainable()
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = AdamW(params, lr=cfg.lr, weight_decay=0.0)
        rng = random.Random(cfg.seed)
        train = list(train)
        dev = list(dev)

        def dev_eval() -> Tuple[float, float]:
            self.model.eval()
            ms = []
            with torch.no_grad():
                for s in range(0, len(dev), cfg.batch_size):
                    ms.extend(self._dpo_margins(dev[s:s + cfg.batch_size], ref).float().cpu().tolist())
            m = np.asarray(ms, float)
            loss = float(np.mean(np.logaddexp(0.0, -cfg.beta * m))) if m.size else float("nan")
            return (float(np.mean(m > 0)) if m.size else float("nan")), loss

        acc0, loss0 = dev_eval()
        best, best_step, stale, step = acc0, 0, 0, 0
        self.save_adapter(best_dir)
        hist = [{"step": 0, "dev_acc": acc0, "dev_loss": loss0}]
        logger.info("%s | %d train / %d dev preference pairs | beta=%.3g lr=%.2e | DEV implicit-reward accuracy "
                    "at the reference = %.4f (should be ~0.5)", tag, len(train), len(dev), cfg.beta, cfg.lr, acc0)
        every = max(1, len(train) // (cfg.batch_size * cfg.grad_accum * max(1, cfg.evals_per_epoch)))
        for ep in range(cfg.epochs):
            order = list(range(len(train)))
            rng.shuffle(order)
            micro = 0
            for s in range(0, len(order) - cfg.batch_size + 1, cfg.batch_size):
                self.model.eval()          # no LoRA dropout: policy and reference see the same network
                m = self._dpo_margins([train[i] for i in order[s:s + cfg.batch_size]], ref)
                loss = -F.logsigmoid(cfg.beta * m).mean() / cfg.grad_accum
                loss.backward()
                micro += 1
                if micro % cfg.grad_accum:
                    continue
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % every == 0:
                    acc, dl = dev_eval()
                    hist.append({"step": step, "dev_acc": acc, "dev_loss": dl})
                    mark = ""
                    if acc > best + 1e-4:
                        best, best_step, stale = acc, step, 0
                        self.save_adapter(best_dir)
                        mark = "  <- best"
                    else:
                        stale += 1
                    logger.info("%s | step %5d (epoch %d) | DEV preference accuracy %.4f | DEV DPO loss %.4f%s",
                                tag, step, ep, acc, dl, mark)
                    if stale >= cfg.patience:
                        break
            if stale >= cfg.patience:
                break
        self.load_adapter(best_dir)
        logger.info("%s | best DEV preference accuracy %.4f at step %d", tag, best, best_step)
        return {"best_dev_acc": best, "best_step": best_step, "history": hist, "n_train": len(train),
                "n_dev": len(dev)}


def project_features(H: np.ndarray, dim: int, seed: int) -> np.ndarray:
    d = H.shape[1]
    if d <= dim:
        return np.asarray(H, np.float32)
    rng = np.random.default_rng(seed)
    R = rng.normal(0.0, 1.0 / math.sqrt(dim), size=(d, dim))
    return np.asarray(H @ R, np.float32)


class StubPolicy:
    def __init__(self, logger: logging.Logger, seed: int = 0, dim: int = 64):
        self.logger = logger
        self.gen_batch = 32
        self.dim = dim
        self.seed = seed
        self.vocab = ("great", "sure", "booked", "sorry", "unfortunately", "confirmed", "table",
                      "train", "hotel", "reference", "number", "please", "thanks", "delay",
                      "cancelled", "available", "expensive", "cheap", "north", "centre")
        self._adapter: Dict[str, Any] = {}

    def save_adapter(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "stub.json").write_text(json.dumps(self._adapter))

    def load_adapter(self, path: Path) -> None:
        f = Path(path) / "stub.json"
        if f.exists():
            self._adapter = json.loads(f.read_text())

    def fit_supervised(self, train, dev, cfg, logger, best_dir: Path, tag: str, resample=None) -> Dict[str, Any]:
        if resample is not None:
            train = resample(1)
        logger.info("%s | STUB supervised fit over %d train / %d dev pairs", tag, len(train), len(dev))
        self._adapter = {"tag": tag, "n": len(train)}
        self.save_adapter(best_dir)
        return {"best_dev_nll": 1.0, "best_step": 1, "history": []}

    def fit_dpo(self, train, dev, cfg, logger, best_dir: Path, tag: str) -> Dict[str, Any]:
        logger.info("%s | STUB DPO fit over %d train / %d dev pairs", tag, len(train), len(dev))
        self._adapter = {"tag": tag, "n": len(train)}
        self.save_adapter(best_dir)
        return {"best_dev_acc": float("nan"), "best_step": 0, "history": [], "n_train": len(train),
                "n_dev": len(dev)}

    def snapshot_trainable(self) -> Dict[str, Any]:
        return {}

    @contextmanager
    def frozen_weights(self, snap: Optional[Dict[str, Any]]):
        yield

    def generate(self, prompts: Sequence[str], gen: GenConfig, seed: Optional[int] = None) -> List[str]:
        out = []
        for i, p in enumerate(prompts):
            h = int(hashlib.sha256(f"{p}|{seed}|{i}|{self._adapter.get('tag', '')}".encode()).hexdigest()[:12], 16)
            rng = np.random.default_rng(h % (2 ** 32))
            n = int(rng.integers(6, 34))
            words = list(rng.choice(np.asarray(self.vocab, dtype=object), size=n, replace=True))
            out.append(" ".join(str(w) for w in words).capitalize() + ".")
        return out

    def candidate_logprobs(self, prefixes: Sequence[str], candidates: Sequence[str],
                           reduce: str = "sum") -> np.ndarray:
        out = np.zeros((len(prefixes), len(candidates)), float)
        cs = np.asarray(StubSentiment()(list(candidates)), float)
        for i, p in enumerate(prefixes):
            resp = p.rsplit("Agent:", 1)[-1].rsplit("\nCustomer:", 1)[0]
            rs = float(StubSentiment()([resp])[0])
            h = int(hashlib.sha256(p.encode()).hexdigest()[:12], 16)
            rng = np.random.default_rng(h % (2 ** 32))
            out[i] = -2.0 - 1.6 * (cs - 0.75 * rs) ** 2 + rng.normal(0, 0.05, len(candidates))
        return out

    def features(self, prompts, responses, dim: int = 256, seed: int = 12345) -> np.ndarray:
        d = min(dim, self.dim)
        X = np.zeros((len(prompts), d), np.float32)
        for i, (p, r) in enumerate(zip(prompts, responses)):
            h = int(hashlib.sha256(f"{p}||{r}".encode()).hexdigest()[:16], 16)
            rng = np.random.default_rng(h % (2 ** 32))
            X[i] = rng.normal(0, 1, d)
            X[i, 0] = math.log1p(len(str(r).split()))
            X[i, 1] = sum(1.0 for w in str(r).lower().split() if w in ("great", "confirmed", "booked", "available"))
            X[i, 2] = sum(1.0 for w in str(r).lower().split() if w in ("sorry", "unfortunately", "delay", "cancelled"))
        return X


class StubSentiment:
    POS = ("great", "sure", "booked", "confirmed", "available", "cheap", "thanks", "please")
    NEG = ("sorry", "unfortunately", "delay", "cancelled", "expensive")

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros(len(texts), float)
        for i, t in enumerate(texts):
            w = str(t).lower().split()
            if not w:
                continue
            p = sum(1 for x in w if x.strip(".,!?") in self.POS)
            n = sum(1 for x in w if x.strip(".,!?") in self.NEG)
            out[i] = math.tanh((p - n) / max(1.0, math.sqrt(len(w))))
        return out


class OutcomePanel:
    def __init__(self, texts: Sequence[str], sentiments: Sequence[float],
                 center: Optional[Sequence[float]] = None, temperature: float = 1.0,
                 calibration: Optional[Dict[str, Any]] = None):
        self.texts = [norm_text(t) for t in texts]
        self.sent = np.asarray(sentiments, float)
        self.center = np.asarray(center, float) if center is not None else None
        self.temperature = float(temperature)
        self.calibration = calibration or {}

    @property
    def calibrated(self) -> bool:
        return self.center is not None

    @staticmethod
    def expectation(Zc: np.ndarray, sent: np.ndarray, temperature: float,
                    projector: Optional["LogitNullspaceProjector"] = None) -> np.ndarray:
        """O = softmax_T(proj(z)) . s  for centred logits z (rows = contexts)."""
        z = np.asarray(Zc, float)
        if projector is not None and projector.fitted:
            z = projector.apply(z)
        z = z / max(float(temperature), 1e-3)
        z = z - z.max(1, keepdims=True)
        w = np.exp(z)
        w = w / np.maximum(w.sum(1, keepdims=True), EPS)
        return w @ np.asarray(sent, float)

    def calibrate(self, policy, turns: Sequence[Turn], sentiment, logger: logging.Logger,
                  grid: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
                  projector: Optional["LogitNullspaceProjector"] = None,
                  quiet: bool = False, cache_file: Optional[Path] = None,
                  admissible: Optional[Callable[[float], bool]] = None) -> "OutcomePanel":
        """Select the softmax temperature on held-out TRAIN turns.

        v9: the (turns x panel) log-probability matrix does not depend on the projector or the
        temperature, so it is cached per turn set; re-calibrating after a projector change is a
        numpy operation instead of a 40-minute panel sweep (v8 recomputed it three times)."""
        turns = [t for t in turns if t.next_user_text]
        if len(turns) < 100:
            raise ValueError("OutcomePanel.calibrate: need >= 100 turns with a next customer turn")
        key = sha_of([t.uid for t in turns] + self.texts)
        cache = getattr(self, "_cal_cache", None)
        disk = None
        if (cache is None or cache[0] != key) and cache_file is not None and Path(cache_file).exists():
            try:
                z = np.load(cache_file, allow_pickle=False)
                if str(z["key"]) == key:
                    disk = (key, np.asarray(z["lp"], float), np.asarray(z["target"], float))
                    logger.info("panel calibration log-probabilities loaded from %s", cache_file)
            except Exception as e:
                logger.warning("ignoring unreadable calibration cache %s (%s)", cache_file, e)
        if disk is not None:
            self._cal_cache = disk
            cache = disk
        if cache is not None and cache[0] == key:
            lp, target = cache[1], cache[2]
        else:
            prefixes = [customer_prompt(t, t.gold_response) for t in turns]
            lp = policy.candidate_logprobs(prefixes, self.texts, reduce="sum")
            target = np.asarray(sentiment([t.next_user_text for t in turns]), float)
            self._cal_cache = (key, lp, target)
            if cache_file is not None:
                np.savez_compressed(cache_file, key=np.asarray(key), lp=lp, target=target)
        self.center = lp.mean(0)
        z0 = lp - self.center[None, :]
        best = (-2.0, float(grid[0]), 0.0)
        best_any = (-2.0, float(grid[0]), 0.0)
        n_adm = 0
        for T in grid:
            ok_T = True if admissible is None else bool(admissible(float(T)))
            n_adm += int(ok_T)
            zz = z0
            if projector is not None and projector.fitted:
                zz = projector.apply(z0)
            z = zz / max(T, 1e-3)
            z = z - z.max(1, keepdims=True)
            w = np.exp(z)
            w = w / np.maximum(w.sum(1, keepdims=True), EPS)
            r = spearman(w @ self.sent, target)
            ess = float(np.mean(1.0 / np.maximum((w ** 2).sum(1), EPS)))
            if math.isfinite(r) and r > best_any[0]:
                best_any = (r, float(T), ess)
            if ok_T and math.isfinite(r) and r > best[0]:
                best = (r, float(T), ess)
        constrained = admissible is not None
        if constrained and n_adm == 0:
            logger.warning("no panel temperature on the grid meets the FIT-half stability constraint; using the "
                           "unconstrained optimum T=%.3g and letting the gate report the consequence", best_any[1])
            best = best_any
        self.temperature = best[1]
        self.calibration = {"n": len(turns), "rho": best[0], "temperature": best[1],
                            "effective_panel_size": best[2], "panel_size": len(self.texts),
                            "projector_k": int(projector.k) if (projector is not None and projector.fitted) else 0,
                            "stability_constrained": constrained, "n_admissible_T": n_adm if constrained else None,
                            "unconstrained_T": best_any[1], "unconstrained_rho": best_any[0]}
        if best[1] >= max(grid) or best[1] <= min(grid):
            logger.warning("panel temperature T=%.3g sits on the edge of the search grid %s; the optimum may lie "
                           "outside it", best[1], list(grid))
        if not quiet:
            logger.info("panel calibrated on %d held-out TRAIN turns | per-candidate log-probability centring "
                        "removes each candidate's intrinsic length and frequency exactly | T=%.3g gives rho=%.4f "
                        "against the observed next-customer sentiment | effective panel size %.1f of %d%s",
                        len(turns), best[1], best[0], best[2], len(self.texts),
                        (f" | stability-constrained ({n_adm}/{len(grid)} temperatures admissible; unconstrained "
                         f"optimum T={best_any[1]:.3g} rho={best_any[0]:.4f})") if constrained else "")
        if best[2] > 0.8 * len(self.texts):
            logger.warning("the calibrated panel weights are nearly uniform (effective size %.1f of %d): the "
                           "expected-outcome estimator is close to non-responsive and will under-discriminate "
                           "variants", best[2], len(self.texts))
        return self

    def __len__(self) -> int:
        return len(self.texts)

    @staticmethod
    def build(turns: Sequence[Turn], sentiment, size: int, seed: int, logger: logging.Logger,
              pool_cap: int = 6000, policy=None, probe_prompt: Optional[str] = None,
              reduce: str = "mean") -> "OutcomePanel":
        cands: List[str] = []
        seen = set()
        for t in turns:
            u = norm_text(t.next_user_text or "")
            k = u.lower()
            if 3 <= len(u.split()) <= 40 and k not in seen:
                seen.add(k)
                cands.append(u)
            if len(cands) >= pool_cap:
                break
        if len(cands) < size:
            raise ValueError(f"OutcomePanel.build: only {len(cands)} candidates for size {size}")
        v = np.asarray(sentiment(cands), float)
        rng = np.random.default_rng(seed)

        # If a policy is provided, compute each candidate's intrinsic log-probability
        # under a fixed neutral prefix. This is the per-candidate bias that will be
        # subtracted at scoring time, so stratifying over it during panel construction
        # removes the correlation between intrinsic likelihood and sentiment.
        if policy is not None and probe_prompt is not None:
            lp = policy.candidate_logprobs([probe_prompt], cands, reduce=reduce)[0]
            lp = np.asarray(lp, float)
            nb = max(2, int(round(math.sqrt(size))))
            se = np.linspace(0, 1, nb + 1)
            lp_edges = np.quantile(lp, se)
            v_edges  = np.quantile(v,  se)
            # Ensure strict monotonic edges (degenerate quantiles break the grid).
            lp_edges = np.maximum.accumulate(lp_edges + 1e-12 * np.arange(lp_edges.size))
            v_edges  = np.maximum.accumulate(v_edges  + 1e-12 * np.arange(v_edges.size))
            picked: List[int] = []
            for i in range(nb):
                for j in range(nb):
                    in_lp = (lp >= lp_edges[i]) & (lp <= lp_edges[i + 1])
                    in_v  = (v  >= v_edges[j])  & (v  <= v_edges[j + 1])
                    cell = np.flatnonzero(in_lp & in_v)
                    if cell.size:
                        picked.append(int(cell[rng.integers(cell.size)]))
            # The 2D grid may return fewer than `size` if some cells are empty;
            # top up with random unpicked candidates.
            picked = sorted(set(picked))
            if len(picked) < size:
                taken = set(picked)
                rest = [i for i in range(len(cands)) if i not in taken]
                rng.shuffle(rest)
                picked.extend(rest[: size - len(picked)])
            picked = sorted(set(picked))[:size]
            texts = [cands[i] for i in picked]
            sents = v[picked]
            lp_picked = lp[picked]
            r2 = float(np.corrcoef(lp_picked, sents)[0, 1]) if len(picked) > 2 else float("nan")
            logger.info("outcome panel | %d candidates drawn from %d real next-customer turns, "
                        "2D-stratified over (sentiment, intrinsic %s log-prob) | sentiment [%.3f, %.3f] "
                        "mean %.3f sd %.3f | log-prob [%.3f, %.3f] | corr(log-prob, sentiment)=%+.4f",
                        len(texts), len(cands), reduce, float(sents.min()), float(sents.max()),
                        float(sents.mean()), float(sents.std()), float(lp_picked.min()),
                        float(lp_picked.max()), r2)
            return OutcomePanel(texts, sents)

        # Fallback: original 1D sentiment-stratified sampler (used by selftest and by
        # any caller that does not pass a policy).
        order = np.argsort(v)
        picked = []
        edges = np.linspace(0, len(order), size + 1).astype(int)
        for a, b in zip(edges[:-1], edges[1:]):
            if b <= a:
                continue
            picked.append(int(order[int(rng.integers(a, b))]))
        picked = sorted(set(picked))
        texts = [cands[i] for i in picked]
        sents = v[picked]
        logger.info("outcome panel | %d candidates drawn from %d real next-customer turns, stratified over "
                    "sentiment [%.3f, %.3f] | mean %.3f sd %.3f", len(texts), len(cands), float(sents.min()),
                    float(sents.max()), float(sents.mean()), float(sents.std()))
        return OutcomePanel(texts, sents)

    def state(self) -> Dict[str, Any]:
        return {"texts": self.texts, "sentiments": self.sent.tolist(),
                "center": self.center.tolist() if self.center is not None else None,
                "temperature": self.temperature, "calibration": self.calibration}

    @staticmethod
    def load(d: Dict[str, Any]) -> "OutcomePanel":
        return OutcomePanel(d["texts"], d["sentiments"], d.get("center"),
                            float(d.get("temperature", 1.0)), d.get("calibration"))


class UserSimulator:
    def __init__(self, policy, sentiment, logger: logging.Logger, n_rollouts: int = 12,
                 temperature: float = 0.9, max_new_tokens: int = 40, mode: str = "expected",
                 panel: Optional[OutcomePanel] = None, panel_temperature: float = 1.0,
                 projector: Optional[LogitNullspaceProjector] = None, n_orbit: int = 1,
                 orbit_seed: int = 4242):
        self.policy = policy
        self.sentiment = sentiment
        self.logger = logger
        self.n_rollouts = int(n_rollouts)
        self.gen = GenConfig(max_new_tokens=max_new_tokens, min_new_tokens=4, temperature=temperature)
        self.control_variate: Optional[FrozenControlVariate] = None
        self.length_control: Optional[InterventionalLengthCalibration] = None
        self._lp_cache: Dict[str, np.ndarray] = {}
        self._lp_cache_tag: Optional[str] = None
        self.cache_hits = 0
        self.cache_misses = 0
        self.mode = str(mode)
        self.panel = panel
        self.panel_temperature = float(panel_temperature)
        self.projector = projector
        self.n_orbit = max(1, int(n_orbit))
        self.orbit_seed = int(orbit_seed)

    def fit(self, train: Sequence[Turn], dev: Sequence[Turn], cfg: SFTConfig, best_dir: Path,
            augment_levels: int = 2, seed: int = 0, bank: str = "train") -> Dict[str, Any]:
        src = [t for t in train if t.next_user_text]
        dv = [(customer_prompt(t, t.gold_response), t.next_user_text)
              for t in dev if t.next_user_text][: cfg.dev_examples]
        if len(src) < cfg.batch_size or len(dv) < 8:
            raise ValueError(f"insufficient simulator data (train={len(src)}, dev={len(dv)})")

        def resample(epoch: int) -> List[Tuple[str, str, Optional[str]]]:
            aug = LengthInvarianceAugmenter(n_levels=augment_levels, seed=seed * 1000 + epoch, bank=bank)
            items: List[Tuple[str, str, Optional[str]]] = []
            rng = np.random.default_rng(seed * 7919 + epoch)
            for t in src:
                lv = int(rng.integers(0, augment_levels + 1))
                main = customer_prompt(t, t.gold_response)
                alt = customer_prompt(t, aug.lengthen(t.gold_response, lv)) if lv > 0 else None
                items.append((main, t.next_user_text, alt))
            return items

        base_w = float(np.mean([len(t.gold_response.split()) for t in src]))
        a0 = LengthInvarianceAugmenter(n_levels=augment_levels, seed=seed, bank=bank)
        top_w = float(np.mean([len(a0.lengthen(t.gold_response, augment_levels).split()) for t in src]))
        self.logger.info("length invariance | %d unique targets held fixed (no duplication); one length rung in "
                         "0..%d resampled per example per epoch | mean agent turn %.1f -> %.1f words at the top "
                         "rung | invariance penalty lambda=%.3g on the squared mean-logprob difference",
                         len(src), augment_levels, base_w, top_w, cfg.invariance_coef)
        return self.policy.fit_supervised(resample(0), dv, cfg, self.logger, best_dir, "SIM", resample=resample)

    def _raw_sampled(self, turns: Sequence[Turn], responses: Sequence[str], crn_seed) -> np.ndarray:
        prompts, index = [], []
        for i, (t, r) in enumerate(zip(turns, responses)):
            for _ in range(self.n_rollouts):
                prompts.append(customer_prompt(t, r))
                index.append(i)
        replies = self.policy.generate(prompts, self.gen, seed=crn_seed)
        v = np.asarray(self.sentiment(replies), float)
        idx = np.asarray(index, int)
        agg = np.bincount(idx, weights=v, minlength=len(turns))
        cnt = np.bincount(idx, minlength=len(turns)).astype(float)
        return agg / np.maximum(cnt, 1.0)

    def _raw_logprobs(self, prefixes: Sequence[str]) -> np.ndarray:
        """Uncentred panel log-probabilities with an exact memo.

        The panel estimator is deterministic given (simulator adapter, panel texts, prefix), so
        a prefix scored once never needs to be scored again.  v8 re-scored identical prefixes in
        the probe, the control-variate fit, the length fit, the gate and every re-calibration;
        the memo removes that redundancy without changing a single number."""
        tag = sha_of(self.panel.texts)
        if tag != self._lp_cache_tag:
            self._lp_cache, self._lp_cache_tag = {}, tag
        keys = [hashlib.sha1(p.encode("utf-8")).hexdigest() for p in prefixes]
        todo, seen = [], set()
        for k, p in zip(keys, prefixes):
            if k not in self._lp_cache and k not in seen:
                todo.append((k, p))
                seen.add(k)
        self.cache_misses += len(todo)
        self.cache_hits += len(keys) - len(todo)
        if todo:
            lp = self.policy.candidate_logprobs([p for _, p in todo], self.panel.texts, reduce="sum")
            for (k, _), row in zip(todo, lp):
                self._lp_cache[k] = np.asarray(row, float)
        return np.stack([self._lp_cache[k] for k in keys], 0) if keys else np.zeros((0, len(self.panel)))

    def panel_logits(self, turns: Sequence[Turn], responses: Sequence[str]) -> np.ndarray:
        """Per-candidate centred log-probabilities, BEFORE the invariance projection.
        This is the quantity the projector is fitted on."""
        if self.panel is None or not self.panel.calibrated:
            raise RuntimeError("panel_logits requires a calibrated OutcomePanel")
        prefixes = [customer_prompt(t, r) for t, r in zip(turns, responses)]
        return self._raw_logprobs(prefixes) - self.panel.center[None, :]

    def _panel_expectation(self, turns: Sequence[Turn], responses: Sequence[str]) -> np.ndarray:
        return OutcomePanel.expectation(self.panel_logits(turns, responses), self.panel.sent,
                                        self.panel.temperature, self.projector)

    def _raw_expected(self, turns: Sequence[Turn], responses: Sequence[str]) -> np.ndarray:
        """Expected panel outcome, optionally Reynolds-averaged over the length orbit.

        The projector is an exact linear erasure; orbit averaging is the residual,
        second-order defence.  Writing G for the filler group generated by the augmenter,
        the orbit mean (1/|G|) sum_g O(g.x) is by construction constant along any exact
        orbit and, for a near-additive residual response, shrinks the induced variance by
        roughly 1/K for K near-independent rungs.  It is deliberately OFF by default
        (n_orbit=1): it costs K forward passes per scored response and it only confers
        invariance to this specific filler family, whereas the projector generalises to any
        perturbation whose logit signature lies in the fitted span."""
        if self.panel is None:
            raise RuntimeError("expected-outcome mode requires an OutcomePanel")
        if not self.panel.calibrated:
            raise RuntimeError("expected-outcome mode requires a calibrated panel; run the simulator stage")
        if self.n_orbit <= 1:
            return self._panel_expectation(turns, responses)
        acc = np.zeros(len(turns), float)
        for lvl in range(self.n_orbit):
            if lvl == 0:
                rs = list(responses)
            else:
                aug = LengthInvarianceAugmenter(n_levels=lvl, seed=self.orbit_seed + 7919 * lvl, bank=FIT_BANK)
                rs = [aug.lengthen(r, lvl) for r in responses]
            acc += self._panel_expectation(turns, rs)
        return acc / float(self.n_orbit)

    def outcome(self, turns: Sequence[Turn], responses: Sequence[str],
                crn_seed: Optional[int] = None) -> np.ndarray:
        return self.rollout(turns, responses, crn_seed)

    def raw(self, turns: Sequence[Turn], responses: Sequence[str], crn_seed: Optional[int] = None) -> np.ndarray:
        """Outcome estimate BEFORE the control variate and the length calibration."""
        if self.mode == "expected":
            return self._raw_expected(turns, responses)
        if self.mode == "sample":
            return self._raw_sampled(turns, responses, crn_seed)
        raise ValueError(f"unknown outcome mode {self.mode}")

    def rollout(self, turns: Sequence[Turn], responses: Sequence[str], crn_seed: Optional[int] = None) -> np.ndarray:
        turns = list(turns)
        responses = list(responses)
        if len(turns) != len(responses):
            raise ValueError("rollout: length mismatch")
        if not turns:
            return np.zeros(0, float)
        raw = self.raw(turns, responses, crn_seed)
        cur = np.asarray(self.sentiment([t.user_text for t in turns]), float)
        y = raw - cur
        if self.control_variate is None:
            self.control_variate = FrozenControlVariate().fit(y, cur, self.logger)
        y = self.control_variate.apply(y, cur)
        if self.length_control is not None and self.length_control.fitted:
            y = self.length_control.apply(responses, y)
        if not np.all(np.isfinite(y)):
            raise RuntimeError("non-finite simulated outcome")
        return y.astype(np.float64)

    def fit_length_calibration(self, turns: Sequence[Turn], base: Sequence[str], perturbed: Sequence[str],
                               crn_seed: int = 9091, seed: int = 0) -> InterventionalLengthCalibration:
        """Fit h(L) on paired (base, padded) responses of the SAME contexts.  The control variate
        and any existing calibration cancel exactly in the paired difference, so the raw
        estimator is differenced directly; common random numbers are used in sample mode."""
        turns = list(turns)
        d = self.raw(turns, list(perturbed), crn_seed) - self.raw(turns, list(base), crn_seed)
        lc = InterventionalLengthCalibration().fit(list(base), list(perturbed), d,
                                                   [t.dialogue_id for t in turns], self.logger, seed=seed)
        self.length_control = lc
        return lc


def _pairwise_flip_rate(a: np.ndarray, b: np.ndarray, gid: np.ndarray, tol: float = 1e-9) -> float:
    """Share of within-group pairs whose order under `a` is reversed under `b` (pairs tied under `a`
    are skipped).  Vectorised per group; v13's version was quadratic in the number of groups."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    gid = np.asarray(gid)
    if a.size < 2:
        return float("nan")
    order = np.argsort(gid, kind="mergesort")
    g, a, b = gid[order], a[order], b[order]
    cuts = np.flatnonzero(g[1:] != g[:-1]) + 1
    flips = tot = 0
    for sa, sb in zip(np.split(a, cuts), np.split(b, cuts)):
        if sa.size < 2:
            continue
        iu = np.triu_indices(sa.size, 1)
        da = (sa[:, None] - sa[None, :])[iu]
        db = (sb[:, None] - sb[None, :])[iu]
        m = np.abs(da) > tol
        tot += int(m.sum())
        flips += int(np.sum((da * db < 0) & m))
    return flips / tot if tot else float("nan")


def intervention_stats(y0: np.ndarray, y1: np.ndarray, y0b: np.ndarray, yA: np.ndarray, yB: np.ndarray,
                       gid: Optional[np.ndarray] = None, cluster: Optional[Sequence[Any]] = None,
                       seed: int = 7777, margin_frac: float = 0.15, max_tau: float = 0.35,
                       max_excess_flip: float = 0.05, max_abs_flip: float = 0.10, n_boot: int = 800,
                       mode: str = "expected", bank: str = "heldout", level: int = 1,
                       words_added: float = float("nan"), surface_word_delta: float = float("nan"),
                       projector_k: int = 0, n_orbit: int = 1,
                       logger: Optional[logging.Logger] = None, label: str = "") -> Dict[str, Any]:
    """All gate statistics of a paired length intervention, given the five outcome vectors.

    Split out of paired_length_intervention in v13 so that an ablation can hold the SAMPLED
    responses fixed and vary only the correction applied to the outcomes (none / v8
    observational / v12 interventional) or the filler bank, at the cost of a single GPU sweep."""
    log = logger if logger is not None else logging.getLogger("caro.intervention")
    y0, y1, y0b = np.asarray(y0, float), np.asarray(y1, float), np.asarray(y0b, float)
    yA, yB = np.asarray(yA, float), np.asarray(yB, float)
    n = int(y0.size)
    if gid is None:
        gid = np.arange(n)
    if cluster is None:
        cluster = np.arange(n)
    d = y1 - y0
    d0 = y0b - y0
    dsurf = yB - yA
    n = int(d.size)
    if gid is None:
        gid = np.arange(n)
    gid = np.asarray(gid)
    cluster = np.asarray(list(cluster))

    var_d = float(np.var(d, ddof=1)) if n > 1 else float("nan")
    var_d0 = float(np.var(d0, ddof=1)) if n > 1 else 0.0
    var_dsurf = float(np.var(dsurf, ddof=1)) if n > 1 else 0.0
    sigma_mc2 = max(0.0, var_d0 / 2.0)
    var_y = float(np.var(y0, ddof=1)) if n > 1 else float("nan")
    gsz = np.asarray([int((gid == g).sum()) for g in np.unique(gid)])
    var_within_raw = within_group_var(y0, gid) if (gsz > 1).any() else var_y
    var_signal = max(var_within_raw - sigma_mc2, EPS)
    excess_var = max(0.0, var_d - var_d0)
    tau_c = float(math.sqrt(excess_var / var_signal))
    # v7 BUG, fixed here: tau_raw was reported as sqrt(var_d / var_y) against the POOLED
    # outcome variance while tau_corrected used the mean WITHIN-context variance.  The two
    # were printed side by side as if the only difference were the noise correction, so the
    # "correction" appeared to inflate the ratio (0.251 -> 0.316) when in fact the
    # denominator had silently changed by a factor of ~1.6.  The within-context variance is
    # the correct denominator -- the reward model only ever ranks candidates inside one
    # context -- so tau_raw now uses it too, and the pooled figure is reported separately
    # and clearly labelled.
    tau_raw = float(math.sqrt(var_d / var_signal)) if math.isfinite(var_d) else float("nan")
    tau_pooled = float(math.sqrt(var_d / max(var_y, EPS))) if math.isfinite(var_d) else float("nan")
    tau_surf = float(math.sqrt(var_dsurf / var_signal)) if math.isfinite(var_dsurf) else float("nan")

    sd_within = within_group_sd(y0, gid) if n > 1 else float("nan")
    if not math.isfinite(sd_within) or sd_within <= EPS:
        sd_within = math.sqrt(var_signal)
    margin = float(margin_frac * sd_within)

    se = float(np.std(d, ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    _, lo90, hi90 = cluster_bootstrap_ci(lambda ix: float(np.mean(d[ix])), cluster, n_boot, seed + 1, conf=0.90)
    tost = bool(math.isfinite(lo90) and math.isfinite(hi90) and lo90 > -margin and hi90 < margin)

    def _tau(ix, copy):
        vd = float(np.var(d[ix], ddof=1))
        vd0 = float(np.var(d0[ix], ddof=1))
        vw = within_group_var(y0[ix], boot_groups(gid[ix], copy))
        vs = max(vw - max(0.0, vd0 / 2.0), EPS)
        return float(math.sqrt(max(0.0, vd - vd0) / vs))

    _, tlo, thi = cluster_bootstrap_ci(_tau, cluster, n_boot, seed + 2, conf=0.90, with_copy=True)
    flip_perturb = _pairwise_flip_rate(y0, y1, gid)
    _, flo, fhi = cluster_bootstrap_ci(lambda ix, cp: _pairwise_flip_rate(y0[ix], y1[ix], boot_groups(gid[ix], cp)),
                                       cluster, max(200, n_boot // 2), seed + 4, conf=0.90, with_copy=True)
    flip_null = _pairwise_flip_rate(y0, y0b, gid)
    flip_surf = _pairwise_flip_rate(yA, yB, gid)
    excess_flip = (flip_perturb - flip_null) if math.isfinite(flip_perturb) and math.isfinite(flip_null) else float("nan")

    tau_ok = (not math.isfinite(thi)) or thi <= max_tau
    deterministic = var_d0 <= 1e-12
    # Symmetric rule across estimator modes.  The absolute ceiling always binds; the excess
    # over the replication null is an ADDITIONAL requirement wherever that null is defined.
    # v7 applied the absolute ceiling only in the deterministic branch, so a sampled
    # estimator could subtract its own Monte-Carlo noise and pass at any flip rate.
    if not math.isfinite(flip_perturb):
        flip_ok, flip_rule = True, "undefined"
    elif deterministic:
        flip_ok = flip_perturb <= max_abs_flip
        flip_rule = (f"absolute (deterministic estimator, no replication null): "
                     f"{flip_perturb:.4f} <= {max_abs_flip:.2f}")
    else:
        flip_ok = (flip_perturb <= max_abs_flip) and (excess_flip <= max_excess_flip)
        flip_rule = (f"absolute {flip_perturb:.4f} <= {max_abs_flip:.2f} AND excess over "
                     f"replication null {excess_flip:+.4f} <= {max_excess_flip:.2f}")
    out = {
        "n": n, "mode": mode, "filler_bank": bank, "level": int(level), "label": label,
        "delta_mean": float(np.mean(d)), "delta_se": se, "delta_ci90": [lo90, hi90],
        "equivalence_margin": margin, "margin_frac": margin_frac, "tost_passes": tost,
        "sd_within_group": sd_within,
        "var_delta": var_d, "var_delta_null": var_d0, "sigma_mc_sq": sigma_mc2,
        "var_delta_surface_null": var_dsurf,
        "var_outcome_pooled": var_y, "var_signal": var_signal,
        "tau_raw_within": tau_raw, "tau_pooled_reference_only": tau_pooled,
        "tau_surface_null": tau_surf,
        "tau_noise_floor": float(math.sqrt(var_d0 / var_signal)) if math.isfinite(var_d0) else float("nan"),
        "tau_corrected": tau_c, "tau_corrected_ci90": [tlo, thi], "tau_ok": tau_ok,
        "flip_perturb": flip_perturb, "flip_ci90": [flo, fhi], "flip_null": flip_null, "flip_surface_null": flip_surf,
        "n_groups": int(np.unique(gid).size), "variants_per_group": float(np.mean(gsz)),
        "excess_flip": excess_flip, "flip_ok": flip_ok,
        "flip_rule": flip_rule, "deterministic": bool(deterministic), "var_within_raw": var_within_raw,
        "words_added": float(words_added), "surface_null_word_delta": float(surface_word_delta),
        "projector_k": int(projector_k), "n_orbit": int(n_orbit),
    }
    out["passes"] = bool(tost and tau_ok and flip_ok)
    log.info("paired length intervention [%s, %s filler bank%s] | +%.1f content-free words | delta=%+.5f "
             "CI90[%+.5f,%+.5f] vs equivalence margin +-%.5f -> TOST %s", mode, bank,
                    (", " + label) if label else "", out["words_added"], out["delta_mean"],
                    lo90, hi90, margin, "PASS" if tost else "FAIL")
    log.info("  variance decomposition [all ratios share the WITHIN-context denominator] | "
                    "Var(delta)=%.5f Var(replication null)=%.5f Var(surface null)=%.5f | "
                    "Var(signal)=%.5f | tau_raw=%.3f noise_floor=%.3f tau_corrected=%.3f CI90[%.3f,%.3f] -> %s",
                    var_d, var_d0, var_dsurf, var_signal, tau_raw, out["tau_noise_floor"], tau_c,
                    tlo, thi, "PASS" if tau_ok else "FAIL")
    log.info("  attribution | length-matched surface null tau=%.3f (vs length tau=%.3f) | "
                    "pooled-denominator tau=%.3f is reported for reference only and is NOT the gate",
                    tau_surf, tau_raw, tau_pooled)
    log.info("  rank stability | within-group flip rate: length %.4f, replication null %.4f, "
                    "length-matched surface null %.4f | rule = %s -> %s",
                    flip_perturb, flip_null, flip_surf, flip_rule, "PASS" if flip_ok else "FAIL")
    if (not deterministic and math.isfinite(flip_null) and math.isfinite(flip_perturb) and not flip_ok
            and flip_null >= flip_perturb):
        log.warning("  attribution: re-running the SAME responses flips %.1f%% of within-context pairs, more "
                    "than the length contrast (%.1f%%): the failure is the estimator's Monte Carlo noise, not "
                    "length or surface sensitivity. Use the deterministic panel estimator.",
                    100 * flip_null, 100 * flip_perturb)
    elif math.isfinite(flip_surf) and math.isfinite(flip_perturb) and not flip_ok:
        if flip_surf >= 0.6 * flip_perturb:
            log.warning("  attribution: most of the instability survives at MATCHED length -- the "
                               "estimator is surface-brittle rather than length-biased; erasing the length "
                               "subspace alone will not fix it")
        else:
            log.info("  attribution: the instability is specific to the length contrast, which is "
                            "what the logit nullspace projection targets")
    return out


def paired_length_intervention(sim: "UserSimulator", turns: Sequence[Turn], responses: Sequence[str],
                               gid: Optional[np.ndarray] = None, cluster: Optional[Sequence[Any]] = None,
                               seed: int = 7777, margin_frac: float = 0.15, max_tau: float = 0.35,
                               max_excess_flip: float = 0.05, max_abs_flip: float = 0.10,
                               n_boot: int = 800, bank: str = "heldout", level: int = 1,
                               y_base: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Paired content-free length intervention with two negative controls.

    Three contrasts are measured on the SAME contexts:
      d      = y(longer) - y(base)        the length contrast under test
      d_rep  = y(base)'  - y(base)        Monte-Carlo replication null (identically zero,
                                          by construction, for the deterministic panel
                                          estimator)
      d_surf = y(long_B) - y(long_A)      length-matched surface null: long_A and long_B
                                          carry an identical word multiset and differ only
                                          in where the filler sits

    d_surf is an ATTRIBUTION diagnostic, not a licence.  It answers "is this a length
    effect or generic surface brittleness?"; it is never subtracted from a gate, because a
    surface-brittle outcome is just as unusable as a length-biased one for within-context
    ranking.  The gates below are strictly stronger than v7's: the absolute flip ceiling
    now applies in BOTH estimator modes (v7 applied it only when the replication null was
    exactly zero, which let a noisy sampled estimator subtract its own noise and pass)."""
    aug = LengthInvarianceAugmenter(n_levels=level, seed=seed, bank=bank)
    responses = list(responses)
    longer = [aug.lengthen(r, level) for r in responses]
    augm = LengthInvarianceAugmenter(n_levels=level, seed=seed + 4242, bank=bank)
    pairs = [augm.matched_pair(r, level) for r in responses]
    longA = [a for a, _ in pairs]
    longB = [b for _, b in pairs]
    y0 = (np.asarray(y_base, float) if y_base is not None else sim.outcome(turns, responses, crn_seed=seed))
    y1 = sim.outcome(turns, longer, crn_seed=seed)
    if sim.mode == "expected":
        # The panel expectation is a deterministic function of the prompt: the replication
        # null is identically zero, so it is set exactly instead of paying a full panel sweep.
        y0b = y0.copy()
    else:
        y0b = sim.outcome(turns, responses, crn_seed=seed + 101)
    yA = sim.outcome(turns, longA, crn_seed=seed + 202)
    yB = sim.outcome(turns, longB, crn_seed=seed + 202)
    if gid is None:
        gid = np.arange(len(responses))
    if cluster is None:
        cluster = np.asarray([t.dialogue_id for t in turns])
    out = intervention_stats(y0, y1, y0b, yA, yB, gid=gid, cluster=cluster, seed=seed,
                             margin_frac=margin_frac, max_tau=max_tau, max_excess_flip=max_excess_flip,
                             max_abs_flip=max_abs_flip, n_boot=n_boot, mode=sim.mode, bank=bank, level=level,
                             words_added=float(np.mean([len(b.split()) - len(a.split())
                                                        for a, b in zip(responses, longer)])),
                             surface_word_delta=float(np.mean([abs(len(a.split()) - len(b.split()))
                                                               for a, b in zip(longA, longB)])),
                             projector_k=int(sim.projector.k) if getattr(sim, "projector", None) is not None else 0,
                             n_orbit=int(getattr(sim, "n_orbit", 1)), logger=sim.logger)
    # v14 (R6): the exact gate sample, so that the ablation can re-score THE SAME responses.
    out["_arrays"] = {"uid": np.asarray([t.uid for t in turns]), "gid": np.asarray(gid),
                      "cluster": np.asarray([str(c) for c in cluster]),
                      "base": np.asarray(responses), "longer": np.asarray(longer),
                      "longA": np.asarray(longA), "longB": np.asarray(longB),
                      "y0": y0, "y1": y1, "yA": yA, "yB": yB}
    return out


def select_outcome_mode(sim: "UserSimulator", turns: Sequence[Turn], logger: logging.Logger,
                        candidates: Sequence[str] = ("sample", "expected")) -> Dict[str, Any]:
    """Choose the outcome estimator on TRAIN-split labelled turns (never on validation data).

    v15 rule.  The corpus labels are WITHIN-context rankings of counterfactual responses, so the
    estimator must first of all reproduce its own rankings when re-run on the same responses.  The
    deterministic panel expectation ("expected") does so by construction; the Monte Carlo estimator
    ("sample") does not: in the v14 validate run its replication null flipped 34.6% of within-context
    pairs, 3.5x the pre-registered 10% ceiling, so no invariance gate could pass whatever the length
    behaviour (tau_corrected was 0 and the TOST passed; only the flip rule failed).  Hence:
      1. "expected" is selected whenever it is available and its human anchor is positive with a
         Fisher 95% CI excluding zero;
      2. otherwise the estimator with the larger human anchor is selected, and a sampled estimator is
         flagged as unlikely to pass the flip gate.
    For "sample" the split-half noise (two disjoint halves of the rollouts) is reported, so the
    choice is documented with the number that motivates it.  v14 maximised the anchor alone.
    This is a change of the pre-registered selection rule and must be reported as such."""
    turns = [t for t in turns if t.next_user_text and t.human_valence is not None]
    if len(turns) < 100:
        raise ValueError("select_outcome_mode: need >= 100 labelled turns with a next customer turn")
    gold = [t.gold_response for t in turns]
    val = np.asarray([t.human_valence for t in turns], float)
    shift_target = (np.asarray(sim.sentiment([t.next_user_text for t in turns]), float)
                    - np.asarray(sim.sentiment([t.user_text for t in turns]), float))
    keep_mode, keep_cv, keep_lc, keep_r = sim.mode, sim.control_variate, sim.length_control, sim.n_rollouts
    sim.control_variate, sim.length_control = None, None
    report: Dict[str, Any] = {}
    try:
        for m in candidates:
            if m == "expected" and (sim.panel is None or not sim.panel.calibrated):
                continue
            sim.mode = m
            try:
                if m == "sample":
                    sim.n_rollouts = max(1, keep_r // 2)
                    r1 = sim.raw(turns, gold, crn_seed=31337)
                    r2 = sim.raw(turns, gold, crn_seed=41337)
                    sim.n_rollouts = keep_r
                    y = 0.5 * (r1 + r2) - np.asarray(sim.sentiment([t.user_text for t in turns]), float)
                    noise_sd = float(np.std(r1 - r2, ddof=1) / 2.0)       # sd of the full-R estimate
                    extra = {"deterministic": False, "mc_noise_sd": noise_sd,
                             "split_half_r": pearson(r1, r2)}
                else:
                    y = sim.raw(turns, gold, crn_seed=31337) - np.asarray(sim.sentiment([t.user_text for t in turns]), float)
                    extra = {"deterministic": True, "mc_noise_sd": 0.0, "split_half_r": 1.0}
            except Exception as e:
                logger.warning("outcome mode '%s' unavailable: %s", m, e)
                continue
            rho = spearman(y, val)
            report[m] = {"human_anchor_rho": rho, "human_anchor_ci95": list(fisher_ci(rho, len(turns))),
                         "shift_anchor_rho": spearman(y, shift_target), "sd": float(np.std(y)), **extra}
    finally:
        sim.control_variate, sim.length_control = keep_cv, keep_lc
        sim.mode, sim.n_rollouts = keep_mode, keep_r
    if not report:
        raise RuntimeError("select_outcome_mode: no usable outcome estimator")
    ex_ = report.get("expected")
    if ex_ is not None and math.isfinite(ex_["human_anchor_ci95"][0]) and ex_["human_anchor_ci95"][0] > 0:
        best, why = "expected", "deterministic and its human anchor CI excludes zero"
    else:
        best = max(report, key=lambda m: (report[m]["human_anchor_rho"]
                                          if math.isfinite(report[m]["human_anchor_rho"]) else -2.0))
        why = "largest human anchor (the deterministic estimator is unavailable or carries no human signal)"
    logger.info("outcome estimator selection on %d TRAIN labelled turns (gold responses):", len(turns))
    for m, v in report.items():
        logger.info("  %-9s | human anchor rho=%+.4f CI95[%+.4f,%+.4f] | shift anchor rho=%+.4f | sd=%.4f | "
                    "Monte Carlo noise sd=%.4f%s", m, v["human_anchor_rho"], *v["human_anchor_ci95"],
                    v["shift_anchor_rho"], v["sd"], v["mc_noise_sd"], "  <- selected" if m == best else "")
    logger.info("  rule: %s", why)
    if best == "sample":
        logger.warning("the SAMPLED estimator was selected: its own Monte Carlo noise flips within-context "
                       "rankings, so the flip gate is expected to fail. Increase --sim-rollouts or fix the panel.")
    report["selected"] = best
    report["rule"] = why
    return report


def validate_simulator(sim: UserSimulator, turns: Sequence[Turn], proposal, n_variants: int, gen: GenConfig,
                       logger: logging.Logger, min_icc: float = 0.25, min_anchor_rho: float = 0.15,
                       max_within_length_rho: float = 0.20, n_boot: int = 800, seed: int = 0,
                       margin_frac: float = 0.15, max_excess_flip: float = 0.05,
                       max_tau: float = 0.35, max_abs_flip: float = 0.10, gate_bank: str = "heldout",
                       n_intervention: int = 600) -> Dict[str, Any]:
    """Pre-registered gate on contexts disjoint from every fitted nuisance component.

    v9 changes: (i) no observational length residualisation is applied here or anywhere else
    (see InterventionalLengthCalibration for why it is a bad control); the within-context
    length association of the FINAL outcome is still gated at the same threshold, now without a
    post-hoc spline in front of it; (ii) the paired intervention perturbs with the held-out
    filler bank; (iii) within-group bootstrap statistics keep duplicated clusters distinct."""
    turns = [t for t in turns if t.next_user_text]
    n = len(turns)
    if n < 30:
        raise ValueError("validate_simulator: need >= 30 contexts with a next customer turn")
    y_gold = sim.rollout(turns, [t.gold_response for t in turns], crn_seed=101)
    y_real = (np.asarray(sim.sentiment([t.next_user_text for t in turns]), float)
              - np.asarray(sim.sentiment([t.user_text for t in turns]), float))
    rho_shift = spearman(y_gold, y_real)
    val = np.asarray([t.human_valence if t.human_valence is not None else np.nan for t in turns], float)
    ok = np.isfinite(val)
    human_n = int(ok.sum())
    gold_L = loglen([t.gold_response for t in turns])
    rho_len_label = spearman(gold_L[ok], val[ok]) if human_n >= 20 else float("nan")

    prompts = [agent_prompt(t) for t in turns for _ in range(n_variants)]
    variants = proposal.generate(prompts, gen, seed=202)
    flat = [t for t in turns for _ in range(n_variants)]
    gid = np.repeat(np.arange(n), n_variants)
    cluster = np.asarray([t.dialogue_id for t in flat])
    if sim.mode == "expected":
        r1 = sim.outcome(flat, variants, crn_seed=303)
        r2 = r1
        logger.info("expected-outcome mode | the estimator is a deterministic panel expectation, so the "
                    "split-half replicate is exact and Monte Carlo variance is zero by construction")
    else:
        full = sim.n_rollouts
        half = max(1, full // 2)
        try:
            sim.n_rollouts = half
            r1 = sim.outcome(flat, variants, crn_seed=303)
            r2 = sim.outcome(flat, variants, crn_seed=404)
        finally:
            sim.n_rollouts = full
    y = 0.5 * (r1 + r2)
    Y = y.reshape(n, n_variants)
    # v14 (R8): v13 reported ICC = 1 and split-half reliability = 1 for the deterministic panel
    # estimator.  Both are true by construction (the replicate IS the estimate, var_repl = 0), so
    # they carry no evidence and are now reported as not applicable.  (The v13 variable called
    # var_between was in fact the WITHIN-context variance.)  The informative responsiveness
    # statistic is how much of the outcome variance lies between responses to the same context
    # rather than between contexts, using the unbiased one-way random-effects decomposition.
    var_within_ctx = float(np.mean(Y.var(1, ddof=1))) if n_variants > 1 else 0.0
    var_ctx_means = float(np.var(Y.mean(1), ddof=1)) if n > 1 else 0.0
    var_between_ctx = max(0.0, var_ctx_means - var_within_ctx / max(1, n_variants))
    response_share = float(var_within_ctx / max(var_within_ctx + var_between_ctx, EPS))
    if sim.mode == "expected":
        icc: Optional[float] = None
        reliability: Optional[float] = None
    else:
        var_repl = float(np.mean((r1 - r2) ** 2) / 2.0)
        icc = float(max(0.0, min(1.0, (var_within_ctx - var_repl / 2.0) / max(var_within_ctx, EPS))))
        rel_half = pearson(r1, r2)
        reliability = (float(2 * rel_half / (1 + rel_half)) if math.isfinite(rel_half) and rel_half > -1
                       else float("nan"))

    Lv = loglen(variants)
    y_res = y
    w_before = within_group_spearman(y, Lv, gid)
    w_after = w_before
    pt, lo_b, hi_b = cluster_bootstrap_ci(
        lambda ix, cp: within_group_spearman(y_res[ix], Lv[ix], boot_groups(gid[ix], cp))["mean"],
        cluster, n_boot, seed + 1, with_copy=True)

    y_gold_res = y_gold
    gcl = np.asarray([t.dialogue_id for t in turns])[ok]
    a_pt, a_lo, a_hi = cluster_bootstrap_ci(
        lambda ix: spearman(y_gold_res[ok][ix], val[ok][ix]), gcl, n_boot, seed + 2) if human_n >= 30 else (
        float("nan"), float("nan"), float("nan"))
    # Human-label length reference: does length itself predict the human label WITHIN a context
    # comparison?  Only gold responses carry labels, so this is the pooled association; it is
    # reported so that a residual observational length trend in the outcome can be compared with
    # the trend humans themselves exhibit rather than being forced to zero.
    rho_len_outcome_gold = spearman(y_gold[ok], gold_L[ok]) if human_n >= 20 else float("nan")

    echo = spearman(np.asarray(sim.sentiment(list(variants)), float), y_res)
    emoji = np.asarray([len(EMOJI_RE.findall(v)) for v in variants], float)
    echo_emoji = spearman(emoji, y_res) if np.ptp(emoji) > 0 else float("nan")

    bank_sent = {b: float(np.mean(sim.sentiment(list(FILLER_BANKS[b][0]) + list(FILLER_BANKS[b][1]))))
                 for b in FILLER_BANKS}
    logger.info("filler-bank affect check | mean sentiment of the fillers: %s (all should be near 0)",
                ", ".join(f"{k}={v:+.3f}" for k, v in bank_sent.items()))

    n_pi = min(len(flat), n_intervention)
    intervention = paired_length_intervention(sim, flat[:n_pi], list(variants)[:n_pi],
                                              gid=gid[:n_pi], cluster=cluster[:n_pi], seed=seed + 3,
                                              margin_frac=margin_frac, max_tau=max_tau,
                                              max_excess_flip=max_excess_flip, max_abs_flip=max_abs_flip,
                                              bank=gate_bank,
                                              y_base=(y[:n_pi] if sim.mode == "expected" else None))

    fails: List[str] = []
    warns: List[str] = []
    if icc is not None and icc < min_icc:
        fails.append(f"responsiveness ICC={icc:.3f} < {min_icc:.2f}")
    if not math.isfinite(rho_shift) or rho_shift < min_anchor_rho:
        fails.append(f"sentiment-shift anchor rho={rho_shift:.3f} < {min_anchor_rho:.2f}")
    if human_n < 30:
        fails.append(f"only {human_n} turns carry a human emotion label")
    elif not math.isfinite(a_lo) or a_lo <= 0.0:
        fails.append(f"human-label anchor rho={a_pt:+.4f} clustered CI[{a_lo:+.4f},{a_hi:+.4f}] does not exclude zero")
    if math.isfinite(lo_b) and math.isfinite(hi_b) and lo_b * hi_b > 0 and min(abs(lo_b), abs(hi_b)) > max_within_length_rho:
        direction = "brevity" if pt < 0 else "verbosity"
        fails.append(f"the outcome tracks length within context at rho={pt:+.3f} CI[{lo_b:+.3f},{hi_b:+.3f}] "
                     f"while length predicts the human label at only rho={rho_len_label:+.3f}; a corpus built "
                     f"on it would rank candidates by {direction}")
    if not intervention["passes"]:
        bits = []
        if not intervention["tost_passes"]:
            bits.append(f"mean shift {intervention['delta_mean']:+.5f} CI90"
                        f"[{intervention['delta_ci90'][0]:+.5f},{intervention['delta_ci90'][1]:+.5f}] is not "
                        f"equivalent to zero within the margin +-{intervention['equivalence_margin']:.5f} "
                        f"(={intervention['margin_frac']:.2f} x within-context outcome sd)")
        if not intervention["tau_ok"]:
            bits.append(f"noise-corrected within-context variance ratio "
                        f"{intervention['tau_corrected']:.3f} CI90"
                        f"[{intervention['tau_corrected_ci90'][0]:.3f},"
                        f"{intervention['tau_corrected_ci90'][1]:.3f}] exceeds {max_tau:.2f} "
                        f"(raw {intervention['tau_raw_within']:.3f}, replication floor "
                        f"{intervention['tau_noise_floor']:.3f}, length-matched surface null "
                        f"{intervention['tau_surface_null']:.3f})")
        if not intervention["flip_ok"]:
            bits.append(f"pure length flips {100 * intervention['flip_perturb']:.1f}% of within-context "
                        f"pairs against {100 * intervention['flip_surface_null']:.1f}% for a length-matched "
                        f"surface null [{intervention['flip_rule']}]")
        fails.append(f"paired length intervention failed after adding {intervention['words_added']:.0f} "
                     f"content-free words: " + "; ".join(bits))
    if abs(bank_sent.get(FIT_BANK, 0.0)) > 0.25:
        warns.append(f"the {FIT_BANK} filler bank used to FIT h(L) is not affect-neutral (mean "
                     f"{bank_sent[FIT_BANK]:+.3f}): h(L) then absorbs an affect effect as if it were length")
    if math.isfinite(bank_sent.get(gate_bank, 0.0)) and abs(bank_sent.get(gate_bank, 0.0)) > 0.25:
        warns.append(f"the {gate_bank} filler bank is not affect-neutral under the sentiment model "
                     f"(mean {bank_sent[gate_bank]:+.3f}); the length intervention is then partly an affect "
                     f"intervention")
    if math.isfinite(echo) and abs(echo) > 0.85:
        warns.append(f"echo rho={echo:.3f}: the simulated reply largely paraphrases the agent turn")
    if math.isfinite(echo_emoji) and abs(echo_emoji) > 0.25:
        fails.append(f"emoji rho={echo_emoji:.3f}: the outcome responds to emoji count")

    gate_arrays = intervention.pop("_arrays", None)
    out = {
        "n_turns": n, "n_variants": n_variants, "_gate_arrays": gate_arrays,
        "anchor_spearman": float(rho_shift), "anchor_ci": list(fisher_ci(rho_shift, n)),
        "human_label_rho": float(a_pt), "human_label_ci_clustered": [a_lo, a_hi], "human_label_n": human_n,
        "length_predicts_human_label_rho": float(rho_len_label),
        "responsiveness_icc": icc, "replicate_reliability": reliability,
        "icc_note": ("not applicable: the panel estimator is deterministic, so the replicate equals the "
                     "estimate and ICC = reliability = 1 by construction" if icc is None else "split-half"),
        "response_variance_share": response_share, "var_within_context": var_within_ctx,
        "var_between_context": var_between_ctx,
        "length_rho_within_before": w_before["mean"], "length_rho_within_after": w_after["mean"],
        "length_rho_within_ci_clustered": [lo_b, hi_b],
        "length_rho_pooled_before": float(spearman(y, Lv)), "length_rho_pooled_after": float(spearman(y_res, Lv)),
        "length_control": (sim.length_control.diag if sim.length_control is not None else {}),
        "paired_intervention": intervention, "filler_bank_sentiment": bank_sent,
        "gold_outcome_length_rho": float(rho_len_outcome_gold),
        "projector": (sim.projector.diag if getattr(sim, "projector", None) is not None
                      and sim.projector.fitted else {"k": 0}),
        "n_orbit": int(getattr(sim, "n_orbit", 1)),
        "echo_spearman": float(echo), "emoji_spearman": float(echo_emoji),
        "outcome_sd": float(np.std(y_res)),
        "passes": False, "failures": fails, "warnings": warns,
    }
    out["passes"] = len(fails) == 0
    logger.info("simulator | shift anchor rho=%.4f | human anchor rho=%+.4f CI[%+.4f,%+.4f] | ICC %s | "
                "within-context share of outcome variance %.3f | observational within-context length rho %+.3f "
                "CI[%+.3f,%+.3f] (no post-hoc residualisation; human label vs length rho %+.3f) | pass=%s",
                rho_shift, a_pt, a_lo, a_hi, ("n/a (deterministic)" if icc is None else f"{icc:.3f}"),
                response_share, w_after["mean"], lo_b, hi_b, rho_len_label, out["passes"])
    for f in fails:
        logger.error("SIMULATOR FAIL: %s", f)
    for w in warns:
        logger.warning("SIMULATOR WARN: %s", w)
    return out


@dataclass
class ResponseGroup:
    dialogue_id: str
    uid: str
    texts: List[str]
    feats: np.ndarray
    outcomes: np.ndarray
    ok: np.ndarray
    gold_valence: float = float("nan")
    # v9 counterfactual padding pairs (features only; no simulator call is needed because the
    # simulator is CERTIFIED padding-invariant by the stage-3 gate, so the counterfactual label
    # of pad(r) is the label of r).  pad_* use the TRAIN filler bank and enter reward training
    # through counterfactual logit pairing; probe_* use the HELD-OUT bank and are used only by
    # the reward validity gate.
    pad_feats: Optional[np.ndarray] = None
    pad_src: Optional[np.ndarray] = None
    probe_feats: Optional[np.ndarray] = None
    probe_src: Optional[np.ndarray] = None

    @property
    def informative(self) -> bool:
        y = self.outcomes[self.ok.astype(bool)]
        return bool(y.size >= 2 and float(np.ptp(y)) > 1e-6)


def build_corpus(turns: Sequence[Turn], proposal, sim: UserSimulator, n_variants: int, gen: GenConfig,
                 logger: logging.Logger, feat_dim: int, batch: int = 16, crn_base: int = 90001,
                 include_gold: bool = True, max_malformed: float = 0.5,
                 n_pad: int = 1, n_probe: int = 1) -> Tuple[List[ResponseGroup], Dict[str, Any]]:
    groups: List[ResponseGroup] = []
    n_bad = n_resp = 0
    reasons: Dict[str, int] = {}
    t0 = time.time()
    for s in range(0, len(turns), batch):
        chunk = list(turns[s:s + batch])
        prompts = [agent_prompt(t) for t in chunk for _ in range(n_variants)]
        variants = proposal.generate(prompts, gen, seed=crn_base + s)
        per = [list(variants[i * n_variants:(i + 1) * n_variants]) for i in range(len(chunk))]
        if include_gold:
            for i, t in enumerate(chunk):
                per[i].append(t.gold_response)
        flat_t, flat_r = [], []
        for i, t in enumerate(chunk):
            for r in per[i]:
                flat_t.append(t)
                flat_r.append(r)
        oks = []
        for r in flat_r:
            good, rs = hygiene_ok(r)
            oks.append(good)
            for x in rs:
                reasons[x] = reasons.get(x, 0) + 1
        ok = np.asarray(oks, bool)
        n_resp += len(flat_r)
        n_bad += int((~ok).sum())
        # Counterfactual padding pairs, drawn among the SAMPLED variants (never the gold turn).
        prng = np.random.default_rng(crn_base + 13 * s)
        aug_tr = LengthInvarianceAugmenter(n_levels=1, seed=crn_base + 17 * s, bank=FIT_BANK)
        aug_ho = LengthInvarianceAugmenter(n_levels=1, seed=crn_base + 19 * s, bank="heldout")
        extra_t, extra_r, pad_src, probe_src = [], [], [], []
        for i, t in enumerate(chunk):
            m = min(n_variants, len(per[i]))
            ps = [int(x) for x in prng.choice(m, size=min(n_pad, m), replace=False)] if n_pad > 0 else []
            qs = [int(x) for x in prng.choice(m, size=min(n_probe, m), replace=False)] if n_probe > 0 else []
            pad_src.append(ps)
            probe_src.append(qs)
            for j in ps:
                extra_t.append(t)
                extra_r.append(aug_tr.lengthen(per[i][j], 1))
            for j in qs:
                extra_t.append(t)
                extra_r.append(aug_ho.lengthen(per[i][j], 1))
        all_t = flat_t + extra_t
        all_r = flat_r + extra_r
        feats_all = proposal.features([agent_prompt(t) for t in all_t], all_r, dim=feat_dim)
        feats = feats_all[:len(flat_r)]
        efeats = feats_all[len(flat_r):]
        y = sim.rollout(flat_t, flat_r, crn_seed=crn_base + 7 * s)
        c = 0
        e = 0
        for i, t in enumerate(chunk):
            k = len(per[i])
            hv = t.human_valence
            npd, npr = len(pad_src[i]), len(probe_src[i])
            groups.append(ResponseGroup(t.dialogue_id, t.uid, flat_r[c:c + k],
                                        np.asarray(feats[c:c + k], np.float32),
                                        np.asarray(y[c:c + k], np.float64),
                                        ok[c:c + k].copy(),
                                        float(hv) if hv is not None else float("nan"),
                                        np.asarray(efeats[e:e + npd], np.float32),
                                        np.asarray(pad_src[i], np.int64),
                                        np.asarray(efeats[e + npd:e + npd + npr], np.float32),
                                        np.asarray(probe_src[i], np.int64)))
            c += k
            e += npd + npr
        if (s // max(1, batch)) % 10 == 0:
            logger.info("  corpus | %d/%d contexts | %.1fs | malformed %.1f%%",
                        min(s + batch, len(turns)), len(turns), time.time() - t0,
                        100 * n_bad / max(1, n_resp))
    if n_bad > max_malformed * n_resp:
        raise RuntimeError(f"{100 * n_bad / max(1, n_resp):.1f}% of sampled responses failed hygiene; "
                           f"fix decoding before fitting a reward model. causes={reasons}")
    texts = [t for g in groups for t in g.texts]
    yall = np.concatenate([g.outcomes for g in groups])
    gid = np.concatenate([np.full(len(g.texts), i) for i, g in enumerate(groups)])
    cluster = np.concatenate([np.full(len(g.texts), g.dialogue_id, dtype=object) for g in groups])
    # v9: NO second, observational length residualisation.  v8 applied a cross-fitted
    # within-context spline here ON TOP of the simulator's own length control, i.e. it removed
    # the length-content association twice and paid reward for padding twice over.  The only
    # length correction is the interventional one inside sim.rollout; the remaining
    # observational association is reported as a diagnostic.
    Lall = loglen(texts)
    w_len = within_group_spearman(yall, Lall, gid)
    _, wl_lo, wl_hi = cluster_bootstrap_ci(
        lambda ix, cp: within_group_spearman(yall[ix], Lall[ix], boot_groups(gid[ix], cp))["mean"],
        cluster, 400, 7, with_copy=True)
    inf = [g for g in groups if g.informative]
    diag = {
        "n_groups": len(groups), "n_informative": len(inf), "K": len(groups[0].texts) if groups else 0,
        "malformed_fraction": float(n_bad / max(1, n_resp)), "hygiene_causes": reasons,
        "length_control": (sim.length_control.diag if sim.length_control is not None else {}),
        "within_length_rho": w_len["mean"], "within_length_rho_ci": [wl_lo, wl_hi],
        "within_var": float(np.mean([g.outcomes.var(ddof=1) for g in groups])) if groups else 0.0,
        "between_var": float(np.var([g.outcomes.mean() for g in groups], ddof=1)) if len(groups) > 1 else 0.0,
    }
    logger.info("corpus | %d groups (%d informative, %.1f%%) | K=%d | within var=%.4f between=%.4f | malformed %.2f%% "
                "| observational within-context length rho %+.3f CI[%+.3f,%+.3f] (diagnostic, not removed)",
                diag["n_groups"], diag["n_informative"], 100 * diag["n_informative"] / max(1, diag["n_groups"]),
                diag["K"], diag["within_var"], diag["between_var"], 100 * diag["malformed_fraction"],
                w_len["mean"], wl_lo, wl_hi)
    if len(inf) < 50:
        raise RuntimeError(f"only {len(inf)} informative groups; the outcome model does not discriminate variants")
    return groups, diag


def save_corpus(groups: Sequence[ResponseGroup], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    D = groups[0].feats.shape[1] if groups else 0

    def cat(xs, dim, dtype):
        xs = [x for x in xs if x is not None and np.size(x)]
        return np.concatenate(xs, 0).astype(dtype) if xs else np.zeros((0,) + dim, dtype)

    np.savez_compressed(
        path,
        meta=np.asarray(json.dumps([{"dialogue_id": g.dialogue_id, "uid": g.uid, "texts": g.texts,
                                     "gold_valence": g.gold_valence} for g in groups])),
        sizes=np.asarray([len(g.texts) for g in groups], np.int32),
        feats=np.concatenate([g.feats for g in groups], 0),
        outcomes=np.concatenate([g.outcomes for g in groups], 0),
        ok=np.concatenate([g.ok for g in groups], 0),
        pad_sizes=np.asarray([0 if g.pad_src is None else len(g.pad_src) for g in groups], np.int32),
        pad_feats=cat([g.pad_feats for g in groups], (D,), np.float32),
        pad_src=cat([g.pad_src for g in groups], (), np.int64),
        probe_sizes=np.asarray([0 if g.probe_src is None else len(g.probe_src) for g in groups], np.int32),
        probe_feats=cat([g.probe_feats for g in groups], (D,), np.float32),
        probe_src=cat([g.probe_src for g in groups], (), np.int64),
    )


def load_corpus(path: Path) -> List[ResponseGroup]:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    sizes = z["sizes"]
    F, Y, O = z["feats"], z["outcomes"], z["ok"]
    has_pad = "pad_sizes" in z.files
    out = []
    c = cp = cq = 0
    for gi, (m, k) in enumerate(zip(meta, sizes)):
        k = int(k)
        g = ResponseGroup(m["dialogue_id"], m["uid"], list(m["texts"]),
                          F[c:c + k], Y[c:c + k], O[c:c + k].astype(bool),
                          float(m.get("gold_valence", float("nan"))))
        if has_pad:
            npd, npr = int(z["pad_sizes"][gi]), int(z["probe_sizes"][gi])
            g.pad_feats, g.pad_src = z["pad_feats"][cp:cp + npd], z["pad_src"][cp:cp + npd]
            g.probe_feats, g.probe_src = z["probe_feats"][cq:cq + npr], z["probe_src"][cq:cq + npr]
            cp += npd
            cq += npr
        out.append(g)
        c += k
    return out


def group_split(groups: Sequence[ResponseGroup], frac: float, seed: int) -> Tuple[List[ResponseGroup], List[ResponseGroup]]:
    ids = sorted({g.dialogue_id for g in groups})
    rng = random.Random(seed)
    rng.shuffle(ids)
    cut = set(ids[: int(round(frac * len(ids)))])
    a = [g for g in groups if g.dialogue_id in cut]
    b = [g for g in groups if g.dialogue_id not in cut]
    return a, b


class Standardizer:
    def __init__(self):
        self.mu: Optional[np.ndarray] = None
        self.sd: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mu = X.mean(0)
        self.sd = X.std(0)
        self.sd = np.where(self.sd < 1e-6, 1.0, self.sd)
        return self

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, np.float64) - self.mu) / self.sd

    def state(self) -> Dict[str, Any]:
        return {"mu": self.mu.tolist(), "sd": self.sd.tolist()}

    @staticmethod
    def load(d: Dict[str, Any]) -> "Standardizer":
        s = Standardizer()
        s.mu = np.asarray(d["mu"], float)
        s.sd = np.asarray(d["sd"], float)
        return s


@dataclass
class ScorerParams:
    w: np.ndarray
    b: float
    W1: np.ndarray
    b1: np.ndarray
    v: np.ndarray
    c: float

    def copy(self) -> "ScorerParams":
        return ScorerParams(self.w.copy(), float(self.b), self.W1.copy(),
                                 self.b1.copy(), self.v.copy(), float(self.c))


class ResidualMLPScorer:
    """s(x) = w.x + b + v.tanh(W1 x + b1) + c: a one-hidden-layer MLP with a linear skip path.
    (v13 called this RatioScorer; nothing in it is a ratio.)"""

    def __init__(self, in_dim: int, hidden: int = 96, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.p = ScorerParams(
            w=rng.normal(0, 1e-3, in_dim), b=0.0,
            W1=rng.normal(0, 1.0 / math.sqrt(in_dim), (hidden, in_dim)),
            b1=np.zeros(hidden), v=np.zeros(hidden), c=0.0)

    def forward(self, X: np.ndarray):
        z = X @ self.p.W1.T + self.p.b1
        h = np.tanh(z)
        s = X @ self.p.w + self.p.b + h @ self.p.v + self.p.c
        return s, h

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)[0]

    def grads(self, X: np.ndarray, ds: np.ndarray, h: np.ndarray, l2: float) -> ScorerParams:
        gw = X.T @ ds + l2 * self.p.w
        gb = float(ds.sum())
        gv = h.T @ ds + l2 * self.p.v
        gc = float(ds.sum())
        dh = np.outer(ds, self.p.v) * (1.0 - h ** 2)
        gW1 = dh.T @ X + l2 * self.p.W1
        gb1 = dh.sum(0)
        return ScorerParams(gw, gb, gW1, gb1, gv, gc)

    def state(self) -> Dict[str, Any]:
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in asdict(self.p).items()}

    @staticmethod
    def load(d: Dict[str, Any]) -> "ResidualMLPScorer":
        m = ResidualMLPScorer(len(d["w"]), len(d["b1"]))
        m.p = ScorerParams(np.asarray(d["w"], float), float(d["b"]), np.asarray(d["W1"], float),
                                np.asarray(d["b1"], float), np.asarray(d["v"], float), float(d["c"]))
        return m


def plackett_luce_loss_grad(s: np.ndarray, y: np.ndarray, ok: np.ndarray,
                            tie_tol: float = 1e-6) -> Tuple[float, np.ndarray]:
    m = ok.astype(bool)
    idx = np.flatnonzero(m)
    if idx.size < 2:
        return 0.0, np.zeros_like(s)
    sub_s = s[idx]
    sub_y = y[idx]
    order = np.argsort(-sub_y, kind="mergesort")
    ys = sub_y[order]
    keep = 1
    for i in range(1, ys.size):
        if ys[i - 1] - ys[i] > tie_tol:
            keep = i + 1
        else:
            break
    if keep < 2:
        return 0.0, np.zeros_like(s)
    perm = order
    sp = sub_s[perm]
    K = sp.size
    mx = float(sp.max())
    e = np.exp(sp - mx)
    suffix = np.cumsum(e[::-1])[::-1]
    loss = 0.0
    g = np.zeros(K)
    for i in range(keep - 1):
        loss += -(sp[i] - (math.log(max(suffix[i], EPS)) + mx))
        g[i:] += e[i:] / max(suffix[i], EPS)
        g[i] -= 1.0
    gs = np.zeros_like(s)
    gs[idx[perm]] = g
    return float(loss), gs


def standardised_within_slope(v: np.ndarray, Lc: np.ndarray) -> float:
    """Slope of standardised v on within-group centred log-length: scale-free, so it is
    comparable between a Plackett-Luce score (whose scale is arbitrary) and an outcome."""
    v = np.asarray(v, float)
    den = float(np.sum(Lc ** 2))
    sd = float(np.std(v))
    if den <= EPS or sd <= EPS:
        return 0.0
    return float(np.sum((v - v.mean()) * Lc) / (sd * den))


def group_length_targets(groups: Sequence["ResponseGroup"]) -> List[Tuple[np.ndarray, float]]:
    """Per group: within-group centred log-length and the LABEL's standardised length slope.

    The label slope is the calibration target for reward training.  Pulling the reward's length
    slope to ZERO would be the same bad control the outcome estimator already refuses: the
    labels' length association is content-driven and is signal.  Pulling it to the LABEL slope
    removes only the part the reward invents on top of the labels, which is the part that a
    proxy-exploiting policy could move without changing content."""
    out = []
    for g in groups:
        L = loglen(list(g.texts))
        Lc = L - L.mean()
        out.append((Lc, standardised_within_slope(np.asarray(g.outcomes, float), Lc)))
    return out


def within_group_pairwise_accuracy(scores: Sequence[np.ndarray], outcomes: Sequence[np.ndarray],
                                   oks: Sequence[np.ndarray]) -> Tuple[float, int]:
    hit = tot = 0
    for s, y, o in zip(scores, outcomes, oks):
        m = o.astype(bool)
        s2, y2 = s[m], y[m]
        for i in range(len(s2)):
            for j in range(i + 1, len(s2)):
                if abs(y2[i] - y2[j]) <= 1e-9:
                    continue
                tot += 1
                if (s2[i] - s2[j]) * (y2[i] - y2[j]) > 0:
                    hit += 1
    return (hit / tot if tot else float("nan"), tot)


@dataclass
class RewardConfig:
    hidden: int = 96
    n_ensemble: int = 5
    lr: float = 3e-3
    l2: float = 1e-4
    epochs: int = 120
    patience: int = 12
    kappa: float = 1.0
    alpha: float = 0.2
    batch_groups: int = 64
    clp: float = -1.0         # counterfactual logit pairing weight; < 0 = select on DEV from clp_grid
    clp_grid: Tuple[float, ...] = (1.0, 3.0, 10.0, 30.0)
    length_cal: float = -1.0  # length-calibration weight; < 0 = select on DEV from length_cal_grid
    length_cal_grid: Tuple[float, ...] = (0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
    length_tol: float = 0.05  # feasible |reward slope - label slope| for constrained early stopping


class RewardModel:
    def __init__(self, in_dim: int, cfg: RewardConfig, std: Standardizer):
        self.cfg = cfg
        self.std = std
        self.in_dim = in_dim
        self.models: List[ResidualMLPScorer] = []
        self.recal: Tuple[float, float] = (1.0, 0.0)   # OLS linear recalibration a*s + b (NOT Platt scaling)
        self.q: Optional[float] = None

    def _train_one(self, train: Sequence[ResponseGroup], dev: Sequence[ResponseGroup], seed: int,
                   logger: logging.Logger) -> ResidualMLPScorer:
        cfg = self.cfg
        m = ResidualMLPScorer(self.in_dim, cfg.hidden, seed)
        state = {k: (np.zeros_like(v) if isinstance(v, np.ndarray) else 0.0) for k, v in asdict(m.p).items()}
        mom = {k: (np.zeros_like(v) if isinstance(v, np.ndarray) else 0.0) for k, v in asdict(m.p).items()}
        rng = random.Random(seed)
        Xs = [self.std(g.feats) for g in train]
        Xp = [(self.std(g.pad_feats) if (cfg.clp > 0 and g.pad_feats is not None and len(g.pad_feats)) else None)
              for g in train]
        LT = group_length_targets(train) if cfg.length_cal > 0 else None
        best = -1.0
        best_p = m.p.copy()
        best_key = None
        best_gap = float("nan")
        LTD = group_length_targets(dev) if cfg.length_cal > 0 else None
        stale = 0
        t = 0
        for ep in range(cfg.epochs):
            order = list(range(len(train)))
            rng.shuffle(order)
            for s in range(0, len(order), cfg.batch_groups):
                sel = order[s:s + cfg.batch_groups]
                acc = {k: (np.zeros_like(v) if isinstance(v, np.ndarray) else 0.0)
                       for k, v in asdict(m.p).items()}
                nb = 0
                for i in sel:
                    g = train[i]
                    if Xp[i] is not None:
                        # Counterfactual logit pairing (Garg et al., 2019): the simulator is
                        # certified padding-invariant, so pad(r) has r's label and any score
                        # difference between them is a pure-length artefact of the features.
                        X = np.concatenate([Xs[i], Xp[i]], 0)
                        sc_all, h = m.forward(X)
                        K = Xs[i].shape[0]
                        _, ds_k = plackett_luce_loss_grad(sc_all[:K], g.outcomes, g.ok)
                        ds = np.concatenate([ds_k, np.zeros(Xp[i].shape[0])])
                        diff = sc_all[K:] - sc_all[g.pad_src]
                        ds[K:] += 2.0 * cfg.clp * diff
                        np.add.at(ds, g.pad_src, -2.0 * cfg.clp * diff)
                    else:
                        X = Xs[i]
                        sc, h = m.forward(X)
                        _, ds = plackett_luce_loss_grad(sc, g.outcomes, g.ok)
                    if LT is not None:
                        # Length calibration: match the reward's standardised within-group length
                        # slope to the label's, so the reward keeps the content-driven length
                        # association that the labels license and drops the excess it would
                        # otherwise invent (a proxy a policy can move without changing content).
                        Lc, b_y = LT[i]
                        K0 = Xs[i].shape[0]
                        sK = sc_all[:K0] if Xp[i] is not None else sc[:K0]
                        den = float(np.sum(Lc ** 2))
                        sd_s = float(np.std(sK))
                        if den > EPS and sd_s > EPS:
                            b_s = float(np.sum((sK - sK.mean()) * Lc) / (sd_s * den))
                            ds[:K0] += 2.0 * cfg.length_cal * (b_s - b_y) * Lc / (sd_s * den)
                    if not np.any(ds):
                        continue
                    gr = m.grads(X, ds, h, cfg.l2)
                    for k, v in asdict(gr).items():
                        acc[k] = acc[k] + v
                    nb += 1
                if nb == 0:
                    continue
                t += 1
                for k in acc:
                    gk = acc[k] / nb
                    mom[k] = 0.9 * mom[k] + 0.1 * gk
                    state[k] = 0.999 * state[k] + 0.001 * (gk ** 2 if isinstance(gk, np.ndarray) else gk * gk)
                    mh = mom[k] / (1 - 0.9 ** t)
                    vh = state[k] / (1 - 0.999 ** t)
                    upd = cfg.lr * mh / (np.sqrt(vh) + 1e-8 if isinstance(vh, np.ndarray) else math.sqrt(vh) + 1e-8)
                    setattr(m.p, k, getattr(m.p, k) - upd)
            sc_dev = [m(self.std(g.feats)) for g in dev]
            a, _ = within_group_pairwise_accuracy(sc_dev, [g.outcomes for g in dev], [g.ok for g in dev])
            # Early stopping is CONSTRAINED when length calibration is on: a checkpoint counts only
            # if its length slope is already within tolerance of the labels'.  Selecting purely on
            # accuracy would silently undo the penalty, because the most length-reliant checkpoint
            # is usually the most accurate one on a noisy dev split.
            gap = float("nan")
            if LTD is not None:
                gap = float(abs(np.mean([standardised_within_slope(sd_g, Lc) - b_y
                                         for sd_g, (Lc, b_y) in zip(sc_dev, LTD)])))
            feasible = (not math.isfinite(gap)) or gap <= cfg.length_tol
            key = (1 if feasible else 0, a if feasible else -gap)
            if math.isfinite(a) and (best_key is None or key > best_key):
                best, best_key, best_p, stale, best_gap = a, key, m.p.copy(), 0, gap
            else:
                stale += 1
                if stale >= cfg.patience:
                    break
        m.p = best_p
        logger.info("  reward member seed=%d | dev within-group pairwise accuracy=%.4f%s", seed, best,
                    "" if not math.isfinite(best_gap) else f" | dev length-slope gap={best_gap:.4f}")
        return m

    def fit(self, train: Sequence[ResponseGroup], dev: Sequence[ResponseGroup], calib: Sequence[ResponseGroup],
            logger: logging.Logger, seed: int = 0) -> Dict[str, Any]:
        self.models = [self._train_one(train, dev, seed + 17 * i, logger) for i in range(self.cfg.n_ensemble)]
        S = [self.raw(g.feats) for g in calib]
        spread = np.concatenate([s["sd"] for s in S]) if S else np.zeros(1)
        self.q = float(np.quantile(spread, 1.0 - self.cfg.alpha)) if spread.size else None
        mu = np.concatenate([s["mean"] for s in S]) if S else np.zeros(1)
        yy = np.concatenate([g.outcomes for g in calib]) if calib else np.zeros(1)
        if mu.size > 10 and np.std(mu) > EPS:
            a = float(np.cov(mu, yy, ddof=1)[0, 1] / max(np.var(mu, ddof=1), EPS))
            if a <= 0:
                # A non-positive slope would INVERT the ranking the members learnt; this can only
                # come from a calibration split on which the ensemble does not discriminate, and
                # recalibrating through it would hand GRPO the opposite of the validated reward.
                logger.warning("reward recalibration slope a=%+.4f <= 0 on the calibration split; keeping the "
                               "identity map instead of inverting the reward", a)
                a = 1.0
                self.recal = (1.0, 0.0)
            else:
                self.recal = (a, float(np.mean(yy) - a * np.mean(mu)))
        acc, npair = within_group_pairwise_accuracy(
            [self.score(g.feats)["reward"] for g in calib], [g.outcomes for g in calib], [g.ok for g in calib])
        logger.info("reward model | %d members | calibration pairwise accuracy=%.4f over %d pairs | "
                    "abstention threshold q=%.4f | linear recalibration a=%.4f b=%.4f",
                    len(self.models), acc, npair, self.q if self.q else float("nan"), *self.recal)
        return {"calib_pairwise_accuracy": acc, "n_pairs": npair, "q": self.q, "recal": list(self.recal)}

    def raw(self, feats: np.ndarray) -> Dict[str, np.ndarray]:
        X = self.std(feats)
        S = np.stack([m(X) for m in self.models], 0)
        return {"mean": S.mean(0), "sd": S.std(0) if len(self.models) > 1 else np.zeros(S.shape[1])}

    def score(self, feats: np.ndarray) -> Dict[str, np.ndarray]:
        r = self.raw(feats)
        a, b = self.recal
        pess = r["mean"] - self.cfg.kappa * r["sd"]
        w = np.ones_like(pess)
        if self.q is not None:
            w = (r["sd"] <= self.q).astype(float)
        return {"reward": a * pess + b, "weight": w, "sd": r["sd"]}

    def state(self) -> Dict[str, Any]:
        return {"cfg": asdict(self.cfg), "std": self.std.state(), "in_dim": self.in_dim,
                "models": [m.state() for m in self.models], "recal": list(self.recal), "q": self.q}

    @staticmethod
    def load(d: Dict[str, Any]) -> "RewardModel":
        cfg = RewardConfig(**d["cfg"])
        rm = RewardModel(int(d["in_dim"]), cfg, Standardizer.load(d["std"]))
        rm.models = [ResidualMLPScorer.load(x) for x in d["models"]]
        rm.recal = tuple(d.get("recal", d.get("platt", (1.0, 0.0))))   # v13 files used the key "platt"
        rm.q = d["q"]
        return rm


def permutation_test(rm: RewardModel, groups: Sequence[ResponseGroup], n_perm: int, seed: int) -> Dict[str, Any]:
    """Within-group label permutation test of pairwise accuracy.  The scores do not depend on the
    permutation, so they are computed once (v13 re-scored every group on every permutation).  With
    B permutations the smallest attainable p is 1/(B+1); a p at that floor is reported as a bound."""
    sc = [rm.score(g.feats)["reward"] for g in groups]
    obs, _ = within_group_pairwise_accuracy(sc, [g.outcomes for g in groups], [g.ok for g in groups])
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        perm = [rng.permutation(g.outcomes) for g in groups]
        a, _ = within_group_pairwise_accuracy(sc, perm, [g.ok for g in groups])
        if math.isfinite(a) and a >= obs:
            cnt += 1
    p = float((cnt + 1) / (n_perm + 1))
    return {"observed": float(obs), "p_value": p, "n_perm": int(n_perm), "at_floor": cnt == 0,
            "p_text": format_p(p, n_perm)}


def reward_padding_test(rm: "RewardModel", groups: Sequence[ResponseGroup], margin_frac: float, seed: int,
                        logger: Optional[logging.Logger] = None, tag: str = "heldout bank") -> Dict[str, Any]:
    """TOST on the reward change caused by held-out content-free padding, within-group sd units."""
    d, cl = [], []
    r_all, gid = [], []
    for gi, g in enumerate(groups):
        rb = rm.score(g.feats)["reward"]
        r_all.append(rb)
        gid.append(np.full(len(rb), gi))
        if g.probe_feats is None or not len(g.probe_feats):
            continue
        rp = rm.score(g.probe_feats)["reward"]
        for q, src in enumerate(np.asarray(g.probe_src, int)):
            d.append(float(rp[q] - rb[src]))
            cl.append(g.dialogue_id)
    out: Dict[str, Any] = {"n": len(d)}
    if len(d) < 30:
        if logger is not None:
            logger.warning("reward padding intervention skipped: only %d probe pairs (rebuild the corpus with %s)",
                           len(d), VERSION)
        return out
    dd = np.asarray(d, float)
    sd_w = within_group_sd(np.concatenate(r_all), np.concatenate(gid))
    m = margin_frac * sd_w
    pt, lo, hi = cluster_bootstrap_ci(lambda ix: float(np.mean(dd[ix])), np.asarray(cl, object), 800, seed, conf=0.90)
    out.update({"delta_mean": pt, "delta_ci90": [lo, hi], "margin": float(m), "sd_within_reward": float(sd_w),
                "tost_passes": bool(math.isfinite(lo) and lo > -m and hi < m)})
    if logger is not None:
        logger.info("reward padding intervention [%s] | n=%d | delta=%+.5f CI90[%+.5f,%+.5f] vs margin +-%.5f "
                    "(=%.2f x within-group reward sd) -> TOST %s", tag, len(d), pt, lo, hi, m, margin_frac,
                    "PASS" if out["tost_passes"] else "FAIL")
    return out


def reward_length_excess(rm: "RewardModel", groups: Sequence[ResponseGroup], seed: int,
                         logger: Optional[logging.Logger] = None, tag: str = "TEST",
                         n_boot: int = 800) -> Dict[str, Any]:
    """Within-context length dependence of the REWARD, measured against the LABELS'.

    An absolute threshold here would re-introduce exactly the bad control this pipeline removed
    from the outcome estimator: the labels themselves track length (content drives both), so a
    reward that reproduces the label association is faithful, and forcing it to zero would
    delete content signal.  What is not licensed is the EXCESS the reward adds on top of the
    labels -- a length component with no counterpart in the labels is, by construction, a proxy
    a policy can move without changing content.  The paired difference is bootstrapped over
    dialogues, with duplicated clusters kept distinct."""
    r = np.concatenate([rm.score(g.feats)["reward"] for g in groups])
    y = np.concatenate([np.asarray(g.outcomes, float) for g in groups])
    L = loglen([t for g in groups for t in g.texts])
    gid = np.concatenate([np.full(len(g.texts), i) for i, g in enumerate(groups)])
    cluster = np.concatenate([np.full(len(g.texts), g.dialogue_id, dtype=object) for g in groups])

    def _stat(ix, cp):
        gg = boot_groups(gid[ix], cp)
        return (within_group_spearman(r[ix], L[ix], gg)["mean"]
                - within_group_spearman(y[ix], L[ix], gg)["mean"])

    rho_r = within_group_spearman(r, L, gid)["mean"]
    rho_y = within_group_spearman(y, L, gid)["mean"]
    pt, lo, hi = cluster_bootstrap_ci(_stat, cluster, n_boot, seed, conf=0.90, with_copy=True)
    out = {"rho_reward": float(rho_r), "rho_label": float(rho_y), "excess": float(pt),
           "excess_ci90": [lo, hi], "n_groups": len(groups)}
    if logger is not None:
        logger.info("reward length calibration [%s] | within-context length rho: reward %+.3f vs labels %+.3f | "
                    "excess %+.3f CI90[%+.3f,%+.3f]", tag, rho_r, rho_y, pt, lo, hi)
    return out


def reward_gold_anchor(rm: "RewardModel", groups: Sequence[ResponseGroup], seed: int = 0,
                       n_boot: int = 800) -> Dict[str, Any]:
    """Human anchor of the REWARD: Spearman correlation, across contexts, between the reward margin
    of the gold agent turn over the mean of the sampled variants and the human emotion label of
    the customer's next turn, with a dialogue-clustered bootstrap CI.

    What it is and is not.  It is BETWEEN-context (only the gold turn carries a label), so it tests
    whether the reward's view of the gold turn tracks how the real customer reacted; it does NOT
    test within-context ranking, which EmoWOZ cannot label (the human study does, see R3).  The
    margin removes the context's baseline, which is why this number is expected to be smaller than
    the simulator's own between-context anchor."""
    marg, lab, cl = [], [], []
    for g in groups:
        if not math.isfinite(g.gold_valence) or len(g.texts) < 2:
            continue
        r = rm.score(g.feats)["reward"]
        marg.append(float(r[-1] - np.mean(r[:-1])))
        lab.append(float(g.gold_valence))
        cl.append(g.dialogue_id)
    if len(marg) < 30:
        return {"rho": float("nan"), "ci95": [float("nan"), float("nan")], "n": len(marg)}
    m, y = np.asarray(marg), np.asarray(lab)
    pt, lo, hi = cluster_bootstrap_ci(lambda ix: spearman(m[ix], y[ix]), np.asarray(cl, object), n_boot, seed)
    return {"rho": float(pt), "ci95": [lo, hi], "n": len(marg)}


def validity_gate(rm: RewardModel, test: Sequence[ResponseGroup], logger: logging.Logger,
                  n_perm: int = 2000, seed: int = 0, max_excess_length_rho: float = 0.10,
                  abs_length_backstop: float = 0.60, min_accuracy: float = 0.55,
                  margin_frac: float = 0.15, min_gold_anchor: float = 0.10) -> Dict[str, Any]:
    r = np.concatenate([rm.score(g.feats)["reward"] for g in test])
    L = loglen([t for g in test for t in g.texts])
    gid = np.concatenate([np.full(len(g.texts), i) for i, g in enumerate(test)])
    cluster = np.concatenate([np.full(len(g.texts), g.dialogue_id, dtype=object) for g in test])
    acc, npair = within_group_pairwise_accuracy([rm.score(g.feats)["reward"] for g in test],
                                                [g.outcomes for g in test], [g.ok for g in test])
    wl = within_group_spearman(r, L, gid)
    pt, lo, hi = cluster_bootstrap_ci(
        lambda ix, cp: within_group_spearman(r[ix], L[ix], boot_groups(gid[ix], cp))["mean"],
        cluster, 800, seed, with_copy=True)
    lex = reward_length_excess(rm, test, seed + 7, logger, tag="TEST")
    # Interventional padding test on the REWARD MODEL with the held-out filler bank.  The
    # simulator gate certifies the LABELS are padding-invariant; this certifies that the learnt
    # reward did not re-acquire a length proxy through its features, which is the channel GRPO
    # would actually exploit (Goodhart through a proxy feature).
    pad = reward_padding_test(rm, test, margin_frac, seed + 5, logger)
    perm = permutation_test(rm, test, n_perm, seed)
    anc = reward_gold_anchor(rm, test, seed + 11)
    anchor = anc["rho"]
    fails = []
    if not math.isfinite(acc) or acc < min_accuracy:
        fails.append(f"within-group pairwise accuracy={acc:.4f} < {min_accuracy:.2f}")
    if perm["p_value"] > 0.01:
        fails.append(f"permutation p {perm['p_text']}: the ordering is not distinguishable from chance")
    e_lo, e_hi = lex["excess_ci90"]
    if (math.isfinite(e_lo) and math.isfinite(e_hi) and e_lo * e_hi > 0
            and min(abs(e_lo), abs(e_hi)) > max_excess_length_rho):
        fails.append(f"the reward's within-context length dependence exceeds the labels' by "
                     f"{lex['excess']:+.3f} CI90[{e_lo:+.3f},{e_hi:+.3f}] (reward {lex['rho_reward']:+.3f} vs "
                     f"labels {lex['rho_label']:+.3f}); that excess is a length proxy with no counterpart in "
                     f"the outcome. Raise --reward-length-cal")
    if math.isfinite(pt) and abs(pt) > abs_length_backstop:
        fails.append(f"absolute backstop: the reward is almost a pure length ranker "
                     f"(within-context rho={pt:+.3f} CI[{lo:+.3f},{hi:+.3f}])")
    if float(np.std(r)) < 0.01:
        fails.append(f"reward standard deviation {float(np.std(r)):.5f} is degenerate")
    if pad.get("tost_passes") is False:
        fails.append(f"reward responds to held-out content-free padding: delta={pad['delta_mean']:+.5f} "
                     f"CI90[{pad['delta_ci90'][0]:+.5f},{pad['delta_ci90'][1]:+.5f}] outside +-{pad['margin']:.5f}; "
                     f"GRPO would learn to pad (or truncate). Raise --reward-clp")
    warns = []
    if math.isfinite(anchor) and anchor < min_gold_anchor:
        warns.append(f"gold anchor rho={anchor:+.4f} CI95[{anc['ci95'][0]:+.4f},{anc['ci95'][1]:+.4f}] (n={anc['n']}) "
                     f"is below the pre-registered {min_gold_anchor:.2f}: the reward ranks SIMULATED outcomes "
                     f"(accuracy {acc:.4f}) but its link to human labels is weak -- report it, do not tune it away")
    out = {"pairwise_accuracy": acc, "n_pairs": npair, "permutation": perm,
           "length_rho_within": wl["mean"], "length_rho_ci": [lo, hi], "length_excess": lex,
           "gold_anchor_rho": float(anchor), "gold_anchor": anc, "gold_anchor_below_threshold":
           bool(math.isfinite(anchor) and anchor < min_gold_anchor),
           "reward_sd": float(np.std(r)), "padding_intervention": pad,
           "passes": len(fails) == 0, "failures": fails, "warnings": warns}
    logger.info("validity gate | accuracy=%.4f (%d pairs) | permutation p %s | length rho within=%+.3f "
                "CI[%+.3f,%+.3f] (labels %+.3f, excess %+.3f) | gold-anchor rho=%+.4f CI95[%+.4f,%+.4f] | pass=%s",
                acc, npair, perm["p_text"], wl["mean"], lo, hi, lex["rho_label"], lex["excess"], anchor,
                anc["ci95"][0], anc["ci95"][1], out["passes"])
    for w in warns:
        logger.warning("REWARD WARN: %s", w)
    for f in fails:
        logger.error("REWARD FAIL: %s", f)
    return out


@dataclass
class GRPOConfig:
    steps: int = 200
    n_contexts: int = 8
    group_size: int = 6
    lr: float = 1e-5
    kl_coef: float = 0.05
    kl_target: float = 0.02
    clip: float = 0.2
    temperature: float = 1.0
    max_grad_norm: float = 1.0
    log_every: int = 10
    kl_coef_max: float = 0.5
    kl_coef_min: float = 0.01       # the anchor never switches off entirely
    bad_advantage: float = 1.0      # fixed advantage for a hygiene failure
    kl_abort_factor: float = 25.0   # abort if the token-level KL exceeds this multiple of the target
    snr_kappa: float = 1.0          # a group must out-spread this multiple of the ensemble sd
    adv_clip: float = 3.0
    adv_scale_floor: float = 0.005  # floor on the running within-group reward sd used as the scale
    length_drift_factor: float = 1.25   # abort if mean length exceeds this multiple of the SFT baseline
    hygiene_floor: float = 0.85         # abort if hygiene falls below this fraction of the baseline


def group_relative_advantage(reward: np.ndarray, weight: np.ndarray, gid: np.ndarray,
                             ok: Optional[np.ndarray] = None, bad_advantage: float = 1.0,
                             scale: float = 1.0, rm_sd: Optional[np.ndarray] = None,
                             snr_kappa: float = 1.0, adv_clip: float = 3.0) -> Dict[str, Any]:
    """Group-centred advantages on a single global scale, with a signal-to-noise gate.

    Two things a per-group standardisation gets wrong, both visible in the v11 run:
      * it divides by the group's own sd, so a group whose spread is reward-model noise is
        rescaled to the same unit advantages as a group with a genuine quality difference.  The
        policy then random-walks on noise -- the observed signature is |adv| growing while the
        reward spread shrinks, with response length and hygiene drifting for no reward reason.
      * it ignores the reward model's stated uncertainty.  The ensemble already reports a
        per-sample sd; a group whose whole spread sits inside that sd carries no usable ranking
        information and should contribute nothing, not a normalised gradient.
    So: centre within the group, divide by ONE running scale (the typical within-group reward sd),
    clip, and zero out groups that fail the signal-to-noise test.  Malformed samples keep a fixed
    negative advantage and stay out of the group statistics."""
    reward = np.asarray(reward, float)
    weight = np.asarray(weight, float)
    good = (np.ones_like(reward, bool) if ok is None else np.asarray(ok, bool))
    gid = np.asarray(gid)
    adv = np.zeros_like(reward)
    n_groups = n_gated = 0
    spreads = []
    for g in np.unique(gid):
        m = gid == g
        use = m & good & (weight > 0)
        if int(use.sum()) < 2:
            continue
        n_groups += 1
        w = weight[use]
        r = reward[use]
        mu = float(np.average(r, weights=w))
        spread = float(np.max(r) - np.min(r))
        spreads.append(float(math.sqrt(max(np.average((r - mu) ** 2, weights=w), 0.0))))
        if rm_sd is not None and snr_kappa > 0:
            noise = float(np.mean(np.asarray(rm_sd, float)[use]))
            if spread <= snr_kappa * noise:
                n_gated += 1                      # spread is inside the model's own uncertainty
                continue
        adv[use] = np.clip((r - mu) / max(float(scale), 1e-6), -adv_clip, adv_clip)
    adv = adv * np.where(good, weight, 1.0)
    adv[~good] = -abs(bad_advantage)
    return {"adv": adv, "n_groups": n_groups, "n_gated": n_gated,
            "within_sd": float(np.mean(spreads)) if spreads else float("nan")}


class LearnedReward:
    """The CARO reward: the validated reward-model ensemble applied to FROZEN SFT features.

    v14 (D1).  The reward model was fitted on features of the frozen SFT proposal policy.  v13
    computed the features during GRPO with the policy being trained, so every update also moved the
    reward's input representation: the function being optimised drifted away from the one that
    passed the validity gate, and the policy could raise its reward by shifting its own hidden
    states rather than its text.  Features are now computed with the trainable weights swapped
    for the frozen snapshot taken at the start of RL, which is exactly the SFT adapter that built
    the corpus, so the RL reward is the validated reward."""
    name = "learned"

    def __init__(self, rm: RewardModel, feat_dim: int, abstain: bool = True):
        self.rm = rm
        self.feat_dim = int(feat_dim)
        self.abstain = bool(abstain)

    def __call__(self, policy, ref_snapshot, prompts: Sequence[str], responses: Sequence[str]) -> Dict[str, np.ndarray]:
        with policy.frozen_weights(ref_snapshot):
            feats = policy.features(prompts, responses, dim=self.feat_dim)
        sc = self.rm.score(feats)
        if not self.abstain:            # ablation: every sample is trusted, whatever the ensemble spread
            sc = {**sc, "weight": np.ones_like(sc["weight"])}
        return sc


class SentimentReward:
    """Baseline reward: sentiment of the AGENT's own wording, from the same sentiment engine that
    labels the simulated customer replies.  It is the control for "the simulator adds nothing
    beyond rewarding positive agent language".  It has no ensemble, so no abstention and no
    signal-to-noise gate (sd = 0)."""
    name = "sentiment"

    def __init__(self, sentiment):
        self.sentiment = sentiment

    def __call__(self, policy, ref_snapshot, prompts: Sequence[str], responses: Sequence[str]) -> Dict[str, np.ndarray]:
        v = np.asarray(self.sentiment(list(responses)), float)
        return {"reward": v, "weight": np.ones_like(v), "sd": np.zeros_like(v)}


def train_grpo(policy, reward_fn, turns: Sequence[Turn], cfg: GRPOConfig, gen: GenConfig,
               logger: logging.Logger, out_dir: Path, tag: str, seed: int,
               stub: bool = False) -> Dict[str, Any]:
    rng = random.Random(seed)
    hist: List[Dict[str, float]] = []
    if stub:
        logger.warning("%s | STUB mode: sampling, rewards and advantages are computed but no gradient step "
                       "is applied (no autograd backend)", tag)
    else:
        from torch.optim import AdamW
        opt = AdamW([p for p in policy.model.parameters() if p.requires_grad], lr=cfg.lr)
    kl_coef = cfg.kl_coef
    ref_snapshot = None
    if not stub:
        ref_snapshot = policy.snapshot_trainable()
        logger.info("%s | KL reference frozen at the SFT policy (%d tensors); the pre-SFT base model is NOT "
                    "the anchor | kl target=%.3g per token, coef in [%.3g, %.3g] | reward=%s%s", tag,
                    len(ref_snapshot), cfg.kl_target, cfg.kl_coef_min, cfg.kl_coef_max, reward_fn.name,
                    " (features from the FROZEN SFT weights)" if reward_fn.name == "learned" else "")
    aborted = False
    adv_scale = float("nan")
    base_len = base_hyg = float("nan")
    for step in range(1, cfg.steps + 1):
        chunk = [turns[rng.randrange(len(turns))] for _ in range(cfg.n_contexts)]
        prompts = [agent_prompt(t) for t in chunk for _ in range(cfg.group_size)]
        gsample = GenConfig(max_new_tokens=gen.max_new_tokens, min_new_tokens=gen.min_new_tokens,
                            temperature=cfg.temperature, top_p=gen.top_p)
        responses = policy.generate(prompts, gsample, seed=seed * 100003 + step)
        sc = reward_fn(policy, ref_snapshot, prompts, responses)
        good = np.asarray([hygiene_ok(r)[0] for r in responses], bool)
        reward = np.asarray(sc["reward"], float)
        weight = sc["weight"]
        gid = np.repeat(np.arange(len(chunk)), cfg.group_size)
        cur_len = float(np.mean([len(r.split()) for r in responses]))
        cur_hyg = float(good.mean())
        if step == 1:
            base_len, base_hyg = cur_len, cur_hyg
            logger.info("%s | SFT baseline from the first sampled batch | mean length %.1f words | hygiene %.2f "
                        "| drift limits: length <= %.1f words, hygiene >= %.2f", tag, base_len, base_hyg,
                        cfg.length_drift_factor * base_len, cfg.hygiene_floor * base_hyg)
        ares = group_relative_advantage(reward, weight, gid, ok=good, bad_advantage=cfg.bad_advantage,
                                        scale=(cfg.adv_scale_floor if not math.isfinite(adv_scale)
                                               else max(adv_scale, cfg.adv_scale_floor)),
                                        rm_sd=sc.get("sd"), snr_kappa=cfg.snr_kappa, adv_clip=cfg.adv_clip)
        adv = ares["adv"]
        if math.isfinite(ares["within_sd"]):
            adv_scale = (ares["within_sd"] if not math.isfinite(adv_scale)
                         else 0.9 * adv_scale + 0.1 * ares["within_sd"])
        kl_val = 0.0
        if not stub:
            _, kl_val = policy.grpo_step(prompts, responses, adv, ref_snapshot, cfg.clip, kl_coef,
                                         opt, cfg.max_grad_norm)
            kl_coef = float(np.clip(kl_coef * (1.5 if kl_val > 2 * cfg.kl_target else 0.75
                                               if kl_val < 0.5 * cfg.kl_target else 1.0),
                                    cfg.kl_coef_min, cfg.kl_coef_max))
        rec = {"step": step, "reward_mean": float(reward[good].mean()) if good.any() else float("nan"),
               "reward_sd": float(reward[good].std()) if int(good.sum()) > 1 else float("nan"),
               "adv_abs": float(np.abs(adv).mean()), "weight": float(weight.mean()),
               "hygiene": cur_hyg, "len": cur_len, "kl": kl_val, "kl_coef": kl_coef,
               "adv_scale": float(adv_scale), "groups": ares["n_groups"], "gated": ares["n_gated"]}
        hist.append(rec)
        if (not stub) and step > 5 and math.isfinite(base_len) and cur_len > cfg.length_drift_factor * base_len:
            logger.error("%s | step %d: mean response length %.1f words has drifted past %.2f x the SFT "
                         "baseline %.1f with no reward justification (the reward tracks length NEGATIVELY); "
                         "stopping RL", tag, step, cur_len, cfg.length_drift_factor, base_len)
            aborted = True
        if (not stub) and step > 5 and math.isfinite(base_hyg) and cur_hyg < cfg.hygiene_floor * base_hyg:
            logger.error("%s | step %d: hygiene %.2f has fallen below %.2f x the SFT baseline %.2f; stopping RL",
                         tag, step, cur_hyg, cfg.hygiene_floor, base_hyg)
            aborted = True
        if (not stub) and math.isfinite(kl_val) and kl_val > cfg.kl_abort_factor * cfg.kl_target:
            logger.error("%s | step %d: token-level KL from the SFT reference is %.4f, above %g x the target "
                         "%.3g -- stopping RL and keeping the current adapter rather than letting the policy "
                         "drift away from the supervised initialisation", tag, step, kl_val,
                         cfg.kl_abort_factor, cfg.kl_target)
            aborted = True
        if step % cfg.log_every == 0 or step == cfg.steps:
            logger.info("%s | step %4d/%d | reward %.4f+-%.4f | |adv| %.3f (scale %.4f, %d/%d groups gated "
                        "as noise) | kept %.2f | hygiene %.2f | len %.1f | kl %.4f (coef %.4f)", tag, step,
                        cfg.steps, rec["reward_mean"], rec["reward_sd"], rec["adv_abs"], rec["adv_scale"],
                        rec["gated"], rec["groups"], rec["weight"], rec["hygiene"], rec["len"],
                        rec["kl"], rec["kl_coef"])
        if aborted:
            break
    if stub and hasattr(policy, "_adapter"):
        policy._adapter = {**policy._adapter, "tag": tag}    # stub arms must generate distinguishable text
    policy.save_adapter(out_dir / f"policy_{tag}")
    return {"history": hist, "aborted": aborted, "steps_run": len(hist),
            "baseline_length": base_len, "baseline_hygiene": base_hyg}


def eval_gen_seed(seed: int, offset: int, draw: int = 0) -> int:
    """Generation seed of one evaluation batch.  Shared by every arm, so arms are compared under
    common random numbers; `draw` indexes extra samples for the best-of-N / length-matched arms."""
    return 555000 + seed * 7919 + offset + 1000003 * draw


def evaluate(policy, sim: UserSimulator, turns: Sequence[Turn], gen: GenConfig, logger: logging.Logger,
             tag: str, seed: int, batch: int = 32, responses: Optional[Sequence[str]] = None,
             extra: Optional[Sequence[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Score one arm on the evaluation turns.  `responses` lets an eval-only arm (best-of-N,
    length-matched) supply responses it selected itself; they are scored exactly like generated
    ones.  Every row also carries info_recall against the gold agent turn (D4)."""
    rows: List[Dict[str, Any]] = []
    for s in range(0, len(turns), batch):
        chunk = list(turns[s:s + batch])
        if responses is not None:
            resp = list(responses[s:s + batch])
        else:
            resp = policy.generate([agent_prompt(t) for t in chunk], gen, seed=eval_gen_seed(seed, s))
        y = sim.rollout(chunk, resp, crn_seed=eval_gen_seed(seed, s))
        sent = np.asarray(sim.sentiment(resp), float)
        for j, (t, r, yy, ss) in enumerate(zip(chunk, resp, y, sent)):
            good, reasons = hygiene_ok(r)
            row = {"uid": t.uid, "dialogue_id": t.dialogue_id, "arm": tag, "seed": seed,
                   "response": r, "outcome": float(yy), "sentiment": float(ss),
                   "words": len(r.split()), "hygiene_ok": float(good), "reasons": reasons,
                   "info_recall": info_recall(r, t.gold_response), "gold_words": len(t.gold_response.split()),
                   **dict(zip(("slot_precision", "slot_recall", "slot_f1"), slot_prf(r, t.gold_response))),
                   "human_valence": t.human_valence}
            if extra is not None:
                row.update(extra[s + j])
            rows.append(row)
    ir = np.asarray([r["info_recall"] for r in rows], float)
    logger.info("eval %s seed=%d | %d turns | outcome %.4f+-%.4f | words %.1f | hygiene %.3f | info recall %.3f "
                "(%d turns with salient gold tokens)", tag, seed, len(rows),
                float(np.mean([r["outcome"] for r in rows])), float(np.std([r["outcome"] for r in rows])),
                float(np.mean([r["words"] for r in rows])), float(np.mean([r["hygiene_ok"] for r in rows])),
                float(np.nanmean(ir)) if np.isfinite(ir).any() else float("nan"), int(np.isfinite(ir).sum()))
    return rows


@dataclass
class DPOConfig:
    beta: float = 0.1
    lr: float = 5e-6
    epochs: int = 2
    batch_size: int = 4
    grad_accum: int = 4
    evals_per_epoch: int = 4
    patience: int = 4
    min_gap_sd: float = 0.5     # a pair is kept only if y(chosen)-y(rejected) >= this x within-group sd
    seed: int = 0


@dataclass
class Config:
    data_dir: Path = Path("emowoz_data")
    out: Path = Path("caro_run")
    device: str = "cuda"
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    models_dir: Optional[str] = None
    seed: int = 42
    seeds: Tuple[int, ...] = (42, 43, 44)
    stub: bool = False
    download: bool = False
    load_4bit: bool = True
    lora_r: int = 16
    gen_batch: int = 32
    feat_dim: int = 256
    sft: SFTConfig = field(default_factory=SFTConfig)
    sim_sft: SFTConfig = field(default_factory=lambda: SFTConfig(epochs=3, max_gap=0.35))
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    sft_turns: Optional[int] = None
    sim_turns: Optional[int] = 20000
    validate_contexts: int = 800
    validate_variants: int = 4
    sim_rollouts: int = 12
    sim_rollouts_corpus: int = 6
    outcome_mode: str = "auto"
    panel_size: int = 48          # v14 (R11): v13's Config said 24 but its CLI default was 48, which the run used
    panel_temperature: float = 1.0
    length_margin_frac: float = 0.15
    max_excess_flip: float = 0.05
    max_abs_flip: float = 0.10
    max_tau: float = 0.35
    gate_bank: str = "heldout"
    reward_dev_excess_tol: float = 0.05
    reward_max_excess_length_rho: float = 0.10
    score_batch: int = 64
    corpus_pad: int = 1
    ablate_contexts: int = 400
    projector_k_max: int = 8
    projector_k_grid: Tuple[int, ...] = (0, 1, 2, 3, 4, 6, 8, 12)
    projector_fit_contexts: int = 600
    projector_fit_frac: float = 0.45
    projector_probe_contexts: int = 200
    projector_min_anchor_retention: float = 0.80
    projector_min_residual_gain: float = 0.02
    orbit_levels: int = 1
    corpus_contexts: int = 3000
    corpus_variants: int = 5
    corpus_temperature: float = 1.0
    eval_turns: int = 800
    strict: bool = True
    arms: Tuple[str, ...] = ("sft", "sentiment_only", "dpo", "caro", "sft_bon_rm", "sft_lenmatch")
    dpo: DPOConfig = field(default_factory=DPOConfig)
    bon_n: int = 4                           # samples per context for sft_bon_rm and sft_lenmatch
    reward_dev_anchor_tol: float = 0.05     # R4: DEV anchor may not fall more than this below the plain ranker's
    reward_min_gold_anchor: float = 0.10
    reward_n_perm: int = 2000
    sens_reward_grid: str = "compact"       # compact (12 cells + the selected one) | full (the DEV selection grid)
    ablate_fit_seeds: int = 3               # R7: refits per reward-ablation variant
    ablate_accuracy_margin: float = 0.01    # R7: TOST margin on pairwise accuracy for "no effect"
    ablate_sim_noreg: bool = False          # R6: also retrain a simulator without the invariance regulariser
    n_signflip: int = 20000                 # R9: Monte Carlo draws when exact enumeration is too large
    external_model: str = "microsoft/Phi-3.5-mini-instruct"                  # R2: out-of-family customer
    external_emotion_model: str = "j-hartmann/emotion-english-distilroberta-base"   # R2: independent labeller
    external_rollouts: int = 4
    human_contexts: int = 150               # R3: contexts per arm comparison in the human packet
    human_validity_pairs: int = 200         # R3: within-context simulator-validity items
    human_overlap: float = 0.3              # R3: share of items annotated twice (for agreement)
    human_comparisons: Tuple[str, ...] = ("caro:sft", "caro:sentiment_only", "caro:dpo")
    stability_safety: float = 0.8           # v15: FIT-half flip limit = this x max_abs_flip when choosing T (0 = off)
    sim_train_bank: str = "train"           # v15: filler bank for the simulator's invariance regulariser
    noninferiority_info_margin: float = 0.05   # D4: information-retention non-inferiority margin (recall units)
    cross_eval_contexts: int = 500             # held-out contexts for cross-evaluator agreement
    cross_eval_k: int = 4                      # responses per context (K=4 -> 6 pairs per context)


def build_policy(cfg: Config, logger: logging.Logger):
    if cfg.stub:
        return StubPolicy(logger, cfg.seed)
    return Policy(cfg.base_model, cfg.device, cfg.models_dir, logger, lora_r=cfg.lora_r,
                  load_4bit=cfg.load_4bit, gen_batch=cfg.gen_batch, score_batch=cfg.score_batch)


def build_sentiment(cfg: Config, logger: logging.Logger):
    if cfg.stub:
        return StubSentiment()
    return SentimentScorer(cfg.sentiment_model, cfg.device, cfg.models_dir, logger)


class Experiment:
    def __init__(self, cfg: Config, tag: str):
        self.cfg = cfg
        cfg.out.mkdir(parents=True, exist_ok=True)
        self.logger = make_logger(cfg.out, tag)
        seed_everything(cfg.seed)
        self.logger.info("CARO %s | tag=%s | device=%s | out=%s | stub=%s | seeds=%s", VERSION,
                         tag, cfg.device, cfg.out, cfg.stub, cfg.seeds)
        if cfg.download and not cfg.stub:
            download_emowoz(cfg.data_dir, self.logger)
        cache = cfg.out / "turns.json"
        if cache.exists():
            self.turns = [Turn(**d) for d in load_json(cache)]
            self.logger.info("loaded %d cached turns from %s", len(self.turns), cache)
        else:
            self.turns = load_emowoz(cfg.data_dir, self.logger)
            dump_json([asdict(t) for t in self.turns], cache)
        self._sent = None

    @property
    def sentiment(self):
        if self._sent is None:
            self._sent = build_sentiment(self.cfg, self.logger)
        return self._sent

    def gen(self, temperature: float = 0.9) -> GenConfig:
        return GenConfig(temperature=temperature)

    def panel(self, policy=None) -> Optional[OutcomePanel]:
        if self.cfg.outcome_mode == "sample":
            return None
        f = self.cfg.out / "outcome_panel.json"
        if f.exists():
            return OutcomePanel.load(load_json(f))
        tr = filter_turns(self.turns, "train", require_next=True, limit=8000, seed=self.cfg.seed)
        # The simulator adapter is what will be used at scoring time, so stratify the panel against
        # the log-probs it produces.  v15: re-use the caller's already-loaded simulator instead of
        # loading a second copy of the 3B model (v14 loaded it twice in the validate stage).
        probe_pol = policy
        if probe_pol is None:
            for cand in ("simulator", "sft_policy"):
                d = self.cfg.out / cand
                if d.exists():
                    probe_pol = self.policy_with(cand)
                    break
        probe_prompt = f"{CUSTOMER_SYSTEM}\nCustomer: hello\nAgent:"
        pan = OutcomePanel.build(
            tr, self.sentiment, self.cfg.panel_size, self.cfg.seed, self.logger,
            policy=probe_pol, probe_prompt=probe_prompt, reduce="mean",
        )
        dump_json(pan.state(), f)
        return pan

    def simulator(self, policy, rollouts: Optional[int] = None, load_nuisance: bool = True) -> "UserSimulator":
        mode = self.cfg.outcome_mode
        if mode == "auto":
            f = self.cfg.out / "outcome_mode.json"
            if not f.exists():
                # v15: v14 silently fell back to the SAMPLED estimator here.  That is what made the
                # validate stage fail: a Monte Carlo estimator re-run on the same responses flipped 35%
                # of within-context pairs, so no invariance gate with a 10% flip ceiling can pass.
                raise FileNotFoundError(
                    f"{f} is missing: the outcome estimator has not been selected for this simulator. Run "
                    f"the validate stage (it selects it before gating), or the simulator stage in full.")
            mode = load_json(f)["selected"]
        pj = None
        p = self.cfg.out / "logit_projector.json"
        if load_nuisance and p.exists():
            st = load_json(p)
            if int(st.get("k", 0)) > 0:
                pj = LogitNullspaceProjector.load(st)
        ob = self.cfg.out / "orbit.json"
        n_orbit = int(load_json(ob).get("n_orbit", 1)) if ob.exists() else self.cfg.orbit_levels
        sim = UserSimulator(policy, self.sentiment, self.logger,
                            rollouts if rollouts is not None else self.cfg.sim_rollouts,
                            mode=mode, panel=self.panel(policy),
                            panel_temperature=self.cfg.panel_temperature,
                            projector=pj, n_orbit=n_orbit)
        if pj is not None:
            self.logger.info("loaded logit nullspace projector | k=%d | residual length-perturbation "
                             "energy %.4f | signal retained %.4f | orbit levels=%d", pj.k,
                             pj.diag.get("residual_perturbation", float("nan")),
                             pj.diag.get("signal_retention", float("nan")), n_orbit)
        if not load_nuisance:
            return sim
        f = self.cfg.out / "length_control.json"
        if f.exists():
            sim.length_control = InterventionalLengthCalibration.load(load_json(f))   # raises on a v8 file
            self.logger.info("loaded interventional length calibration | %s",
                             {k: sim.length_control.diag.get(k) for k in ("n_pairs", "mean_delta_raw",
                                                                         "mean_delta_residual_cross_fitted")})
        g = self.cfg.out / "control_variate.json"
        if g.exists():
            sim.control_variate = FrozenControlVariate.load(load_json(g))
        return sim

    def sim_train_bank(self) -> str:
        """Filler bank the simulator in cfg.out was regularised on.  Simulators trained before v15 carry
        no record and used the courtesy bank."""
        f = self.cfg.out / "simulator_meta.json"
        return str(load_json(f).get("train_bank", "courtesy_v13")) if f.exists() else "courtesy_v13"

    def ensure_estimator(self, sim_pol) -> str:
        """Build and calibrate the outcome panel and select the estimator if that has not been done for
        this simulator.  Selection uses only TRAIN-split turns, so it cannot see the validation gate."""
        cfg = self.cfg
        if cfg.outcome_mode != "sample":
            panel = self.panel(sim_pol)
            if panel is not None and not panel.calibrated:
                cal = filter_turns(self.turns, "train", require_next=True, limit=1200, seed=cfg.seed + 11,
                                   logger=self.logger, what="panel calibration turns")
                panel.calibrate(sim_pol, cal, self.sentiment, self.logger, cache_file=cfg.out / "panel_cal_cache.npz")
                dump_json(panel.state(), cfg.out / "outcome_panel.json")
        f = cfg.out / "outcome_mode.json"
        if cfg.outcome_mode == "auto" and not f.exists():
            sel_turns = filter_turns(self.turns, "train", require_next=True, require_label=True, limit=400,
                                     seed=cfg.seed + 13, logger=self.logger, what="estimator selection turns")
            probe = UserSimulator(sim_pol, self.sentiment, self.logger, cfg.sim_rollouts, mode="sample",
                                  panel=self.panel(sim_pol), panel_temperature=cfg.panel_temperature)
            dump_json(select_outcome_mode(probe, sel_turns, self.logger), f)
        return load_json(f)["selected"] if cfg.outcome_mode == "auto" else cfg.outcome_mode

    def policy_with(self, adapter: Optional[str]):
        p = build_policy(self.cfg, self.logger)
        if adapter:
            d = self.cfg.out / adapter
            if not d.exists():
                raise FileNotFoundError(f"missing adapter {d}")
            p.load_adapter(d)
        return p


def stage_sft(cfg: Config) -> Dict[str, Any]:
    ex = Experiment(cfg, "1_sft")
    tr = filter_turns(ex.turns, "train", limit=cfg.sft_turns, seed=cfg.seed, logger=ex.logger, what="SFT train turns")
    dv = filter_turns(ex.turns, "valid", limit=cfg.sft.dev_examples * 2, seed=cfg.seed, logger=ex.logger,
                      what="SFT dev turns")
    pol = build_policy(cfg, ex.logger)
    res = pol.fit_supervised([(agent_prompt(t), t.gold_response) for t in tr],
                             [(agent_prompt(t), t.gold_response) for t in dv][: cfg.sft.dev_examples],
                             cfg.sft, ex.logger, cfg.out / "sft_best", "SFT")
    pol.save_adapter(cfg.out / "sft_policy")
    dump_json(res, cfg.out / "sft.json")
    return res


def stage_simulator(cfg: Config) -> Dict[str, Any]:
    ex = Experiment(cfg, "2_simulator")
    # A retrained simulator invalidates every artefact that was fitted on the old one.
    for stale in ("panel_cal_cache.npz", "outcome_panel.json", "logit_projector.json", "length_control.json",
                  "control_variate.json", "orbit.json", "simulator_validation.json", "validation_gate_sample.npz",
                  "invariance_selection.json"):
        f = cfg.out / stale
        if f.exists():
            f.unlink()
            ex.logger.info("removed stale artefact %s", f.name)
    tr = filter_turns(ex.turns, "train", require_next=True, limit=cfg.sim_turns, seed=cfg.seed,
                      logger=ex.logger, what="simulator train turns")
    dv = filter_turns(ex.turns, "valid", require_next=True, limit=1500, seed=cfg.seed, logger=ex.logger,
                      what="simulator dev turns")
    pol = build_policy(cfg, ex.logger)
    for stale in ("outcome_mode.json",):
        if (cfg.out / stale).exists():
            (cfg.out / stale).unlink()
    res = sim_fit = UserSimulator(pol, ex.sentiment, ex.logger, cfg.sim_rollouts, mode="sample").fit(
        tr, dv, cfg.sim_sft, cfg.out / "sim_best", augment_levels=2, seed=cfg.seed, bank=cfg.sim_train_bank)
    del sim_fit
    pol.save_adapter(cfg.out / "simulator")
    dump_json({"train_bank": cfg.sim_train_bank, "invariance_coef": cfg.sim_sft.invariance_coef,
               "version": VERSION}, cfg.out / "simulator_meta.json")
    dump_json(res, cfg.out / "simulator.json")
    res["outcome_mode"] = ex.ensure_estimator(pol)
    return res


def make_length_pairs(turns: Sequence[Turn], proposal, gen: GenConfig, n_variants: int, seed: int,
                      bank: str, levels: Sequence[int] = (1, 2)):
    """Sample natural variants and pad each with content-free filler from `bank` at a random rung.
    Returns (flat turns, base responses, padded responses, gid, cluster)."""
    turns = list(turns)
    prompts = [agent_prompt(t) for t in turns for _ in range(n_variants)]
    base = list(proposal.generate(prompts, gen, seed=seed))
    flat = [t for t in turns for _ in range(n_variants)]
    aug = LengthInvarianceAugmenter(n_levels=max(levels), seed=seed, bank=bank)
    rng = np.random.default_rng(seed + 1)
    lv = rng.choice(np.asarray(levels, int), size=len(base))
    pert = [aug.lengthen(r, int(l)) for r, l in zip(base, lv)]
    gid = np.repeat(np.arange(len(turns)), n_variants)
    cluster = np.asarray([t.dialogue_id for t in flat])
    return flat, base, pert, gid, cluster


def _invariance_stats(y0: np.ndarray, y1: np.ndarray, gid: np.ndarray, cluster: np.ndarray,
                      n_boot: int, seed: int) -> Dict[str, float]:
    """Deterministic-estimator invariance statistics (no replication null by construction)."""
    d = y1 - y0
    vs = max(within_group_var(y0, gid), EPS)
    tau = float(math.sqrt(float(np.var(d, ddof=1)) / vs))

    def _t(ix, cp):
        return float(math.sqrt(float(np.var(d[ix], ddof=1)) / max(within_group_var(y0[ix], boot_groups(gid[ix], cp)), EPS)))

    _, lo, hi = cluster_bootstrap_ci(_t, cluster, n_boot, seed, conf=0.90, with_copy=True)
    return {"tau": tau, "tau_lo": lo, "tau_hi": hi, "flip": _pairwise_flip_rate(y0, y1, gid),
            "delta_mean": float(np.mean(d)), "sd_within": within_group_sd(y0, gid)}


def stability_constraint(sent: np.ndarray, Z0: np.ndarray, Z1: np.ndarray, base: Sequence[str],
                         pert: Sequence[str], gid: np.ndarray, cluster: np.ndarray,
                         pj: Optional["LogitNullspaceProjector"], limit: float, seed: int) -> Callable[[float], bool]:
    """Admissibility of a panel temperature: after the interventional length calibration fitted on
    the SAME fitting pairs, the within-context flip rate of the (base, padded) pairs must not exceed
    `limit`.  Only FIT-half data enter, so the TEST gate stays untouched.  `limit` is the
    pre-registered ceiling times a safety factor < 1: a temperature that only just passes on the
    fitting half is the one most likely to fail on a new sample (v13 passed the gate at flip 0.083
    and then measured 0.16-0.19 on a fresh sample)."""
    base, pert = list(base), list(pert)

    def ok(T: float) -> bool:
        y0 = OutcomePanel.expectation(Z0, sent, T, pj)
        y1 = OutcomePanel.expectation(Z1, sent, T, pj)
        try:
            h = InterventionalLengthCalibration().fit(base, pert, y1 - y0, cluster, None, seed=seed)
            y0, y1 = h.apply(base, y0), h.apply(pert, y1)
        except ValueError:
            pass
        f = _pairwise_flip_rate(y0, y1, gid)
        return (not math.isfinite(f)) or f <= limit
    return ok


def _crossfit_probe(sim: "UserSimulator", k: int, Z0: np.ndarray, Z1: np.ndarray, base: Sequence[str],
                    pert: Sequence[str], gid: np.ndarray, cluster: np.ndarray, Zg: Optional[np.ndarray],
                    gold_val: Optional[np.ndarray], gold_cluster: Optional[np.ndarray], cal: Sequence[Turn],
                    cfg: "Config", logger: logging.Logger) -> Dict[str, Any]:
    """Two-fold, dialogue-disjoint probe of one erasure rank k, computed ENTIRELY from cached
    panel logits: no forward pass is spent.  For each fold the projector, the panel temperature
    and the interventional length calibration are fitted on the OTHER fold (temperature on the
    held-out TRAIN calibration turns), and invariance / anchor are measured out-of-fold.  v8's
    probe was in-sample for the projector and cost ~2 GPU-hours per rank."""
    uniq = np.unique(cluster)
    rng = np.random.default_rng(cfg.seed + 97)
    f_of = {u: i % 2 for i, u in enumerate(uniq[rng.permutation(uniq.size)])}
    fold = np.asarray([f_of[c] for c in cluster], int)
    gfold = (np.asarray([f_of.get(c, 0) for c in gold_cluster], int) if gold_cluster is not None else None)
    y0c = np.full(len(base), np.nan)
    y1c = np.full(len(base), np.nan)
    yg = np.full(0 if Zg is None else Zg.shape[0], np.nan)
    T_keep = sim.panel.temperature
    temps = []
    try:
        for f in (0, 1):
            tr, te = fold != f, fold == f
            pj = LogitNullspaceProjector.fit(Z0[tr], Z1[tr], k) if k > 0 else None
            itr = np.flatnonzero(tr)
            adm = stability_constraint(sim.panel.sent, Z0[tr], Z1[tr], [base[i] for i in itr], [pert[i] for i in itr],
                                       gid[tr], cluster[tr], pj, cfg.stability_safety * cfg.max_abs_flip, cfg.seed)
            sim.panel.calibrate(sim.policy, cal, sim.sentiment, logger, projector=pj, quiet=True, admissible=adm)
            T = sim.panel.temperature
            temps.append(T)
            y0 = OutcomePanel.expectation(Z0, sim.panel.sent, T, pj)
            y1 = OutcomePanel.expectation(Z1, sim.panel.sent, T, pj)
            h = InterventionalLengthCalibration().fit([base[i] for i in np.flatnonzero(tr)],
                                                      [pert[i] for i in np.flatnonzero(tr)],
                                                      (y1 - y0)[tr], cluster[tr], None, seed=cfg.seed)
            ie = np.flatnonzero(te)
            y0c[ie] = h.apply([base[i] for i in ie], y0[ie])
            y1c[ie] = h.apply([pert[i] for i in ie], y1[ie])
            if Zg is not None and gfold is not None:
                m = gfold == f
                yg[m] = OutcomePanel.expectation(Zg[m], sim.panel.sent, T, pj)
    finally:
        sim.panel.temperature = T_keep
    st = _invariance_stats(y0c, y1c, gid, cluster, 200, cfg.seed + 2)
    anchor = (spearman(yg, gold_val) if (Zg is not None and np.isfinite(yg).all() and len(yg) >= 60)
              else float("nan"))
    passes = bool((not math.isfinite(st["tau_hi"]) or st["tau_hi"] <= cfg.max_tau) and
                  (not math.isfinite(st["flip"]) or st["flip"] <= cfg.max_abs_flip))
    return {**st, "anchor": float(anchor), "passes": passes, "temperatures": temps}


def stage_validate(cfg: Config) -> Dict[str, Any]:
    """Fit every nuisance component on one dialogue-disjoint half, gate on the other.

    Pre-registered order (never revisited after seeing the gate):
      1. split the validation pool into dialogue-disjoint FIT and TEST halves;
      0. [v15] make sure the outcome estimator was selected (TRAIN-split data only), instead of silently
         falling back to the sampled estimator as v14 did;
      2. on FIT, sample natural variants and pad them with the affect-neutral CALIB bank (rungs 1-2);
      3. [expected mode] profile the length-response subspace of the panel logits and choose the
         SMALLEST erasure rank k whose two-fold out-of-fold probe passes the tau and flip
         ceilings while retaining >= projector_min_anchor_retention of the k=0 human anchor;
      4. freeze the projector and the temperature -- v15: the temperature maximises the calibration
         anchor SUBJECT TO a FIT-half flip rate <= stability_safety x the flip ceiling -- and fit the
         INTERVENTIONAL length calibration h(L) and the control variate on FIT;
      5. gate on TEST, perturbing with the disjoint HELD-OUT filler bank.
    Nothing in step 5 can feed back into steps 1-4."""
    ex = Experiment(cfg, "3_validate")
    sim_pol = ex.policy_with("simulator")
    selected = ex.ensure_estimator(sim_pol)       # v15: never fall back silently to the sampled estimator
    sim = ex.simulator(sim_pol, load_nuisance=False)
    sim.length_control = None
    sim.projector = None
    sim.control_variate = None
    sim_bank = ex.sim_train_bank()
    ex.logger.info("outcome estimator in use: %s (selected: %s) | simulator regularised on the '%s' filler bank | "
                   "corrections fitted on the '%s' bank | gate on the '%s' bank", sim.mode, selected, sim_bank,
                   FIT_BANK, cfg.gate_bank)
    if cfg.gate_bank in (sim_bank, FIT_BANK):
        ex.logger.warning("the gate re-uses a filler bank that a training or fitting step has seen (circular); use "
                          "--gate-bank heldout for a real test")
    if sim_bank == "courtesy_v13":
        ex.logger.warning("this simulator was regularised to be INVARIANT to courtesy phrases (the pre-v15 bank, "
                          "mean sentiment ~+0.4): it was trained to ignore part of the affect signal this work "
                          "rewards. Retrain it (stage simulator) for the paper; it is kept here only for analysis.")
    if sim.mode == "sample":
        ex.logger.warning("the SAMPLED estimator is in use: its Monte Carlo noise alone flips a large share of "
                          "within-context pairs, so the flip gate is expected to fail (the v14 run: 34.6%%)")
    prop = ex.policy_with("sft_policy")
    pool = filter_turns(ex.turns, "valid", require_next=True,
                        limit=cfg.validate_contexts + cfg.projector_fit_contexts,
                        seed=cfg.seed, logger=ex.logger, what="validation pool")
    fitset, te = split_turns_by_dialogue(pool, cfg.projector_fit_frac, cfg.seed + 5)
    te = te[: cfg.validate_contexts]
    fitset = fitset[: cfg.projector_fit_contexts]
    ex.logger.info("dialogue-disjoint split | %d fitting contexts (%d dialogues) | %d gating contexts "
                   "(%d dialogues) | intersection of dialogue ids = %d",
                   len(fitset), len({t.dialogue_id for t in fitset}), len(te),
                   len({t.dialogue_id for t in te}),
                   len({t.dialogue_id for t in fitset} & {t.dialogue_id for t in te}))
    cal = filter_turns(ex.turns, "train", require_next=True, limit=1200, seed=cfg.seed + 11,
                       logger=ex.logger, what="panel re-calibration turns")
    if sim.mode == "expected":
        sim.panel.calibrate(sim_pol, cal, ex.sentiment, ex.logger, projector=None,
                            cache_file=cfg.out / "panel_cal_cache.npz")

    flat_f, base_f, pert_f, gid_f, cl_f = make_length_pairs(fitset, prop, ex.gen(0.9), 2, 6161, FIT_BANK)
    ex.logger.info("FIT-half paired interventions | %d (base, padded) pairs over %d contexts | %s bank, "
                   "rungs {1,2} | +%.1f words mean", len(base_f), len(fitset), FIT_BANK,
                   float(np.mean([len(b.split()) - len(a.split()) for a, b in zip(base_f, pert_f)])))

    sel: Dict[str, Any] = {"enabled": False, "trace": []}
    k_sel, pj = 0, None
    Z0 = Z1 = None
    if sim.mode == "expected" and sim.panel is not None:
        Z0 = sim.panel_logits(flat_f, base_f)
        Z1 = sim.panel_logits(flat_f, pert_f)
    if sim.mode == "expected" and cfg.projector_k_max > 0 and sim.panel is not None:
        rows = LogitNullspaceProjector.profile(Z0, Z1, cfg.projector_k_max, ex.logger)
        res = {r["k"]: r["residual_perturbation"] for r in rows}
        grid, last = [0], res.get(0, 1.0)
        for k in sorted(x for x in cfg.projector_k_grid if 0 < x <= cfg.projector_k_max):
            if k in res and last - res[k] >= cfg.projector_min_residual_gain:
                grid.append(k)
                last = res[k]
        ex.logger.info("escalation grid k in %s (ranks with residual gain < %.3f skipped; each probe is now "
                       "free: two-fold out-of-fold on cached logits)", grid, cfg.projector_min_residual_gain)
        lab = [t for t in fitset if t.human_valence is not None]
        Zg = sim.panel_logits(lab, [t.gold_response for t in lab]) if len(lab) >= 60 else None
        gval = np.asarray([t.human_valence for t in lab], float) if Zg is not None else None
        gcl = np.asarray([t.dialogue_id for t in lab]) if Zg is not None else None
        base_anchor = float("nan")
        chosen = None
        for k in grid:
            try:
                pr = _crossfit_probe(sim, k, Z0, Z1, base_f, pert_f, gid_f, cl_f, Zg, gval, gcl, cal, cfg,
                                     ex.logger)
            except ValueError as e:
                ex.logger.warning("invariance search | k=%d cannot be probed on this sample (%s); skipped", k, e)
                continue
            if k == 0:
                base_anchor = pr["anchor"]
            keep = (abs(pr["anchor"]) >= cfg.projector_min_anchor_retention * abs(base_anchor)
                    if math.isfinite(pr["anchor"]) and math.isfinite(base_anchor) and base_anchor != 0 else True)
            r0 = next((r for r in rows if r["k"] == k), {"residual_perturbation": 1.0, "signal_retention": 1.0})
            row = {"k": k, "residual": r0["residual_perturbation"], "retention": r0["signal_retention"],
                   "anchor_kept": bool(keep), **pr}
            sel["trace"].append(row)
            ex.logger.info("invariance search | k=%d | residual perturbation %.4f | signal retained %.4f | "
                           "out-of-fold tau=%.3f CI90[%.3f,%.3f] flip=%.4f delta=%+.5f | anchor %+.4f%s -> %s",
                           k, row["residual"], row["retention"], pr["tau"], pr["tau_lo"], pr["tau_hi"], pr["flip"],
                           pr["delta_mean"], pr["anchor"], "" if keep else " [anchor retention violated]",
                           "ACCEPT" if (pr["passes"] and keep) else "reject")
            if pr["passes"] and keep:
                chosen = k
                break
        if chosen is None:
            chosen = int(grid[-1])
            ex.logger.error("no erasure rank on the grid satisfied the out-of-fold probe; freezing the largest "
                            "probed rank k=%d and letting the TEST gate report the residual failure honestly "
                            "rather than widening any threshold", chosen)
        k_sel = int(chosen)
        pj = LogitNullspaceProjector.fit(Z0, Z1, k_sel) if k_sel > 0 else None
        sel.update({"enabled": True, "selected_k": k_sel, "profile": rows, "base_anchor": base_anchor,
                    "grid": grid})
    sim.projector = pj
    sim.n_orbit = max(1, int(cfg.orbit_levels))
    if sim.mode == "expected":
        adm = None
        if Z0 is not None and cfg.stability_safety > 0:
            adm = stability_constraint(sim.panel.sent, Z0, Z1, base_f, pert_f, gid_f, cl_f, pj,
                                       cfg.stability_safety * cfg.max_abs_flip, cfg.seed)
        sim.panel.calibrate(sim_pol, cal, ex.sentiment, ex.logger, projector=pj, admissible=adm)
        dump_json(sim.panel.state(), cfg.out / "outcome_panel.json")
    dump_json((pj.state() if pj is not None else {"k": 0}), cfg.out / "logit_projector.json")
    dump_json({"n_orbit": sim.n_orbit}, cfg.out / "orbit.json")
    dump_json(sel, cfg.out / "invariance_selection.json")

    # Interventional length calibration and control variate, both on the FIT half only.
    lc = sim.fit_length_calibration(flat_f, base_f, pert_f, crn_seed=9091, seed=cfg.seed)
    sim.control_variate = None
    sim.rollout(flat_f, base_f, crn_seed=9091)
    dump_json(lc.state(), cfg.out / "length_control.json")
    ex.logger.info("nuisance components frozen | k=%d | T=%.3g | h(L) interventional | control variate b=%+.4f | "
                   "panel-logit memo: %d hits / %d misses so far",
                   k_sel, sim.panel.temperature if sim.panel is not None else float("nan"),
                   sim.control_variate.b, sim.cache_hits, sim.cache_misses)

    out = validate_simulator(sim, te, prop, cfg.validate_variants, ex.gen(0.9), ex.logger, seed=cfg.seed,
                             margin_frac=cfg.length_margin_frac, max_excess_flip=cfg.max_excess_flip,
                             max_abs_flip=cfg.max_abs_flip, max_tau=cfg.max_tau, gate_bank=cfg.gate_bank)
    cv = sim.control_variate.state() if sim.control_variate else {}
    out["control_variate"] = cv
    out["invariance_selection"] = sel
    out["pre_registration"] = {"margin_frac": cfg.length_margin_frac, "max_tau": cfg.max_tau,
                               "max_abs_flip": cfg.max_abs_flip, "max_excess_flip": cfg.max_excess_flip,
                               "gate_bank": cfg.gate_bank, "version": VERSION}
    ga = out.pop("_gate_arrays", None)
    if ga is not None:
        np.savez_compressed(cfg.out / "validation_gate_sample.npz", **ga)
    ex.logger.info("panel-logit memo | %d hits / %d misses (%.1f%% of panel sweeps avoided)", sim.cache_hits,
                   sim.cache_misses, 100.0 * sim.cache_hits / max(1, sim.cache_hits + sim.cache_misses))
    dump_json(out, cfg.out / "simulator_validation.json")
    dump_json(cv, cfg.out / "control_variate.json")
    if not out["passes"] and cfg.strict:
        raise RuntimeError("simulator validation failed; the counterfactual corpus cannot identify the causal "
                           "effect. failures: " + "; ".join(out["failures"]))
    return out


def stage_corpus(cfg: Config) -> Dict[str, Any]:
    ex = Experiment(cfg, "4_corpus")
    vf = cfg.out / "simulator_validation.json"
    if cfg.strict:
        if not vf.exists():
            raise FileNotFoundError("no simulator_validation.json: run the validate stage first")
        v = load_json(vf)
        if not v.get("passes", False) or v.get("pre_registration", {}).get("version") not in COMPATIBLE_VERSIONS:
            raise RuntimeError("refusing to build a counterfactual corpus on a simulator that has not passed the "
                               f"{VERSION} validation gate (use --no-strict only for debugging)")
    sim_pol = ex.policy_with("simulator")
    sim = ex.simulator(sim_pol, cfg.sim_rollouts_corpus)
    prop = ex.policy_with("sft_policy")
    tr = filter_turns(ex.turns, "train", require_next=True, limit=cfg.corpus_contexts, seed=cfg.seed + 1,
                      logger=ex.logger, what="corpus contexts")
    groups, diag = build_corpus(tr, prop, sim, cfg.corpus_variants, ex.gen(cfg.corpus_temperature),
                                ex.logger, cfg.feat_dim, n_pad=cfg.corpus_pad, n_probe=1)
    save_corpus(groups, cfg.out / "corpus.npz")
    dump_json(diag, cfg.out / "corpus_diagnostics.json")
    return diag


def stage_reward(cfg: Config) -> Dict[str, Any]:
    """Reward model with counterfactual logit pairing and length calibration.

    Both regularisation weights are chosen on the DEV split only, by a pre-registered ascending
    sweep (length-calibration weight outer, pairing weight inner; the first pair that satisfies
    every DEV criterion wins), so the TEST gate never enters model selection.  DEV criteria:
      * held-out-bank padding TOST passes (no reward for content-free padding);
      * |reward length rho - label length rho| <= dev_excess_tol (no length dependence beyond
        what the labels license);
      * v14 (R4): the DEV gold anchor is not more than reward_dev_anchor_tol below that of the
        unregularised Plackett-Luce ranker.  In v13 the selected model had the LOWEST anchor of all
        ablation variants (0.061 vs 0.088-0.138): the invariance penalties were free to buy
        padding-invariance with human-relevant signal.  The anchor is noisy on DEV (a few hundred
        labelled contexts), so the criterion is a point-estimate tolerance, fixed in advance.
    Fixed weights can be forced with --reward-clp / --reward-length-cal."""
    ex = Experiment(cfg, "5_reward")
    groups = load_corpus(cfg.out / "corpus.npz")
    train, dev, calib, test = reward_splits(groups, cfg.seed)
    ex.logger.info("reward splits | train=%d dev=%d calib=%d test=%d groups (disjoint dialogues)",
                   len(train), len(dev), len(calib), len(test))
    X = np.concatenate([g.feats for g in train], 0)
    std = Standardizer().fit(X)
    plain = RewardModel(X.shape[1], RewardConfig(**{**asdict(cfg.reward), "clp": 0.0, "length_cal": 0.0}), std)
    plain.fit(train, dev, calib, ex.logger, seed=cfg.seed)
    plain_anchor = reward_gold_anchor(plain, dev, cfg.seed + 5, n_boot=300)
    ex.logger.info("DEV anchor of the unregularised ranker: rho=%+.4f CI95[%+.4f,%+.4f] (n=%d); regularised "
                   "candidates may not fall more than %.3f below it", plain_anchor["rho"], *plain_anchor["ci95"],
                   plain_anchor["n"], cfg.reward_dev_anchor_tol)
    clp_grid = [cfg.reward.clp] if cfg.reward.clp >= 0 else list(cfg.reward.clp_grid)
    mu_grid = [cfg.reward.length_cal] if cfg.reward.length_cal >= 0 else list(cfg.reward.length_cal_grid)
    trace: List[Dict[str, Any]] = []
    rm = info = None
    chosen = False
    for mu in mu_grid:
        for lam in clp_grid:
            rc = RewardConfig(**{**asdict(cfg.reward), "clp": float(lam), "length_cal": float(mu)})
            cand = RewardModel(X.shape[1], rc, std)
            cinfo = cand.fit(train, dev, calib, ex.logger, seed=cfg.seed)
            dv_pad = reward_padding_test(cand, dev, cfg.length_margin_frac, cfg.seed + 3, ex.logger,
                                         tag=f"DEV, clp={lam:g}, length_cal={mu:g}")
            dv_len = reward_length_excess(cand, dev, cfg.seed + 4, ex.logger,
                                          tag=f"DEV, clp={lam:g}, length_cal={mu:g}", n_boot=300)
            dv_anc = reward_gold_anchor(cand, dev, cfg.seed + 5, n_boot=300)
            ok_pad = bool(dv_pad.get("tost_passes", True))
            ok_len = bool(abs(dv_len["excess"]) <= cfg.reward_dev_excess_tol)
            ok_anc = bool(not (math.isfinite(dv_anc["rho"]) and math.isfinite(plain_anchor["rho"]))
                          or dv_anc["rho"] >= plain_anchor["rho"] - cfg.reward_dev_anchor_tol)
            trace.append({"clp": float(lam), "length_cal": float(mu), "dev_padding": dv_pad,
                          "dev_length": dv_len, "dev_anchor": dv_anc, "dev_padding_ok": ok_pad,
                          "dev_length_ok": ok_len, "dev_anchor_ok": ok_anc,
                          "calib_accuracy": cinfo["calib_pairwise_accuracy"]})
            ex.logger.info("DEV selection | clp=%g length_cal=%g | padding %s | length %s | anchor %+.4f %s",
                           lam, mu, "ok" if ok_pad else "FAIL", "ok" if ok_len else "FAIL", dv_anc["rho"],
                           "ok" if ok_anc else "FAIL")
            rm, info = cand, cinfo
            if ok_pad and ok_len and ok_anc:
                chosen = True
                break
        if chosen:
            break
    if not chosen:
        ex.logger.error("no (length_cal, clp) pair on the grid met every DEV criterion; keeping the last one "
                        "(length_cal=%g, clp=%g) and letting the TEST gate report the residual failure",
                        rm.cfg.length_cal, rm.cfg.clp)
    ex.logger.info("reward regularisation selected on DEV | counterfactual logit pairing clp=%g | length "
                   "calibration weight=%g | calibration accuracy=%.4f", rm.cfg.clp, rm.cfg.length_cal,
                   info["calib_pairwise_accuracy"])
    gate = validity_gate(rm, test, ex.logger, n_perm=cfg.reward_n_perm, seed=cfg.seed,
                         margin_frac=cfg.length_margin_frac,
                         max_excess_length_rho=cfg.reward_max_excess_length_rho,
                         min_gold_anchor=cfg.reward_min_gold_anchor)
    dump_json(rm.state(), cfg.out / "reward_model.json")
    dump_json({"fit": info, "gate": gate, "selection": trace, "selection_met_all_criteria": chosen,
               "plain_ranker_dev_anchor": plain_anchor,
               "selected": {"clp": rm.cfg.clp, "length_cal": rm.cfg.length_cal}},
              cfg.out / "reward.json")
    if not gate["passes"] and cfg.strict:
        raise RuntimeError("reward validity gate failed: " + "; ".join(gate["failures"]))
    return {"fit": info, "gate": gate}


def reward_splits(groups: Sequence[ResponseGroup], seed: int):
    """The one corpus split used by the reward stage, its ablations, its sensitivity analysis and
    the DPO baseline, so that every method sees exactly the same training data."""
    fit_pool, held = group_split(groups, 0.7, seed)
    train, dev = group_split(fit_pool, 0.8, seed + 1)
    calib, test = group_split(held, 0.5, seed + 2)
    return train, dev, calib, test


def dpo_pairs(groups: Sequence[ResponseGroup], turn_of: Dict[str, Turn], min_gap_sd: float
              ) -> Tuple[List[Tuple[str, str, str]], Dict[str, Any]]:
    """Best-vs-worst preference pairs per context from the counterfactual corpus.

    chosen = the hygienic response with the highest simulated outcome; rejected = a malformed
    response if the group has one (the same hygiene pressure GRPO applies through its fixed
    negative advantage), else the lowest-outcome response.  A pair is kept only if the outcome gap
    is at least min_gap_sd x the corpus's mean within-context sd, so the baseline is not trained on
    pairs the simulator cannot tell apart.  The gold turn is excluded: it is not a policy sample."""
    sds = [float(np.std(g.outcomes[:-1], ddof=1)) for g in groups if len(g.outcomes) > 2]
    thr = min_gap_sd * (float(np.mean(sds)) if sds else 0.0)
    out, n_bad_rej, n_small = [], 0, 0
    for g in groups:
        t = turn_of.get(g.uid)
        if t is None:
            continue
        texts, y, ok = list(g.texts[:-1]), np.asarray(g.outcomes[:-1], float), np.asarray(g.ok[:-1], bool)
        if ok.sum() < 1 or len(texts) < 2:
            continue
        good = np.flatnonzero(ok)
        c = int(good[np.argmax(y[good])])
        bad = np.flatnonzero(~ok)
        if bad.size:
            r = int(bad[0])
            n_bad_rej += 1
        else:
            r = int(np.argmin(y))
            if y[c] - y[r] < thr:
                n_small += 1
                continue
        if c != r:
            out.append((agent_prompt(t), texts[c], texts[r]))
    return out, {"n_pairs": len(out), "min_gap": thr, "malformed_rejected": n_bad_rej, "dropped_small_gap": n_small,
                 "mean_words_chosen": float(np.mean([len(c.split()) for _, c, _ in out])) if out else float("nan"),
                 "mean_words_rejected": float(np.mean([len(r.split()) for _, _, r in out])) if out else float("nan")}


def stage_train(cfg: Config, arm: str, seed: int) -> Dict[str, Any]:
    ex = Experiment(cfg, f"6_train_{arm}_{seed}")
    if arm in EVAL_ONLY_ARMS:
        ex.logger.info("arm '%s' requires no training; it re-uses the SFT adapter", arm)
        return {"arm": arm, "seed": seed, "skipped": True}
    if arm not in TRAINED_ARMS:
        raise ValueError(f"unknown arm {arm}; choose from {KNOWN_ARMS}")
    seed_everything(seed)
    pol = ex.policy_with("sft_policy")
    if arm == "dpo":
        groups = load_corpus(cfg.out / "corpus.npz")
        train, dev, _, _ = reward_splits(groups, cfg.seed)
        turn_of = {t.uid: t for t in ex.turns}
        tr_pairs, tr_diag = dpo_pairs(train, turn_of, cfg.dpo.min_gap_sd)
        dv_pairs, dv_diag = dpo_pairs(dev, turn_of, cfg.dpo.min_gap_sd)
        ex.logger.info("DPO pairs from the reward TRAIN/DEV corpus splits | train %s | dev %s", tr_diag, dv_diag)
        if len(tr_pairs) < 50 or len(dv_pairs) < 10:
            raise RuntimeError(f"too few DPO pairs (train {len(tr_pairs)}, dev {len(dv_pairs)})")
        dcfg = DPOConfig(**{**asdict(cfg.dpo), "seed": seed})
        res = pol.fit_dpo(tr_pairs, dv_pairs, dcfg, ex.logger, cfg.out / f"dpo_best_s{seed}", f"DPO_s{seed}")
        pol.save_adapter(cfg.out / f"policy_{arm}_s{seed}")
        res.update({"pairs_train": tr_diag, "pairs_dev": dv_diag})
    else:
        tr = filter_turns(ex.turns, "train", limit=20000, seed=seed, logger=ex.logger, what="GRPO contexts")
        if arm == "sentiment_only":
            reward_fn = SentimentReward(ex.sentiment)
        else:
            rm = RewardModel.load(load_json(cfg.out / "reward_model.json"))
            reward_fn = LearnedReward(rm, cfg.feat_dim, abstain=(arm != "caro_no_abstain"))
        res = train_grpo(pol, reward_fn, tr, cfg.grpo, ex.gen(1.0), ex.logger, cfg.out, f"{arm}_s{seed}",
                         seed, cfg.stub)
    dump_json(res, cfg.out / f"{'dpo' if arm == 'dpo' else 'grpo'}_{arm}_{seed}.json")
    return res


def _sample_n(policy, turns: Sequence[Turn], gen: GenConfig, seed: int, n: int, batch: int = 32) -> List[List[str]]:
    """n SFT samples per context, draw 0 under the same seeds as the plain sft arm."""
    out = [[] for _ in turns]
    for k in range(n):
        for s in range(0, len(turns), batch):
            chunk = list(turns[s:s + batch])
            resp = policy.generate([agent_prompt(t) for t in chunk], gen, seed=eval_gen_seed(seed, s, k))
            for j, r in enumerate(resp):
                out[s + j].append(r)
    return out


def stage_eval(cfg: Config, arm: str, seed: int) -> Dict[str, Any]:
    """Evaluate one arm under one evaluation seed.

    Eval-only controls (no training):
      sft_bon_rm    best of `bon_n` SFT samples under the CARO reward (frozen SFT features, the same
                    reward GRPO optimised).  Tests whether RL adds anything beyond reranking.
      sft_lenmatch  per context, the SFT sample whose word count is closest to the CARO response for
                    the same (seed, context).  A DESIGN-based length control (R12): if CARO still
                    beats it, the gain is not explained by response length.  Needs eval_caro_<seed>."""
    ex = Experiment(cfg, f"7_eval_{arm}_{seed}")
    adapter = "sft_policy" if arm in EVAL_ONLY_ARMS else f"policy_{arm}_s{seed}"
    pol = ex.policy_with(adapter)
    sim_pol = ex.policy_with("simulator")
    sim = ex.simulator(sim_pol)
    te = filter_turns(ex.turns, "test", require_next=True, limit=cfg.eval_turns, seed=cfg.seed,
                      logger=ex.logger, what="evaluation turns")
    gen = ex.gen(0.7)
    if arm in ("sft_bon_rm", "sft_lenmatch"):
        samples = _sample_n(pol, te, gen, seed, cfg.bon_n)
        chosen, extra = [], []
        if arm == "sft_bon_rm":
            rm = RewardModel.load(load_json(cfg.out / "reward_model.json"))
            for t, cand in zip(te, samples):
                f = pol.features([agent_prompt(t)] * len(cand), cand, dim=cfg.feat_dim)
                r = rm.score(f)["reward"]
                ok = np.asarray([hygiene_ok(c)[0] for c in cand], bool)
                r = np.where(ok, r, -np.inf) if ok.any() else r
                k = int(np.argmax(r))
                chosen.append(cand[k])
                extra.append({"bon_pick": k})
        else:
            f = cfg.out / f"eval_caro_{seed}.json"
            if not f.exists():
                raise FileNotFoundError(f"sft_lenmatch needs {f.name}: evaluate the caro arm first")
            target = {r["uid"]: r["words"] for r in load_json(f)}
            for t, cand in zip(te, samples):
                w = np.asarray([len(c.split()) for c in cand], float)
                k = int(np.argmin(np.abs(w - target.get(t.uid, np.median(w)))))
                chosen.append(cand[k])
                extra.append({"lenmatch_target_words": target.get(t.uid), "lenmatch_gap":
                              float(w[k] - target.get(t.uid, w[k]))})
        rows = evaluate(pol, sim, te, gen, ex.logger, arm, seed, responses=chosen, extra=extra)
    else:
        rows = evaluate(pol, sim, te, gen, ex.logger, arm, seed)
    dump_json(rows, cfg.out / f"eval_{arm}_{seed}.json")
    return {"n": len(rows)}


REPORT_METRICS = ("outcome", "words", "hygiene_ok", "sentiment", "info_recall", "slot_f1", "slot_recall")


def load_eval_rows(cfg: "Config", arms: Optional[Sequence[str]] = None, prefix: str = "eval"
                   ) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
    rows: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
    for arm in (arms or cfg.arms):
        for sd in cfg.seeds:
            f = cfg.out / f"{prefix}_{arm}_{sd}.json"
            if f.exists():
                rows.setdefault(arm, {})[sd] = load_json(f)
    return rows


def comparison_plan(arms: Sequence[str]) -> List[Tuple[str, str]]:
    """Every arm against SFT, and CARO against every other arm (the controls that decide whether
    the simulator-derived reward adds anything: sentiment-only reward, offline DPO on the same
    corpus, best-of-N reranking, the length-matched SFT control, RL without abstention)."""
    plan = [(a, "sft") for a in arms if a != "sft"]
    if "caro" in arms:
        plan += [("caro", a) for a in arms if a not in ("sft", "caro")]
    return plan


def contrast_family(rows: Dict[str, Dict[int, List[Dict[str, Any]]]], plan: Sequence[Tuple[str, str]],
                    metrics: Sequence[str], seed: int, margin: Optional[float], info_margin: float,
                    n_mc: int) -> Dict[str, Dict[str, Any]]:
    """All planned contrasts with Holm within each metric family and across everything."""
    out: Dict[str, Dict[str, Any]] = {}
    for a, b in plan:
        if a not in rows or b not in rows:
            continue
        for m in metrics:
            c = pooled_paired_contrast(rows[a], rows[b], m, seed=seed,
                                       margin=(margin if m == "outcome" else None),
                                       noninferiority_margin=(info_margin if m in ("info_recall", "slot_f1") else None),
                                       n_mc=n_mc)
            if c.get("n"):
                out[f"{a}_vs_{b}:{m}"] = {**c, "arm": a, "comparator": b, "metric": m}
    keys = list(out)
    for k, adj in zip(keys, holm([out[k]["p_value"] for k in keys])):
        out[k]["p_holm_all"] = adj
    for m in metrics:
        km = [k for k in keys if out[k]["metric"] == m]
        for k, adj in zip(km, holm([out[k]["p_value"] for k in km])):
            out[k]["p_holm"] = adj
    return out


def stage_report(cfg: Config, prefix: str = "eval", outcome_key: str = "outcome") -> Dict[str, Any]:
    """Primary analysis: arm means, paired contrasts, effect sizes, multiplicity and power.

    Design: evaluation contexts are shared by all arms and all seeds, so every contrast is PAIRED
    within (seed, context); DIALOGUES are the resampling unit, pooled over seeds (D2).  Holm is
    applied within each metric family (the primary family is the outcome) and, more
    conservatively, across every reported contrast.  Information retention (info_recall) is
    tested for NON-INFERIORITY, because the objective is to change how information is delivered,
    not to deliver less of it."""
    ex = Experiment(cfg, "8_report" if prefix == "eval" else f"8_report_{prefix}")
    rows = load_eval_rows(cfg, prefix=prefix)
    if not rows:
        raise FileNotFoundError(f"no {prefix} rows; run the corresponding eval stage first")
    if outcome_key != "outcome":
        for per in rows.values():
            for rs in per.values():
                for r in rs:
                    r["outcome"] = r[outcome_key]
    metrics = tuple(m for m in REPORT_METRICS if any(m in rs[0] for per in rows.values() for rs in per.values() if rs))
    summary: Dict[str, Any] = {"arms": {}, "contrasts": {}, "primary_metric": "outcome", "evaluator": prefix,
                               "n_seeds": len(cfg.seeds), "cluster_unit": "dialogue (pooled over seeds)"}
    for arm, per in rows.items():
        allr = [r for rs in per.values() for r in rs]
        y = np.asarray([r["outcome"] for r in allr], float)
        cl = np.asarray([str(r["dialogue_id"]) for r in allr], dtype=object)
        pt, lo, hi = cluster_bootstrap_ci(lambda ix: float(np.mean(y[ix])), cl, 2000, cfg.seed)
        seed_means = [float(np.mean([r["outcome"] for r in rs])) for rs in per.values()]
        ir = np.asarray([r.get("info_recall", float("nan")) for r in allr], float)
        summary["arms"][arm] = {
            "n_seeds": len(per), "n_rows": len(allr), "outcome_mean": pt, "outcome_ci95": [lo, hi],
            "outcome_sd_across_seeds": float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else float("nan"),
            "words": float(np.mean([r["words"] for r in allr])),
            "hygiene": float(np.mean([r["hygiene_ok"] for r in allr])),
            "sentiment": float(np.mean([r["sentiment"] for r in allr])),
            "info_recall": float(np.nanmean(ir)) if np.isfinite(ir).any() else float("nan"),
            "variance_decomposition": variance_decomposition(per, "outcome")}
    base = "sft" if "sft" in rows else sorted(rows)[0]
    # Equivalence margin for null results: 0.2 x the sd of the baseline outcome ACROSS SEEDS within a
    # context (the evaluation noise a single context carries), falling back to 0.2 x the pooled sd.
    b_rows = [r for rs in rows[base].values() for r in rs]
    margin = 0.2 * within_group_sd(np.asarray([r["outcome"] for r in b_rows], float),
                                   np.asarray([r["uid"] for r in b_rows]))
    if not math.isfinite(margin) or margin <= 0:
        margin = 0.2 * float(np.std([r["outcome"] for r in b_rows]))
    plan = comparison_plan([a for a in cfg.arms if a in rows])
    summary["contrasts"] = contrast_family(rows, plan, metrics, cfg.seed, margin,
                                           cfg.noninferiority_info_margin, cfg.n_signflip)
    summary["equivalence_margin_outcome"] = float(margin)
    dump_json(summary, cfg.out / ("report.json" if prefix == "eval" else f"report_{prefix}.json"))
    ex.logger.info("%-16s %8s %18s %10s %8s %8s %8s", "arm", "outcome", "95% CI", "sd(seed)", "words", "hygiene",
                   "info")
    for arm, v in summary["arms"].items():
        ex.logger.info("%-16s %8.4f  [%+.4f,%+.4f] %10.4f %8.1f %8.3f %8.3f", arm, v["outcome_mean"],
                       v["outcome_ci95"][0], v["outcome_ci95"][1], v["outcome_sd_across_seeds"],
                       v["words"], v["hygiene"], v["info_recall"])
    for nm, v in summary["contrasts"].items():
        tail = ""
        if "equivalent_to_zero" in v:
            tail = (f" | TOST vs +-{v['equivalence_margin']:.4f}: "
                    f"{'equivalent to zero' if v['equivalent_to_zero'] else 'not equivalent'}")
        if "noninferior" in v:
            tail = f" | non-inferior at -{v['noninferiority_margin']:.3f}: {v['noninferior']}"
        ex.logger.info("%-40s delta=%+.4f CI95[%+.4f,%+.4f] | d_z=%+.3f dom=%+.3f | p %s holm=%.4f | "
                       "MDE80=%.4f | %d dialogues%s", nm, v["delta"], v["ci95"][0], v["ci95"][1], v["cohens_dz"],
                       v["paired_dominance"], v["p_text"], v.get("p_holm", float("nan")), v["mde80"],
                       v["n_clusters"], tail)
    for arm, v in summary["arms"].items():
        vd = v["variance_decomposition"]
        if math.isfinite(vd.get("icc_seed", float("nan"))):
            ex.logger.info("  %-14s variance decomposition | between-seed %.5f | within-seed %.5f | "
                           "seed ICC %.3f", arm, vd["between_seed_var"], vd["within_seed_var"], vd["icc_seed"])
    return summary


# ---------------------------------------------------------------------------------------------
# v13: statistics, ablations, sensitivity analysis and paper artefacts
# ---------------------------------------------------------------------------------------------


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Non-parametric effect size in [-1, 1]; robust to the heavy tails of outcome differences."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    order = np.argsort(b)
    bs = b[order]
    gt = np.searchsorted(bs, a, side="left")
    ge = np.searchsorted(bs, a, side="right")
    return float((gt.sum() - (b.size * a.size - ge.sum())) / (a.size * b.size))


def _cluster_sums(d: np.ndarray, cluster: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    _, inv = np.unique(np.asarray(cluster), return_inverse=True)
    S = np.bincount(inv, weights=np.asarray(d, float))
    n = np.bincount(inv).astype(float)
    return S, n


def sign_flip_test(d: np.ndarray, cluster: np.ndarray, n_mc: int = 20000, seed: int = 0,
                   exact_max_clusters: int = 16) -> Dict[str, Any]:
    """Cluster-level sign-flip randomisation test of H0: E[d] = 0 for a paired difference.

    v14 (D3).  The statistic is |sum_g s_g S_g| / N with S_g the SUM of the differences in cluster
    g, which is exactly the row mean whose CI is reported; v13 flipped cluster MEANS, whose
    unweighted average is a different estimand when cluster sizes differ.  Under H0 (d symmetric
    about zero within each cluster, clusters independent) every sign vector is equally likely.
    With <= exact_max_clusters clusters all 2^G vectors are enumerated and p is exact; otherwise
    n_mc draws are used and p >= 1/(n_mc+1) by construction (R9)."""
    S, n = _cluster_sums(d, cluster)
    G = S.size
    N = float(n.sum())
    obs = abs(float(S.sum())) / max(N, 1.0)
    if G <= exact_max_clusters:
        signs = 1.0 - 2.0 * ((np.arange(2 ** G)[:, None] >> np.arange(G)[None, :]) & 1)
        stat = np.abs(signs @ S) / N
        p = float(np.mean(stat >= obs - 1e-15))
        return {"p": p, "exact": True, "n_draws": int(2 ** G), "p_text": format_p(p, exact=True)}
    rng = np.random.default_rng(seed)
    cnt = 0
    for s0 in range(0, n_mc, 2000):
        b = min(2000, n_mc - s0)
        sg = rng.choice(np.array([-1.0, 1.0]), size=(b, G))
        cnt += int(np.sum(np.abs(sg @ S) / N >= obs - 1e-15))
    p = float((cnt + 1) / (n_mc + 1))
    return {"p": p, "exact": False, "n_draws": int(n_mc), "at_floor": cnt == 0, "p_text": format_p(p, n_mc)}


def sign_flip_p(d: np.ndarray, cluster: np.ndarray, n_mc: int = 20000, seed: int = 0) -> float:
    return sign_flip_test(d, cluster, n_mc, seed)["p"]


def cluster_robust_se(d: np.ndarray, cluster: np.ndarray) -> float:
    """CR1 cluster-robust standard error of the row mean of d (Liang & Zeger, 1986)."""
    S, n = _cluster_sums(d, cluster)
    G = S.size
    if G < 3:
        return float("nan")
    mu = float(S.sum() / n.sum())
    u = S - n * mu
    return float(math.sqrt(G / (G - 1.0) * float(np.sum(u ** 2))) / n.sum())


def mde_paired(d: np.ndarray, cluster: np.ndarray, power: float = 0.80, alpha: float = 0.05) -> float:
    """Minimum detectable effect of the paired design at the given power: (z_{1-a/2} + z_power) x
    the cluster-robust SE of the SAME row-mean estimator the contrast reports."""
    se = cluster_robust_se(d, cluster)
    if not math.isfinite(se):
        return float("nan")
    z_a = 1.959963985 if abs(alpha - 0.05) < 1e-9 else float(math.sqrt(2) * _erfinv(1 - alpha))
    z_b = 0.8416212336 if abs(power - 0.80) < 1e-9 else float(math.sqrt(2) * _erfinv(2 * power - 1))
    return float((z_a + z_b) * se)


def _erfinv(x: float) -> float:
    a = 0.147
    ln = math.log(max(1e-12, 1 - x * x))
    t = 2 / (math.pi * a) + ln / 2
    return math.copysign(math.sqrt(max(0.0, math.sqrt(t * t - ln / a) - t)), x)


def pooled_paired_contrast(rows_arm: Dict[int, List[Dict[str, Any]]], rows_base: Dict[int, List[Dict[str, Any]]],
                           metric: str, seed: int = 0, n_boot: int = 2000, margin: Optional[float] = None,
                           noninferiority_margin: Optional[float] = None, n_mc: int = 20000) -> Dict[str, Any]:
    """Contrast of one arm against a comparator, pooled over seeds and paired within (seed, context).

    v14 (D2): the resampling unit is the DIALOGUE, pooled over seeds.  v13 used "seed|dialogue",
    which counts one dialogue as n_seeds independent clusters; since the evaluation contexts are
    the same under every seed, that is pseudo-replication and it shrinks CIs and p-values by up
    to sqrt(n_seeds).  Rows with a NaN metric (e.g. info_recall on a turn without salient gold
    tokens) are dropped pairwise.

    Effect sizes: d_z (mean / sd of the paired differences), and the PAIRED dominance
    P(d > 0) - P(d < 0), which is the paired analogue of Cliff's delta.  v13's cliffs_delta compared
    the two marginal distributions and ignored the pairing; it is still reported for continuity.
    margin -> two-sided TOST for "no difference"; noninferiority_margin -> one-sided test that
    the arm is not worse than the comparator by more than the margin (lower CI bound > -margin)."""
    d_all, cl_all, a_all, b_all, per_seed = [], [], [], [], {}
    for sd, ra in rows_arm.items():
        rb = rows_base.get(sd)
        if not rb:
            continue
        ia = {r["uid"]: r for r in ra}
        ib = {r["uid"]: r for r in rb}
        keys = [k for k in sorted(set(ia) & set(ib))
                if ia[k].get(metric) is not None and ib[k].get(metric) is not None
                and math.isfinite(float(ia[k][metric])) and math.isfinite(float(ib[k][metric]))]
        if len(keys) < 20:
            continue
        d = np.asarray([float(ia[k][metric]) - float(ib[k][metric]) for k in keys], float)
        per_seed[sd] = float(np.mean(d))
        d_all.append(d)
        cl_all.append(np.asarray([str(ia[k]["dialogue_id"]) for k in keys], dtype=object))
        a_all.append(np.asarray([float(ia[k][metric]) for k in keys], float))
        b_all.append(np.asarray([float(ib[k][metric]) for k in keys], float))
    if not d_all:
        return {"n": 0, "delta": float("nan")}
    d = np.concatenate(d_all)
    cl = np.concatenate(cl_all)
    a = np.concatenate(a_all)
    b = np.concatenate(b_all)
    pt, lo, hi = cluster_bootstrap_ci(lambda ix: float(np.mean(d[ix])), cl, n_boot, seed, conf=0.95)
    _, lo90, hi90 = cluster_bootstrap_ci(lambda ix: float(np.mean(d[ix])), cl, n_boot, seed + 7, conf=0.90)
    sd_d = float(np.std(d, ddof=1)) if d.size > 1 else float("nan")
    sf = sign_flip_test(d, cl, n_mc=n_mc, seed=seed + 1)
    out = {"n": int(d.size), "n_clusters": int(np.unique(cl).size), "n_seeds": len(per_seed),
           "delta": pt, "ci95": [lo, hi], "ci90": [lo90, hi90], "se_cluster": cluster_robust_se(d, cl),
           "delta_per_seed": per_seed,
           "delta_sd_across_seeds": (float(np.std(list(per_seed.values()), ddof=1))
                                     if len(per_seed) > 1 else float("nan")),
           "p_value": sf["p"], "p_text": sf["p_text"], "p_exact": sf["exact"], "p_draws": sf["n_draws"],
           "cohens_dz": float(np.mean(d) / sd_d) if sd_d and math.isfinite(sd_d) and sd_d > 0 else float("nan"),
           "paired_dominance": float(np.mean(d > 0) - np.mean(d < 0)),
           "cliffs_delta": cliffs_delta(a, b),
           "mde80": mde_paired(d, cl)}
    if margin is not None and math.isfinite(lo90) and math.isfinite(hi90):
        out["equivalence_margin"] = float(margin)
        out["equivalent_to_zero"] = bool(lo90 > -margin and hi90 < margin)
    if noninferiority_margin is not None and math.isfinite(lo90):
        out["noninferiority_margin"] = float(noninferiority_margin)
        out["noninferior"] = bool(lo90 > -noninferiority_margin)
    return out


def variance_decomposition(rows: Dict[int, List[Dict[str, Any]]], metric: str) -> Dict[str, float]:
    """How much of the arm's metric variance sits between seeds versus between contexts."""
    per = {sd: np.asarray([r[metric] for r in rs], float) for sd, rs in rows.items() if rs}
    if len(per) < 2:
        return {"between_seed_var": float("nan"), "within_seed_var": float("nan"), "icc_seed": float("nan")}
    means = np.asarray([v.mean() for v in per.values()])
    within = float(np.mean([np.var(v, ddof=1) for v in per.values()]))
    between = float(np.var(means, ddof=1))
    return {"between_seed_var": between, "within_seed_var": within,
            "icc_seed": float(between / max(between + within, EPS))}


def latex_table(path: Path, caption: str, label: str, header: Sequence[str],
                rows: Sequence[Sequence[Any]], note: str = "") -> None:
    """Write a booktabs table.  Everything a reviewer sees in the log is also written as LaTeX."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def esc(x: Any) -> str:
        t = f"{x:.4f}" if isinstance(x, float) else str(x)
        return t.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")

    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\begin{tabular}{l" + "r" * (len(header) - 1) + "}", r"\toprule",
             " & ".join(esc(h) for h in header) + r" \\", r"\midrule"]
    lines += [" & ".join(esc(c) for c in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines.append(r"\vspace{2pt}{\footnotesize " + note.replace("_", r"\_") + "}")
    lines += [r"\caption{" + caption.replace("_", r"\_") + "}", r"\label{" + label + "}", r"\end{table}", ""]
    path.write_text("\n".join(lines))


class ObservationalLengthControl:
    """The v8 baseline, kept ONLY so the ablation can price it.

    Cross-fitted within-context spline of the outcome on log-length: the estimator this pipeline
    argues against.  It is never used in the pipeline itself -- fitting it here and reporting what
    it does to the padding intervention is what turns "adjusting for length is a bad control" from
    a claim into a measurement."""

    def __init__(self, n_folds: int = 5, n_knots: int = 6):
        self.n_folds = int(n_folds)
        self.n_knots = int(n_knots)
        self.basis: Optional[NaturalCubicBasis] = None
        self.beta: Optional[np.ndarray] = None
        self.diag: Dict[str, Any] = {}

    def fit(self, texts: Sequence[str], y: np.ndarray, gid: np.ndarray, lam: float = 1e-2,
            logger: Optional[logging.Logger] = None) -> "ObservationalLengthControl":
        y = np.asarray(y, float)
        L = loglen(list(texts))
        self.basis = NaturalCubicBasis.from_data(L, self.n_knots)
        B = self.basis.design(L)
        _, inv = np.unique(np.asarray(gid), return_inverse=True)
        # within-context demeaning of both sides: this is exactly the v8 fit
        Bc = B - np.stack([B[inv == g].mean(0) for g in inv], 0)
        yc = y - np.asarray([y[inv == g].mean() for g in inv], float)
        A = Bc.T @ Bc + lam * np.eye(B.shape[1])
        self.beta = np.linalg.solve(A, Bc.T @ yc)
        self.diag = {"kind": "observational", "n": int(y.size), "lam": float(lam),
                     "within_rho_before": within_group_spearman(y, L, np.asarray(gid))["mean"],
                     "within_rho_after": within_group_spearman(self.apply(texts, y), L, np.asarray(gid))["mean"]}
        if logger is not None:
            logger.info("v8 observational length control refitted for the ablation | within-context length rho "
                        "%+.3f -> %+.3f on its own fitting data", self.diag["within_rho_before"],
                        self.diag["within_rho_after"])
        return self

    def apply(self, texts: Sequence[str], y: np.ndarray) -> np.ndarray:
        if self.basis is None or self.beta is None:
            return np.asarray(y, float)
        return np.asarray(y, float) - self.basis.design(loglen(list(texts))) @ self.beta


_CELL_KEYS = ("delta_mean", "delta_ci90", "equivalence_margin", "tost_passes", "tau_corrected",
              "tau_corrected_ci90", "tau_ok", "flip_perturb", "flip_ci90", "flip_ok", "flip_surface_null",
              "passes", "words_added", "n", "n_groups", "variants_per_group")


def _correction_cells(flat, base, padded: Dict[str, Tuple[List[str], List[str], List[str]]], raw, obs, h_int,
                      gid, cluster, cfg: "Config", mode: str, sample: str, simulator: str,
                      logger: logging.Logger) -> List[Dict[str, Any]]:
    """Cross {none, v8 observational, v12 interventional} with the filler banks on ONE set of responses.
    `raw(texts)` returns the outcome WITHOUT any length correction."""
    y0 = raw(base)
    cells = []
    for bank, (longer, longA, longB) in padded.items():
        y1, yA, yB = raw(longer), raw(longA), raw(longB)
        for name, fn in (("none", None), ("v8_observational", obs), ("v12_interventional", h_int)):
            if fn is None and name != "none":
                continue
            if name == "v12_interventional" and not fn.fitted:
                continue
            ap = (lambda tx, yy: yy) if fn is None else (lambda tx, yy, f=fn: f.apply(tx, yy))
            st = intervention_stats(ap(base, y0), ap(longer, y1), ap(base, y0), ap(longA, yA), ap(longB, yB),
                                    gid=gid, cluster=cluster, seed=cfg.seed + 3,
                                    margin_frac=cfg.length_margin_frac, max_tau=cfg.max_tau,
                                    max_excess_flip=cfg.max_excess_flip, max_abs_flip=cfg.max_abs_flip,
                                    mode=mode, bank=bank, level=1,
                                    words_added=float(np.mean([len(b.split()) - len(a.split())
                                                               for a, b in zip(base, longer)])),
                                    logger=logger, label=f"{simulator}/{sample}/correction={name}")
            cells.append({"simulator": simulator, "sample": sample, "correction": name, "bank": bank,
                          **{k: st[k] for k in _CELL_KEYS}})
    return cells


def _pad_sets(base: Sequence[str], banks: Sequence[str], seed: int) -> Dict[str, Tuple[List[str], List[str], List[str]]]:
    out = {}
    for bank in banks:
        aug = LengthInvarianceAugmenter(n_levels=1, seed=seed, bank=bank)
        augm = LengthInvarianceAugmenter(n_levels=1, seed=seed + 4242, bank=bank)
        longer = [aug.lengthen(r, 1) for r in base]
        mp = [augm.matched_pair(r, 1) for r in base]
        out[bank] = (longer, [a for a, _ in mp], [b for _, b in mp])
    return out


def _ablate_one_simulator(cfg: "Config", ex: "Experiment", label: str, prop) -> Dict[str, Any]:
    """All ablation cells for the simulator whose artefacts live in cfg.out."""
    sim_pol = ex.policy_with("simulator")
    sim = ex.simulator(sim_pol)                     # frozen projector / temperature / h(L) / control variate
    h_int = sim.length_control
    sim.length_control = None                        # raw outcomes; corrections are applied afterwards
    turn_of = {t.uid: t for t in ex.turns}
    cells: List[Dict[str, Any]] = []
    replica: Dict[str, Any] = {"available": False}
    arrays: Dict[str, np.ndarray] = {}

    # (a) Replica of the validation gate: the SAME contexts, responses and held-out fillers.
    gf = cfg.out / "validation_gate_sample.npz"
    if gf.exists():
        z = np.load(gf, allow_pickle=False)
        flat = [turn_of[str(u)] for u in z["uid"]]
        base = [str(x) for x in z["base"]]
        gid, cl = z["gid"], z["cluster"]
        padded = {"heldout": ([str(x) for x in z["longer"]], [str(x) for x in z["longA"]], [str(x) for x in z["longB"]])}
        padded.update(_pad_sets(base, [b for b in (ex.sim_train_bank(), FIT_BANK) if b != "heldout"], cfg.seed + 3))
        raw = lambda tx, fl=flat: sim.rollout(fl, list(tx), crn_seed=cfg.seed + 3)
        y0r = raw(base)
        y1r = raw(padded["heldout"][0])
        if h_int is not None and h_int.fitted:
            y0c, y1c = h_int.apply(base, y0r), h_int.apply(padded["heldout"][0], y1r)
            dev_ = float(max(np.max(np.abs(y0c - z["y0"])), np.max(np.abs(y1c - z["y1"]))))
        else:
            dev_ = float(max(np.max(np.abs(y0r - z["y0"])), np.max(np.abs(y1r - z["y1"]))))
        replica = {"available": True, "n_pairs": len(base), "max_abs_deviation_from_gate": dev_,
                   "reproduces_gate": bool(dev_ < 1e-3)}
        (ex.logger.info if replica["reproduces_gate"] else ex.logger.warning)(
            "[%s] gate replica | %d pairs | max |outcome - stored gate outcome| = %.2e -> %s", label, len(base), dev_,
            "reproduces the gate" if replica["reproduces_gate"] else
            "DOES NOT reproduce the gate (4-bit batch-shape noise or changed artefacts; see the deviation)")
        # the v8 observational control is fitted on the dialogue-disjoint FIT half, as v8 did
        obs = _fit_observational(cfg, ex, sim, prop)
        cells += _correction_cells(flat, base, padded, raw, obs, h_int, gid, cl, cfg, sim.mode, "gate_replica",
                                   label, ex.logger)
    else:
        ex.logger.warning("[%s] no validation_gate_sample.npz (written by the v14 validate stage); the gate "
                          "replica is skipped", label)
        obs = _fit_observational(cfg, ex, sim, prop)

    # (b) An independent, larger sample of TEST-half contexts not used by the gate, with the SAME
    #     number of variants per context as the gate (v13 used 2 instead of 4).
    # Contexts: validation-split turns from dialogues used neither to FIT the nuisance components nor
    # by the gate, so the sample is independent of both.
    pool = filter_turns(ex.turns, "valid", require_next=True,
                        limit=cfg.validate_contexts + cfg.projector_fit_contexts,
                        seed=cfg.seed, logger=ex.logger, what="validation pool")
    fitset, _ = split_turns_by_dialogue(pool, cfg.projector_fit_frac, cfg.seed + 5)
    used = {turn_of[str(u)].dialogue_id for u in np.load(gf, allow_pickle=False)["uid"]} if gf.exists() else set()
    used |= {t.dialogue_id for t in fitset}
    rest = [t for t in filter_turns(ex.turns, "valid", require_next=True, seed=cfg.seed + 77)
            if t.dialogue_id not in used]
    te = filter_turns(rest, None, limit=cfg.ablate_contexts, seed=cfg.seed + 78)
    k = int(cfg.validate_variants)
    if len(te) < 30:
        ex.logger.warning("[%s] only %d validation contexts are independent of the fit half and the gate; the "
                          "independent sample is skipped", label, len(te))
        return {"cells": cells, "replica": replica, "arrays": arrays, "observational_diag": obs.diag,
                "interventional_diag": (h_int.diag if h_int is not None else {}), "n_independent_contexts": len(te),
                "variants_per_context": k}
    ex.logger.info("[%s] independent sample | %d contexts from %d dialogues disjoint from the fit half and the gate | "
                   "%d variants per context (as in the gate)", label, len(te), len({t.dialogue_id for t in te}), k)
    base = list(prop.generate([agent_prompt(t) for t in te for _ in range(k)], ex.gen(0.9), seed=4321))
    flat = [t for t in te for _ in range(k)]
    gid = np.repeat(np.arange(len(te)), k)
    cl = np.asarray([t.dialogue_id for t in flat])
    raw = lambda tx, fl=flat: sim.rollout(fl, list(tx), crn_seed=4321)
    padded = _pad_sets(base, list(dict.fromkeys([ex.sim_train_bank(), FIT_BANK, "heldout"])), 911)
    cells += _correction_cells(flat, base, padded, raw, obs, h_int, gid, cl, cfg, sim.mode, "independent", label,
                               ex.logger)
    arrays.update({"y0": raw(base), "gid": gid, "cluster": np.asarray([str(c) for c in cl])})
    for bank, (longer, longA, longB) in padded.items():
        arrays[f"y1_{bank}"], arrays[f"yA_{bank}"], arrays[f"yB_{bank}"] = raw(longer), raw(longA), raw(longB)
    return {"cells": cells, "replica": replica, "arrays": arrays, "observational_diag": obs.diag,
            "interventional_diag": (h_int.diag if h_int is not None else {}), "n_independent_contexts": len(te),
            "variants_per_context": k}


def _fit_observational(cfg: "Config", ex: "Experiment", sim: "UserSimulator", prop) -> "ObservationalLengthControl":
    pool = filter_turns(ex.turns, "valid", require_next=True,
                        limit=cfg.validate_contexts + cfg.projector_fit_contexts, seed=cfg.seed)
    fitset, _ = split_turns_by_dialogue(pool, cfg.projector_fit_frac, cfg.seed + 5)
    fitset = fitset[: max(60, cfg.ablate_contexts // 2)]
    k = int(cfg.validate_variants)
    bf = list(prop.generate([agent_prompt(t) for t in fitset for _ in range(k)], ex.gen(0.9), seed=4322))
    ff = [t for t in fitset for _ in range(k)]
    yf = sim.rollout(ff, bf, crn_seed=4322)
    return ObservationalLengthControl().fit(bf, yf, np.repeat(np.arange(len(fitset)), k), logger=ex.logger)


def stage_ablate_simulator(cfg: Config) -> Dict[str, Any]:
    """Length correction x filler bank (x simulator training regime), responses held fixed per sample.

    v14 (R6) changes.  (i) A GATE REPLICA re-scores the exact responses and fillers of the validation
    gate, so the ablation's "interventional / held-out" cell must reproduce the gate; v13 drew a new
    sample with 2 variants per context and reported tau 0.49 against the gate's 0.21 without any
    way to tell sampling variability from a real difference.  (ii) An INDEPENDENT sample of further
    gate-half contexts with the gate's variants-per-context, reported with tau and flip CIs; if it
    fails where the gate passed, the gate passed on a favourable sample and the paper must say so.
    Note that the variants-per-context count alone does not bias the denominator: the mean of
    per-context unbiased variances is unbiased for any group size >= 2 (it changes only its
    precision).  (iii) Optionally (--ablate-sim-noreg) a simulator retrained WITHOUT the invariance
    regulariser is put through the same cells.  v13's "none" and "interventional" cells were
    almost identical, which suggests the regulariser, not h(L), supplies the invariance; only
    this retrained control can attribute it."""
    ex = Experiment(cfg, "9_ablate_simulator")
    prop = ex.policy_with("sft_policy")
    main = _ablate_one_simulator(cfg, ex, "main", prop)
    cells = list(main["cells"])
    out: Dict[str, Any] = {"cells": cells, "replica": main["replica"],
                           "n_independent_contexts": main["n_independent_contexts"],
                           "variants_per_context": main["variants_per_context"],
                           "observational_diag": main["observational_diag"],
                           "interventional_diag": main["interventional_diag"]}
    if main["arrays"]:
        np.savez_compressed(cfg.out / "ablation_simulator_arrays.npz", **main["arrays"])
    if cfg.ablate_sim_noreg:
        import dataclasses
        c2 = dataclasses.replace(cfg, out=cfg.out / "ablation_noreg", strict=False, ablate_sim_noreg=False,
                                 sim_sft=dataclasses.replace(cfg.sim_sft, invariance_coef=0.0))
        c2.out.mkdir(parents=True, exist_ok=True)
        for name in ("turns.json",):
            if not (c2.out / name).exists():
                (c2.out / name).write_text((cfg.out / name).read_text())
        link = c2.out / "sft_policy"
        if not link.exists():
            link.symlink_to((cfg.out / "sft_policy").resolve(), target_is_directory=True)
        if not (c2.out / "simulator_validation.json").exists():
            stage_simulator(c2)
            stage_validate(c2)
        ex2 = Experiment(c2, "9_ablate_simulator_noreg")
        nr = _ablate_one_simulator(c2, ex2, "no_invariance_regulariser", prop)
        cells += nr["cells"]
        out["noreg"] = {"replica": nr["replica"],
                        "validation": {k: v for k, v in load_json(c2.out / "simulator_validation.json").items()
                                       if k in ("passes", "failures", "human_label_rho", "paired_intervention")}}
    dump_json(out, cfg.out / "ablation_simulator.json")
    ex.logger.info("%-26s %-13s %-20s %-8s %9s %9s %16s %16s %5s", "simulator", "sample", "correction", "bank",
                   "delta", "margin", "tau [CI90]", "flip [CI90]", "TOST")
    for c in cells:
        ex.logger.info("%-26s %-13s %-20s %-8s %+9.5f %9.5f %.3f[%.2f,%.2f] %.3f[%.2f,%.2f] %5s", c["simulator"],
                       c["sample"], c["correction"], c["bank"], c["delta_mean"], c["equivalence_margin"],
                       c["tau_corrected"], *c["tau_corrected_ci90"], c["flip_perturb"], *c["flip_ci90"],
                       "PASS" if c["tost_passes"] else "FAIL")
    return out


def _per_group_accuracy(rm: "RewardModel", groups: Sequence[ResponseGroup]) -> np.ndarray:
    return np.asarray([within_group_pairwise_accuracy([rm.score(g.feats)["reward"]], [g.outcomes], [g.ok])[0]
                       for g in groups], float)


def stage_ablate_reward(cfg: Config) -> Dict[str, Any]:
    """Reward-stage ablations on the frozen corpus (CPU only).

    Every variant sees the same splits, features and labels; one component is removed at a time.
    v14 (R7):
      * "no abstention" is gone.  The abstention threshold only sets the GRPO sample weight, so on
        the corpus that row was identical to "full" by construction and could not evaluate anything.
        Abstention is ablated where it acts, as the RL arm caro_no_abstain.
      * "no pessimism" (kappa = 0) replaces it: the ensemble-sd penalty DOES change the ranking.
      * Each variant is refitted with cfg.ablate_fit_seeds initialisation seeds; per-context accuracy
        is averaged over the refits before the paired test, and the between-refit sd is reported,
        so a difference is not an artefact of one initialisation.
      * "No effect" is claimed only through a TOST: the 90% CI of the accuracy difference must sit
        inside +-ablate_accuracy_margin.  A non-significant p is not evidence of no effect."""
    ex = Experiment(cfg, "9_ablate_reward")
    groups = load_corpus(cfg.out / "corpus.npz")
    train, dev, calib, test = reward_splits(groups, cfg.seed)
    X = np.concatenate([g.feats for g in train], 0)
    std = Standardizer().fit(X)
    sel = load_json(cfg.out / "reward.json").get("selected", {}) if (cfg.out / "reward.json").exists() else {}
    clp0 = float(sel.get("clp", 1.0))
    mu0 = float(sel.get("length_cal", 0.0))
    base = asdict(cfg.reward)
    variants = {
        "full": {"clp": clp0, "length_cal": mu0},
        "no counterfactual pairing": {"clp": 0.0, "length_cal": mu0},
        "no length calibration": {"clp": clp0, "length_cal": 0.0},
        "neither (plain PL ranker)": {"clp": 0.0, "length_cal": 0.0},
        "no pessimism (kappa=0)": {"clp": clp0, "length_cal": mu0, "kappa": 0.0},
        "single model (no ensemble)": {"clp": clp0, "length_cal": mu0, "n_ensemble": 1},
    }
    fit_seeds = [cfg.seed + 1000 * k for k in range(max(1, cfg.ablate_fit_seeds))]
    cl = np.asarray([g.dialogue_id for g in test], dtype=object)
    res: Dict[str, Any] = {}
    acc_vec: Dict[str, np.ndarray] = {}
    for name, over in variants.items():
        rc = RewardConfig(**{**base, **over})
        per_fit, pads, lexs, ancs = [], [], [], []
        for fs in fit_seeds:
            rm = RewardModel(X.shape[1], rc, std)
            rm.fit(train, dev, calib, ex.logger, seed=fs)
            per_fit.append(_per_group_accuracy(rm, test))
            pads.append(reward_padding_test(rm, test, cfg.length_margin_frac, cfg.seed + 5, None))
            lexs.append(reward_length_excess(rm, test, cfg.seed + 7, None, n_boot=300))
            ancs.append(reward_gold_anchor(rm, test, cfg.seed + 11, n_boot=300))
        P = np.vstack(per_fit)
        per_group = np.nanmean(P, 0)
        acc_vec[name] = per_group
        ok = np.isfinite(per_group)
        pt, lo, hi = cluster_bootstrap_ci(lambda ix: float(np.mean(per_group[ok][ix])), cl[ok], 800, cfg.seed)
        fit_acc = [float(np.nanmean(r)) for r in P]
        res[name] = {"accuracy": pt, "accuracy_ci95": [lo, hi],
                     "accuracy_sd_across_refits": float(np.std(fit_acc, ddof=1)) if len(fit_acc) > 1 else float("nan"),
                     "n_refits": len(fit_seeds),
                     "padding_delta": float(np.mean([p.get("delta_mean", float("nan")) for p in pads])),
                     "padding_tost_all_refits": bool(all(p.get("tost_passes", False) for p in pads)),
                     "length_rho": float(np.mean([l["rho_reward"] for l in lexs])),
                     "label_length_rho": float(np.mean([l["rho_label"] for l in lexs])),
                     "length_excess": float(np.mean([l["excess"] for l in lexs])),
                     "gold_anchor": float(np.mean([a["rho"] for a in ancs])),
                     "gold_anchor_ci95_first_refit": ancs[0]["ci95"],
                     "gold_anchor_sd_across_refits": (float(np.std([a["rho"] for a in ancs], ddof=1))
                                                      if len(ancs) > 1 else float("nan")),
                     **{k: getattr(rc, k) for k in ("clp", "length_cal", "kappa", "n_ensemble")}}
    names = [n for n in res if n != "full"]
    pvals = []
    for n in names:
        a, b = acc_vec[n], acc_vec["full"]
        m = np.isfinite(a) & np.isfinite(b)
        d = a[m] - b[m]
        pt, lo, hi = cluster_bootstrap_ci(lambda ix: float(np.mean(d[ix])), cl[m], 800, cfg.seed + 2)
        _, lo90, hi90 = cluster_bootstrap_ci(lambda ix: float(np.mean(d[ix])), cl[m], 800, cfg.seed + 4, conf=0.90)
        sf = sign_flip_test(d, cl[m], n_mc=cfg.n_signflip, seed=cfg.seed + 3)
        res[n].update({"delta_accuracy_vs_full": pt, "delta_ci95": [lo, hi], "delta_ci90": [lo90, hi90],
                       "p_value": sf["p"], "p_text": sf["p_text"],
                       "equivalent_to_full": bool(lo90 > -cfg.ablate_accuracy_margin and hi90 < cfg.ablate_accuracy_margin),
                       "equivalence_margin": cfg.ablate_accuracy_margin})
        pvals.append(sf["p"])
    for n, adj in zip(names, holm(pvals)):
        res[n]["p_holm"] = adj
    dump_json(res, cfg.out / "ablation_reward.json")
    ex.logger.info("%-28s %8s %8s %8s %6s %10s %10s %8s", "reward variant", "acc", "d(acc)", "p holm", "TOST",
                   "pad delta", "len excess", "anchor")
    for n, v in res.items():
        ex.logger.info("%-28s %8.4f %8s %8s %6s %+10.5f %+10.3f %+8.3f", n, v["accuracy"],
                       ("--" if n == "full" else f"{v['delta_accuracy_vs_full']:+.4f}"),
                       ("--" if n == "full" else f"{v['p_holm']:.4f}"),
                       ("--" if n == "full" else ("equiv" if v["equivalent_to_full"] else "no")),
                       v["padding_delta"], v["length_excess"], v["gold_anchor"])
    return res


def stage_sensitivity(cfg: Config) -> Dict[str, Any]:
    """Would the conclusions change under different analysis choices?

    Three families, all cheap: (i) gate thresholds re-applied to the stored intervention vectors;
    (ii) panel temperature re-applied to the cached calibration log-probabilities; (iii) reward
    regularisation weights re-fitted on the frozen corpus.  Nothing here is allowed to change the
    pre-registered values -- it reports how far they would have to move to flip a verdict."""
    ex = Experiment(cfg, "9_sensitivity")
    out: Dict[str, Any] = {}
    silent = logging.getLogger("caro.silent")
    silent.handlers[:] = [logging.NullHandler()]
    silent.propagate = False

    f = cfg.out / "ablation_simulator_arrays.npz"
    if f.exists():
        z = np.load(f, allow_pickle=False)
        gid = z["gid"]
        cl = z["cluster"]
        rows = []
        for bank in FILLER_BANKS:
            if f"y1_{bank}" not in z.files:
                continue
            for mf in (0.10, 0.15, 0.20, 0.25):
                # tau, flip and their CIs do not depend on the thresholds: compute once per margin
                st = intervention_stats(z["y0"], z[f"y1_{bank}"], z["y0"], z[f"yA_{bank}"],
                                        z[f"yB_{bank}"], gid=gid, cluster=cl, seed=cfg.seed,
                                        margin_frac=mf, max_tau=cfg.max_tau, max_abs_flip=cfg.max_abs_flip,
                                        max_excess_flip=cfg.max_excess_flip, n_boot=400,
                                        mode="expected", bank=bank, logger=silent)
                thi = st["tau_corrected_ci90"][1]
                for mt in (0.25, 0.35, 0.50):
                    for mfl in (0.08, 0.10, 0.12, 0.15):
                        tau_ok = (not math.isfinite(thi)) or thi <= mt
                        flip_ok = (not math.isfinite(st["flip_perturb"])) or st["flip_perturb"] <= mfl
                        rows.append({"bank": bank, "margin_frac": mf, "max_tau": mt, "max_abs_flip": mfl,
                                     "tost": st["tost_passes"], "tau_ok": tau_ok, "flip_ok": flip_ok,
                                     "passes": bool(st["tost_passes"] and tau_ok and flip_ok),
                                     "tau": st["tau_corrected"], "tau_hi": thi, "flip": st["flip_perturb"]})
        out["gate_thresholds"] = rows
        ok = [r for r in rows if r["bank"] == cfg.gate_bank]
        if ok:
            frac = float(np.mean([r["passes"] for r in ok]))
            ex.logger.info("gate-threshold sensitivity | %d threshold combinations on the %s bank | the gate "
                           "passes in %.0f%% of them", len(ok), cfg.gate_bank, 100 * frac)
            tight = [r for r in ok if not r["passes"]]
            if tight:
                ex.logger.info("  the verdict first flips at margin_frac=%.2f / max_tau=%.2f / max_abs_flip=%.2f",
                               max(r["margin_frac"] for r in tight), min(r["max_tau"] for r in tight),
                               min(r["max_abs_flip"] for r in tight))
    else:
        ex.logger.warning("no ablation_simulator_arrays.npz: run the ablate-simulator stage for the "
                          "gate-threshold sensitivity analysis")

    c = cfg.out / "panel_cal_cache.npz"
    p = cfg.out / "outcome_panel.json"
    if c.exists() and p.exists():
        z = np.load(c, allow_pickle=False)
        pan = OutcomePanel.load(load_json(p))
        lp, target = z["lp"], z["target"]
        z0 = lp - lp.mean(0, keepdims=True)
        rows = []
        for T in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
            w = np.exp((z0 / T) - (z0 / T).max(1, keepdims=True))
            w = w / np.maximum(w.sum(1, keepdims=True), EPS)
            rows.append({"T": T, "rho": float(spearman(w @ pan.sent, target)),
                         "effective_panel_size": float(np.mean(1.0 / np.maximum((w ** 2).sum(1), EPS)))})
        out["panel_temperature"] = {"selected": pan.temperature, "panel_size": len(pan.texts), "grid": rows}
        ex.logger.info("panel-temperature sensitivity | selected T=%.3g | rho by T: %s", pan.temperature,
                       ", ".join(f"{r['T']:g}:{r['rho']:.3f}" for r in rows))

    if (cfg.out / "corpus.npz").exists():
        # v14 (R5): the grid always contains the DEV-selected cell (v13's did not: it selected
        # clp=3 / length_cal=1 and then reported a grid without it); every cell reports TEST
        # accuracy (v13 reported calibration-split accuracy here but TEST accuracy in the ablation
        # table) and the gate criteria it fails.  The anchor is reported, never gated on here.
        groups = load_corpus(cfg.out / "corpus.npz")
        train, dev, calib, test = reward_splits(groups, cfg.seed)
        X = np.concatenate([g.feats for g in train], 0)
        std = Standardizer().fit(X)
        sel = load_json(cfg.out / "reward.json").get("selected", {}) if (cfg.out / "reward.json").exists() else {}
        if cfg.sens_reward_grid == "full":
            cells = [(c, m) for m in cfg.reward.length_cal_grid for c in (0.0,) + tuple(cfg.reward.clp_grid)]
        else:
            cells = [(c, m) for c in (0.0, 1.0, 10.0) for m in (0.0, 3.0, 30.0, 300.0)]
        if sel and (float(sel["clp"]), float(sel["length_cal"])) not in cells:
            cells.append((float(sel["clp"]), float(sel["length_cal"])))
        grid = []
        for clp, mu in cells:
            rc = RewardConfig(**{**asdict(cfg.reward), "clp": clp, "length_cal": mu})
            rm = RewardModel(X.shape[1], rc, std)
            rm.fit(train, dev, calib, ex.logger, seed=cfg.seed)
            acc, _ = within_group_pairwise_accuracy([rm.score(g.feats)["reward"] for g in test],
                                                    [g.outcomes for g in test], [g.ok for g in test])
            pad = reward_padding_test(rm, test, cfg.length_margin_frac, cfg.seed + 5, None)
            lex = reward_length_excess(rm, test, cfg.seed + 7, None, n_boot=300)
            anc = reward_gold_anchor(rm, test, cfg.seed + 11, n_boot=300)
            e_lo, e_hi = lex["excess_ci90"]
            why = []
            if not (math.isfinite(acc) and acc >= 0.55):
                why.append("accuracy")
            if not pad.get("tost_passes", False):
                why.append("padding")
            if (math.isfinite(e_lo) and math.isfinite(e_hi) and e_lo * e_hi > 0
                    and min(abs(e_lo), abs(e_hi)) > cfg.reward_max_excess_length_rho):
                why.append("length excess")
            grid.append({"clp": clp, "length_cal": mu, "test_accuracy": float(acc),
                         "padding_delta": pad.get("delta_mean", float("nan")),
                         "padding_tost": pad.get("tost_passes", None), "length_excess": lex["excess"],
                         "length_excess_ci90": lex["excess_ci90"], "gold_anchor": anc["rho"],
                         "selected": bool(sel and clp == float(sel["clp"]) and mu == float(sel["length_cal"])),
                         "gate_would_pass": not why, "fails_on": why})
        out["reward_regularisation"] = grid
        n_ok = int(sum(g["gate_would_pass"] for g in grid))
        ex.logger.info("reward-regularisation sensitivity | %d cells (%s grid, selected cell included) | gate "
                       "holds in %d of them | failing criteria: %s", len(grid), cfg.sens_reward_grid, n_ok,
                       {k: sum(k in g["fails_on"] for g in grid) for k in ("accuracy", "padding", "length excess")})

    hist = []
    for arm in cfg.arms:
        for sd in cfg.seeds:
            g = cfg.out / f"grpo_{arm}_{sd}.json"
            if g.exists():
                h = load_json(g)
                if h.get("history"):
                    last = h["history"][-1]
                    hist.append({"arm": arm, "seed": sd, "steps": h.get("steps_run", len(h["history"])),
                                 "aborted": h.get("aborted", False), "final_kl": last.get("kl"),
                                 "final_len": last.get("len"), "baseline_len": h.get("baseline_length"),
                                 "final_hygiene": last.get("hygiene"),
                                 "gated_fraction": float(np.mean([r.get("gated", 0) / max(1, r.get("groups", 1))
                                                                  for r in h["history"]]))})
    if hist:
        out["rl_runs"] = hist
        ex.logger.info("RL drift summary | %s", "; ".join(
            f"{h['arm']}/s{h['seed']}: len {h['baseline_len']:.1f}->{h['final_len']:.1f}, kl {h['final_kl']:.4f}, "
            f"{100 * h['gated_fraction']:.0f}% groups gated" for h in hist
            if h["final_len"] is not None and h["baseline_len"] is not None))
    dump_json(out, cfg.out / "sensitivity.json")
    return out


def _paired_arrays(rows_a: Dict[int, List[Dict[str, Any]]], rows_b: Dict[int, List[Dict[str, Any]]]):
    dy, dw, wb, cl = [], [], [], []
    for sd, ra in rows_a.items():
        rb = rows_b.get(sd)
        if not rb:
            continue
        ib = {r["uid"]: r for r in rb}
        for r in ra:
            q = ib.get(r["uid"])
            if q is None:
                continue
            dy.append(r["outcome"] - q["outcome"])
            dw.append(r["words"] - q["words"])
            wb.append(q["words"])
            cl.append(str(r["dialogue_id"]))
    return np.asarray(dy, float), np.asarray(dw, float), np.asarray(wb, float), np.asarray(cl, dtype=object)


def length_adjusted_contrast(rows_a, rows_b, seed: int = 0, n_boot: int = 2000, match_words: int = 2,
                             n_mc: int = 20000) -> Dict[str, Any]:
    """Is an outcome gain explained by the arm writing shorter (or longer) responses?  (R12)

    Three views of the same paired rows, from weakest to strongest:
      * regression: d_y = alpha + beta * d_words over (seed, context) pairs; alpha is the contrast
        at equal length.  d_words is POST-treatment (the arm chose it), so alpha is a descriptive
        decomposition under a linearity assumption, not a causal direct effect.
      * matching: the mean d_y over pairs whose lengths differ by at most `match_words` words, with a
        cluster bootstrap CI and a sign-flip p.  Selection into this stratum is also post-treatment.
      * strata of d_words and of the comparator's length (tertiles), to show the shape.
    The design-based answer is the sft_lenmatch arm, which is reported by the main report."""
    dy, dw, wb, cl = _paired_arrays(rows_a, rows_b)
    if dy.size < 30:
        return {"n": int(dy.size)}

    def ols(ix):
        X = np.stack([np.ones(ix.size), dw[ix]], 1)
        return np.linalg.lstsq(X, dy[ix], rcond=None)[0]

    a_pt, a_lo, a_hi = cluster_bootstrap_ci(lambda ix: float(ols(ix)[0]), cl, n_boot, seed)
    b_pt, b_lo, b_hi = cluster_bootstrap_ci(lambda ix: float(ols(ix)[1]), cl, n_boot, seed + 1)
    m = np.abs(dw) <= match_words
    matched: Dict[str, Any] = {"n": int(m.sum()), "n_clusters": int(np.unique(cl[m]).size) if m.any() else 0}
    if m.sum() >= 30:
        dm, cm = dy[m], cl[m]
        pt, lo, hi = cluster_bootstrap_ci(lambda ix: float(np.mean(dm[ix])), cm, n_boot, seed + 2)
        sf = sign_flip_test(dm, cm, n_mc=n_mc, seed=seed + 3)
        matched.update({"delta": pt, "ci95": [lo, hi], "p_value": sf["p"], "p_text": sf["p_text"]})
    bins = [(-np.inf, -10), (-10, -5), (-5, -match_words - 1e-9), (-match_words, match_words),
            (match_words + 1e-9, 5), (5, 10), (10, np.inf)]
    strata = []
    for lo_, hi_ in bins:
        mm = (dw >= lo_) & (dw <= hi_)
        strata.append({"d_words": [lo_, hi_], "n": int(mm.sum()),
                       "mean_delta": float(dy[mm].mean()) if mm.any() else float("nan")})
    qs = np.quantile(wb, [1 / 3, 2 / 3])
    terc = []
    for k, (lo_, hi_) in enumerate([(-np.inf, qs[0]), (qs[0], qs[1]), (qs[1], np.inf)]):
        mm = (wb > lo_) & (wb <= hi_) if k else (wb <= hi_)
        terc.append({"comparator_words": [float(lo_), float(hi_)], "n": int(mm.sum()),
                     "mean_delta": float(dy[mm].mean()) if mm.any() else float("nan"),
                     "mean_d_words": float(dw[mm].mean()) if mm.any() else float("nan")})
    return {"n": int(dy.size), "n_clusters": int(np.unique(cl).size), "raw_delta": float(dy.mean()),
            "mean_d_words": float(dw.mean()), "alpha_equal_length": a_pt, "alpha_ci95": [a_lo, a_hi],
            "beta_per_word": b_pt, "beta_ci95": [b_lo, b_hi],
            "share_explained_by_length": (float(1.0 - a_pt / dy.mean()) if abs(dy.mean()) > EPS else float("nan")),
            "matched": matched, "strata_d_words": strata, "strata_comparator_length": terc}


def stage_length_analysis(cfg: Config) -> Dict[str, Any]:
    """R12: length-stratified and length-adjusted versions of every contrast against SFT, the CARO vs
    sft_lenmatch design-based control, and the reward's own length preference for context."""
    ex = Experiment(cfg, "9_length_analysis")
    rows = load_eval_rows(cfg)
    out: Dict[str, Any] = {"contrasts": {}}
    for a in rows:
        for b in ("sft", "sft_lenmatch"):
            if a == b or b not in rows or (b == "sft_lenmatch" and a != "caro"):
                continue
            r = length_adjusted_contrast(rows[a], rows[b], seed=cfg.seed, n_mc=cfg.n_signflip)
            out["contrasts"][f"{a}_vs_{b}"] = r
            if r.get("n", 0) >= 30:
                mt = r["matched"]
                ex.logger.info("%-26s raw %+.4f (d words %+.1f) | at equal length %+.4f CI95[%+.4f,%+.4f] | "
                               "%+.5f per word | matched |dw|<=2: %s", f"{a}_vs_{b}", r["raw_delta"],
                               r["mean_d_words"], r["alpha_equal_length"], *r["alpha_ci95"], r["beta_per_word"],
                               (f"{mt['delta']:+.4f} CI95[{mt['ci95'][0]:+.4f},{mt['ci95'][1]:+.4f}] n={mt['n']} p {mt['p_text']}"
                                if "delta" in mt else f"n={mt['n']} (too few)"))
    if "caro" in rows and "sft_lenmatch" in rows:
        gaps = [abs(r.get("lenmatch_gap", 0.0)) for rs in rows["sft_lenmatch"].values() for r in rs]
        out["lenmatch_quality"] = {"mean_abs_word_gap": float(np.mean(gaps)), "share_within_2_words":
                                   float(np.mean(np.asarray(gaps) <= 2))}
        ex.logger.info("sft_lenmatch quality | mean |word gap| to CARO %.2f | %.1f%% within 2 words",
                       out["lenmatch_quality"]["mean_abs_word_gap"], 100 * out["lenmatch_quality"]["share_within_2_words"])
    if (cfg.out / "reward.json").exists():
        g = load_json(cfg.out / "reward.json").get("gate", {}).get("length_excess", {})
        out["reward_length_preference"] = g
        if g:
            ex.logger.info("reward length preference (TEST, within context) | reward rho %+.3f vs labels %+.3f | "
                           "excess %+.3f CI90[%+.3f,%+.3f]", g["rho_reward"], g["rho_label"], g["excess"],
                           *g["excess_ci90"])
    dump_json(out, cfg.out / "length_analysis.json")
    return out


def external_customer_prompts(judge, turns: Sequence[Turn], responses: Sequence[str]) -> List[str]:
    """Prompt for the out-of-family customer.  Uses the judge's chat template when it has one, so an
    instruct model is queried the way it was trained; the stub uses the plain customer prompt."""
    tok = getattr(judge, "tok", None)
    if tok is None or not getattr(tok, "chat_template", None):
        return [customer_prompt(t, r) for t, r in zip(turns, responses)]
    out = []
    for t, r in zip(turns, responses):
        conv = f"{t.history}\nCustomer: {t.user_text}\nAgent: {norm_text(r)}".strip()
        msgs = [{"role": "system", "content": CUSTOMER_SYSTEM},
                {"role": "user", "content": f"Conversation so far:\n{conv}\n\nWrite only the customer's next "
                                            f"turn, in one or two sentences."}]
        out.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    return out


def stage_eval_external(cfg: Config) -> Dict[str, Any]:
    """R2: re-score every evaluated response with an evaluator that shares NOTHING with training.

    The customer is a different LLM family (cfg.external_model, no simulator adapter, no EmoWOZ
    fine-tuning) and its replies are labelled by an emotion classifier (cfg.external_emotion_model)
    that no stage used; the outcome is the same shape as the training outcome (mean reply valence
    minus the valence of the customer's current turn).  This removes the "the reward model is a
    model of the evaluator" objection for automatic evaluation.  It does not replace the human
    study: both evaluators are still machines, and agreement between them is only evidence that the
    effect is not specific to one simulator."""
    ex = Experiment(cfg, "7_eval_external")
    rows = load_eval_rows(cfg)
    if not rows:
        raise FileNotFoundError("no evaluation rows; run the eval stage first")
    if cfg.stub:
        judge, emo = StubPolicy(ex.logger, cfg.seed + 99), StubEmotion()
    else:
        judge = Policy(cfg.external_model, cfg.device, cfg.models_dir, ex.logger, load_4bit=cfg.load_4bit,
                       gen_batch=cfg.gen_batch, score_batch=cfg.score_batch, with_lora=False)
        emo = EmotionValenceScorer(cfg.external_emotion_model, cfg.device, cfg.models_dir, ex.logger)
    turn_of = {t.uid: t for t in ex.turns}
    gen = GenConfig(max_new_tokens=40, min_new_tokens=4, temperature=0.9)
    K = max(1, int(cfg.external_rollouts))
    for arm, per in rows.items():
        for sd, rs in per.items():
            f = cfg.out / f"ext_{arm}_{sd}.json"
            if f.exists():
                continue
            ts = [turn_of[r["uid"]] for r in rs]
            prompts = external_customer_prompts(judge, ts, [r["response"] for r in rs])
            flat = [p for p in prompts for _ in range(K)]
            # common random numbers: the same seed stream for every arm, because rows are in the
            # same (context) order in every arm's eval file
            replies = judge.generate(flat, gen, seed=777000 + 7919 * sd)
            v = np.asarray(emo(replies), float).reshape(len(rs), K).mean(1)
            cur = np.asarray(emo([t.user_text for t in ts]), float)
            agent_v = np.asarray(emo([r["response"] for r in rs]), float)
            out = []
            for r, vv, cc, av, rep in zip(rs, v, cur, agent_v, np.asarray(replies, dtype=object).reshape(len(rs), K)):
                out.append({**r, "ext_outcome": float(vv - cc), "ext_agent_valence": float(av),
                            "ext_reply_example": str(rep[0])})
            dump_json(out, f)
            ex.logger.info("external eval %s seed=%d | %d rows | ext outcome %.4f | simulator outcome %.4f | "
                           "rank agreement (Spearman, rows) %+.3f", arm, sd, len(out),
                           float(np.mean([o["ext_outcome"] for o in out])), float(np.mean([o["outcome"] for o in out])),
                           spearman(np.asarray([o["ext_outcome"] for o in out]), np.asarray([o["outcome"] for o in out])))
    return stage_report(cfg, prefix="ext", outcome_key="ext_outcome")


HUMAN_QUESTIONS = {
    "q_satisfaction": "Imagine you are this customer. After which reply would you feel better about how the "
                      "conversation is going?",
    "q_information": "Which reply gives the customer more of the information they need to get their task done?",
    "q_appropriate": "Which reply responds more appropriately to how the customer seems to feel (acknowledges it "
                     "where needed, without sounding insincere or over-the-top)?",
}


def binomial_n(p1: float, p0: float = 0.5, alpha: float = 0.05, power: float = 0.80) -> int:
    """Non-tie judgements needed for a two-sided test of a win rate p1 against p0."""
    za, zb = 1.959963985, 0.8416212336
    if abs(power - 0.80) > 1e-9:
        zb = float(math.sqrt(2) * _erfinv(2 * power - 1))
    return int(math.ceil(((za * math.sqrt(p0 * (1 - p0)) + zb * math.sqrt(p1 * (1 - p1))) / (p1 - p0)) ** 2))


def _context_text(t: Turn) -> str:
    return (t.history + "\n" if t.history else "") + f"Customer: {t.user_text}"


def stage_human_export(cfg: Config) -> Dict[str, Any]:
    """R3: a blinded, randomised pairwise annotation packet.

    Three item kinds, shuffled together so annotators cannot tell them apart:
      arm_pair  the responses of two arms to the SAME context (same eval seed): the human test of the
                main result, and of information retention (q_information, non-inferiority);
      validity  two SFT samples to the SAME corpus TEST context, one preferred by the simulator: the
                within-context validity test that EmoWOZ cannot provide (the simulator's between-context
                anchor says nothing about ranking alternatives to one context);
      catch     the real agent turn against a rude catch response: attention check (pre-registered
                exclusion: an annotator below 80% correct on catch items is dropped).
    A share of items is flagged for double annotation, for inter-annotator agreement.  Nothing in the
    packet reveals the arm, the simulator score or the item kind; the key is written separately."""
    ex = Experiment(cfg, "10_human_export")
    rng = np.random.default_rng(cfg.seed + 2024)
    d = cfg.out / "human_study"
    d.mkdir(parents=True, exist_ok=True)
    turn_of = {t.uid: t for t in ex.turns}
    rows = load_eval_rows(cfg)
    sd0 = cfg.seeds[0]
    items: List[Dict[str, Any]] = []
    for comp in cfg.human_comparisons:
        a, b = comp.split(":")
        if a not in rows or b not in rows or sd0 not in rows[a] or sd0 not in rows[b]:
            ex.logger.warning("human export | comparison %s skipped: eval rows missing", comp)
            continue
        ia = {r["uid"]: r for r in rows[a][sd0]}
        ib = {r["uid"]: r for r in rows[b][sd0]}
        uids = sorted(u for u in set(ia) & set(ib) if norm_text(ia[u]["response"]) != norm_text(ib[u]["response"]))
        # one context per dialogue first, so the items are as independent as the pool allows
        by_d: Dict[str, List[str]] = {}
        for u in uids:
            by_d.setdefault(ia[u]["dialogue_id"], []).append(u)
        pick = [by_d[k][int(rng.integers(len(by_d[k])))] for k in rng.permutation(sorted(by_d))]
        pick = pick[: cfg.human_contexts]
        for u in pick:
            items.append({"kind": "arm_pair", "comparison": comp, "uid": u, "dialogue_id": ia[u]["dialogue_id"],
                          "A": (a, ia[u]["response"], ia[u]["outcome"]), "B": (b, ib[u]["response"], ib[u]["outcome"])})
        ex.logger.info("human export | %s | %d contexts (one per dialogue where possible)", comp, len(pick))
    if (cfg.out / "corpus.npz").exists():
        _, _, _, test = reward_splits(load_corpus(cfg.out / "corpus.npz"), cfg.seed)
        cand = []
        for g in test:
            t = turn_of.get(g.uid)
            ok = np.flatnonzero(np.asarray(g.ok[:-1], bool))
            if t is None or ok.size < 2:
                continue
            i, j = [int(x) for x in rng.choice(ok, 2, replace=False)]
            if norm_text(g.texts[i]) == norm_text(g.texts[j]):
                continue
            cand.append((g, t, i, j, abs(float(g.outcomes[i] - g.outcomes[j]))))
        # half of the items from the top tercile of simulator margin, half at random: the first half
        # has power where the simulator is confident, the second estimates agreement at typical margins
        cand.sort(key=lambda c: -c[4])
        n_top_pool = max(1, len(cand) // 3) if cand else 0
        n_top = min(n_top_pool, cfg.human_validity_pairs // 2)
        top_idx = [int(k) for k in rng.permutation(n_top_pool)[:n_top]]
        taken = set(top_idx)
        rest_idx = [k for k in range(len(cand)) if k not in taken]
        rest_idx = [rest_idx[int(k)] for k in rng.permutation(len(rest_idx))[: max(0, cfg.human_validity_pairs - n_top)]]
        chosen = top_idx + rest_idx
        for k in chosen:
            g, t, i, j, m = cand[k]
            items.append({"kind": "validity", "comparison": "simulator", "uid": g.uid, "dialogue_id": g.dialogue_id,
                          "A": ("sample_i", g.texts[i], float(g.outcomes[i])),
                          "B": ("sample_j", g.texts[j], float(g.outcomes[j])), "sim_margin": m,
                          "stratum": "top_margin" if k in taken else "random"})
        ex.logger.info("human export | %d within-context simulator-validity items (%d from the top margin tercile)",
                       len(chosen), n_top)
    n_catch = max(4, int(round(0.05 * len(items))))
    pool = [t for t in ex.turns if t.split == "test" and len(t.gold_response.split()) >= 5]
    for k in range(n_catch):
        t = pool[int(rng.integers(len(pool)))]
        items.append({"kind": "catch", "comparison": "catch", "uid": t.uid, "dialogue_id": t.dialogue_id,
                      "A": ("gold", t.gold_response, float("nan")),
                      "B": ("catch", CATCH_RESPONSES[k % len(CATCH_RESPONSES)], float("nan"))})
    order = rng.permutation(len(items))
    packet, key = [], {}
    for pos, k in enumerate(order):
        it = items[int(k)]
        iid = f"item{pos:05d}"
        flip = bool(rng.random() < 0.5)
        L, R = (it["B"], it["A"]) if flip else (it["A"], it["B"])
        t = turn_of[it["uid"]]
        packet.append({"item_id": iid, "context": _context_text(t), "reply_1": L[1], "reply_2": R[1],
                       "double_annotate": int(rng.random() < cfg.human_overlap or it["kind"] == "catch")})
        key[iid] = {"kind": it["kind"], "comparison": it["comparison"], "uid": it["uid"],
                    "dialogue_id": it["dialogue_id"], "left": L[0], "right": R[0], "left_sim": L[2], "right_sim": R[2],
                    **({"sim_margin": it["sim_margin"], "stratum": it["stratum"]} if it["kind"] == "validity" else {})}
    import csv
    with open(d / "items.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(packet[0]))
        w.writeheader()
        w.writerows(packet)
    with open(d / "annotations_template.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["annotator_id", "item_id"] + list(HUMAN_QUESTIONS))
    dump_json(key, d / "key_DO_NOT_SHARE.json")
    (d / "instructions.md").write_text(
        "# Annotation instructions\n\nEach item shows a customer-service conversation and two possible agent "
        "replies (reply_1, reply_2). The replies were produced by different systems in random order. For each "
        "question answer 1 (reply_1), 2 (reply_2) or 0 (no real difference). Judge the reply only; do not guess "
        "which system wrote it.\n\n" + "\n".join(f"* **{k}**: {v}" for k, v in HUMAN_QUESTIONS.items()) +
        "\n\nItems with double_annotate=1 must be labelled by two different annotators. Some items are checks.\n",
        encoding="utf-8")
    need = binomial_n(0.60)
    n_arm = {c: sum(1 for it in items if it["comparison"] == c) for c in cfg.human_comparisons}
    ex.logger.info("human export | %d items (%d catch) -> %s | detecting a 60%% win rate at 80%% power needs ~%d "
                   "NON-TIE judgements per comparison; planned %s", len(packet), n_catch, d, need, n_arm)
    for c, n in n_arm.items():
        if n < need:
            ex.logger.warning("human export | comparison %s has %d items < %d: underpowered for a 60%% win rate "
                              "unless each item is judged by several annotators", c, n, need)
    meta = {"n_items": len(packet), "n_catch": n_catch, "per_comparison": n_arm, "power_n_60pct": need,
            "questions": HUMAN_QUESTIONS, "eval_seed": sd0}
    dump_json(meta, d / "packet_meta.json")
    return meta


def krippendorff_alpha_interval(units: Dict[str, List[float]]) -> float:
    """Krippendorff's alpha with the interval metric (Krippendorff, 2011), for units with >= 2 values."""
    vals = [np.asarray(v, float) for v in units.values() if len(v) >= 2]
    if not vals:
        return float("nan")
    n = float(sum(v.size for v in vals))
    do = sum(float(np.sum((v[:, None] - v[None, :]) ** 2)) / (v.size - 1) for v in vals) / n
    allv = np.concatenate(vals)
    de = float(np.sum((allv[:, None] - allv[None, :]) ** 2)) / (n * (n - 1))
    return float(1.0 - do / de) if de > EPS else float("nan")


def stage_human_analyze(cfg: Config) -> Dict[str, Any]:
    """R3: analyse human_study/annotations*.csv against the hidden key.

    Per item, each annotation is coded +1 (left preferred), -1 (right) or 0 (tie) and re-expressed from
    the perspective of the arm under test (or of the simulator-preferred reply); an item's score is
    the mean over its annotators.  Win rate = P(score > 0) + 0.5 P(score = 0).  Tests are sign-flip
    tests on item scores clustered by dialogue, Holm-adjusted across comparisons x questions; CIs
    are dialogue-clustered bootstraps.  q_information is additionally tested for non-inferiority."""
    import csv
    ex = Experiment(cfg, "10_human_analyze")
    d = cfg.out / "human_study"
    key = load_json(d / "key_DO_NOT_SHARE.json")
    ann = []
    for f in sorted(d.glob("annotations*.csv")):
        if f.name == "annotations_template.csv":
            continue
        with open(f, encoding="utf-8") as fh:
            ann.extend(csv.DictReader(fh))
    if not ann:
        raise FileNotFoundError(f"no filled annotations*.csv in {d}")
    code = {"1": 1.0, "2": -1.0, "0": 0.0}
    # attention checks: the gold reply must beat the catch reply
    acc: Dict[str, List[float]] = {}
    for a in ann:
        k = key.get(a["item_id"])
        if k and k["kind"] == "catch":
            v = code.get(str(a["q_satisfaction"]).strip())
            if v is not None:
                gold_left = k["left"] == "gold"
                acc.setdefault(a["annotator_id"], []).append(float((v > 0) == gold_left and v != 0))
    keep = {u for u, v in acc.items() if np.mean(v) >= 0.8}
    dropped = sorted(set(a["annotator_id"] for a in ann) - keep) if acc else []
    ex.logger.info("human analysis | %d annotations from %d annotators | %d fail the attention check and are "
                   "excluded: %s", len(ann), len({a['annotator_id'] for a in ann}), len(dropped), dropped)
    per_item: Dict[str, Dict[str, List[float]]] = {}
    for a in ann:
        if acc and a["annotator_id"] not in keep:
            continue
        k = key.get(a["item_id"])
        if not k or k["kind"] == "catch":
            continue
        for q in HUMAN_QUESTIONS:
            v = code.get(str(a.get(q, "")).strip())
            if v is None:
                continue
            if k["kind"] == "arm_pair":
                a_arm = k["comparison"].split(":")[0]
                v = v if k["left"] == a_arm else -v
            else:
                v = v if k["left_sim"] >= k["right_sim"] else -v   # +1 = human agrees with the simulator
            per_item.setdefault(a["item_id"], {}).setdefault(q, []).append(v)
    res: Dict[str, Any] = {"excluded_annotators": dropped, "comparisons": {}, "validity": {}, "agreement": {}}
    names, pv = [], []
    for comp in sorted({k["comparison"] for k in key.values() if k["kind"] in ("arm_pair", "validity")}):
        for q in HUMAN_QUESTIONS:
            ids = [i for i, k in key.items() if k["comparison"] == comp and q in per_item.get(i, {})]
            if len(ids) < 10:
                continue
            sc = np.asarray([np.mean(per_item[i][q]) for i in ids], float)
            cl = np.asarray([key[i]["dialogue_id"] for i in ids], dtype=object)
            wr = lambda ix: float(np.mean(sc[ix] > 0) + 0.5 * np.mean(sc[ix] == 0))
            pt, lo, hi = cluster_bootstrap_ci(wr, cl, 2000, cfg.seed)
            _, lo90, hi90 = cluster_bootstrap_ci(wr, cl, 2000, cfg.seed + 1, conf=0.90)
            sf = sign_flip_test(sc, cl, n_mc=cfg.n_signflip, seed=cfg.seed + 2)
            r = {"n_items": len(ids), "win_rate": pt, "ci95": [lo, hi], "ci90": [lo90, hi90], "p_value": sf["p"],
                 "p_text": sf["p_text"], "tie_rate": float(np.mean(sc == 0)), "mean_score": float(sc.mean())}
            if q == "q_information" and comp != "simulator":
                r["noninferior_at_0.45"] = bool(lo90 > 0.5 - cfg.noninferiority_info_margin)
            if comp == "simulator":
                ms = np.asarray([key[i]["sim_margin"] for i in ids], float)
                st = np.asarray([key[i]["stratum"] for i in ids])
                r["spearman_sim_margin_vs_agreement"] = spearman(ms, sc)
                r["by_stratum"] = {s_: float(np.mean(sc[st == s_] > 0) + 0.5 * np.mean(sc[st == s_] == 0))
                                   for s_ in np.unique(st)}
                res["validity"][q] = r
            else:
                res["comparisons"].setdefault(comp, {})[q] = r
            names.append((comp, q))
            pv.append(sf["p"])
            ex.logger.info("human | %-22s %-15s win %.3f CI95[%.3f,%.3f] ties %.2f | p %s | n=%d", comp, q, pt, lo, hi,
                           r["tie_rate"], sf["p_text"], len(ids))
    for (comp, q), adj in zip(names, holm(pv)):
        (res["validity"][q] if comp == "simulator" else res["comparisons"][comp][q])["p_holm"] = adj
    for q in HUMAN_QUESTIONS:
        units = {i: v[q] for i, v in per_item.items() if len(v.get(q, [])) >= 2}
        res["agreement"][q] = {"krippendorff_alpha_interval": krippendorff_alpha_interval(units), "n_units": len(units)}
        ex.logger.info("human | agreement %-15s Krippendorff alpha (interval) %.3f over %d double-annotated items", q,
                       res["agreement"][q]["krippendorff_alpha_interval"], len(units))
    dump_json(res, cfg.out / "human_study_results.json")
    return res


def _get(path: Path) -> Dict[str, Any]:
    return load_json(path) if path.exists() else {}


def stage_claims(cfg: Config) -> Dict[str, Any]:
    """Derive every headline claim from the artefacts, with its verdict and its evidence.

    Nothing here is tuned: each rule is written down before the numbers are read, and a claim whose
    evidence is missing is "untested", never "supported".  The paper should make exactly the claims
    this file supports, in the words it uses."""
    ex = Experiment(cfg, "11_claims")
    rep, ext = _get(cfg.out / "report.json"), _get(cfg.out / "report_ext.json")
    sv, rw = _get(cfg.out / "simulator_validation.json"), _get(cfg.out / "reward.json")
    abs_, abr = _get(cfg.out / "ablation_simulator.json"), _get(cfg.out / "ablation_reward.json")
    la, hs = _get(cfg.out / "length_analysis.json"), _get(cfg.out / "human_study_results.json")
    sen = _get(cfg.out / "sensitivity.json")
    C = rep.get("contrasts", {})
    claims: List[Dict[str, Any]] = []

    def add(cid: str, text: str, verdict: str, evidence: Any):
        claims.append({"id": cid, "claim": text, "verdict": verdict, "evidence": evidence})

    def sig_pos(c: Optional[Dict[str, Any]]) -> Optional[bool]:
        if not c:
            return None
        return bool(c["ci95"][0] > 0 and c.get("p_holm", 1.0) < 0.05)

    for comp, text in (("caro_vs_sft:outcome", "CARO raises the simulated customer-affect outcome over SFT"),
                       ("caro_vs_sentiment_only:outcome", "the simulator-derived reward beats rewarding positive agent "
                                                          "wording (sentiment-only RL)"),
                       ("caro_vs_dpo:outcome", "online RL on the learnt reward beats offline DPO on the same corpus"),
                       ("caro_vs_sft_bon_rm:outcome", "RL adds beyond best-of-N reranking with the same reward"),
                       ("caro_vs_sft_lenmatch:outcome", "the gain is not explained by response length "
                                                        "(design-based length-matched SFT control)")):
        c = C.get(comp)
        v = sig_pos(c)
        add(comp, text, "untested" if v is None else ("supported" if v else "not supported"),
            None if not c else {k: c.get(k) for k in ("delta", "ci95", "p_text", "p_holm", "n_clusters")})
    c = C.get("caro_vs_sft:info_recall")
    add("information", "CARO conveys no less task information than SFT (non-inferiority, recall of gold salient "
        "tokens)", "untested" if not c else ("supported" if c.get("noninferior") else "not supported"),
        None if not c else {k: c.get(k) for k in ("delta", "ci90", "noninferiority_margin", "noninferior")})
    e = ext.get("contrasts", {}).get("caro_vs_sft:outcome")
    add("external", "the gain replicates under an out-of-family customer and an independent emotion labeller",
        "untested" if not e else ("supported" if sig_pos(e) else "not supported"),
        None if not e else {k: e.get(k) for k in ("delta", "ci95", "p_text", "p_holm")})
    hv = hs.get("validity", {}).get("q_satisfaction")
    add("within_context_validity", "the simulator ranks alternative responses to the SAME context as humans do",
        "untested (needs the human study; EmoWOZ cannot test it)" if not hv else
        ("supported" if hv["ci95"][0] > 0.5 else "not supported"), hv)
    hc = hs.get("comparisons", {}).get("caro:sft", {}).get("q_satisfaction")
    add("human_preference", "humans prefer CARO over SFT on customer satisfaction",
        "untested" if not hc else ("supported" if hc["ci95"][0] > 0.5 and hc.get("p_holm", 1) < 0.05 else
                                   "not supported"), hc)
    add("between_context_anchor", "the simulator's outcome for the GOLD response tracks the real customer's "
        "next-turn emotion label across contexts (between-context only)",
        "untested" if not sv else ("supported" if (sv.get("human_label_ci_clustered") or [0])[0] > 0 else
                                   "not supported"),
        None if not sv else {"rho": sv.get("human_label_rho"), "ci95": sv.get("human_label_ci_clustered"),
                             "n": sv.get("human_label_n")})
    g = rw.get("gate", {})
    anc = g.get("gold_anchor", {})
    add("reward_human_signal", f"the reward carries human-anchored signal (gold anchor >= "
        f"{cfg.reward_min_gold_anchor:.2f} with a CI excluding 0)",
        "untested" if not anc else ("supported" if anc.get("rho", 0) >= cfg.reward_min_gold_anchor
                                    and anc.get("ci95", [0])[0] > 0 else "not supported"), anc or None)
    cells = [c for c in abs_.get("cells", []) if c.get("simulator") == "main"]

    def cell(sample, corr, bank):
        return next((c for c in cells if c["sample"] == sample and c["correction"] == corr and c["bank"] == bank), None)

    for sample in ("gate_replica", "independent"):
        n0, v8, v12 = cell(sample, "none", "heldout"), cell(sample, "v8_observational", "heldout"), \
            cell(sample, "v12_interventional", "heldout")
        if n0 and v8:
            harmful = abs(v8["delta_mean"]) > abs(n0["delta_mean"]) or (n0["tost_passes"] and not v8["tost_passes"]) \
                or v8["tau_corrected"] > n0["tau_corrected"]
            add(f"observational_harmful[{sample}]", "the observational (v8) length correction makes padding "
                "sensitivity worse than no correction", "supported" if harmful else "not supported",
                {"none": {k: n0[k] for k in ("delta_mean", "tau_corrected", "tost_passes")},
                 "v8": {k: v8[k] for k in ("delta_mean", "tau_corrected", "tost_passes")}})
        if n0 and v12:
            rel = abs(v12["delta_mean"]) / max(abs(n0["delta_mean"]), EPS)
            add(f"interventional_adds[{sample}]", "the interventional h(L) removes a material share (>= 25%) of "
                "the residual padding shift left by the simulator's own regulariser",
                "supported" if rel <= 0.75 else "not supported",
                {"none_delta": n0["delta_mean"], "h_delta": v12["delta_mean"], "ratio": rel,
                 "none_tau": n0["tau_corrected"], "h_tau": v12["tau_corrected"]})
        sb = _get(cfg.out / "simulator_meta.json").get("train_bank", "courtesy_v13")
        tr, ho = cell(sample, "none", sb), cell(sample, "none", "heldout")
        if tr and ho:
            add(f"heldout_stricter[{sample}]", "the held-out filler bank is a stricter test than the bank the "
                "simulator was regularised on",
                "supported" if abs(ho["delta_mean"]) > abs(tr["delta_mean"]) else "not supported (the paper may only "
                "say the train bank is not an independent test)", {"train": tr["delta_mean"], "heldout": ho["delta_mean"]})
    nr = next((c for c in abs_.get("cells", []) if c.get("simulator") == "no_invariance_regulariser"
               and c["sample"] == "independent" and c["correction"] == "none" and c["bank"] == "heldout"), None)
    mn = cell("independent", "none", "heldout")
    add("regulariser_supplies_invariance", "the simulator's invariance regulariser (not h(L)) supplies most of its "
        "padding invariance: without it the uncorrected held-out shift is >= 25% larger or fails the TOST",
        "untested (run ablate-simulator --ablate-sim-noreg)" if not (nr and mn) else
        ("supported" if (abs(nr["delta_mean"]) >= 1.25 * abs(mn["delta_mean"]) or not nr["tost_passes"])
         else "not supported"),
        None if not (nr and mn) else {"with_regulariser": mn["delta_mean"], "without": nr["delta_mean"],
                                      "tost_without": nr["tost_passes"]})
    ind = cell("independent", "v12_interventional", "heldout")
    if ind:
        add("gate_replicates", "the simulator invariance gate also passes on an independent sample (pre-registered "
            "ceilings)", "supported" if ind["passes"] else "not supported: the gate passed on a favourable sample",
            {k: ind[k] for k in ("tau_corrected", "tau_corrected_ci90", "flip_perturb", "flip_ci90", "tost_passes")})
    grid = sen.get("reward_regularisation", [])
    if grid:
        n_ok = sum(x["gate_would_pass"] for x in grid)
        add("reward_gate_robust", "the reward gate holds across the regularisation grid (>= 75% of cells)",
            "supported" if n_ok >= 0.75 * len(grid) else f"not supported ({n_ok}/{len(grid)} cells)",
            {"cells": len(grid), "holds": n_ok})
    for n, v in abr.items():
        if n == "full":
            continue
        add(f"reward_ablation[{n}]", f"the variant '{n}' differs from the full reward in held-out ranking accuracy",
            "supported" if v.get("p_holm", 1) < 0.05 else
            ("no effect (TOST equivalent)" if v.get("equivalent_to_full") else "inconclusive (neither different nor "
                                                                                  "equivalent)"),
            {k: v.get(k) for k in ("delta_accuracy_vs_full", "delta_ci90", "p_holm", "equivalent_to_full")})
    lc = la.get("contrasts", {}).get("caro_vs_sft", {})
    if lc.get("n", 0) >= 30:
        add("gain_at_equal_length", "the CARO-SFT gain persists at equal length (regression intercept; descriptive)",
            "supported" if lc["alpha_ci95"][0] > 0 else "not supported",
            {k: lc.get(k) for k in ("raw_delta", "alpha_equal_length", "alpha_ci95", "share_explained_by_length")})
    add("icc", "the panel estimator's ICC / split-half reliability are evidence of reliability",
        "not supported: they equal 1 by construction for a deterministic estimator (report as n/a)", None)
    dump_json(claims, cfg.out / "claims.json")
    lines = ["# Claims audit (auto-generated; edit the rules, never the verdicts)", "",
             "| id | claim | verdict |", "|---|---|---|"]
    lines += [f"| {c['id']} | {c['claim']} | {c['verdict']} |" for c in claims]
    (cfg.out / "claims.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for c in claims:
        ex.logger.info("CLAIM %-40s %s", c["id"], c["verdict"])
    return {"claims": claims}


def stage_paper(cfg: Config) -> Dict[str, Any]:
    """Assemble every table a submission needs, the claims audit and a reproducibility manifest."""
    ex = Experiment(cfg, "9_paper")
    tables = cfg.out / "tables"
    rep = load_json(cfg.out / "report.json") if (cfg.out / "report.json").exists() else stage_report(cfg)
    rows = [[a, v["outcome_mean"], v["outcome_ci95"][0], v["outcome_ci95"][1], v["outcome_sd_across_seeds"],
             v["words"], v["hygiene"], v["info_recall"], v["sentiment"]] for a, v in rep["arms"].items()]
    latex_table(tables / "main_results.tex",
                "Simulated customer-affect outcome by arm (evaluator: the CARO simulator, which also produced the "
                "reward labels; see the external and human evaluations). CIs are 95\\% dialogue-clustered bootstrap "
                "intervals pooled over seeds; sd(seed) is the between-seed standard deviation of the arm mean; "
                "info is the recall of the gold turn's task-critical tokens; sent. is the sentiment of the agent's "
                "own wording.", "tab:main",
                ["arm", "outcome", "CI lo", "CI hi", "sd(seed)", "words", "hygiene", "info", "sent."], rows)

    def contrast_rows(con):
        return [[k, v["delta"], v["ci95"][0], v["ci95"][1], v["cohens_dz"], v["paired_dominance"],
                 v["p_text"], v.get("p_holm", float("nan")), v["mde80"], v["n_clusters"]] for k, v in con.items()]
    hdr = ["contrast", "delta", "CI lo", "CI hi", "d_z", "dom.", "p", "p Holm", "MDE80", "dialogues"]
    latex_table(tables / "contrasts.tex",
                "Paired contrasts, pooled over seeds and paired within (seed, context); dialogues are the resampling "
                "unit. p is a dialogue-level sign-flip randomisation test on cluster sums (a p at the Monte Carlo "
                "floor is shown as a bound); Holm is within each metric family. dom. is the paired dominance "
                "P(d>0)-P(d<0). MDE80 is the effect detectable at 80\\% power with the cluster-robust SE.",
                "tab:contrasts", hdr, contrast_rows(rep["contrasts"]))
    ext = _get(cfg.out / "report_ext.json")
    if ext:
        latex_table(tables / "contrasts_external.tex",
                    f"Paired contrasts under the external evaluator (customer: {cfg.external_model}; emotion "
                    f"labeller: {cfg.external_emotion_model}); neither model is used by any training stage.",
                    "tab:contrasts-ext", hdr, contrast_rows({k: v for k, v in ext["contrasts"].items()
                                                             if k.endswith(":outcome")}))
    ab = _get(cfg.out / "ablation_reward.json")
    if ab:
        latex_table(tables / "ablation_reward.tex",
                    "Reward-model ablations on the frozen corpus, each refitted over "
                    f"{cfg.ablate_fit_seeds} initialisation seeds. Accuracy is held-out within-context pairwise "
                    "accuracy; TOST reports equivalence to the full model within "
                    f"+-{cfg.ablate_accuracy_margin:g} accuracy; padding delta is the reward change under held-out "
                    "content-free padding; length excess is the reward's within-context length correlation minus the "
                    "labels'; anchor is the gold-turn human anchor (between-context). Abstention only acts in RL and "
                    "is ablated there (arm caro_no_abstain).", "tab:ablation-reward",
                    ["variant", "accuracy", "sd(refit)", "d vs full", "p Holm", "TOST", "padding delta",
                     "length excess", "anchor"],
                    [[n, v["accuracy"], v["accuracy_sd_across_refits"], v.get("delta_accuracy_vs_full", float("nan")),
                      v.get("p_holm", float("nan")), ("--" if n == "full" else ("equiv." if v["equivalent_to_full"]
                                                                               else "no")),
                      v["padding_delta"], v["length_excess"], v["gold_anchor"]] for n, v in ab.items()])
    sa = _get(cfg.out / "ablation_simulator.json")
    if sa:
        latex_table(tables / "ablation_simulator.tex",
                    "Length correction x filler bank on a paired do(length) intervention, with the responses held "
                    "fixed within each sample. gate replica = the exact responses and fillers of the validation gate; "
                    f"independent = {sa.get('n_independent_contexts')} further contexts with the same "
                    f"{sa.get('variants_per_context')} variants per context. The train bank is the one the simulator "
                    "was regularised on, so a gate that uses it is not an independent test. Pre-registered ceilings: "
                    f"tau {cfg.max_tau:g} (upper CI90), flip {cfg.max_abs_flip:g}.", "tab:ablation-sim",
                    ["simulator", "sample", "correction", "bank", "delta", "margin", "tau", "tau CI90 hi", "flip",
                     "TOST", "gate"],
                    [[c["simulator"], c["sample"], c["correction"], c["bank"], c["delta_mean"], c["equivalence_margin"],
                      c["tau_corrected"], c["tau_corrected_ci90"][1], c["flip_perturb"],
                      "pass" if c["tost_passes"] else "fail", "pass" if c["passes"] else "fail"] for c in sa["cells"]])
    sen = _get(cfg.out / "sensitivity.json")
    if sen.get("reward_regularisation"):
        latex_table(tables / "sensitivity_reward.tex",
                    "Sensitivity of the reward gate to its two regularisation weights (TEST split). The row marked "
                    "* is the cell selected on DEV.", "tab:sens-reward",
                    ["clp", "length cal", "accuracy", "padding delta", "length excess", "anchor", "gate", "fails on"],
                    [[f"{g['clp']:g}" + ("*" if g["selected"] else ""), g["length_cal"], g["test_accuracy"],
                      g["padding_delta"], g["length_excess"], g["gold_anchor"],
                      "holds" if g["gate_would_pass"] else "fails", ", ".join(g["fails_on"]) or "--"]
                     for g in sen["reward_regularisation"]])
    if sen.get("panel_temperature"):
        P = sen["panel_temperature"].get("panel_size")
        latex_table(tables / "sensitivity_panel.tex",
                    f"Panel temperature (panel of P = {P} real customer replies): agreement with the observed "
                    f"next-customer sentiment and the effective panel size (at most P).", "tab:sens-panel",
                    ["T", "rho", "effective panel size"],
                    [[g["T"], g["rho"], g["effective_panel_size"]] for g in sen["panel_temperature"]["grid"]])
    la = _get(cfg.out / "length_analysis.json")
    if la.get("contrasts"):
        latex_table(tables / "length_analysis.tex",
                    "Length analysis. alpha is the intercept of the paired outcome difference regressed on the word "
                    "difference (the contrast at equal length; word count is post-treatment, so this is descriptive); "
                    "matched is the mean difference among pairs within 2 words. The design-based control is the "
                    "sft_lenmatch arm in the contrasts table.", "tab:length",
                    ["contrast", "raw delta", "d words", "alpha", "alpha CI lo", "alpha CI hi", "per word",
                     "matched delta", "matched n"],
                    [[k, v["raw_delta"], v["mean_d_words"], v["alpha_equal_length"], v["alpha_ci95"][0],
                      v["alpha_ci95"][1], v["beta_per_word"], v["matched"].get("delta", float("nan")), v["matched"]["n"]]
                     for k, v in la["contrasts"].items() if v.get("n", 0) >= 30])
    sv = _get(cfg.out / "simulator_validation.json")
    rw = _get(cfg.out / "reward.json")
    obj = []
    if sv:
        obj.append(["C2: rho(gold outcome, human label), between contexts", sv.get("human_label_rho"),
                    "[{:+.3f},{:+.3f}]".format(*sv.get("human_label_ci_clustered", [float("nan")] * 2)), "--",
                    f"n={sv.get('human_label_n')}"])
    for key_, lab in (("caro_vs_sft:outcome", "C3: delta outcome (simulator)"),
                      ("caro_vs_sentiment_only:outcome", "C3: vs sentiment-only"),
                      ("caro_vs_dpo:outcome", "C3: vs DPO"), ("caro_vs_sft_lenmatch:outcome", "C3: vs length-matched SFT"),
                      ("caro_vs_sft:info_recall", "C3: delta information recall")):
        c = rep["contrasts"].get(key_)
        if c:
            obj.append([lab, c["delta"], "[{:+.4f},{:+.4f}]".format(*c["ci95"]), c.get("p_holm", float("nan")),
                        f"d_z={c['cohens_dz']:.3f}"])
    if ext.get("contrasts", {}).get("caro_vs_sft:outcome"):
        c = ext["contrasts"]["caro_vs_sft:outcome"]
        obj.append(["C3: delta outcome (external evaluator)", c["delta"], "[{:+.4f},{:+.4f}]".format(*c["ci95"]),
                    c.get("p_holm", float("nan")), f"d_z={c['cohens_dz']:.3f}"])
    if rw:
        g = rw.get("gate", {})
        n_ok = sum(x["gate_would_pass"] for x in sen.get("reward_regularisation", []))
        obj.append(["C4: reward gate", str(g.get("passes")), "--", "--",
                    f"{n_ok}/{len(sen.get('reward_regularisation', []))} grid cells"])
        ga = g.get("gold_anchor", {})
        if ga:
            obj.append(["C4: reward gold anchor", ga.get("rho"), "[{:+.3f},{:+.3f}]".format(*ga["ci95"]), "--",
                        f"n={ga.get('n')}"])
    if obj:
        latex_table(tables / "objective.tex",
                    "Objective check. C2 is a BETWEEN-context correlation on gold responses: it shows that the "
                    "simulator tracks the customer's affective trajectory, not that it ranks alternative responses "
                    "to one context (that is tested by the human study). C3 rows under the simulator are "
                    "in-distribution for the reward.", "tab:objective", ["clause", "value", "95% CI", "p Holm", "effect"],
                    obj)
    claims = stage_claims(cfg)
    panel = _get(cfg.out / "outcome_panel.json")
    manifest = {
        "version": VERSION, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config": json.loads(json.dumps(asdict(cfg), default=_json_default)),
        "realised_panel_size": len(panel.get("texts", [])) if panel else None,
        "panel_temperature": panel.get("temperature") if panel else None,
    }
    if panel and len(panel.get("texts", [])) != cfg.panel_size:
        ex.logger.warning("the stored outcome panel has %d candidates but this configuration says %d: the panel was "
                          "built by a different configuration; report the realised size", len(panel["texts"]),
                          cfg.panel_size)
    checklist = {
        "version": VERSION,
        "pre_registered_thresholds": sv.get("pre_registration", {}),
        "simulator_gate": {k: sv.get(k) for k in ("passes", "human_label_rho", "anchor_spearman",
                                                  "length_rho_within_after", "response_variance_share",
                                                  "icc_note")},
        "simulator_padding_intervention": sv.get("paired_intervention", {}),
        "reward_gate": rw.get("gate", {}).get("passes"),
        "reward_selection_on_dev_only": rw.get("selected", {}),
        "splits": "dialogue-disjoint throughout: simulator fit / gate, corpus train / dev / calib / test, "
                  "evaluation on the EmoWOZ test split",
        "seeds": list(cfg.seeds), "arms": list(cfg.arms),
        "base_model": cfg.base_model, "quantisation": "4-bit" if cfg.load_4bit else "bf16",
        "lora_rank": cfg.lora_r, "realised_panel_size": manifest["realised_panel_size"],
        "known_limitations": [
            "the primary outcome is produced by the simulator family whose labels trained the reward: the reward is "
            "a model of the evaluator, so simulator-evaluated gains are in-distribution proxy improvements; the "
            "external evaluator and the human study are the tests that are not",
            "EmoWOZ labels only the gold response, so the simulator's human anchor is between-context; within-context "
            "ranking validity rests on the human study",
            "the reward's gold-response anchor against human emotion labels is weak (see claims.json)",
            "ICC and split-half reliability are 1 by construction for the deterministic panel estimator and are "
            "reported as not applicable",
            "the length correction identifies the average causal effect of appended content-free text, not of every "
            "way a response could be made longer",
            "information retention is measured by recall of salient gold tokens, a transparent lexical proxy",
            "the GRPO clip is inactive (one update per batch): the optimiser is REINFORCE with a k3 KL penalty",
        ],
    }
    dump_json(checklist, cfg.out / "reproducibility_checklist.json")
    dump_json(manifest, cfg.out / "run_manifest.json")
    ex.logger.info("wrote %d LaTeX tables to %s, the claims audit and the reproducibility manifest",
                   len(list(tables.glob("*.tex"))) if tables.exists() else 0, tables)
    return {"tables": sorted(str(x.name) for x in tables.glob("*.tex")) if tables.exists() else [],
            "checklist": checklist, "claims": claims["claims"]}


def stage_analysis(cfg: Config, with_gpu_ablation: bool = True, with_external: bool = True) -> Dict[str, Any]:
    """Everything a submission needs after the pipeline has run: primary analysis, external
    evaluation, ablations, sensitivity, length analysis, human-study packet, claims and LaTeX."""
    log = logging.getLogger("caro")
    stage_report(cfg)
    if with_external:
        try:
            stage_eval_external(cfg)
        except Exception as e:                      # a missing model should not sink the analysis
            log.error("external evaluation skipped: %s", e)
    try:
        stage_cross_eval(cfg)
    except Exception as e:
        log.error("cross-evaluator agreement skipped: %s", e)
    stage_ablate_reward(cfg)
    if with_gpu_ablation:
        try:
            stage_ablate_simulator(cfg)
        except Exception as e:
            log.error("simulator ablation skipped: %s", e)
    stage_sensitivity(cfg)
    stage_length_analysis(cfg)
    try:
        stage_human_export(cfg)
    except Exception as e:
        log.error("human-study export skipped: %s", e)
    if list((cfg.out / "human_study").glob("annotations*.csv")) and any(
            f.name != "annotations_template.csv" for f in (cfg.out / "human_study").glob("annotations*.csv")):
        stage_human_analyze(cfg)
    return stage_paper(cfg)


def ordered_arms(arms: Sequence[str]) -> List[str]:
    """Training/evaluation order: sft_lenmatch needs CARO's eval rows, so it goes last."""
    unknown = [a for a in arms if a not in KNOWN_ARMS]
    if unknown:
        raise ValueError(f"unknown arms {unknown}; choose from {KNOWN_ARMS}")
    return sorted(arms, key=lambda a: (a == "sft_lenmatch", a == "sft_bon_rm", a not in ("sft",)))


def stage_all(cfg: Config) -> None:
    stage_sft(cfg)
    stage_simulator(cfg)
    stage_validate(cfg)
    stage_corpus(cfg)
    stage_reward(cfg)
    if "sft_lenmatch" in cfg.arms and "caro" not in cfg.arms:
        raise ValueError("arm sft_lenmatch needs the caro arm")
    # v14 (D2): every arm, SFT included, is evaluated under every evaluation seed.  v13 evaluated SFT
    # once and copied its rows to the other seeds, so "seed variance" of the baseline was zero by
    # construction and the copies entered the analysis as if they were new evidence.
    for arm in ordered_arms(cfg.arms):
        for sd in cfg.seeds:
            stage_train(cfg, arm, sd)
            stage_eval(cfg, arm, sd)
    stage_analysis(cfg)


def make_synthetic_emowoz(data_dir: Path, n_dialogues: int = 420, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    pos = ["great", "confirmed", "booked", "available", "cheap", "thanks"]
    neg = ["sorry", "unfortunately", "delay", "cancelled", "expensive"]
    neu = ["table", "train", "hotel", "reference", "number", "north", "centre", "please"]
    mw: Dict[str, Any] = {}
    dm: Dict[str, Any] = {}
    split = {"train": {"multiwoz": [], "dialmage": []},
             "dev": {"multiwoz": [], "dialmage": []},
             "test": {"multiwoz": [], "dialmage": []}}
    for i in range(n_dialogues):
        src = "multiwoz" if i % 4 else "dialmage"
        did = (f"PMUL{i:04d}.json" if src == "multiwoz" else f"DMAGE{i}.json")
        n_turns = int(rng.integers(4, 12)) * 2
        log = []
        mood = 0.0
        for k in range(n_turns):
            if k % 2 == 0:
                w = list(rng.choice(np.asarray(neu, dtype=object), size=int(rng.integers(4, 12))))
                txt = " ".join(str(x) for x in w).capitalize() + "?"
                if mood > 0.3:
                    emo = int(rng.choice([6, 5, 0], p=[0.5, 0.2, 0.3]))
                elif mood < -0.3:
                    emo = int(rng.choice([2, 1, 4, 0], p=[0.5, 0.2, 0.1, 0.2]))
                else:
                    emo = int(rng.choice([0, 3], p=[0.85, 0.15]))
                log.append({"text": txt, "emotion": [{"emotion": emo} for _ in range(4)]})
            else:
                good = rng.random() < 0.55
                pool = pos if good else neg
                w = list(rng.choice(np.asarray(pool + neu, dtype=object), size=int(rng.integers(6, 26))))
                txt = " ".join(str(x) for x in w).capitalize() + "."
                mood = 0.6 * mood + (0.8 if good else -0.8) + float(rng.normal(0, 0.3))
                log.append({"text": txt, "emotion": -1})
        rec = {"log": log}
        (mw if src == "multiwoz" else dm)[did] = rec
        s = "train" if i % 10 < 8 else ("dev" if i % 10 == 8 else "test")
        split[s][src].append(did)
    data_dir.mkdir(parents=True, exist_ok=True)
    dump_json(mw, data_dir / "emowoz-multiwoz.json")
    dump_json(dm, data_dir / "emowoz-dialmage.json")
    dump_json(split, data_dir / "data-split.json")


STAGES = ["all", "sft", "simulator", "validate", "corpus", "reward", "train", "eval", "report", "eval-external", "cross-eval",
          "ablate-reward", "ablate-simulator", "sensitivity", "length-analysis", "human-export", "human-analyze",
          "claims", "paper", "analysis", "selftest"]

# (CLI flag, dotted Config path, type, help).  Defaults are READ FROM Config (R11), so the CLI and the
# dataclass cannot drift apart again (v13: Config.panel_size=24, CLI default 48).
_FLAGS: List[Tuple[str, str, Any, str]] = [
    ("--data-dir", "data_dir", str, ""), ("--out", "out", str, ""), ("--base-model", "base_model", str, ""),
    ("--sentiment-model", "sentiment_model", str, ""), ("--models-dir", "models_dir", str, ""),
    ("--seed", "seed", int, ""), ("--lora-r", "lora_r", int, ""), ("--gen-batch", "gen_batch", int, ""),
    ("--feat-dim", "feat_dim", int, ""), ("--sft-turns", "sft_turns", int, ""), ("--sim-turns", "sim_turns", int, ""),
    ("--sft-epochs", "sft.epochs", int, ""), ("--sim-epochs", "sim_sft.epochs", int, ""),
    ("--validate-contexts", "validate_contexts", int, ""), ("--validate-variants", "validate_variants", int, ""),
    ("--max-abs-flip", "max_abs_flip", float, "pre-registered absolute flip ceiling"),
    ("--max-tau", "max_tau", float, "pre-registered ceiling on the upper CI90 of tau"),
    ("--max-excess-flip", "max_excess_flip", float, ""),
    ("--projector-k-max", "projector_k_max", int, "0 disables the logit nullspace projection"),
    ("--projector-fit-contexts", "projector_fit_contexts", int, ""),
    ("--projector-probe-contexts", "projector_probe_contexts", int, ""),
    ("--projector-min-anchor-retention", "projector_min_anchor_retention", float, ""),
    ("--projector-min-residual-gain", "projector_min_residual_gain", float, ""),
    ("--orbit-levels", "orbit_levels", int, "Reynolds averaging over K length rungs; 1 disables it"),
    ("--panel-size", "panel_size", int, "number of real customer replies in the outcome panel"),
    ("--panel-temperature", "panel_temperature", float, ""),
    ("--length-margin-frac", "length_margin_frac", float, "pre-registered TOST margin in within-context sd units"),
    ("--score-batch", "score_batch", int, ""), ("--corpus-pad", "corpus_pad", int, ""),
    ("--reward-clp", "reward.clp", float, "pairing weight (0 disables; <0 selects on DEV)"),
    ("--reward-length-cal", "reward.length_cal", float, "length-calibration weight (0 disables; <0 selects on DEV)"),
    ("--reward-max-excess-length-rho", "reward_max_excess_length_rho", float, ""),
    ("--reward-dev-anchor-tol", "reward_dev_anchor_tol", float, "R4: DEV anchor tolerance vs the plain ranker"),
    ("--reward-min-gold-anchor", "reward_min_gold_anchor", float, ""),
    ("--reward-n-perm", "reward_n_perm", int, ""),
    ("--invariance-coef", "sim_sft.invariance_coef", float, ""),
    ("--sim-rollouts", "sim_rollouts", int, ""), ("--sim-rollouts-corpus", "sim_rollouts_corpus", int, ""),
    ("--corpus-contexts", "corpus_contexts", int, ""), ("--corpus-variants", "corpus_variants", int, ""),
    ("--eval-turns", "eval_turns", int, ""),
    ("--grpo-steps", "grpo.steps", int, ""), ("--grpo-contexts", "grpo.n_contexts", int, ""),
    ("--grpo-group", "grpo.group_size", int, ""), ("--grpo-kl-coef", "grpo.kl_coef", float, ""),
    ("--grpo-kl-target", "grpo.kl_target", float, ""), ("--grpo-snr-kappa", "grpo.snr_kappa", float, ""),
    ("--grpo-length-drift", "grpo.length_drift_factor", float, ""),
    ("--grpo-bad-advantage", "grpo.bad_advantage", float, ""),
    ("--dpo-beta", "dpo.beta", float, ""), ("--dpo-lr", "dpo.lr", float, ""), ("--dpo-epochs", "dpo.epochs", int, ""),
    ("--bon-n", "bon_n", int, "samples per context for sft_bon_rm / sft_lenmatch"),
    ("--ablate-contexts", "ablate_contexts", int, ""), ("--ablate-fit-seeds", "ablate_fit_seeds", int, ""),
    ("--ablate-accuracy-margin", "ablate_accuracy_margin", float, ""),
    ("--sens-reward-grid", "sens_reward_grid", str, "compact | full"),
    ("--n-signflip", "n_signflip", int, ""),
    ("--reward-ensemble", "reward.n_ensemble", int, ""), ("--reward-kappa", "reward.kappa", float, ""),
    ("--reward-alpha", "reward.alpha", float, ""),
    ("--external-model", "external_model", str, "R2: out-of-family customer LLM"),
    ("--external-emotion-model", "external_emotion_model", str, "R2: independent emotion classifier"),
    ("--external-rollouts", "external_rollouts", int, ""),
    ("--human-contexts", "human_contexts", int, ""), ("--human-validity-pairs", "human_validity_pairs", int, ""),
    ("--noninferiority-info-margin", "noninferiority_info_margin", float, ""),
    ("--stability-safety", "stability_safety", float, "FIT-half flip limit as a fraction of --max-abs-flip "
                                                      "when choosing the panel temperature (0 disables)"),
    ("--cross-eval-contexts", "cross_eval_contexts", int, "held-out contexts for cross-evaluator agreement"),
    ("--cross-eval-k", "cross_eval_k", int, "responses per context for cross-evaluator agreement"),
]


def _cfg_get(cfg: Any, path: str) -> Any:
    for part in path.split("."):
        cfg = getattr(cfg, part)
    return cfg


def _cfg_set(cfg: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    for part in parts[:-1]:
        cfg = getattr(cfg, part)
    setattr(cfg, parts[-1], value)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    d = Config()
    p = argparse.ArgumentParser(description=f"CARO {VERSION} end-to-end pipeline on EmoWOZ")
    p.add_argument("stage", nargs="?", default="all", choices=STAGES)
    for flag, path, typ, hlp in _FLAGS:
        dv = _cfg_get(d, path)
        p.add_argument(flag, dest=path.replace(".", "__"), type=typ, default=(str(dv) if isinstance(dv, Path) else dv),
                       help=(hlp + " " if hlp else "") + f"(default: {dv})")
    p.add_argument("--device", default=None)
    p.add_argument("--download", action="store_true")
    p.add_argument("--stub", action="store_true")
    p.add_argument("--no-4bit", dest="load_4bit", action="store_false")
    p.add_argument("--no-strict", dest="strict", action="store_false")
    p.add_argument("--gap-aborts", action="store_true")
    p.add_argument("--ablate-sim-noreg", action="store_true",
                   help="R6: also retrain a simulator WITHOUT the invariance regulariser (GPU-hours)")
    p.add_argument("--seeds", type=int, nargs="+", default=list(d.seeds))
    p.add_argument("--arm", default=None)
    p.add_argument("--arms", nargs="+", default=list(d.arms), choices=list(KNOWN_ARMS))
    p.add_argument("--outcome-mode", default=d.outcome_mode, choices=["auto", "expected", "sample"])
    p.add_argument("--sim-train-bank", default=d.sim_train_bank, choices=["train", "courtesy_v13"],
                   help="filler bank the simulator is regularised on; courtesy_v13 reproduces pre-v15 simulators")
    p.add_argument("--gate-bank", default=d.gate_bank, choices=sorted(FILLER_BANKS),
                   help="filler bank for the validation gates; 'train' is circular and only for debugging")
    return p.parse_args(list(argv))


def build_config(a: argparse.Namespace) -> Config:
    device = a.device
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    cfg = Config()
    for _, path, _, _ in _FLAGS:
        v = getattr(a, path.replace(".", "__"))
        if path in ("data_dir", "out"):
            v = Path(v)
        _cfg_set(cfg, path, v)
    cfg.device = device
    cfg.seeds = tuple(a.seeds)
    cfg.arms = tuple(a.arms)
    cfg.stub = bool(a.stub)
    cfg.download = bool(a.download)
    cfg.load_4bit = bool(a.load_4bit)
    cfg.strict = bool(a.strict)
    cfg.outcome_mode = a.outcome_mode
    cfg.gate_bank = a.gate_bank
    cfg.sim_train_bank = a.sim_train_bank
    cfg.ablate_sim_noreg = bool(a.ablate_sim_noreg)
    cfg.sim_sft.gap_aborts = bool(a.gap_aborts)
    cfg.sft.invariance_coef = 0.0          # the agent SFT has no invariance penalty; only the simulator does
    return cfg


def _unit_tests() -> None:
    """Statistical unit tests of the v9 corrections, on synthetic data with known ground truth."""
    rng = np.random.default_rng(0)
    # (1) Bad control.  Content q drives both length and outcome; length has NO causal effect.
    #     The observational within-context slope is strongly non-zero; the interventional
    #     first-difference estimator must return ~0.
    G, K = 400, 4
    q = rng.normal(size=(G, K))
    words = np.clip(np.round(15 + 6 * q + rng.normal(0, 2, (G, K))), 3, 80).astype(int)
    y = -0.05 * q + rng.normal(0, 0.01, (G, K))                 # outcome depends on content only
    texts = [" ".join(["w"] * int(n)) for n in words.ravel()]
    gid = np.repeat(np.arange(G), K)
    obs = within_group_spearman(y.ravel(), loglen(texts), gid)["mean"]
    padded = [t + " f f f f f f f f f" for t in texts]
    d = rng.normal(0, 0.002, G * K)                             # do(length) has no effect
    h = InterventionalLengthCalibration().fit(texts, padded, d, gid)
    added = float(np.mean(h.curve(padded) - h.curve(texts)))
    assert obs < -0.3, f"synthetic confounding not present (rho={obs:+.3f})"
    assert abs(added) < 0.002, f"interventional calibration invented a length effect ({added:+.5f})"
    # (2) It must recover a real pure-length effect: O += 0.02 * log-length.
    L0, L1 = loglen(texts), loglen(padded)
    d2 = 0.02 * (L1 - L0) + rng.normal(0, 0.002, G * K)
    h2 = InterventionalLengthCalibration().fit(texts, padded, d2, gid)
    resid = float(np.mean(d2 - (h2.curve(padded) - h2.curve(texts))))
    assert abs(resid) < 5e-4, f"interventional calibration failed to remove a real effect ({resid:+.5f})"
    # (3) Copy-aware bootstrap: resampling a cluster twice must not merge its groups.
    yy = rng.normal(size=G * K)
    cl = np.repeat(np.arange(G), K)
    _, lo_c, hi_c = cluster_bootstrap_ci(lambda ix, cp: within_group_var(yy[ix], boot_groups(cl[ix], cp)),
                                         cl, 300, 1, with_copy=True)
    v = within_group_var(yy, cl)
    assert lo_c < v < hi_c, "copy-aware within-group variance CI does not cover the point estimate"
    # (4) Length calibration.  Labels carry a genuine content-driven length association; the
    #     features expose a clean length column and a noisy content column, so an unregularised
    #     Plackett-Luce ranker leans on length harder than the labels license.  The DEV sweep
    #     must pull the excess back inside tolerance at an acceptable accuracy cost.
    q = logging.getLogger("caro.unit")
    q.addHandler(logging.NullHandler())
    q.setLevel(logging.CRITICAL)
    r4 = np.random.default_rng(7)
    gs = []
    for gi in range(300):
        words = np.clip(np.round(np.exp(r4.normal(2.8, 0.35, 5))), 3, 90).astype(int)
        content = r4.normal(size=5)
        yy = 0.6 * content - 0.15 * np.log(words) + r4.normal(0, 0.1, 5)
        feats = np.stack([np.log1p(words), content + r4.normal(0, 0.8, 5),
                          r4.normal(size=5), r4.normal(size=5)], 1).astype(np.float32)
        gs.append(ResponseGroup(f"d{gi // 3}", f"u{gi}", [" ".join(["w"] * int(n)) for n in words],
                                feats, yy, np.ones(5, bool), float("nan")))
    tr, dv = group_split(gs, 0.7, 0)
    Xt = np.concatenate([g.feats for g in tr], 0)
    sd0 = Standardizer().fit(Xt)
    sweep = []
    for mu in (0.0, 1.0, 3.0, 10.0):
        rmu = RewardModel(Xt.shape[1], RewardConfig(n_ensemble=1, epochs=60, clp=0.0, length_cal=mu), sd0)
        rmu.fit(tr, dv, dv, q, seed=1)
        ex = reward_length_excess(rmu, dv, 1, None, n_boot=100)["excess"]
        ac = within_group_pairwise_accuracy([rmu.score(g.feats)["reward"] for g in dv],
                                            [g.outcomes for g in dv], [g.ok for g in dv])[0]
        sweep.append((mu, ex, ac))
        if abs(ex) <= 0.05:
            break
    assert abs(sweep[0][1]) > 0.05, "synthetic case has no length excess to correct"
    assert abs(sweep[-1][1]) <= 0.05, f"DEV sweep did not bring the excess inside tolerance: {sweep}"
    assert sweep[-1][2] > sweep[0][2] - 0.10, f"length calibration cost too much accuracy: {sweep}"
    print(f"unit tests OK | length excess {sweep[0][1]:+.3f} -> {sweep[-1][1]:+.3f} at weight "
          f"{sweep[-1][0]:g} (pairwise accuracy {sweep[0][2]:.3f} -> {sweep[-1][2]:.3f})")
    # (5) Advantage construction: hygiene must not enter the group statistics, weak groups must
    #     produce weak gradients, and a group inside the model's own uncertainty must produce none.
    r5 = np.array([0.104, 0.108, 0.101, 0.112, 0.099, 0.300])   # last entry stands in for a malformed sample
    ok5 = np.array([True] * 5 + [False])
    w5 = np.ones(6)
    g5 = np.zeros(6, int)
    a5 = group_relative_advantage(r5, w5, g5, ok=ok5, bad_advantage=1.0, scale=0.02, rm_sd=np.zeros(6))
    a_ref = group_relative_advantage(r5[:5], w5[:5], g5[:5], ok=ok5[:5], scale=0.02, rm_sd=np.zeros(5))
    assert abs(a5["adv"][5] + 1.0) < 1e-9, "a hygiene failure must carry the fixed negative advantage"
    assert np.max(np.abs(a5["adv"][:5] - a_ref["adv"])) < 1e-9, "malformed samples still enter the statistics"
    # a weak group must give weak advantages on the shared scale, not unit-normalised ones
    weak = np.array([0.1000, 0.1004, 0.0997, 0.1002, 0.0999])
    strong = weak + np.array([0.0, 0.0, 0.0, 0.0, 0.06])
    ok4 = np.ones(5, bool)
    aw = group_relative_advantage(weak, np.ones(5), np.zeros(5, int), ok=ok4, scale=0.02, rm_sd=np.zeros(5))
    as_ = group_relative_advantage(strong, np.ones(5), np.zeros(5, int), ok=ok4, scale=0.02, rm_sd=np.zeros(5))
    assert np.abs(aw["adv"]).max() < 0.1 < np.abs(as_["adv"]).max(), (
        "the shared scale does not separate a noise group from a signal group: "
        f"{np.abs(aw['adv']).max():.3f} vs {np.abs(as_['adv']).max():.3f}")
    # signal-to-noise gate: spread inside the ensemble sd contributes nothing
    ag = group_relative_advantage(weak, np.ones(5), np.zeros(5, int), ok=ok4, scale=0.02,
                                  rm_sd=np.full(5, 0.05), snr_kappa=1.0)
    assert ag["n_gated"] == 1 and not np.any(ag["adv"]), "the signal-to-noise gate did not fire"
    print(f"unit tests OK | advantages: noise group max |adv|={np.abs(aw['adv']).max():.3f}, signal group "
          f"{np.abs(as_['adv']).max():.3f}, uncertainty-gated group {np.abs(ag['adv']).max():.3f}")
    print(f"unit tests OK | bad-control demo: observational rho={obs:+.3f}, interventional shift={added:+.5f} | "
          f"real-effect residual={resid:+.6f}")
    # (6) Sign-flip test: exact enumeration gives the textbook p for an all-positive sample, the Monte
    #     Carlo version agrees with it, and the statistic targets the ROW mean (D3).
    sf = sign_flip_test(np.ones(8), np.arange(8))
    assert sf["exact"] and abs(sf["p"] - 2 / 256) < 1e-12, sf
    r6 = np.random.default_rng(3)
    d6, c6 = r6.normal(0.1, 1, 120), np.repeat(np.arange(12), 10)
    pe = sign_flip_test(d6, c6)["p"]
    pm = sign_flip_test(d6, c6, exact_max_clusters=0, n_mc=20000, seed=1)["p"]
    assert abs(pe - pm) < 0.02, (pe, pm)
    # (7) Pseudo-replication (D2): identical rows under three seeds must not triple the cluster count.
    base_rows = [{"uid": f"d{i // 2}#{i % 2}", "dialogue_id": f"d{i // 2}", "outcome": float(r6.normal()),
                  "words": 10} for i in range(80)]
    arm_rows = [{**r, "outcome": r["outcome"] + 0.1 + float(r6.normal(0, 0.5))} for r in base_rows]
    c7 = pooled_paired_contrast({s_: arm_rows for s_ in (1, 2, 3)}, {s_: base_rows for s_ in (1, 2, 3)}, "outcome",
                                n_boot=200, n_mc=2000)
    assert c7["n_clusters"] == 40, f"clusters must be dialogues, got {c7['n_clusters']}"
    c7b = pooled_paired_contrast({1: arm_rows}, {1: base_rows}, "outcome", n_boot=200, n_mc=2000)
    assert abs(c7["se_cluster"] - c7b["se_cluster"]) < 1e-9, "tripling identical rows must not shrink the SE"
    # (8) Agreement and information retention.
    assert abs(krippendorff_alpha_interval({"a": [1, 1], "b": [-1, -1], "c": [0, 0]}) - 1.0) < 1e-12
    assert info_recall("Booked at the Gonville, ref XYZ123.", "Your Gonville booking is done, reference XYZ123.") > 0.3
    assert info_recall("So sorry, glad to help!", "Your Gonville booking is done, reference XYZ123.") == 0.0
    assert math.isnan(info_recall("Hello.", "Thank you, goodbye!"))
    # (9) Stability-constrained temperature: an inadmissible optimum must not be chosen, and an empty
    #     admissible set must fall back to the unconstrained optimum (reported, never silently widened).
    r9 = np.random.default_rng(9)
    turns9 = [Turn(f"d{i}", f"d{i}#0", "train", "x", 0, "", "hi", "ok.", "fine.", 0, 0) for i in range(150)]
    pan = OutcomePanel([f"c{i}" for i in range(12)], np.linspace(-1, 1, 12))
    lp9 = r9.normal(size=(150, 12))
    target9 = lp9 @ np.linspace(-1, 1, 12) + r9.normal(0, 0.5, 150)
    pan._cal_cache = (sha_of([t.uid for t in turns9] + pan.texts), lp9, target9)
    q9 = logging.getLogger("caro.unit")
    pan.calibrate(None, turns9, None, q9, quiet=True)
    t_free = pan.temperature
    pan.calibrate(None, turns9, None, q9, quiet=True, admissible=lambda T: T != t_free)
    assert pan.temperature != t_free and pan.calibration["stability_constrained"]
    pan.calibrate(None, turns9, None, q9, quiet=True, admissible=lambda T: False)
    assert pan.temperature == t_free and pan.calibration["n_admissible_T"] == 0
    print(f"unit tests OK | exact sign-flip p={sf['p']:.5f} | exact vs MC {pe:.4f}/{pm:.4f} | dialogue clusters "
          f"{c7['n_clusters']} with SE unchanged by seed copies")


def _fake_annotations(cfg: Config) -> None:
    """Selftest only: two careful annotators who follow the simulator with noise, and one who clicks at
    random, so that agreement, the attention-check exclusion and every estimator are exercised."""
    import csv
    key = load_json(cfg.out / "human_study" / "key_DO_NOT_SHARE.json")
    rng = np.random.default_rng(5)
    rows = []
    for ann in ("a1", "a2", "bad_annotator"):
        for iid, k in key.items():
            if ann == "a2" and rng.random() > 0.5 and k["kind"] != "catch":
                continue
            ans = {}
            for q in HUMAN_QUESTIONS:
                if ann == "bad_annotator":
                    ans[q] = str(int(rng.integers(0, 3)))
                elif k["kind"] == "catch":
                    ans[q] = "1" if k["left"] == "gold" else "2"
                else:
                    ls, rs_ = k["left_sim"], k["right_sim"]
                    pref = ("1" if ls > rs_ else "2") if rng.random() < 0.7 else str(int(rng.integers(0, 3)))
                    ans[q] = pref
            rows.append({"annotator_id": ann, "item_id": iid, **ans})
    with open(cfg.out / "human_study" / "annotations_selftest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["annotator_id", "item_id"] + list(HUMAN_QUESTIONS))
        w.writeheader()
        w.writerows(rows)


def run_selftest() -> None:
    _unit_tests()
    root = Path("caro_selftest")
    data = root / "data"
    make_synthetic_emowoz(data, 420, 0)
    cfg = build_config(parse_args([
        "all", "--stub", "--no-strict", "--data-dir", str(data), "--out", str(root / "run"),
        "--seeds", "42", "43", "--arms", "sft", "sentiment_only", "dpo", "caro", "caro_no_abstain", "sft_bon_rm",
        "sft_lenmatch", "--ablate-fit-seeds", "2", "--human-contexts", "40", "--human-validity-pairs", "40",
        "--n-signflip", "2000", "--reward-n-perm", "200",
        "--sim-rollouts", "4", "--sim-rollouts-corpus", "3",
        "--validate-contexts", "80", "--validate-variants", "4", "--ablate-contexts", "60",
        "--corpus-contexts", "300", "--corpus-variants", "4",
        "--eval-turns", "150", "--grpo-steps", "6", "--grpo-contexts", "4", "--grpo-group", "4",
        "--reward-ensemble", "3", "--feat-dim", "64",
    ]))
    stage_all(cfg)
    rep = load_json(cfg.out / "report.json")
    assert rep["arms"], "no arms in report"
    assert set(rep["arms"]) == set(KNOWN_ARMS), f"arms missing from the report: {set(KNOWN_ARMS) - set(rep['arms'])}"
    for k in ("caro_vs_sentiment_only:outcome", "caro_vs_dpo:outcome", "caro_vs_sft_lenmatch:outcome",
              "caro_vs_sft:info_recall"):
        assert k in rep["contrasts"], f"missing contrast {k}"
    assert "noninferior" in rep["contrasts"]["caro_vs_sft:info_recall"]
    assert rep["cluster_unit"].startswith("dialogue")
    sft_rows = [load_json(cfg.out / f"eval_sft_{sd}.json") for sd in cfg.seeds]
    assert [r["response"] for r in sft_rows[0]] != [r["response"] for r in sft_rows[1]], \
        "SFT rows are identical across seeds: they were copied, not evaluated"
    for name in ("report_ext.json", "length_analysis.json", "claims.json", "run_manifest.json",
                 "human_study/items.csv", "human_study/key_DO_NOT_SHARE.json", "ablation_simulator.json",
                 "validation_gate_sample.npz"):
        assert (cfg.out / name).exists(), f"missing {name}"
    sa = load_json(cfg.out / "ablation_simulator.json")
    assert sa["replica"]["available"] and sa["replica"]["reproduces_gate"], f"gate replica failed: {sa['replica']}"
    assert sa["variants_per_context"] == cfg.validate_variants
    sv0 = load_json(cfg.out / "simulator_validation.json")
    assert sv0["responsiveness_icc"] is None and sv0["replicate_reliability"] is None or sv0.get("icc_note") == "split-half"
    sen = load_json(cfg.out / "sensitivity.json")
    assert any(g["selected"] for g in sen["reward_regularisation"]), "the selected reward cell is not in the grid"
    abr = load_json(cfg.out / "ablation_reward.json")
    assert "no abstention" not in abr and "no pessimism (kappa=0)" in abr
    # simulate a filled human-annotation file from the key and run the human analysis end to end
    _fake_annotations(cfg)
    hs = stage_human_analyze(cfg)
    assert hs["validity"] and hs["comparisons"], "human analysis produced no estimates"
    assert "bad_annotator" in hs["excluded_annotators"], "the attention check did not exclude a random annotator"
    for name in ("simulator_validation.json", "reward.json", "length_control.json"):
        assert (cfg.out / name).exists(), f"missing {name}"
    sv = load_json(cfg.out / "simulator_validation.json")
    assert sv["paired_intervention"]["filler_bank"] == "heldout", "gate did not use the held-out bank"
    assert load_json(cfg.out / "length_control.json")["kind"] == "interventional"
    corpus = load_corpus(cfg.out / "corpus.npz")
    assert len(corpus) >= 100
    assert all(g.probe_feats is not None and len(g.probe_feats) for g in corpus), "corpus lacks probe pairs"
    for name in ("ablation_reward.json", "sensitivity.json", "reproducibility_checklist.json"):
        assert (cfg.out / name).exists(), f"missing analysis artefact {name}"
    ab = load_json(cfg.out / "ablation_reward.json")
    assert "full" in ab and "p_holm" in ab["neither (plain PL ranker)"], "reward ablation incomplete"
    assert "equivalent_to_full" in ab["neither (plain PL ranker)"], "reward ablation lacks the TOST"
    tex = sorted((cfg.out / "tables").glob("*.tex"))
    assert len(tex) >= 3, f"expected LaTeX tables, found {[t.name for t in tex]}"
    con = list(rep["contrasts"].values())
    assert con and "cliffs_delta" in con[0] and "mde80" in con[0], "report lacks effect sizes / power"
    rw = load_json(cfg.out / "reward.json")
    assert rw["gate"]["padding_intervention"]["n"] >= 30, "reward padding test did not run"
    assert "length_excess" in rw["gate"] and rw["selection"], "reward length-calibration report missing"
    pi = sv["paired_intervention"]
    print(f"\nSELFTEST OK | {len(corpus)} groups | simulator held-out padding delta={pi['delta_mean']:+.5f} "
          f"(margin +-{pi['equivalence_margin']:.5f}, TOST {'PASS' if pi['tost_passes'] else 'FAIL'}) | reward "
          f"padding delta={rw['gate']['padding_intervention'].get('delta_mean', float('nan')):+.5f} | reward "
          f"length rho {rw['gate']['length_excess']['rho_reward']:+.3f} vs labels "
          f"{rw['gate']['length_excess']['rho_label']:+.3f} (clp={rw['selected']['clp']:g}, "
          f"length_cal={rw['selected']['length_cal']:g})")
    print(f"analysis artefacts | {len(tex)} LaTeX tables: {[t.name for t in tex]}")
    print(json.dumps(rep["arms"], indent=2))


def main(argv: Optional[Sequence[str]] = None) -> None:
    a = parse_args(sys.argv[1:] if argv is None else argv)
    if a.stage == "selftest":
        run_selftest()
        return
    cfg = build_config(a)
    arms = [a.arm] if a.arm else ordered_arms(cfg.arms)
    # Stage names are resolved from globals() when main() runs.  That only works because the
    # `if __name__ == "__main__"` guard is the LAST statement of the module: Python executes the file
    # top to bottom, so a function defined below the guard does not exist yet when main() is called.
    _dispatch = {
        "all": "stage_all", "sft": "stage_sft", "simulator": "stage_simulator",
        "validate": "stage_validate", "corpus": "stage_corpus", "reward": "stage_reward",
        "report": "stage_report", "eval-external": "stage_eval_external",
        "cross-eval": "stage_cross_eval",
        "ablate-reward": "stage_ablate_reward", "ablate-simulator": "stage_ablate_simulator",
        "sensitivity": "stage_sensitivity", "length-analysis": "stage_length_analysis",
        "human-export": "stage_human_export", "human-analyze": "stage_human_analyze",
        "claims": "stage_claims", "paper": "stage_paper", "analysis": "stage_analysis",
    }
    simple = {k: globals()[v] for k, v in _dispatch.items()}
    if a.stage in simple:
        simple[a.stage](cfg)
    elif a.stage == "train":
        for arm in arms:
            for sd in cfg.seeds:
                stage_train(cfg, arm, sd)
    elif a.stage == "eval":
        for arm in arms:
            for sd in cfg.seeds:
                stage_eval(cfg, arm, sd)




# ---------------------------------------------------------------------------------------------
# v14 automatic-evidence additions (appended; self-contained top-level definitions)
# ---------------------------------------------------------------------------------------------
SLOT_PATTERNS = {
    "time":     re.compile(r"\b\d{1,2}[:.]\d{2}\s*(?:am|pm)?\b|\b\d{1,2}\s*(?:am|pm)\b", re.I),
    "date":     re.compile(
        r"\b(?:mon|tue|wed|thu|fri|sat|sun)\w*\b"
        r"|\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\b"
        r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2}\b", re.I),
    "price":    re.compile(r"[£$€]\s?\d+(?:\.\d{1,2})?|\b\d+\s?(?:pounds?|dollars?|euros?)\b", re.I),
    "ref":      re.compile(r"\b(?:ref(?:erence)?|code|number|booking)\s*[:#]?\s*[A-Z0-9]{4,}\b"
                           r"|\b[A-Z]{2,}\d{3,}\b"),
    "postcode": re.compile(r"\b[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}\b", re.I),
    "phone":    re.compile(r"\b(?:\+?\d[\d\s\-]{7,}\d)\b"),
    "venue":    re.compile(r"\b(?:hotel|restaurant|museum|college|hospital|park|station|airport|cafe|pub|"
                           r"theatre|theater|cinema|gallery|club|inn|lodge)\b", re.I),
}


def slot_set(text: str) -> Dict[str, set]:
    t = norm_text(text)
    return {k: set(p.findall(t)) for k, p in SLOT_PATTERNS.items()}


def slot_prf(response: str, reference: str) -> Tuple[float, float, float]:
    rr, rg = slot_set(response), slot_set(reference)
    gold = set().union(*rg.values()) if rg else set()
    pred = set().union(*rr.values()) if rr else set()
    if not gold:
        return float("nan"), float("nan"), float("nan")
    hit = len(gold & pred)
    p = hit / max(len(pred), 1)
    r = hit / len(gold)
    f = 2 * p * r / max(p + r, 1e-9)
    return float(p), float(r), float(f)


BEHAVIOUR_FAMILIES = {
    "apology":         re.compile(r"\b(?:sorry|apolog|unfortunately|afraid)\w*\b", re.I),
    "acknowledgement": re.compile(r"\b(?:i understand|understood|i see|noted|of course|certainly|"
                                  r"sure thing|absolutely|no problem|you're right|thank you)\b", re.I),
    "offer_help":      re.compile(r"\b(?:let me|i can|i will|i'll|would you like|shall i|happy to|"
                                  r"glad to|feel free)\b", re.I),
    "question":        re.compile(r"\?"),
    "hedge":           re.compile(r"\b(?:perhaps|maybe|might|possibly|it seems|it appears|"
                                  r"i think|i believe|if you like)\b", re.I),
    "repetition":      re.compile(r"\b(\w{3,})\s+\1\b", re.I),
}


def behaviour_features(text: str) -> Dict[str, float]:
    t = norm_text(text)
    w = max(1, len(t.split()))
    return {f"beh_{k}": 100.0 * len(p.findall(t)) / w for k, p in BEHAVIOUR_FAMILIES.items()}


# ---------------------------------------------------------------------------------------------
# v14 automatic-evidence addition: cross-evaluator within-context agreement.
# ---------------------------------------------------------------------------------------------
def _release_gpu() -> None:
    """Collect garbage and release cached GPU memory.  The CALLER must `del` its references first:
    deleting a name inside a helper does not drop the caller's reference to the model."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def stage_cross_eval(cfg: "Config") -> Dict[str, Any]:
    """Automatic analogue of the within-context validity test.

    For held-out contexts, sample K responses from the frozen SFT policy, score every response with
    BOTH automatic evaluators (the CARO simulator and the out-of-family evaluator) and compare their
    within-context PREFERENCES pairwise.  This measures the property the reward needs -- ranking
    alternatives to the SAME context -- under a second, independent automatic evaluator.  It is NOT
    human evidence and is reported as such.

    Models are loaded one at a time and released (SFT -> simulator -> judge), so peak GPU memory is
    one model, not three."""
    ex = Experiment(cfg, "7_cross_eval")
    out_file = cfg.out / "cross_eval.json"
    if out_file.exists():
        ex.logger.info("cross_eval.json already present; reusing (delete to recompute)")
        return load_json(out_file)
    te = filter_turns(ex.turns, "test", require_next=True, limit=cfg.cross_eval_contexts,
                      seed=cfg.seed, logger=ex.logger, what="cross-eval contexts")
    if len(te) < 50:
        raise RuntimeError(f"cross_eval needs >= 50 test contexts, got {len(te)}")
    K = max(2, int(cfg.cross_eval_k))
    flat_t = [t for t in te for _ in range(K)]
    seed0 = eval_gen_seed(cfg.seed, 0, 0)

    sft = ex.policy_with("sft_policy")
    responses = sft.generate([agent_prompt(t) for t in flat_t], ex.gen(0.7), seed=seed0)
    del sft
    _release_gpu()
    sim = ex.simulator(ex.policy_with("simulator"))
    y_sim = np.asarray(sim.rollout(flat_t, responses, crn_seed=seed0), float)
    del sim
    _release_gpu()
    if cfg.stub:
        judge, emo = StubPolicy(ex.logger, cfg.seed + 99), StubEmotion()
    else:
        judge = Policy(cfg.external_model, cfg.device, cfg.models_dir, ex.logger,
                       load_4bit=cfg.load_4bit, gen_batch=cfg.gen_batch,
                       score_batch=cfg.score_batch, with_lora=False)
        emo = EmotionValenceScorer(cfg.external_emotion_model, cfg.device, cfg.models_dir, ex.logger)
    reps = judge.generate(external_customer_prompts(judge, flat_t, responses),
                          GenConfig(max_new_tokens=40, min_new_tokens=4, temperature=0.9), seed=seed0)
    y_ext = np.asarray(emo(reps), float) - np.asarray(emo([t.user_text for t in flat_t]), float)
    del judge, emo
    _release_gpu()

    Ys, Ye = y_sim.reshape(len(te), K), y_ext.reshape(len(te), K)
    iu = np.triu_indices(K, 1)
    per_ctx, kept_cl = [], []
    agree = tot = 0
    for c, t in enumerate(te):
        a = (Ys[c][:, None] - Ys[c][None, :])[iu]
        b = (Ye[c][:, None] - Ye[c][None, :])[iu]
        m = np.abs(a) > 1e-9
        if not m.any():
            continue
        hits = int(np.sum(a[m] * b[m] > 0))
        agree += hits
        tot += int(m.sum())
        per_ctx.append(hits / int(m.sum()))
        kept_cl.append(t.dialogue_id)          # cluster of THIS context (v14-as-sent sliced the first n)
    pca = np.asarray(per_ctx, float)
    kcl = np.asarray(kept_cl, dtype=object)
    rate = agree / max(tot, 1)
    _, lo, hi = cluster_bootstrap_ci(lambda ix: float(np.mean(pca[ix])), kcl, 2000, cfg.seed, conf=0.95)
    sf = sign_flip_test(pca - 0.5, kcl, n_mc=cfg.n_signflip, seed=cfg.seed + 1)
    # Between-context anchors on CONTEXT means: the label is per context, so correlating it with the K
    # responses of each context would count every context K times.
    hv = np.asarray([t.human_valence if t.human_valence is not None else np.nan for t in te], float)
    rho_sim, rho_ext = spearman(Ys.mean(1), hv), spearman(Ye.mean(1), hv)
    out = {"n_contexts": int(len(te)), "n_contexts_scored": int(pca.size), "K": int(K), "n_pairs": int(tot),
           "within_context_agreement": float(rate),
           "within_context_agreement_ctx_mean": float(pca.mean()) if pca.size else float("nan"),
           "within_context_agreement_ci95": [float(lo), float(hi)],
           "p_vs_chance": sf["p"], "p_vs_chance_text": sf["p_text"],
           "between_context_rho_simulator": float(rho_sim),
           "between_context_rho_external": float(rho_ext),
           "n_labelled_contexts": int(np.isfinite(hv).sum()),
           "external_model": cfg.external_model,
           "external_emotion_model": cfg.external_emotion_model,
           "note": "Automatic two-evaluator within-context agreement (chance = 0.5). Not human evidence."}
    dump_json(out, out_file)
    ex.logger.info("cross-evaluator within-context agreement | %d contexts x %d variants = %d pairs | "
                   "agreement %.3f CI95[%.3f,%.3f] (chance 0.5, p %s) | between-context rho (context means) "
                   "sim %.3f ext %.3f", len(te), K, tot, rate, lo, hi, sf["p_text"], rho_sim, rho_ext)
    return out


if __name__ == "__main__":      # keep this the last statement of the file (see main())
    main()
