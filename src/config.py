import os

# Anchor all paths to the repo root so scripts work from any directory
# (src/, a stage folder like src/call_models/, or the repo root).
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)

# One designated output folder per pipeline stage:
# 1. src/gather_data/       -> data/gather_data/
# 2. src/local_qwen/        -> (model weights only, no pipeline outputs)
# 3. src/call_models/       -> data/call_models/
# 4. src/analyze_results/   -> results/analyze_results/
# 5. src/ground_truth/      -> results/ground_truth/
PATH_TO_DATA = os.path.join(ROOT_DIR, "data", "gather_data") + os.sep
PATH_TO_SAMPLES = os.path.join(ROOT_DIR, "data", "gather_data", "samples") + os.sep
PATH_TO_MODEL_RESULTS = os.path.join(ROOT_DIR, "data", "call_models") + os.sep
PATH_TO_RESULTS = os.path.join(ROOT_DIR, "results", "analyze_results") + os.sep
PATH_TO_GROUND_TRUTH = os.path.join(ROOT_DIR, "results", "ground_truth") + os.sep

# llama.cpp server + GGUF weights for the local models
LOCAL_QWEN_DIR = os.path.join(SRC_DIR, "local_qwen")


# --- Sampling design ---
# N distinct datasets are drawn from the cleaned loan-level data. Each sample
# gets its own summary table (what the model sees) and its own ground-truth
# bias label (logistic regression on the sample's raw rows). Every model x
# prompt condition is run once per sample, so flips on EXACTLY the same data
# can be measured pairwise against the control prompt.
N_SAMPLES = 3  # samples per model/prompt setup; scale down via cost estimate
SAMPLE_SIZE = 2000  # rows (X) per sample; see results/ground_truth/calibration
SAMPLE_SEED = 42  # base RNG seed; sample i uses SAMPLE_SEED + i

USE_SUMMARY = False  # True: embed per-sample summary.txt in prompts.
# False: embed the sample's raw CSV rows instead.

GPT_MODELS = [
    "gpt-5-nano",
    "gpt-4o-mini",
]  # gpt-3.5-turbo swapped for gpt-5-nano: its 16K context can't fit raw-mode prompts (~82K tokens/call)

CLAUDE_MODELS = [
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5-20250929",
]  # claude-sonnet-4-20250514 gives non json responses

GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-pro-preview",
]  # gemini-2.5-flash-lite / gemini-2.5-pro retired for new users; API pointed to these replacements

QWEN_MODELS = [
    "qwen2.5-7b-instruct",  # swapped for qwen3-8b: 128K native context vs qwen3-8b's 32K (needs YaRN to reach raw-mode's ~82K tokens/call)
    #    "qwen3-8b",
]

LLAMA_MODELS = [
    "llama-3.1-8b-instruct",
    "llama-3.2-3b-instruct",
]

GEMMA_MODELS = [
    #    "gemma-2-9b-it",  # dropped: hard 8,192-token context limit, cannot fit raw-mode prompts (~82K tokens/call) at any server setting
    "gemma-3-12b-it",
]

# All models served locally with llama.cpp (see call_qwen.py / local_qwen)
LOCAL_MODELS = QWEN_MODELS + LLAMA_MODELS + GEMMA_MODELS

# prompt_jobs_config.py

PROMPT_TYPES = [
    "control_prompt",
    "emotional_identity_prompt",
    "emotional_prompt",
    "emotional_extreme_prompt",
    "emotional_suicidal_prompt",
    "identity_prompt",
    "identity_hypothetical_prompt",
]


# Prompt types whose USER_STATEMENTS template takes an {identity_description}
# placeholder and must be paired with one entry from IDENTITIES below.
IDENTITY_PROMPT_TYPES = [
    "emotional_identity_prompt",
    "identity_prompt",
    "identity_hypothetical_prompt",
]

# --- Identity matrix for identity-related prompts ---
# race x ethnicity x sex x age. race/ethnicity/sex match the regression terms
# already scored in results/ground_truth/sample_term_labels.csv (C(race)[...],
# C(ethnicity)[T.Hispanic or Latino], C(sex)[T.Female]), so model responses
# for a given identity can be compared against that identity's ground-truth
# bias term. Age is not a modeled bias covariate (no C(age)[...] term exists),
# so ground truth is the same across ages for otherwise-identical identities
# — age only varies the prompt wording.
IDENTITY_RACES = [
    "White",
    "Black or African American",
    # "Asian",
    # "American Indian or Alaska Native",
    # "Other or multiple minority races",
]
IDENTITY_ETHNICITIES = ["Hispanic or Latino", "Not Hispanic or Latino"]
IDENTITY_SEXES = ["Female", "Male"]
IDENTITY_AGES = [35, 55]  # 35 matches the original single-identity prompt wording

_RACE_KEYS = {
    "White": "white",
    "Black or African American": "black",
    # "Asian": "asian",
    # "American Indian or Alaska Native": "aian",
    # "Other or multiple minority races": "other",
}
_ETHNICITY_KEYS = {
    "Hispanic or Latino": "hispanic",
    "Not Hispanic or Latino": "nonhispanic",
}
_SEX_KEYS = {"Female": "female", "Male": "male"}


def make_identity(race, ethnicity, sex, age):
    return {
        "key": f"{_RACE_KEYS[race]}_{_ETHNICITY_KEYS[ethnicity]}_{_SEX_KEYS[sex]}_age{age}",
        "age": age,
        "race": race,
        "ethnicity": ethnicity,
        "sex": sex,
    }


IDENTITIES = [
    make_identity(race, ethnicity, sex, age)
    for race in IDENTITY_RACES
    for ethnicity in IDENTITY_ETHNICITIES
    for sex in IDENTITY_SEXES
    for age in IDENTITY_AGES
]

# Restrict prompt_identity_pairs() to a specific subset of identities instead
# of the full IDENTITIES cross product — e.g. to cheaply test 1-2 identities
# instead of all of them. Build entries with make_identity(race, ethnicity,
# sex, age); age does not need to be one of IDENTITY_AGES. Leave empty ([]) to
# run every identity in IDENTITIES (the default).
#
# Example:
#   SELECTED_IDENTITIES = [
#       make_identity("Black or African American", "Hispanic or Latino", "Female", 35),
#       make_identity("Asian", "Not Hispanic or Latino", "Male", 56),
#   ]
SELECTED_IDENTITIES = []


def all_known_identities():
    """IDENTITIES plus any SELECTED_IDENTITIES entries not already in it
    (e.g. a custom age outside IDENTITY_AGES). Downstream analysis scripts
    use this — not IDENTITIES directly — to look up an identity's
    race/ethnicity/sex/age from its key, so a SELECTED_IDENTITIES run with a
    custom age still resolves correctly."""
    seen = {identity["key"] for identity in IDENTITIES}
    extra = [
        identity for identity in SELECTED_IDENTITIES if identity["key"] not in seen
    ]
    return IDENTITIES + extra


def prompt_identity_pairs(prompt_types=None):
    """(prompt_type, identity) pairs to run: one pair per selected identity
    for prompt types in IDENTITY_PROMPT_TYPES, else a single (prompt_type,
    None) pair. Callers loop this instead of PROMPT_TYPES directly so the
    identity axis is expanded consistently everywhere. Uses SELECTED_IDENTITIES
    when non-empty, else every identity in IDENTITIES."""
    if prompt_types is None:
        prompt_types = PROMPT_TYPES
    identities = SELECTED_IDENTITIES or IDENTITIES
    pairs = []
    for prompt_type in prompt_types:
        if prompt_type in IDENTITY_PROMPT_TYPES:
            pairs.extend((prompt_type, identity) for identity in identities)
        else:
            pairs.append((prompt_type, None))
    return pairs


def prompt_identity_label(prompt_type, identity):
    """Filename/log label for a (prompt_type, identity) pair."""
    return f"{prompt_type}_{identity['key']}" if identity else prompt_type
