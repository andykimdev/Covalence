"""
clinical_codes.py

Single source of truth for all clinical coding systems, diagnostic thresholds,
guideline-based treatment mappings, and disease graph edges used by the agent.

Every entry is traceable to a published clinical practice guideline.
If Synthea uses a different code than expected, fix it here and all
modules pick up the change.

Code systems:
    LOINC   - lab tests and vitals (observations)
    SNOMED  - conditions and diagnoses
    RxNorm  - medications (matched by keyword on description, not code)
    ICD-10  - billing codes (not used directly, included for reference)
"""

# ============================================================================
# LOINC CODES (labs and vitals)
# Used by: step 2 (condition inference), step 4 (graph lab evidence scoring)
# ============================================================================

LOINC = {
    # diabetes monitoring
    "A1C": "4548-4",                # Hemoglobin A1c
    "FASTING_GLUCOSE": "2339-0",    # Fasting glucose

    # kidney function
    "EGFR": "33914-3",             # Estimated Glomerular Filtration Rate
    "CREATININE": "2160-0",        # Serum creatinine
    "BUN": "6299-2",               # Blood urea nitrogen
    "UACR": "89579-7",            # Urine albumin-to-creatinine ratio
    "POTASSIUM": "2823-3",        # Serum potassium (ACEi/ARB monitoring)

    # lipid panel
    "LDL": "18262-6",             # LDL Cholesterol
    "HDL": "2085-9",              # HDL Cholesterol
    "TOTAL_CHOLESTEROL": "2093-3", # Total Cholesterol
    "TRIGLYCERIDES": "2571-8",    # Triglycerides

    # blood pressure
    "SYSTOLIC_BP": "8480-6",      # Systolic blood pressure
    "DIASTOLIC_BP": "8462-4",     # Diastolic blood pressure

    # body composition
    "BMI": "39156-5",             # Body mass index
    "BODY_WEIGHT": "29463-7",     # Body weight

    # heart failure
    "NT_PROBNP": "17856-6",       # NT-proBNP (heart failure severity)

    # hematology
    "HEMOGLOBIN": "718-7",        # Hemoglobin (anemia detection)
}


# ============================================================================
# SNOMED CT CODES (conditions/diagnoses)
# Used by: step 1 (ingest), step 2 (inference), step 3 (care gaps), step 4 (graph)
# ============================================================================

SNOMED = {
    # type 2 diabetes
    "T2DM": 44054006,

    # chronic kidney disease (Synthea uses stage-specific codes)
    "CKD_STAGE1": 431855005,
    "CKD_STAGE2": 431856006,
    "CKD_STAGE3": 433144002,
    "CKD_STAGE4": 431857002,
    "CKD_STAGE5": 433146000,
    "CKD_UNSPECIFIED": 709044004,

    # cardiovascular
    "CHF": 88805009,               # Chronic congestive heart failure
    "HYPERTENSION": 59621000,      # Essential hypertension
    "AFIB": 49436004,              # Atrial fibrillation

    # metabolic
    "HYPERLIPIDEMIA": 55822004,
    "OBESITY": 162864005,          # BMI 30+ finding

    # other tracked conditions
    "ANEMIA": 271737000,
    "COPD": 185086009,
    "MDD": 370143000,              # Major depressive disorder

    # diabetic complications (for enrichment context)
    "DIABETIC_NEPHROPATHY": 127013003,
    "DIABETIC_RETINOPATHY": 422034002,
    "PERIPHERAL_NEUROPATHY": 302226006,
}

# convenience set for matching any CKD stage
CKD_ALL_CODES = {
    SNOMED["CKD_STAGE1"],
    SNOMED["CKD_STAGE2"],
    SNOMED["CKD_STAGE3"],
    SNOMED["CKD_STAGE4"],
    SNOMED["CKD_STAGE5"],
    SNOMED["CKD_UNSPECIFIED"],
}


# ============================================================================
# MEDICATION KEYWORDS (matched against Synthea description field)
# Used by: step 3 (care gap detection)
#
# Why keywords instead of RxNorm codes:
# The same drug has multiple RxNorm codes for different dosages and
# formulations. Matching on description keywords is more reliable
# for care gap detection.
# ============================================================================

MED_CLASSES = {
    "statin": [
        "atorvastatin", "simvastatin", "rosuvastatin",
        "pravastatin", "lovastatin", "fluvastatin", "pitavastatin",
    ],
    "ACEi": [
        "lisinopril", "enalapril", "ramipril",
        "benazepril", "captopril", "quinapril",
    ],
    "ARB": [
        "losartan", "valsartan", "irbesartan",
        "candesartan", "olmesartan", "telmisartan",
    ],
    "ARNi": [
        "sacubitril",  # sacubitril/valsartan (Entresto)
    ],
    "SGLT2i": [
        "empagliflozin", "dapagliflozin", "canagliflozin",
        "ertugliflozin",
    ],
    "beta_blocker": [
        "metoprolol", "carvedilol", "atenolol",
        "bisoprolol", "nebivolol",
    ],
    "MRA": [
        "spironolactone", "eplerenone",
    ],
    "antidiabetic": [
        "metformin", "insulin", "glipizide", "glyburide", "glimepiride",
        "sitagliptin", "saxagliptin", "linagliptin", "alogliptin",
        "empagliflozin", "dapagliflozin", "canagliflozin",
        "semaglutide", "liraglutide", "dulaglutide", "exenatide",
        "tirzepatide",
    ],
    "GLP1_agonist": [
        "semaglutide", "liraglutide", "dulaglutide",
        "exenatide", "tirzepatide",
    ],
    "antihypertensive": [
        "lisinopril", "enalapril", "ramipril",
        "losartan", "valsartan", "irbesartan",
        "amlodipine", "nifedipine", "diltiazem",
        "metoprolol", "atenolol", "carvedilol",
        "hydrochlorothiazide", "chlorthalidone",
    ],
    "CCB": [
        "amlodipine", "nifedipine", "diltiazem", "verapamil",
    ],
    "thiazide": [
        "hydrochlorothiazide", "chlorthalidone", "indapamide",
    ],
    "anticoagulant": [
        "apixaban", "rivaroxaban", "warfarin",
        "dabigatran", "edoxaban",
    ],
}

# combined ACEi/ARB/ARNi for care gap checks that accept any RAS inhibitor
MED_CLASSES["RAS_inhibitor"] = (
    MED_CLASSES["ACEi"] + MED_CLASSES["ARB"] + MED_CLASSES["ARNi"]
)


# ============================================================================
# CONDITION INFERENCE RULES (step 2)
# Maps lab values to potential undiagnosed conditions.
# Each rule cites the published guideline that defines the threshold.
#
# The agent flags these as "potential, requires clinical review"
# consistent with FDA CDS guidance.
# ============================================================================

INFERENCE_RULES = [
    {
        "condition": "CKD_STAGE3",
        "snomed": SNOMED["CKD_STAGE3"],
        "description": "Chronic kidney disease stage 3",
        "lab": LOINC["EGFR"],
        "operator": "<",
        "threshold": 60,
        "require_repeat": True,
        "repeat_days_apart": 90,
        "source": "KDIGO 2024 Clinical Practice Guideline for CKD",
        "note": "eGFR < 60 mL/min/1.73m2 on two occasions >= 90 days apart",
    },
    {
        "condition": "T2DM",
        "snomed": SNOMED["T2DM"],
        "description": "Type 2 diabetes mellitus",
        "lab": LOINC["A1C"],
        "operator": ">=",
        "threshold": 6.5,
        "require_repeat": True,
        "repeat_days_apart": 0,
        "source": "ADA Standards of Care in Diabetes, 2025",
        "note": "A1c >= 6.5% (confirmatory testing recommended)",
    },
    {
        "condition": "HYPERTENSION",
        "snomed": SNOMED["HYPERTENSION"],
        "description": "Essential hypertension",
        "lab": LOINC["SYSTOLIC_BP"],
        "operator": ">=",
        "threshold": 130,
        "require_repeat": True,
        "repeat_days_apart": 0,
        "source": "2025 ACC/AHA High Blood Pressure Guideline",
        "note": "SBP >= 130 mmHg on repeated measurements",
    },
    {
        "condition": "HYPERLIPIDEMIA",
        "snomed": SNOMED["HYPERLIPIDEMIA"],
        "description": "Hyperlipidemia",
        "lab": LOINC["LDL"],
        "operator": ">=",
        "threshold": 190,
        "require_repeat": False,
        "repeat_days_apart": 0,
        "source": "2026 ACC/AHA Guideline on Management of Dyslipidemia",
        "note": "LDL >= 190 mg/dL indicates severe hypercholesterolemia",
    },
    {
        "condition": "OBESITY",
        "snomed": SNOMED["OBESITY"],
        "description": "Obesity (BMI >= 30)",
        "lab": LOINC["BMI"],
        "operator": ">=",
        "threshold": 30,
        "require_repeat": False,
        "repeat_days_apart": 0,
        "source": "WHO/CDC obesity classification",
        "note": "BMI >= 30 kg/m2",
    },
    {
        "condition": "ANEMIA",
        "snomed": SNOMED["ANEMIA"],
        "description": "Anemia",
        "lab": LOINC["HEMOGLOBIN"],
        "operator": "<",
        "threshold": 10,
        "require_repeat": False,
        "repeat_days_apart": 0,
        "source": "WHO hemoglobin thresholds for anemia",
        "note": "Hemoglobin < 10 g/dL",
    },
]


# ============================================================================
# GUIDELINE TREATMENT MAPPINGS (step 3)
# Maps conditions to recommended medication classes.
# Used to detect care gaps: condition present + recommended med absent = gap.
# ============================================================================

GUIDELINE_TREATMENTS = {
    "T2DM": {
        "condition_codes": {SNOMED["T2DM"]},
        "required_meds": [
            {
                "class": "statin",
                "keywords": MED_CLASSES["statin"],
                "rationale": "Statin recommended for all T2DM patients aged 40-75",
            },
            {
                "class": "antidiabetic",
                "keywords": MED_CLASSES["antidiabetic"],
                "rationale": "Pharmacotherapy indicated for T2DM",
            },
        ],
        "source": "ADA Standards of Care in Diabetes, 2025",
    },
    "CKD": {
        "condition_codes": CKD_ALL_CODES,
        "required_meds": [
            {
                "class": "ACEi/ARB",
                "keywords": MED_CLASSES["RAS_inhibitor"],
                "rationale": "RAS inhibitor recommended for CKD with diabetes or albuminuria",
            },
            {
                "class": "SGLT2i",
                "keywords": MED_CLASSES["SGLT2i"],
                "rationale": "SGLT2i recommended for T2D + CKD with eGFR >= 20",
            },
        ],
        "source": "KDIGO 2024 Clinical Practice Guideline for CKD",
    },
    "HYPERTENSION": {
        "condition_codes": {SNOMED["HYPERTENSION"]},
        "required_meds": [
            {
                "class": "antihypertensive",
                "keywords": MED_CLASSES["antihypertensive"],
                "rationale": "Antihypertensive therapy recommended, target < 130/80 mmHg",
            },
        ],
        "source": "2025 ACC/AHA High Blood Pressure Guideline",
    },
    "CHF": {
        "condition_codes": {SNOMED["CHF"]},
        "required_meds": [
            {
                "class": "RAS inhibitor (ARNi preferred)",
                "keywords": MED_CLASSES["RAS_inhibitor"],
                "rationale": "ARNi first-line; ACEi if ARNi not feasible; ARB if ACEi intolerant",
            },
            {
                "class": "beta blocker",
                "keywords": MED_CLASSES["beta_blocker"],
                "rationale": "Evidence-based beta blocker for HFrEF",
            },
            {
                "class": "MRA",
                "keywords": MED_CLASSES["MRA"],
                "rationale": "Mineralocorticoid receptor antagonist for HFrEF",
            },
            {
                "class": "SGLT2i",
                "keywords": MED_CLASSES["SGLT2i"],
                "rationale": "SGLT2i recommended regardless of diabetes status",
            },
        ],
        "source": "2022 AHA/ACC/HFSA Guideline for HF Management (2025 ACC Decision Pathway Update)",
    },
    "HYPERLIPIDEMIA": {
        "condition_codes": {SNOMED["HYPERLIPIDEMIA"]},
        "required_meds": [
            {
                "class": "statin",
                "keywords": MED_CLASSES["statin"],
                "rationale": "High-intensity statin for LDL >= 190; moderate-intensity for lower risk",
            },
        ],
        "source": "2026 ACC/AHA Guideline on Management of Dyslipidemia",
    },
    "OBESITY": {
        "condition_codes": {SNOMED["OBESITY"]},
        "required_meds": [
            {
                "class": "GLP-1 agonist",
                "keywords": MED_CLASSES["GLP1_agonist"],
                "rationale": "GLP-1 RA for weight management in obesity with comorbidities",
            },
        ],
        "source": "AGA 2024 Clinical Practice Guideline on Pharmacological Interventions for Adults with Obesity",
    },
}


# ============================================================================
# DISEASE RELATIONSHIP GRAPH (step 4)
# Edges represent Relative Risk (RR) between conditions, extracted from a
# published comorbidity network of 13 million hospital patients (1997-2014).
#
# Source: "Comorbidity Networks From Population-Wide Health Data:
#         Aggregated Data of 8.9M Hospital Patients (1997-2014)"
#         Nature Scientific Data, 2025.
#         ICD-9 3-digit, "All patients" stratum.
#
# Format: (source, target, raw_rr, icd9_pair, co_occurrence)
# raw_rr = Relative Risk from dataset (how much more likely target is
#          given source, compared to general population)
# ============================================================================

DISEASE_GRAPH_EDGES_RAW = [
    # (source, target, RR, ICD-9 pair, co-occurrence count)
    # T2DM cluster
    ("T2DM", "CKD", 2.31, "250->585", 85272),
    ("T2DM", "CHF", 1.62, "250->428", 633475),
    ("T2DM", "HYPERLIPIDEMIA", 1.34, "250->272", 143554),
    ("T2DM", "HYPERTENSION", 1.40, "250->401", 1011337),
    ("T2DM", "OBESITY", 2.24, "250->278", 107949),

    # CKD cluster
    ("CKD", "CHF", 3.21, "428->585 (reverse)", 133470),
    ("CKD", "HYPERTENSION", 1.22, "401->585 (reverse)", 93131),
    ("CKD", "ANEMIA", 2.97, "285->585 (reverse)", 99699),

    # CHF cluster
    ("CHF", "AFIB", 2.08, "427->428 (reverse)", 1198928),
    ("CHF", "HYPERTENSION", 1.18, "401->428 (reverse)", 956919),

    # Hypertension cluster
    ("HYPERTENSION", "HYPERLIPIDEMIA", 1.67, "272->401 (reverse)", 372545),
    ("HYPERTENSION", "CKD", 1.22, "401->585", 93131),

    # Obesity cluster
    ("OBESITY", "T2DM", 2.24, "250->278 (reverse)", 107949),
    ("OBESITY", "HYPERTENSION", 1.73, "278->401", 173569),
    ("OBESITY", "HYPERLIPIDEMIA", 2.60, "272->278 (reverse)", 38887),
]

# Maximum RR in dataset (CKD -> CHF), used for normalization
MAX_RR = 3.21


def normalize_rr(rr: float) -> float:
    """
    Normalize Relative Risk to 0-1 scale.
    RR = 1.0 (no excess risk) -> 0.0
    RR = MAX_RR (strongest association) -> 1.0
    """
    if rr <= 1.0:
        return 0.0
    return min((rr - 1.0) / (MAX_RR - 1.0), 1.0)


# Normalized edges for use in scoring formula
# Format: (source, target, normalized_weight)
DISEASE_GRAPH_EDGES = [
    (source, target, normalize_rr(rr))
    for source, target, rr, _, _ in DISEASE_GRAPH_EDGES_RAW
]

# Pre-computed for reference:
# CKD -> CHF:              RR 3.21 -> normalized 1.00
# CKD -> Anemia:           RR 2.97 -> normalized 0.89
# Obesity -> Hyperlipidemia: RR 2.60 -> normalized 0.72
# T2DM -> CKD:             RR 2.31 -> normalized 0.59
# T2DM -> Obesity:         RR 2.24 -> normalized 0.56
# Obesity -> T2DM:         RR 2.24 -> normalized 0.56
# CHF -> AFib:             RR 2.08 -> normalized 0.49
# Obesity -> Hypertension: RR 1.73 -> normalized 0.33
# Hypertension -> HLD:     RR 1.67 -> normalized 0.30
# T2DM -> CHF:             RR 1.62 -> normalized 0.28
# T2DM -> Hypertension:    RR 1.40 -> normalized 0.18
# T2DM -> Hyperlipidemia:  RR 1.34 -> normalized 0.15
# Hypertension -> CKD:     RR 1.22 -> normalized 0.10
# CKD -> Hypertension:     RR 1.22 -> normalized 0.10
# CHF -> Hypertension:     RR 1.18 -> normalized 0.08

# SNOMED code lookup for graph nodes
GRAPH_NODE_CODES = {
    "T2DM": {SNOMED["T2DM"]},
    "CKD": CKD_ALL_CODES,
    "CHF": {SNOMED["CHF"]},
    "HYPERTENSION": {SNOMED["HYPERTENSION"]},
    "HYPERLIPIDEMIA": {SNOMED["HYPERLIPIDEMIA"]},
    "OBESITY": {SNOMED["OBESITY"]},
    "AFIB": {SNOMED["AFIB"]},
    "ANEMIA": {SNOMED["ANEMIA"]},
    "COPD": {SNOMED["COPD"]},
    "MDD": {SNOMED["MDD"]},
}


# ============================================================================
# LAB EVIDENCE SCORING (step 4)
# For each candidate graph expansion, score how close the patient's labs
# are to the diagnostic threshold. Combined with normalized RR to determine
# whether to expand.
#
# Formula: expansion_score = normalized_rr * lab_evidence_score
# Expand if: expansion_score > EXPANSION_THRESHOLD
# ============================================================================

EXPANSION_THRESHOLD = 0.15

LAB_EVIDENCE_RANGES = {
    "CKD": {
        "lab": LOINC["EGFR"],
        "direction": "below",  # lower = worse
        "ranges": [
            # (min_value, max_value, evidence_score)
            (None, 60, 1.0),    # below threshold (caught by inference)
            (60, 75, 0.6),      # approaching threshold
            (75, 90, 0.2),      # mildly reduced
            (90, None, 0.0),    # normal
        ],
        "missing_data_score": 0.3,
    },
    "HYPERLIPIDEMIA": {
        "lab": LOINC["LDL"],
        "direction": "above",  # higher = worse
        "ranges": [
            (190, None, 1.0),   # above threshold
            (160, 190, 0.6),    # approaching
            (130, 160, 0.3),    # borderline
            (None, 130, 0.0),   # normal
        ],
        "missing_data_score": 0.3,
    },
    "HYPERTENSION": {
        "lab": LOINC["SYSTOLIC_BP"],
        "direction": "above",
        "ranges": [
            (130, None, 1.0),
            (120, 130, 0.5),
            (None, 120, 0.0),
        ],
        "missing_data_score": 0.3,
    },
    "T2DM": {
        "lab": LOINC["A1C"],
        "direction": "above",
        "ranges": [
            (6.5, None, 1.0),
            (5.7, 6.5, 0.5),   # prediabetes range
            (None, 5.7, 0.0),
        ],
        "missing_data_score": 0.3,
    },
    "CHF": {
        "lab": LOINC["NT_PROBNP"],
        "direction": "above",
        "ranges": [
            (300, None, 0.8),   # elevated, suggestive
            (125, 300, 0.4),    # borderline
            (None, 125, 0.0),   # normal
        ],
        "missing_data_score": 0.2,
    },
    "ANEMIA": {
        "lab": LOINC["HEMOGLOBIN"],
        "direction": "below",
        "ranges": [
            (None, 10, 1.0),
            (10, 12, 0.4),
            (12, None, 0.0),
        ],
        "missing_data_score": 0.2,
    },
    "OBESITY": {
        "lab": LOINC["BMI"],
        "direction": "above",
        "ranges": [
            (30, None, 1.0),
            (25, 30, 0.4),      # overweight
            (None, 25, 0.0),
        ],
        "missing_data_score": 0.1,
    },
}


# ============================================================================
# ICD-10 REFERENCE (not used directly by the agent)
# Included for reference if ClinicalTrials.gov eligibility criteria
# mention ICD-10 codes. Standard SNOMED-to-ICD-10 mappings exist.
# ============================================================================

ICD10_REFERENCE = {
    "T2DM": "E11",
    "CKD_STAGE3": "N18.3",
    "CKD_STAGE4": "N18.4",
    "CKD_STAGE5": "N18.5",
    "CHF": "I50.9",
    "HYPERTENSION": "I10",
    "HYPERLIPIDEMIA": "E78.5",
    "OBESITY": "E66.9",
    "AFIB": "I48.91",
    "ANEMIA": "D64.9",
}