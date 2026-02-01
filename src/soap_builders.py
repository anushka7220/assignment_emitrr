import re

ANATOMY_SITES = ["neck", "back", "lower back", "upper back"]

INJURY_TERMS = [
    "whiplash",
    "strain",
    "sprain",
    "injury",
    "pain"
]

TRAUMA_PATTERNS = [
    "car accident",
    "motor vehicle",
    "hit from behind",
    "rear-ended",
    "collision",
    "accident"
]

def build_subjective(sentences):
    if not sentences:
        return {}

    text = " ".join(sentences).lower()

    # -------------------------
    # Chief Complaint
    # -------------------------
    anatomy = [s for s in ANATOMY_SITES if s in text]
    injury = next((t for t in INJURY_TERMS if t in text), None)

    if anatomy and injury:
        chief_complaint = " and ".join(anatomy).title() + f" {injury}"
    elif anatomy:
        chief_complaint = " and ".join(anatomy).title() + " pain"
    elif injury:
        chief_complaint = injury.title()
    else:
        chief_complaint = "Pain"

    # -------------------------
    # History of Present Illness
    # -------------------------
    cause_phrase = None

    for pattern in TRAUMA_PATTERNS:
        if pattern in text:
            cause_phrase = pattern
            break

    if cause_phrase:
        history = f"Following a {cause_phrase}"
    else:
        history = ""

    return {
        "Chief_Complaint": chief_complaint.strip().title(),
        "History_of_Present_Illness": history.strip().capitalize()
    }

def build_objective(sentences):
    if not sentences:
        return {}

    text = " ".join(sentences).lower()

    exam_actions = []
    findings = []

    # -------------------------
    # Physical Exam Extraction
    # -------------------------
    if "range of movement" in text or "range of motion" in text:
        exam_actions.append(
            "Assessed cervical and lumbar range of motion"
        )
        if "full" in text:
            findings.append("full range of motion")

    if "tenderness" in text:
        exam_actions.append("Palpated cervical and lumbar spine")
        if "no tenderness" in text:
            findings.append("no tenderness")

    if not exam_actions:
        physical_exam = "Physical examination performed"
    else:
        physical_exam = "; ".join(exam_actions)
        if findings:
            physical_exam += f", findings: {', '.join(findings)}"

    physical_exam = physical_exam.capitalize() + "."

    # -------------------------
    # Observations (derived)
    # -------------------------
    if "full range of motion" in text and "no tenderness" in text:
        observation = "Patient appears clinically stable"
    elif "tenderness" in text or "reduced" in text:
        observation = "Patient appears mildly unstable with minor functional limitation"
    else:
        observation = "Patient appears stable"

    return {
        "Physical_Exam": physical_exam,
        "Observations": observation
    }

def build_assessment(sentences, subjective=None, objective=None):
    # -------------------------
    # Inputs
    # -------------------------
    subj_text = " ".join((subjective or {}).values()).lower()
    obj_obs = (objective or {}).get("Observations", "").lower()

    # -------------------------
    # Diagnosis (cause-based)
    # -------------------------
    diagnoses = []

    trauma = any(k in subj_text for k in [
        "accident", "collision", "hit from behind", "motor vehicle"
    ])

    neck = "neck" in subj_text
    back = "back" in subj_text

    if trauma and neck:
        diagnoses.append("Whiplash injury")
    if trauma and back:
        diagnoses.append("Lumbar strain")

    if not diagnoses:
        diagnoses.append("Musculoskeletal pain")

    diagnosis = " and ".join(diagnoses)

    # -------------------------
    # Severity (objective-based)
    # -------------------------
    if "clinically stable" in obj_obs:
        severity = "Mild"
    elif "mildly unstable" in obj_obs:
        severity = "Mild"
    elif "unstable" in obj_obs:
        severity = "Severe"
    else:
        severity = "Undetermined"

    return {
        "Diagnosis": diagnosis,
        "Severity": severity
    }


def build_plan(sentences, assessment=None, objective=None):
    plan_text = " ".join(sentences).lower()
    severity = (assessment or {}).get("Severity", "").lower()

    treatments = []

    # -------------------------
    # Treatment extraction
    # -------------------------
    if "physiotherapy" in plan_text:
        treatments.append("Continue physiotherapy as advised")

    if "painkiller" in plan_text or "analgesic" in plan_text:
        treatments.append("Use analgesics as needed for pain relief")

    if not treatments:
        if severity == "mild":
            treatments.append("Conservative management with rest and activity as tolerated")
        else:
            treatments.append("Further evaluation and management required")

    treatment = ", ".join(treatments)

    # -------------------------
    # Follow-up logic
    # -------------------------
    if severity == "mild":
        follow_up = (
            "Patient advised to return if pain worsens or persists beyond four months"
        )
    else:
        follow_up = (
            "Patient advised to seek immediate medical attention"
        )

    return {
        "Treatment": treatment,
        "Follow-Up": follow_up
    }
