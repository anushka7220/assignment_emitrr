import spacy
import re

nlp = spacy.load("en_core_sci_md")

def extract_patient_name(parsed_conversation):
    """
    Extracts patient name from physician greetings like:
    'Good morning, Ms. Jones'
    """
    for turn in parsed_conversation["conversation"]:
        if turn["speaker"] == "Physician":
            text = turn["text"]

            match = re.search(r"(Ms\.|Mr\.|Mrs\.)\s+([A-Z][a-z]+)", text)
            if match:
                title, last_name = match.groups()
                return f"{last_name}"  # first name not explicitly stated

    return "Unknown"

def collect_text_by_speaker(parsed_conversation, speaker):
    """
    Collects text spoken by a specific speaker (Patient or Physician).
    """
    texts = [
        turn["text"]
        for turn in parsed_conversation["conversation"]
        if turn["speaker"] == speaker
    ]
    return " ".join(texts)

INVALID_SYMPTOM_PHRASES = [
    "to manchester",
    "when i",
    "regularly"
]

def extract_current_status(parsed_conversation):
    for turn in parsed_conversation["conversation"]:
        if turn["speaker"] == "Patient":
            text = turn["text"].lower()
            tokens = text.split()

            if "occasional" in tokens or "constant" in tokens:
                idx = tokens.index("occasional")

                # find the next valid word after "occasional"
                if idx + 1 < len(tokens):
                    next_word = tokens[idx + 1]

                    # clean punctuation like "backaches."
                    next_word = re.sub(r"[^\w]", "", next_word)

                    if next_word:
                        return f"Occasional {next_word}"

    return "Symptoms improving"


def extract_prognosis(parsed_conversation):
    for turn in parsed_conversation["conversation"]:
        if turn["speaker"] == "Physician":
            text = turn["text"].lower()

            if "recovery" in text:
                # match patterns like "six months", "6 months"
                match = re.search(r"(\b\d+\b|\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b)\s+months", text)

                if match:
                    duration = match.group(0)  # e.g. "six months" or "6 months"
                    return f"Full recovery expected within {duration}"

                # fallback if recovery mentioned but duration missing
                return "Full recovery expected"

    return "Prognosis not explicitly stated"

def extract_diagnosis(parsed_conversation):
    """
    Extracts diagnosis based on explicit clinical statements.
    Priority:
    1. Physician-confirmed diagnosis
    2. Patient-reported diagnosis from medical visit
    """

    # 1️⃣ Physician-confirmed diagnosis (highest priority)
    for turn in parsed_conversation["conversation"]:
        if turn["speaker"] == "Physician":
            text = turn["text"].lower()

            if "whiplash" in text:
                return "Whiplash injury"

            if "diagnosis" in text or "assessment" in text:
                return turn["text"]

    # 2️⃣ Patient-reported diagnosis from hospital
    for turn in parsed_conversation["conversation"]:
        if turn["speaker"] == "Patient":
            text = turn["text"].lower()

            if "said it was" in text or "told me it was" in text or "diagnosed" in text:
                if "whiplash" in text:
                    return "Whiplash injury"

    return "Diagnosis not explicitly stated"

def generate_medical_summary(parsed_conversation):
    patient_text = collect_text_by_speaker(parsed_conversation, "Patient")
    physician_text = collect_text_by_speaker(parsed_conversation, "Physician")

    doc = nlp(patient_text + " " + physician_text)

    symptoms = set()
    treatments = set()
    prognosis = ""

    for ent in doc.ents:
        text = ent.text.lower()

        if any(k in text for k in ["pain", "ache", "discomfort", "stiffness", "backache"]):
          if not any(bad in text for bad in INVALID_SYMPTOM_PHRASES):
            symptoms.add(ent.text)


        if any(k in text for k in ["physiotherapy", "painkiller", "analgesic"]):
            treatments.add(ent.text)

        if "recovery" in text or "months" in text:
            prognosis = ent.text
    cleaned_symptoms = []

    for s in symptoms:
      s = s.lower()
      if "painkiller" in s or "physio" in s:
          continue
      cleaned_symptoms.append(s.capitalize())

    current_status = extract_current_status(parsed_conversation)
    prognosis_text = extract_prognosis(parsed_conversation)
    summary = {
        "Patient_Name": extract_patient_name(parsed_conversation),
        "Symptoms": cleaned_symptoms,
        "Diagnosis": extract_diagnosis(parsed_conversation),
        "Treatment": list(treatments),
        "Current_Status": current_status,
        "Prognosis": prognosis_text
    }

    return summary
