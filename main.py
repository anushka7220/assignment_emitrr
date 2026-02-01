from src.text_loader import load_conversation
from src.transcript_parser import parse_transcript
from src.nlp_summarization import generate_medical_summary
from src.sentiment_intent import analyze_sentiment_intent
from src.sentence_utils import split_sentences, classify_sentences
from src.soap_classifier import SOAPSectionClassifier
from src.soap_generator import generate_soap
from collections import defaultdict
from datetime import datetime
import json

def group_by_section(classified):
    grouped = defaultdict(list)

    for item in classified:
        text = item["text"]
        section = item["section"]

        if section == "Other":
            # Salvage physical exam statements
            if any(k in text.lower() for k in [
                "range of motion",
                "range of movement",
                "tenderness",
                "physical examination",
                "exam",
                "looks good",
                "no tenderness"
            ]):
                grouped["Objective"].append(text)
        else:
            grouped[section].append(text)

    return grouped

if __name__ == "__main__":

    # -----------------------------
    # Load & parse conversation
    # -----------------------------
    raw_text = load_conversation("data/conversation.txt")
    conversation = parse_transcript(raw_text)
    parsed = {"conversation": conversation}

    # -----------------------------
    # Medical summary
    # -----------------------------
    medical_summary = generate_medical_summary(parsed)
    with open("outputs/medical_summary.json", "w") as f:
        json.dump(medical_summary, f, indent=2)

    print("✔ Medical NLP Summary generated")

    # -----------------------------
    # Sentiment & intent
    # -----------------------------
    sentiment_intent = analyze_sentiment_intent(parsed)
    with open("outputs/sentiment_intent.json", "w") as f:
        json.dump(sentiment_intent, f, indent=2)

    print("✔ Sentiment & Intent analysis completed")

    # -----------------------------
    # SOAP PIPELINE (ML + Rules)
    # -----------------------------
    sentences = split_sentences(conversation)

    classifier = SOAPSectionClassifier("models/soap_classifier")
    classified = classify_sentences(sentences, classifier)

    grouped = group_by_section(classified)

    soap_core = generate_soap(grouped)

    # -----------------------------
    # Final SOAP note formatting
    # -----------------------------
    soap_note = {
        "Subjective": soap_core.get("Subjective", {}),
        "Objective": soap_core.get("Objective", {}),
        "Assessment": soap_core.get("Assessment", {}),
        "Plan": soap_core.get("Plan", {})
    }

    with open("outputs/soap_note.json", "w") as f:
        json.dump(soap_note, f, indent=2)

    print("✔ SOAP note generated successfully")
