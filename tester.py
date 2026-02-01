from src.transcript_parser import parse_transcript
from src.sentence_utils import split_sentences, classify_sentences
from src.soap_classifier import SOAPSectionClassifier
from src.soap_generator import generate_soap
from collections import defaultdict
import json


def group_by_section(classified):
    grouped = defaultdict(list)
    for item in classified:
        if item["section"] == "Other":
            # salvage physical exam statements
            if any(k in item["text"].lower() for k in [
                "range of motion", "tenderness", "movement", "exam", "looks good"
            ]):
                grouped["Objective"].append(item["text"])
        else:
            grouped[item["section"]].append(item["text"])

    return grouped


if __name__ == "__main__":

    raw_text = """
    Physician: Good morning, Ms. Jones. How are you feeling today?
    Patient: Good morning, doctor. I’m doing better, but I still have some discomfort now and then.
    Physician: I understand you were in a car accident last September. Can you walk me through what happened?
    Patient: Yes, it was on September 1st, around 12:30 in the afternoon. I was driving from Cheadle Hulme to Manchester when I had to stop in traffic. Out of nowhere, another car hit me from behind, which pushed my car into the one in front.
    Physician: That sounds like a strong impact. Were you wearing your seatbelt?
    Patient: Yes, I always do.
    Physician: What did you feel immediately after the accident?
    Patient: At first, I was just shocked. But then I realized I had hit my head on the steering wheel, and I could feel pain in my neck and back almost right away.
    Physician: Did you seek medical attention at that time?
    Patient: Yes, I went to Moss Bank Accident and Emergency. They checked me over and said it was a whiplash injury, but they didn’t do any X-rays. They just gave me some advice and sent me home.
    Physician: How did things progress after that?
    Patient: The first four weeks were rough. My neck and back pain were really bad—I had trouble sleeping and had to take painkillers regularly. It started improving after that, but I had to go through ten sessions of physiotherapy to help with the stiffness and discomfort.
    Physician: That makes sense. Are you still experiencing pain now?
    Patient: It’s not constant, but I do get occasional backaches. It’s nothing like before, though.
    Physician: That’s good to hear. Have you noticed any other effects, like anxiety while driving or difficulty concentrating?
    Patient: No, nothing like that. I don’t feel nervous driving, and I haven’t had any emotional issues from the accident.
    Physician: And how has this impacted your daily life? Work, hobbies, anything like that?
    Patient: I had to take a week off work, but after that, I was back to my usual routine. It hasn’t really stopped me from doing anything.
    Physician: That’s encouraging. Let’s go ahead and do a physical examination to check your mobility and any lingering pain.
    Physician: Everything looks good. Your neck and back have a full range of movement, and there’s no tenderness or signs of lasting damage.
    Physician: Given your progress, I’d expect you to make a full recovery within six months of the accident.
    Patient: Thank you, doctor. I appreciate it.

    """

    # 1. Parse transcript
    conversation = parse_transcript(raw_text)

    # 2. Split into sentences
    sentences = split_sentences(conversation)

    # 3. Load ML classifier
    classifier = SOAPSectionClassifier("models/soap_classifier")

    # 4. Classify sentences
    classified = classify_sentences(sentences, classifier)

    # OPTIONAL: see ML decisions
    print("\n--- Sentence Classification ---")
    for c in classified:
        print(f"{c['text']}  -->  {c['section']}")

    # 5. Group by SOAP section
    grouped = group_by_section(classified)

    # 6. Generate SOAP note
    soap_note = generate_soap(grouped)

    # 7. Print SOAP output
    print("\n--- FINAL SOAP NOTE ---")
    print(json.dumps(soap_note, indent=2))
