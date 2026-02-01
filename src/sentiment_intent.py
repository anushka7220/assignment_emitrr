from transformers import pipeline

# Sentiment analysis pipeline (DistilBERT)
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

#collecting patient's text only
def collect_patient_text(parsed_conversation):
    texts = [
        turn["text"]
        for turn in parsed_conversation["conversation"]
        if turn["speaker"] == "Patient"
    ]
    return " ".join(texts)

#classification of sentiment first done on purely rule based approach
#then we moved onto the model based approach 
#then we moved onto the hybrid approach that is mix of rule based and model based
#lets finetune the model for clinical intrepretaions for better results
def classify_sentiment_bert(patient_text):
    
    # Step 1: BERT sentiment
    result = sentiment_pipeline(patient_text)[0]
    label = result["label"]
    score = result["score"]

    # Step 2: Clinical reassurance override
    reassurance_signals = [
        "doing better",
        "nothing like before",
        "thank you",
        "i appreciate it",
        "it started improving",
        "no emotional issues",
        "not constant"
    ]

    if any(phrase in patient_text for phrase in reassurance_signals):
        return "Reassured"

    # Step 3: Anxiety signals
    anxiety_signals = ["worried", "anxious", "afraid", "concerned"]

    if any(word in patient_text for word in anxiety_signals):
        return "Anxious"

    # Step 4: Fallback to model
    if label == "NEGATIVE" and score > 0.6:
        return "Anxious"

    if label == "POSITIVE" and score > 0.6:
        return "Reassured"

    return "Neutral"


#intent detecttion is rule based
def detect_intent(patient_text):
    text = patient_text.lower()

    if any(word in text for word in ["worried", "hope", "will this", "should i"]):
        return "Seeking reassurance"

    if any(word in text for word in ["pain", "ache", "discomfort", "stiffness"]):
        return "Reporting symptoms"

    if any(word in text for word in ["afraid", "anxious", "concern"]):
        return "Expressing concern"

    return "General discussion"

def analyze_sentiment_intent(parsed_conversation):
    patient_text = collect_patient_text(parsed_conversation)

    sentiment = classify_sentiment_bert(patient_text)
    intent = detect_intent(patient_text)

    return {
        "Sentiment": sentiment,
        "Intent": intent
    }
