import nltk

# Download tokenizer once
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


def split_sentences(conversation: list) -> list:
    """
    Splits each conversation turn into individual sentences.

    Args:
        conversation (list): Output of parse_transcript()

    Returns:
        list: [{speaker, sentence}]
    """
    sentences = []

    for turn in conversation:
        for sent in nltk.sent_tokenize(turn["text"]):
            sent = sent.strip()
            if sent:
                sentences.append({
                    "speaker": turn["speaker"],
                    "sentence": sent
                })

    return sentences

def classify_sentences(sentences: list, classifier) -> list:
    """
    Classifies each sentence into a SOAP section.

    Args:
        sentences (list): Output of split_sentences()
        classifier: SOAPSectionClassifier instance

    Returns:
        list: [{speaker, text, section}]
    """
    classified = []

    for item in sentences:
        label = classifier.classify(item["sentence"])

        classified.append({
            "speaker": item["speaker"],
            "text": item["sentence"],
            "section": label
        })

    return classified
