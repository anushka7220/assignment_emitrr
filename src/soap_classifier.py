import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class SOAPSectionClassifier:
    """
    Inference-only SOAP section classifier.

    Loads a trained model from disk and predicts
    one of: Subjective, Objective, Assessment, Plan, Other
    for a single sentence.
    """

    def __init__(self, model_path: str):
        """
        Args:
            model_path (str): Path to trained model directory
                              e.g. "models/soap_classifier"
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()  # IMPORTANT: inference mode

    def classify(self, sentence: str) -> str:
        """
        Classify a single sentence into a SOAP section.

        Args:
            sentence (str): Input sentence

        Returns:
            str: One of ["Subjective", "Objective", "Assessment", "Plan", "Other"]
        """
        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=96
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        label_id = torch.argmax(outputs.logits, dim=1).item()
        return self.model.config.id2label[label_id]
