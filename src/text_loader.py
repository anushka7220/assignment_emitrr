def load_conversation(file_path: str) -> str:
    """
    Loads a conversation transcript from a .txt file.
    
    Args:
        file_path (str): Path to the conversation file
    
    Returns:
        str: Full conversation text
    """
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
        return text