def parse_transcript(text: str) -> list:
    conversation = []

    lines = text.splitlines()

    for idx, line in enumerate(lines):
        line = line.strip()

        if not line:
            continue

        if ":" not in line:
            print(f"SKIPPED (no colon): Line {idx} -> {line}")
            continue

        speaker_part, content = line.split(":", 1)
        speaker = speaker_part.strip().lower()

        if speaker in ["physician", "doctor"]:
            speaker = "Physician"
        elif speaker == "patient":
            speaker = "Patient"
        else:
            print(f"SKIPPED (unknown speaker): Line {idx} -> {line}")
            continue

        conversation.append({
            "line_id": idx,
            "speaker": speaker,
            "text": content.strip()
        })

    return conversation
