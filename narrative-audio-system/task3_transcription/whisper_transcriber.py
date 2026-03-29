import whisper
import sys
from pathlib import Path

# Allow script-style execution (python task3_transcription/...) to import sibling task modules.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from task4_audio_retrieval.retrieval_prototype import build_index, search


def transcribe_audio(audio_path, model_size="base"):
    whisper_model = whisper.load_model(model_size)
    transcription_result = whisper_model.transcribe(audio_path)
    transcript_text = transcription_result["text"]
    print("Transcription:", transcript_text)
    return transcript_text


def transcribe_and_retrieve(audio_path, candidate_documents):
    # Task 3 output (transcript) is used directly as the Task 4 query.
    transcript_text = transcribe_audio(audio_path)
    corpus_embeddings = build_index(candidate_documents)
    best_matching_document = search(transcript_text, candidate_documents, corpus_embeddings)
    print("Retrieved document:", best_matching_document)
    return transcript_text, best_matching_document


if __name__ == "__main__":
    candidate_docs = [
        "The hero embarks on a journey.",
        "A tragic ending unfolds.",
        "Comedic relief scene.",
    ]
    transcribe_and_retrieve("examples/sample_audio.wav", candidate_docs)