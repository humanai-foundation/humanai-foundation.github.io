from sentence_transformers import SentenceTransformer, util

# Loaded once so repeated queries do not reload model weights.
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def build_index(candidate_texts):
    return EMBEDDING_MODEL.encode(candidate_texts, convert_to_tensor=True)


def search(query_text, candidate_texts, corpus_embeddings):
    query_embedding = EMBEDDING_MODEL.encode(query_text, convert_to_tensor=True)
    # util.cos_sim returns a 1 x N score tensor for one query against N documents.
    similarity_scores = util.cos_sim(query_embedding, corpus_embeddings)
    best_match_index = int(similarity_scores.argmax())
    return candidate_texts[best_match_index]


if __name__ == "__main__":
    docs = ["The hero embarks on a journey.", "A tragic ending unfolds.", "Comedic relief scene."]
    doc_embeddings = build_index(docs)
    print(search("tragic story", docs, doc_embeddings))
