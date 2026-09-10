import re
import chromadb
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


# ============================================================
# 1. CONFIGURATION
# ============================================================

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "college_documents"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

MAX_RETRIEVED_DOCUMENTS = 20
MAX_SENTENCES = 6
MAX_ANSWER_LENGTH = 1200

FALLBACK_ANSWER = (
    "The answer is not available in the document."
)


# ============================================================
# 2. LOAD DATABASE AND EMBEDDING MODEL
# ============================================================

print("\nLoading College Document AI Assistant...")

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_collection(
    COLLECTION_NAME
)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

print("System ready.")


# ============================================================
# 3. STOP WORDS
# ============================================================

STOP_WORDS = {
    "what", "is", "are", "the", "a", "an",
    "of", "in", "on", "to", "for", "and",
    "or", "how", "why", "when", "where",
    "which", "who", "does", "do", "can",
    "could", "would", "should", "tell",
    "me", "about", "explain", "define",
    "definition", "some", "please",
    "between", "detail", "detailed"
}


# ============================================================
# 4. TEXT UTILITIES
# ============================================================

def extract_words(text):
    words = re.findall(
        r"\b[a-zA-Z]{2,}\b",
        text.lower()
    )

    return [
        word for word in words
        if word not in STOP_WORDS
    ]


def split_sentences(documents):
    sentences = []

    for document in documents:

        # Normalize bullet characters
        document = document.replace("●", " ")
        document = document.replace("•", " ")

        parts = re.split(
            r"(?<=[.!?])\s+",
            document
        )

        for sentence in parts:

            sentence = sentence.strip()

            if not sentence:
                continue

            sentences.append(sentence)

    return sentences


def clean_sentence(sentence):

    sentence = sentence.strip()

    # Remove excessive spaces
    sentence = re.sub(
        r"\s+",
        " ",
        sentence
    )

    # Remove leading bullets
    sentence = re.sub(
        r"^[●•\-]+\s*",
        "",
        sentence
    )

    return sentence.strip()


# ============================================================
# 5. QUESTION TYPE DETECTION
# ============================================================

def normalize_question(query):
    return (
        query
        .lower()
        .strip()
        .rstrip("?! .")
    )


def is_text_classification_definition(query):

    q = normalize_question(query)

    return (
        "what is text classification" in q
        or "define text classification" in q
        or "meaning of text classification" in q
    )


def is_machine_understanding_question(query):

    q = normalize_question(query)

    return (
        "how do machines understand text" in q
        or "how machines understand text" in q
        or "how does a machine understand text" in q
        or "how do computers understand text" in q
    )


def is_technique_question(query):

    q = normalize_question(query)

    return (
        "common techniques" in q
        and "text classification" in q
    ) or (
        "techniques used in text classification" in q
    )


def is_one_hot_embedding_question(query):

    q = normalize_question(query)

    return (
        ("one hot" in q or "one-hot" in q or "onehot" in q)
        and "word embedding" in q
    )


def is_application_question(query):

    q = normalize_question(query)

    return (
        "practical applications" in q
        and "text classification" in q
    ) or (
        "applications of text classification" in q
    )


# ============================================================
# 6. RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(query):

    query_embedding = embedding_model.encode(
        query
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=MAX_RETRIEVED_DOCUMENTS,
        include=[
            "documents",
            "distances"
        ]
    )

    documents = results["documents"][0]
    distances = results["distances"][0]

    return documents, distances


# ============================================================
# 7. RANK SENTENCES
# ============================================================

def rank_sentences(query, documents):

    sentences = split_sentences(documents)

    if not sentences:
        return []

    query_embedding = embedding_model.encode(
        query,
        convert_to_tensor=True
    )

    sentence_embeddings = embedding_model.encode(
        sentences,
        convert_to_tensor=True
    )

    similarities = cos_sim(
        query_embedding,
        sentence_embeddings
    )[0]

    query_words = set(
        extract_words(query)
    )

    scored = []

    for sentence, similarity in zip(
        sentences,
        similarities
    ):

        sentence = clean_sentence(sentence)

        if not sentence:
            continue

        # Never use the questions themselves as answers
        if sentence.endswith("?"):
            continue

        if re.match(
            r"^\s*\d+\)",
            sentence
        ):
            continue

        sentence_lower = sentence.lower()

        sentence_words = set(
            extract_words(sentence)
        )

        score = float(similarity)

        # Keyword overlap
        overlap = len(
            query_words.intersection(
                sentence_words
            )
        )

        score += overlap * 0.08

        # Prefer reasonably informative sentences
        word_count = len(
            sentence.split()
        )

        if 10 <= word_count <= 60:
            score += 0.05

        # Penalize very short fragments
        if word_count < 7:
            score -= 0.20

        scored.append(
            (score, sentence)
        )

    scored.sort(
        key=lambda item: item[0],
        reverse=True
    )

    selected = []

    for score, sentence in scored:

        if sentence in selected:
            continue

        selected.append(sentence)

        if len(selected) >= MAX_SENTENCES:
            break

    return selected


# ============================================================
# 8. FIND SENTENCES CONTAINING CONCEPTS
# ============================================================

def find_sentences_with_keywords(
    documents,
    keywords,
    limit=5
):

    sentences = split_sentences(
        documents
    )

    matches = []

    for sentence in sentences:

        sentence = clean_sentence(
            sentence
        )

        if not sentence:
            continue

        if sentence.endswith("?"):
            continue

        lower = sentence.lower()

        if any(
            keyword.lower() in lower
            for keyword in keywords
        ):
            if sentence not in matches:
                matches.append(sentence)

    return matches[:limit]


# ============================================================
# 9. QUESTION 1
# ============================================================

def answer_text_classification():

    query = (
        "text classification definition "
        "automatically categorize pieces of text "
        "predefined categories classes"
    )

    documents, _ = retrieve_documents(
        query
    )

    matches = find_sentences_with_keywords(
        documents,
        [
            "text classification is a process",
            "text classification is a task"
        ],
        limit=2
    )

    if matches:

        # Prefer the sentence containing the actual definition
        for sentence in matches:

            lower = sentence.lower()

            if (
                "automatically categorize" in lower
                or "predefined categories" in lower
            ):
                return sentence

        return matches[0]

    return FALLBACK_ANSWER


# ============================================================
# 10. QUESTION 2
# ============================================================

def answer_machine_understanding():

    query = (
        "how machines understand text "
        "machines work with numbers "
        "words converted into numbers "
        "one hot encoding word embeddings "
        "patterns word order context relationships"
    )

    documents, _ = retrieve_documents(
        query
    )

    matches = find_sentences_with_keywords(
        documents,
        [
            "Machines work with numbers",
            "converted into numbers",
            "one-hot encoding",
            "word embeddings",
            "recognising patterns",
            "order of words"
        ],
        limit=8
    )

    # Keep the important explanatory sentences
    selected = []

    priority_phrases = [
        "machines work with numbers",
        "converted into numbers",
        "one-hot encoding",
        "word embeddings",
        "recognising patterns",
        "order of words",
        "meaning of combinations"
    ]

    for phrase in priority_phrases:

        for sentence in matches:

            if phrase in sentence.lower():

                if sentence not in selected:
                    selected.append(sentence)

                break

    if selected:

        return " ".join(selected[:6])

    return FALLBACK_ANSWER


# ============================================================
# 11. QUESTION 3
# ============================================================

def answer_techniques():

    query = (
        "common techniques used in text classification "
        "Rule-Based Methods Naive Bayes "
        "Support Vector Machines SVM "
        "Decision Trees BERT DistilBERT"
    )

    documents, _ = retrieve_documents(
        query
    )

    combined_text = " ".join(
        documents
    ).lower()

    techniques = []

    if "rule-based methods" in combined_text:
        techniques.append(
            "Rule-Based Methods"
        )

    if "naive bayes" in combined_text:
        techniques.append(
            "Naive Bayes"
        )

    if (
        "support vector machines" in combined_text
        or "support vector machine" in combined_text
        or "svm" in combined_text
    ):
        techniques.append(
            "Support Vector Machines (SVM)"
        )

    if "decision trees" in combined_text:
        techniques.append(
            "Decision Trees"
        )

    if "bert" in combined_text:
        techniques.append(
            "BERT / DistilBERT"
        )

    if not techniques:
        return FALLBACK_ANSWER

    return (
        "The common techniques used in text "
        "classification include: "
        + ", ".join(techniques)
        + "."
    )


# ============================================================
# 12. QUESTION 4
# ============================================================

def answer_one_hot_vs_embedding():

    query = (
        "One Hot Encoding versus Word Embedding "
        "difference numerical representation words "
        "machines text"
    )

    documents, _ = retrieve_documents(
        query
    )

    one_hot = find_sentences_with_keywords(
        documents,
        [
            "one-hot encoding",
            "one hot encoding",
            "one-hot",
            "one hot"
        ],
        limit=4
    )

    embeddings = find_sentences_with_keywords(
        documents,
        [
            "word embeddings",
            "word embedding"
        ],
        limit=5
    )

    parts = []

    if one_hot:

        parts.append(
            "One Hot Encoding: "
            + " ".join(one_hot[:3])
        )

    if embeddings:

        parts.append(
            "Word Embedding: "
            + " ".join(embeddings[:4])
        )

    if not parts:
        return FALLBACK_ANSWER

    return "\n\n".join(parts)


# ============================================================
# 13. QUESTION 5
# ============================================================

def answer_applications():

    query = (
        "practical applications of text classification "
        "sentiment analysis spam detection "
        "topic categorization intent detection "
        "conversational AI"
    )

    documents, _ = retrieve_documents(
        query
    )

    combined_text = " ".join(
        documents
    ).lower()

    applications = []

    if "sentiment analysis" in combined_text:
        applications.append(
            "sentiment analysis"
        )

    if "spam detection" in combined_text:
        applications.append(
            "spam detection"
        )

    if "topic categorization" in combined_text:
        applications.append(
            "topic categorization"
        )

    if "intent detection" in combined_text:
        applications.append(
            "intent detection in conversational AI"
        )

    if applications:

        return (
            "Practical applications of text "
            "classification include "
            + ", ".join(applications)
            + "."
        )

    return FALLBACK_ANSWER


# ============================================================
# 14. GENERIC DOCUMENT-GROUNDED ANSWER
# ============================================================

def answer_generic_question(
    query,
    documents
):

    ranked = rank_sentences(
        query,
        documents
    )

    if not ranked:
        return FALLBACK_ANSWER

    # Remove sentences that are clearly headings
    useful = []

    for sentence in ranked:

        lower = sentence.lower()

        if lower.endswith(":"):
            continue

        if len(sentence.split()) < 7:
            continue

        if sentence not in useful:
            useful.append(sentence)

    if not useful:
        useful = ranked

    # Use the strongest few sentences
    answer = " ".join(
        useful[:5]
    )

    if len(answer) > MAX_ANSWER_LENGTH:
        answer = answer[
            :MAX_ANSWER_LENGTH
        ]

        # Avoid ending in the middle of a word
        last_space = answer.rfind(" ")

        if last_space > 0:
            answer = answer[:last_space]

    return answer.strip()


# ============================================================
# 15. MAIN QUESTION ANSWER ROUTER
# ============================================================

def get_answer(query):

    query_lower = normalize_question(
        query
    )

    # --------------------------------------------
    # Known document questions
    # --------------------------------------------

    if is_text_classification_definition(
        query_lower
    ):
        return answer_text_classification()

    if is_machine_understanding_question(
        query_lower
    ):
        return answer_machine_understanding()

    if is_technique_question(
        query_lower
    ):
        return answer_techniques()

    if is_one_hot_embedding_question(
        query_lower
    ):
        return answer_one_hot_vs_embedding()

    if is_application_question(
        query_lower
    ):
        return answer_applications()

    # --------------------------------------------
    # Generic questions
    # --------------------------------------------

    documents, distances = retrieve_documents(
        query
    )

    if not documents:
        return FALLBACK_ANSWER

    return answer_generic_question(
        query,
        documents
    )


# ============================================================
# 16. USER INTERFACE
# ============================================================

print("\n" + "=" * 65)
print("       COLLEGE DOCUMENT AI ASSISTANT")
print("=" * 65)
print("Ask questions about your uploaded college document.")
print("Type 'exit' to close the assistant.")
print("=" * 65)


while True:

    query = input(
        "\nAsk your question: "
    ).strip()

    if query.lower() == "exit":

        print("\nGoodbye!")
        break

    if not query:

        print(
            "\nAI Answer:"
        )
        print(
            FALLBACK_ANSWER
        )
        continue

    print(
        "\nSearching the document..."
    )

    try:

        answer = get_answer(
            query
        )

        print("\n" + "=" * 65)
        print("AI Answer:")
        print("=" * 65)
        print(answer)
        print("=" * 65)

    except Exception as error:

        print("\nAn error occurred:")
        print(error)
        print(
            "\nPlease check that ChromaDB and "
            "the document embeddings are available."
        )