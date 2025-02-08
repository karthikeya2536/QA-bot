import torch
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
from sentence_transformers import SentenceTransformer
import faiss
import re
import logging

# Load Pretrained Models
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
qa_model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")
retrieval_model = SentenceTransformer("all-MiniLM-L6-v2")

# Set up logging
logging.basicConfig(level=logging.INFO)

# Step 1: Load and Preprocess the Book
def load_book(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    return re.split(r"\n{2,}", text)  # Robust paragraph splitting

book_path = "book.txt"
paragraphs = load_book(book_path)

# Step 2: Create Embeddings and FAISS Index
embeddings = retrieval_model.encode(paragraphs, convert_to_numpy=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Step 3: Retrieve Relevant Context
import numpy as np

def retrieve_context(question, top_k=5, similarity_threshold=0.3):
    query_embedding = retrieval_model.encode([question], convert_to_numpy=True)
    distances, idxs = index.search(query_embedding, top_k)

    # Compute similarity scores
    similarities = np.exp(-distances[0])  # Convert L2 distances to similarity scores

    # Ignore results if similarity is too low
    if max(similarities) < similarity_threshold:
        return None

    top_paragraphs = [paragraphs[i] for i in idxs[0]]
    
    # Choose the best match
    best_paragraph = max(top_paragraphs, key=lambda p: retrieval_model.encode([p], convert_to_numpy=True) @ query_embedding.T)
    
    return best_paragraph



# Step 4: Improved Answer Extraction
def answer_question(question, context):
    if context is None:
        return "Sorry, I couldn't find relevant information in the book."

    inputs = tokenizer(question, context, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = qa_model(**inputs)

    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits)

    if start_idx >= end_idx:
        logging.warning("QA model failed to extract an answer. Falling back to retrieval.")
        return extract_best_sentence(context, question)

    input_ids = inputs["input_ids"][0]
    answer_tokens = tokenizer.convert_ids_to_tokens(input_ids[start_idx:end_idx+1])
    answer_tokens = fix_word_tokens(answer_tokens, input_ids, start_idx, end_idx)
    answer = tokenizer.convert_tokens_to_string(answer_tokens).strip()

    if len(answer.split()) < 4:
        logging.warning("Answer too short. Falling back to retrieval.")
        return extract_best_sentence(context, question)

    return answer

# Ensure words aren't split incorrectly
def fix_word_tokens(answer_tokens, input_ids, start_idx, end_idx):
    while start_idx > 0 and answer_tokens[0].startswith("##"):
        start_idx -= 1
        answer_tokens.insert(0, tokenizer.convert_ids_to_tokens([input_ids[start_idx]])[0])

    while end_idx + 1 < len(input_ids) and tokenizer.convert_ids_to_tokens([input_ids[end_idx+1]])[0].startswith("##"):
        end_idx += 1
        answer_tokens.append(tokenizer.convert_ids_to_tokens([input_ids[end_idx]])[0])

    return answer_tokens

# Extracts the best matching sentence if QA fails
def extract_best_sentence(context, question):
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", context)
    best_sentence = max(sentences, key=lambda s: retrieval_model.encode([s], convert_to_numpy=True) @ retrieval_model.encode([question], convert_to_numpy=True).T)
    return best_sentence.strip()

# Step 5: Run in Terminal
if __name__ == "__main__":
    print("AI Question Answering System (Type 'exit' to quit)")
    while True:
        question = input("\nAsk a question: ")
        if question.lower() == "exit":
            print("Goodbye!")
            break
        context = retrieve_context(question)
        answer = answer_question(question, context)
        print(f"\nAnswer: {answer}\nContext: {context if context else 'No relevant context found.'}")
