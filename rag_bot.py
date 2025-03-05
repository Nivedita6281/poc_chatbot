import os
from io import StringIO
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import logging
import boto3
import csv
from fuzzywuzzy import process, fuzz

logger = logging.getLogger(__name__)
s3_client = boto3.client("s3")

def load_csv_from_s3(bucket_name, csv_key):
    file_url_map = {}
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=csv_key)
        csv_content = response["Body"].read().decode("utf-8")
        csv_reader = csv.DictReader(StringIO(csv_content))
        for row in csv_reader:
            file_url_map[row["filename"]] = row["url"]
        logger.info(f"Loaded {len(file_url_map)} entries from {csv_key}.")
        return file_url_map
    except Exception as e:
        logger.error(f"Error reading CSV from S3: {e}")
        return None

def create_rag_bot(vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})  # Balanced retrieval
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

    prompt_template = """You are an expert knowledge assistant. Use the following retrieved context to answer the question accurately.

Context:
{context}

Question: {question}

Instructions:
- If the question is unclear, request clarification.
- Answer using retrieved documents only.
- Provide a detailed response with at least 5 numbered points (unless unnecessary for short, direct answers).
- Provide a step-by-step answer where necessary.
- If context is unavailable, say: "I'm not sure how to answer that."
- Include specific details from the provided context.
  If the information comes from an article, provide a source link from the "Learn More" section at the end.
Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain

# ✅ Extract keywords from question dynamically
def extract_keywords(question):
    stopwords = {"how", "what", "is", "can", "i", "do", "does", "it", "the", "in", "on", "with", "for", "to"}
    words = question.lower().split()
    keywords = [word for word in words if word not in stopwords]
    return " ".join(keywords)

# ✅ Static answers for frequently asked questions
static_answers = {
    "refund": "Find out how to get a refund for games that qualify under the Great Game Guarantee: https://help.ea.com/en-us/help/account/returns-and-cancellations/",
    "delete account": "To delete your EA account, visit: https://help.ea.com/en/help/account/close-ea-account",
    "reset password": "To reset your password, visit: https://ea.com/reset-password",
    "cross-play": "Cross-play in NHL® 25 lets you play with friends across PlayStation® and Xbox. Read more here: https://help.ea.com/in/cross-play-guide",
    "crafting metals": "Find more about Crafting Metals and Legend Tokens here: https://help.ea.com/in/solutions/?product=apex-legends&platform=&topic=metals-tokens-info",
}

# ✅ Handle vague questions before querying FAISS
def ask_question(qa_chain, question, file_urls):
    if not qa_chain:
        return "⚠️ The RAG bot is not initialized. Please check the vector store.", []

    try:
        if not question.strip():
            return "⚠️ Please provide a valid question.", []

        input_key = "query" if "query" in qa_chain.input_keys else "question"

        # ✅ Check for static answers first (predefined FAQ responses)
        best_match = process.extractOne(question.lower(), static_answers.keys(), scorer=fuzz.token_sort_ratio)
        if best_match:
            matched_key, score = best_match[0], best_match[1]
            if score > 65:  # Confidence threshold
                return static_answers[matched_key], []

        # ✅ Extract keywords from the question itself
        keywords = extract_keywords(question)
        
        # ✅ Handle vague questions before retrieving from FAISS
        ambiguous_phrases = [
    "turn it off", "can I change it?", "how does it work?", "where can I find it?",
    "does it work online?", "is it enabled?", "is it better?", "do I need it?",
    "can I use this?", "will this affect my game?", "is it required?","can I buy items?"
]
        if any(phrase in question.lower() for phrase in ambiguous_phrases):
            return "❓ Your question is unclear. Can you provide more details or keywords?", []

        query_text = question
        logger.info(f"Retrieving documents for: {query_text}")

        # ✅ Retrieve documents from FAISS
        response = qa_chain({input_key: query_text})

        if response is None:
            return "⚠️ No response received from the model.", []

        answer = response.get("result") or response.get("answer") or "⚠️ No answer found."
        sources = response.get("source_documents", []) or []

        # ✅ If the response is vague, ask for clarification
        vague_responses = [
            "I'm not sure how to answer that.",
            "Please ask a question related to EA."
        ]
        if answer in vague_responses or "not sure how to answer" in answer.lower():
            return "❓ Could you provide more details or keywords to narrow down your question?", []

        # ✅ Extract source URLs
        source_list = []
        for doc in sources:
            source_path = os.path.basename(doc.metadata.get("source", "Unknown"))
            if file_urls and source_path in file_urls:
                source_list.append(file_urls[source_path])

        return answer, source_list

    except Exception as e:
        logger.error(f"⚠️ Error processing query: {e}", exc_info=True)
        return f"⚠️ Error processing query: {str(e)}", []
