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
        return file_url_map  # Return the map if successful
    except Exception as e:
        logger.error(f"Error reading CSV from S3: {e}")
        return None  # Return None on failure

def create_rag_bot(vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

    # Updated prompt template to emphasize the use of keywords
    prompt_template = """You are an expert knowledge assistant. Use the following pieces of retrieved context to answer the question thoroughly and accurately.

Context:
{context}

Question: {question}

Keywords: {keywords}

Instructions:
- Pay special attention to the provided keywords: {keywords}. Use them to narrow down the context and provide a more accurate answer.
- Answer in complete and concise sentences.
- If the question is unrelated to EA and does not fit the above categories, respond with:
  "I'm not sure how to answer that. Please ask a question related to EA."
- If the question is relevant to EA and has contextual information available, follow these rules:
  Answer based only on the given context.
  Provide a detailed response with at least 5 numbered points (unless unnecessary for short, direct answers).
  Include specific details from the provided context.
  If the information comes from an article, provide a source link from the "Learn More" section at the end.
- Retrieve and reference all relevant parts of the uploaded PDFs before answering. Think step by step and provide a detailed response.
Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question", "keywords"],
        partial_variables={"keywords": ""}  # Default to empty string if keywords not available
    )

    # Use the default stuff chain for document handling
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain
static_answers = {
    "refund ": "Find out how to get a refund for games that qualify under the great game guarantee policy here: \nhttps://help.ea.com/en-us/help/account/returns-and-cancellations/",
    "delete account": "To delete your Ea account, visit https://help.ea.com/en/help/account/close-ea-account",
    "reset password": "To reset your password, visit https://ea.com/reset-password",
    "crafting legend tokens": "Find out more about Crafting Metals and Legend Tokens on this site \nhttps://help.ea.com/in/solutions/?product=apex-legends&platform=&topic=metals-tokens-info",
    "crafting metals": "Find out more about Crafting and Legend Tokens on this site \nhttps://help.ea.com/in/solutions/?product=apex-legends&platform=&topic=metals-tokens-info",
    "reset rank": "Please refer this link to get more help on resetting rank \nhttps://help.ea.com/in/solutions/?product=apex-legends&topic=emerging-technical-support-09",
    "progress restore": "Check out what you can do to get back in the game here: \nhttps://help.ea.com/in/help/apex-legends/apex-legends/game-progress",
}

def get_static_answer(question):
    best_match = process.extractOne(question.lower(), static_answers.keys(), scorer=fuzz.token_sort_ratio)
    if best_match:
        matched_key, score = best_match[0], best_match[1]  # Extract matched string and score
        if score > 65:  # Adjust threshold as needed
            return static_answers[matched_key]
    return None

def ask_question(qa_chain, question, file_urls, keywords=None):
    if not qa_chain:
        return "⚠️ The RAG bot is not initialized. Please check the vector store.", []

    try:
        if not question.strip():
            return "⚠️ Please provide a valid question.", []

        # Check if the QA chain expects 'query' or 'question' as input
        input_key = "query" if "query" in qa_chain.input_keys else "question"
        
        # If keywords are provided, use them directly
        if keywords and keywords.strip():
            logger.info(f"Answering with keywords: {question}, Keywords: {keywords}")
            
            # Create input dict based on expected input key
            if input_key == "question":
                input_dict = {"question": question, "keywords": keywords}
            else:
                # If using 'query', we might need to adapt how keywords are handled
                # One approach is to append keywords to the query
                input_dict = {"query": f"{question} (Keywords: {keywords})"}
            
            response = qa_chain(input_dict)
            
            if response is None:
                return "⚠️ No response received from the model.", []

            answer = response.get("result") or response.get("answer") or "⚠️ No answer found."
            sources = response.get("source_documents", []) or []
            
            # Extract source URLs
            source_list = []
            for doc in sources:
                source_path = os.path.basename(doc.metadata.get("source", "Unknown"))
                if file_urls and source_path in file_urls:
                    source_list.append(file_urls[source_path])
            
            return answer, source_list
            
        # If no keywords provided, try to answer without them
        logger.info(f"Attempting to answer without keywords: {question}")
        
        # Create input dict based on expected input key
        if input_key == "question":
            input_dict = {"question": question, "keywords": ""}
        else:
            input_dict = {"query": question}
        
        response = qa_chain(input_dict)

        if response is None:
            return "⚠️ No response received from the model.", []

        answer = response.get("result") or response.get("answer") or "⚠️ No answer found."
        sources = response.get("source_documents", []) or []

        # Check if the answer indicates uncertainty
        non_contextual_responses = {
            "I'm sorry, I can't provide a response, as your question appears to be inappropriate and not related to EA.",
            "Hi, how can I help you? Please ask me a question related to EA.",
            "Goodbye! If you have any more questions in the future, feel free to ask. Have a great day.",
            "I detected a statement that you intend to cause yourself harm. Please visit https://help.ea.com/en for any help.",
            "I'm not sure how to answer that. Please ask a question related to EA."
        }

        if answer in non_contextual_responses or "not sure how to answer" in answer.lower():
            return "Could you please provide a few keywords to help me narrow down your question?", []
        
        # Extract source URLs
        source_list = []
        for doc in sources:
            source_path = os.path.basename(doc.metadata.get("source", "Unknown"))
            if file_urls and source_path in file_urls:
                source_list.append(file_urls[source_path])
        
        return answer, source_list

    except Exception as e:
        logger.error(f"⚠️ Error processing query: {e}", exc_info=True)
        return f"⚠️ Error processing query: {str(e)}", []