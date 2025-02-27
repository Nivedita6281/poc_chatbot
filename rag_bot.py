import os
from io import StringIO
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import logging
import boto3
import csv

logger = logging.getLogger(__name__)
s3_client = boto3.client("s3")


def load_csv_from_s3(bucket_name, csv_key):
    """Downloads and parses a CSV from S3, returning a dictionary mapping filenames to URLs."""
    file_url_map = {}

    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=csv_key)
        csv_content = response["Body"].read().decode("utf-8")

        # Read CSV content into dictionary
        csv_reader = csv.DictReader(StringIO(csv_content))
        for row in csv_reader:
            file_url_map[row["filename"]] = row["url"]

        logger.info(f"Loaded {len(file_url_map)} entries from {csv_key}.")
    except Exception as e:
        logger.error(f"Error reading CSV from S3: {e}")

    return file_url_map


def create_rag_bot(vector_store):
    """Creates an improved Retrieval-Augmented Generation (RAG) bot with better answer quality."""
    
    # Increase k to retrieve more relevant documents
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    # Use a more capable model with slightly higher temperature for more detailed responses
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)
    
    # Create a comprehensive prompt that encourages detailed answers
    prompt_template = """You are an expert knowledge assistant. Use the following pieces of retrieved context to answer the question thoroughly and accurately.

Context:
{context}

Question: {question}

Instructions:
- Answer in complete and concise sentences.
- If the question is related to self-harm, respond with:
    "I detected a statement that you intend to cause yourself harm. Please visit https://help.ea.com/en for any help."
- If the question contains toxic or inappropriate language, respond with:
    "I'm sorry, I can't provide a response, as your question appears to be inappropriate and not related to EA."
- If the question is a greeting (e.g., "hi," "hello," "how are you?"), respond with:
    "Hi, how can I help you? Please ask me a question related to EA."
- If the question is a goodbye message (e.g., "bye," "see you later"), respond with:
    "Goodbye! If you have any more questions in the future, feel free to ask. Have a great day."
- If there is not enough highly relevant information in the context to answer the question properly, or if you're unsure about the response, respond with:
    "I'm not completely sure what you're asking about. Could you provide some additional keywords or details?"
- If the question is unrelated to EA, respond with:
    "I'm not sure how to answer that. Please ask a question related to EA."
- Use the context provided to answer the question only if it directly addresses the question.
- NEVER generate a general answer when you don't have specific information in the provided context.
- If the question is relevant to EA and has directly relevant contextual information available, follow these rules:
    - Answer based only on the given context.
    - Provide a detailed response with numbered points when appropriate.
    - If the information comes from an article, provide a source link.
Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create custom QA chain with the refactored prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain


def ask_question(qa_chain, question, file_urls, keywords=None):
    """
    Processes a user query and retrieves an answer using the RAG system.
    """
    if not question.strip():
        return "⚠️ Please provide a valid question.", [], False

    try:
        logger.info(f"Processing query: {question}")

        # Enhance question with keywords
        enhanced_question = question
        if keywords:
            keywords_list = [k.strip() for k in keywords.split(',')]
            enhanced_question = f"{question} {' '.join(keywords_list)}"
            logger.info(f"Enhanced question with keywords: {enhanced_question}")

            # Increase document retrieval scope when keywords are provided
            if hasattr(qa_chain, 'retriever'):
                qa_chain.retriever.search_kwargs["k"] = 8

        # Retrieve relevant documents
        retrieved_docs = qa_chain.retriever.get_relevant_documents(enhanced_question) if hasattr(qa_chain, 'retriever') else []
        logger.info(f"Retrieved {len(retrieved_docs)} documents")

        # If no relevant documents are retrieved
        if not retrieved_docs:
            if keywords:
                return "I'm not completely sure what you're asking about. Could you provide even more specific keywords?", [], True
            return "I'm not sure how to answer that. Please ask a question related to EA.", [], False

        # Process query with relevant documents
        return process_query(qa_chain, enhanced_question, file_urls)

    except Exception as e:
        logger.error(f"⚠️ Error processing query: {e}", exc_info=True)
        return f"⚠️ Error processing query: {str(e)}", [], False


def process_query(qa_chain, query, file_urls):
    """Processes a query and returns an answer based on the retrieved context."""
    
    expected_keys = getattr(qa_chain, 'input_keys', ['unknown'])
    logger.info(f"Chain expects these input keys: {expected_keys}")

    # Determine the correct input parameter
    if 'question' in expected_keys:
        response = qa_chain({"question": query})
    elif 'query' in expected_keys:
        response = qa_chain({"query": query})
    else:
        input_param = {expected_keys[0]: query}
        response = qa_chain(input_param)

    if isinstance(response, dict):
        answer = response.get("result") or response.get("answer") or "⚠️ No answer found in response."
        sources = response.get("source_documents", [])

        # Extract valid source links
        source_list = set()
        for doc in sources:
            source_path = os.path.basename(doc.metadata.get("source", "Unknown"))
            if source_path in file_urls:
                source_list.add(file_urls[source_path])

        # Append sources to the answer if sources are found
        if source_list:
            sources_text = "\n\n Learn more at:\n" + "\n".join(source_list)
            answer += sources_text  # Append sources to answer

        logger.info(f"Successfully generated answer: {answer[:100]}...")
        logger.info(f"Sources: {source_list}")

        return answer, list(source_list), False

    else:
        logger.error(f"Unexpected response type: {type(response)}")
        return "⚠️ Received an unexpected response format.", [], False
