import os
from io import StringIO
from langchain.chains import RetrievalQA, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import logging
import boto3
import csv
from fuzzywuzzy import process, fuzz

logger = logging.getLogger(__name__)
s3_client = boto3.client("s3")

# ✅ Session memory storage (Stores last 10 interactions per session)
session_memory = {}

def load_csv_from_s3(bucket_name, csv_key):
    """Loads file URLs from S3 into a dictionary mapping filenames to URLs."""
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
        return {}  # Return empty dict instead of None to avoid NoneType errors

def create_rag_bot(vector_store):
    """Creates a Retrieval-Augmented Generation (RAG) bot with improved retrieval."""
    if vector_store is None:
        raise ValueError("⚠️ FAISS vector store is not loaded. Please upload documents first.")

    # ✅ Use similarity search for better diversity and relevance
    retriever = vector_store.as_retriever(
        search_type="similarity",  
        search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.7}  # Fetch more, then filter
    )
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain

# ✅ Static answers for frequently asked questions
static_answers = {
    "refund": "Find out how to get a refund for games that qualify under the Great Game Guarantee: https://help.ea.com/en-us/help/account/returns-and-cancellations/",
    "delete account": "To delete your EA account, visit: https://help.ea.com/en/help/account/close-ea-account",
    "reset password": "To reset your password, visit: https://ea.com/reset-password",
    "cross-play": "Cross-play in NHL® 25 lets you play with friends across PlayStation® and Xbox. Read more here: https://help.ea.com/in/cross-play-guide",
    "crafting metals": "Find more about Crafting Metals and Legend Tokens here: https://help.ea.com/in/solutions/?product=apex-legends&platform=&topic=metals-tokens-info",
}

# ✅ Ambiguous questions that need clarification
ambiguous_phrases = [
    "can I turn it off?", "can I change it?", "how does it work?", "where can I find it?",
    "does it work online?", "is it enabled?", "is it better?", "do I need it?",
    "can I use this?", "will this affect my game?", "is it required?"
]

def check_document_relevance(doc, question, threshold=0.5):
    """Check if a document is relevant to the question."""
    # Simple relevance check based on content overlap
    doc_content = doc.page_content.lower()
    question_words = set(question.lower().split())
    
    # Count overlapping words
    overlap_count = sum(1 for word in question_words if word in doc_content)
    relevance_score = overlap_count / len(question_words) if question_words else 0
    
    return relevance_score >= threshold

def retrieve_with_retry(qa_chain, question, retries=3):
    """Retries document retrieval multiple times with improved error handling."""
    for attempt in range(retries):
        try:
            response = qa_chain({"query": question})
            
            # Check if we have source documents
            if response and response.get("source_documents"):
                # Filter for relevant documents
                relevant_docs = [doc for doc in response["source_documents"] 
                               if check_document_relevance(doc, question)]
                
                if relevant_docs:
                    # Update response with only relevant documents
                    response["source_documents"] = relevant_docs
                    return response
                else:
                    logger.warning(f"⚠️ Retrieved documents on attempt {attempt+1} but none were relevant. Retrying...")
            else:
                logger.warning(f"⚠️ No documents retrieved on attempt {attempt+1}. Retrying...")
                
        except Exception as e:
            logger.error(f"⚠️ Error during retrieval attempt {attempt+1}: {e}")

    # After all retries, return structured error response instead of None
    return {
        "result": "I couldn't find sufficiently relevant information in our knowledge base for your question.",
        "source_documents": []
    }

def ask_question(qa_chain, question, session_id, file_urls):
    global session_memory
    
    # Initialize session memory if it doesn't exist
    if session_id not in session_memory:
        session_memory[session_id] = []

    # Validate input
    if not question or not question.strip():
        return "⚠️ Please provide a valid question.", []

    # ✅ Check if the same question was asked before (fetch from memory)
    for entry in session_memory[session_id]:
        if entry.startswith(f"Q: {question}"):
            answer = entry.split("A: ")[-1]
            logger.info("✅ Fetching from session memory")
            return answer, []

    # ✅ Get previous conversations (last 10 interactions for context)
    previous_interactions = "\n".join(session_memory[session_id][-10:]) if session_memory[session_id] else "No prior conversation."

    # ✅ Check for static answers with improved matching
    best_match = process.extractOne(question.lower(), static_answers.keys(), 
                                   scorer=fuzz.token_set_ratio)  # Better handles word order
    if best_match:
        matched_key, score = best_match[0], best_match[1]
        if score > 60:  # Adjusted threshold
            logger.info(f"✅ Static answer match found with score {score}: {matched_key}")
            return static_answers[matched_key], []

    # ✅ Handle ambiguous questions before querying vector store
    if any(phrase in question.lower() for phrase in ambiguous_phrases):
        return "❓ Your question seems a bit unclear. Could you provide more specific details about what you're looking for?", []

    try:
        # ✅ FAISS Retrieval with improved Retry logic
        rag_response = retrieve_with_retry(qa_chain, question)
        
        # Get the answer from RAG
        basic_answer = rag_response.get("result", "⚠️ No relevant information found.")
        source_docs = rag_response.get("source_documents", [])
        
        # If no relevant documents were found, inform the user
        if not source_docs:
            return "⚠️ I couldn't find any relevant documents that answer your question. Could you rephrase or provide more details?", []
        # ✅ Improved URL mapping with better error handling
        source_strings = []
        for doc in source_docs:
            try:
                source = doc.metadata.get('source', 'Unknown source')
                filename = os.path.basename(source)
                
                # Handle file URL mapping with fallbacks
                if file_urls:
                    # Try exact match first
                    if filename in file_urls:
                        source_strings.append(file_urls[filename])
                        continue
                        
                    # Try partial matching if exact match fails
                    for key in file_urls:
                        if key in source or source in key:
                            source_strings.append(file_urls[key])
                            break
                    else:  # No match found in the for-else pattern
                        source_strings.append(source.replace("\\", "/"))
                else:
                    # If no file_urls mapping available
                    source_strings.append(source.replace("\\", "/"))
                    
            except Exception as e:
                logger.error(f"Error processing source document: {e}")
                source_strings.append("Source unavailable")

        # ✅ Enhance response with full session memory
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)

        enhance_template = """You are an expert knowledge assistant for EA (Electronic Arts) questions ONLY. 

### IMPORTANT: FIRST determine if the question is about EA (Electronic Arts) games, services, or products:
- If the question is NOT related to EA, respond ONLY with:  
  "This question does not seem related to EA. Please ask about EA-related topics such as EA games, accounts, or services."
- DO NOT answer non-EA questions under ANY circumstances, even if you know the answer
- If Question is unrelated or not related to EA then return an empty list for sources.
 
            
### Previous Conversations:
{history}

### Retrieved Answer:
{answer}

### New Question:
{question}

### Retrieved Sources:
{sources}
Instructions:
- Use previous conversation context when relevant to improve responses.
- If a similar question was asked earlier in this session, retrieve the previous response from memory instead of querying again.
- Do not use old session memory for unrelated questions. Ensure context is relevant before using it.
- Strictly answer using retrieved documents. DO NOT generate answers based on general knowledge.
- Provide a structured response with key points from the retrieved documents.
- If additional sources are needed, suggest where to find more details.
- Should retrive sources even if the response is retriving from session memory.
- Response should retrive if question is related to EA and give atleast 5 points.

Handling Different Cases:
- If the question is **unrelated to EA**, respond with:  
  "This question does not seem related to EA. Please ask about EA-related topics."
- If the question is **unclear or too vague**, ask for clarification:  
  "Can you please provide more details or specify what you're asking?"
- If the question is **toxic, offensive, or inappropriate**, respond with:  
  "I'm sorry, but I can't provide a response to that."
- If the question is **a greeting** (e.g., "Hi," "Hello," "How are you?"), respond with:  
  "Hi! How can I assist you today with EA-related queries?"
- If the question is **a goodbye message** (e.g., "Bye," "See you later"), respond with:  
  "Goodbye! Feel free to ask if you need help with anything EA-related in the future."
Enhanced Answer:"""

        enhance_prompt = PromptTemplate(
            template=enhance_template,
            input_variables=["history", "answer", "question", "sources"]
        )

        enhance_chain = LLMChain(llm=llm, prompt=enhance_prompt)

        # Include sources in the prompt to ensure the model knows what sources were used
        sources_text = "\n".join([f"- {src}" for src in source_strings]) if source_strings else "No specific sources retrieved."
        
        enhanced_response = enhance_chain.run(
            history=previous_interactions,
            answer=basic_answer,
            question=question,
            sources=sources_text
        )

        final_answer = enhanced_response
        if any(phrase in final_answer.lower() for phrase in [
            "not seem related to ea", 
            "not related to ea",
            "provide more context or details",
            "not sure",
            "unclear",
            "seems like your question is a bit unclear"
        ]):
            source_strings = []
       
        # ✅ Store the interaction in session memory
        session_memory[session_id].append(f"Q: {question}\nA: {final_answer}")
        # Keep only last 30 interactions
        if len(session_memory[session_id]) > 30:
            session_memory[session_id] = session_memory[session_id][-30:]
        return final_answer, source_strings

    except Exception as e:
        logger.error(f"⚠️ Error processing query: {e}")
        return f"⚠️ Sorry, I encountered an error while processing your question: {str(e)}", []
    