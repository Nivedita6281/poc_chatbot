import os
from io import StringIO
from langchain.chains import RetrievalQA, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import logging
import boto3
import csv
import numpy as np
from fuzzywuzzy import process, fuzz
from rank_bm25 import BM25Okapi
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

def load_documents():
    """Load and preprocess documents from S3 or local storage."""
    docs = [...]  # Load documents as a list of strings
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)
    return split_docs

def create_hybrid_retriever(documents):
    """Create hybrid search using BM25 and FAISS."""
    tokenized_docs = [doc.page_content.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    return bm25, vector_store

def hybrid_search(query, bm25, vector_store, k=5):
    # BM25: Get top-k docs with scores
    bm25_scores = bm25.get_scores(query.split())
    bm25_docs = [(score, idx) for idx, score in enumerate(bm25_scores)]
    bm25_docs.sort(reverse=True, key=lambda x: x[0])
    
    # FAISS: Get top-k docs
    faiss_docs = vector_store.similarity_search(query, k=k)
    
    # Combine results (weighted average)
    combined = []
    for idx, (score, _) in enumerate(bm25_docs[:k]):
        combined.append(("bm25", idx, score * 0.6))  # BM25 weight: 60%
    for idx, doc in enumerate(faiss_docs):
        combined.append(("faiss", idx, 0.4))  # FAISS weight: 40%
    
    # Sort by combined score
    combined.sort(key=lambda x: x[2], reverse=True)
    return combined[:k]

def create_rag_bot(vector_store):
    """Creates a RAG bot using hybrid retrieval (BM25 + FAISS)."""
    if vector_store is None:
        raise ValueError("⚠️ FAISS vector store is not loaded. Please upload documents first.")

    # ✅ Fix: Use FAISS correctly by extracting retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k":5}  # Adjust k as needed
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
    "hi": "Hi! How can I assist you with EA-related questions today?",
    "hello": "Hello there! What EA-related questions can I help you with?",
    "hey": "Hey! Ready to help with any EA games or services questions.",
    "good morning": "Good morning! What EA-related assistance do you need today?",
    "good afternoon": "Good afternoon! How can I help with EA products or services?",
    "good evening": "Good evening! What EA questions can I answer for you?",
    "how are you":"I'm just a bot, but thanks for asking! How can I help with EA today?",
    "refund": "Find out how to get a refund for games that qualify under the Great Game Guarantee: https://help.ea.com/en-us/help/account/returns-and-cancellations/",
    "delete account": "To delete your EA account, visit: https://help.ea.com/en/help/account/close-ea-account",
    "reset password": "To reset your password, visit: https://ea.com/reset-password",
    "cross-play": "Cross-play in NHL® 25 lets you play with friends across PlayStation® and Xbox. Read more here: https://help.ea.com/in/cross-play-guide",
    "crafting metals": "Find more about Crafting Metals and Legend Tokens here: https://help.ea.com/in/solutions/?product=apex-legends&platform=&topic=metals-tokens-info",
}
# Add this to your existing static answers and ambiguous phrases
CLARIFICATION_LEVELS = {
    'game_features': {
        'triggers': ['fifa', 'madden', 'apex', 'nhl', 'game', 'feature', 'problem', 'issue', 'mode'],
        'level1': "Which specific game are you asking about? (1) FIFA 2) Madden 3) Apex Legends 4) NHL",
        'level2': {
            'fifa': "What aspect of FIFA? (1) Ultimate Team 2) Career Mode 3) Gameplay 4) Online Modes",
            'madden': "What aspect of Madden? (1) Franchise Mode 2) MUT 3) Gameplay",
            'apex legends': "What aspect of Apex? (1) Legends 2) Weapons 3) Battle Pass"
        }
    },
    'technical': {
        'triggers': ['crash', 'lag', 'server', 'performance', 'error', 'bug', 'glitch', 'connection'],
        'level1': "Is this about: 1) Server issues 2) Connectivity 3) Game performance?",
        'level2': {
            'server issues': "Are you experiencing: 1) Login problems 2) Disconnections 3) Matchmaking?",
            'connectivity': "Is this about: 1) NAT type 2) Port forwarding 3) General connection?",
            'performance': "Are you having: 1) FPS drops 2) Crashing 3) Graphical glitches?"
        }
    },
    'account': {
        'triggers': ['account', 'login', 'password', 'ea play', 'subscription', 'sign in', 'profile'],
        'level1': "Are you having issues with: 1) Login 2) Account recovery 3) Subscriptions?",
        'level2': {
            'login': "Is this about: 1) Password reset 2) 2FA 3) Banned account?",
            'account recovery': "Do you need help with: 1) Hacked account 2) Email change 3) Profile recovery?",
            'subscriptions': "Is this about: 1) EA Play 2) Payment issues 3) Cancellation?"
        }
    },
    'purchase': {
        'triggers': ['purchase', 'refund', 'price', 'microtransaction', 'fifa points', 'buy', 'order'],
        'level1': "Is this about: 1) Refunds 2) Payment issues 3) In-game purchases?",
        'level2': {
            'refunds': "Are you asking about: 1) Eligibility 2) Process 3) Timeframe?",
            'payment issues': "Is this about: 1) Declined payment 2) Missing content 3) Incorrect charges?",
            'in-game purchases': "Are you asking about: 1) FIFA Points 2) Apex Coins 3) Missing items?"
        }
    }
}

# ✅ Ambiguous questions that need clarification
ambiguous_phrases = [
    "can I turn it off?", "can I change it?", "how does it work?", "where can I find it?",
    "does it work online?", "is it enabled?", "is it better?", "do I need it?",
    "can I use this?", "will this affect my game?", "is it required?"
]
def check_document_relevance(doc, question, threshold=0.3):  # Lower threshold
    if any(keyword in question.lower() for keyword in ["reaction", "feedback", "community","strategies","teams"]):
        threshold = 0.1  # Lower threshold for subjective queries
    doc_content = doc.page_content.lower()
    question_words = set(question.lower().split())
    overlap_count = sum(1 for word in question_words if word in doc_content)
    relevance_score = overlap_count / len(question_words) if question_words else 0
    return relevance_score >= threshold

import time
def retrieve_with_retry(qa_chain, question, retries=5, delay=1):
    for attempt in range(retries):
        try:
            response = qa_chain({"query": question})
            if response and response.get("source_documents"):
                relevant_docs = [doc for doc in response["source_documents"] 
                               if check_document_relevance(doc, question)]
                if relevant_docs:
                    response["source_documents"] = relevant_docs
                    return response
            time.sleep(delay)  # Add delay between retries
        except Exception as e:
            logger.error(f"Retrieval attempt {attempt+1} failed: {e}")
            time.sleep(delay)
    return {
        "result": "I couldn't find sufficiently relevant information in our knowledge base for your question.",
        "source_documents": []
    }
def is_filtered_query_type(answer):
    """Determine if the answer indicates a greeting, unrelated query, or need for clarification."""
    answer_lower = answer.lower()
    # Check for greeting patterns in the answer
    greeting_indicators = [
        "hi! how can i assist you", 
        "hello there!", 
        "hey! ready to help",
        "good morning! what ea-related",
        "good afternoon! how can i help",
        "good evening! what ea",
        "i'm just a bot, but thanks for asking"
    ]
    # Check for unrelated query patterns
    unrelated_indicators = [
        "not seem related to ea",
        "not related to ea",
        "please ask about ea-related topics",
        "not sure",
        "don't have",
        "sorry",
        "don't know"
    ]
    # Check for clarification request patterns
    clarification_indicators = [
        "provide more context or details",
        "your question seems a bit unclear",
        "could you provide more specific details",
        "can you please provide more details"
    ]
     # Check if any patterns match
    if any(indicator in answer_lower for indicator in greeting_indicators + unrelated_indicators + clarification_indicators):
        return True
    return False
def ask_question(qa_chain, question, session_id, file_urls):
    global session_memory
    
    # Initialize session memory if it doesn't exist
    if session_id not in session_memory:
        session_memory[session_id] = []
    # Check for static answers first
    best_match = process.extractOne(question.lower(), static_answers.keys(), scorer=fuzz.token_set_ratio)
    if best_match and best_match[1] > 60:
        return {"answer": static_answers[best_match[0]]}
    last_interaction = session_memory[session_id][-1] if session_memory[session_id] else None
    
    # Handle clarification responses
    if last_interaction and isinstance(last_interaction, dict) and last_interaction.get('type') == 'clarification':
        current_level = last_interaction['level']
        topic = last_interaction['topic']
        
        # Process level 1 response
        if current_level == 1:
            # Try to match the response to one of our options
            response_text = question.strip().lower()
            best_match, score = process.extractOne(response_text, CLARIFICATION_LEVELS[topic]['level2'].keys())
            
            if score > 60:  # If we have a good match
                session_memory[session_id].append({
                    'type': 'clarification',
                    'level': 2,
                    'topic': topic,
                    'subtopic': best_match,
                    'original': last_interaction['original']
                })
                return {
                    "answer": CLARIFICATION_LEVELS[topic]['level2'][best_match],
                    "needs_clarification": True
                }
        
        # Process level 2 response
        elif current_level == 2:
            focused_question = (
                f"{last_interaction['original']}\n"
                f"Game: {last_interaction['subtopic']}\n"
                f"Specific issue: {question}"
            )
            return _process_question(qa_chain, focused_question, session_id, file_urls)
    
    # Detect topics that need clarification
    detected_topics = []
    question_lower = question.lower()
    
    for topic, config in CLARIFICATION_LEVELS.items():
        if 'triggers' in config:
            if any(trigger in question_lower for trigger in config['triggers']):
                detected_topics.append(topic)
    
    # Check if the question is ambiguous and related to a detected topic
    is_ambiguous = any(phrase in question_lower for phrase in ambiguous_phrases)
    
    if detected_topics and is_ambiguous:
        primary_topic = detected_topics[0]  # Take the first matching topic
        session_memory[session_id].append({
            'type': 'clarification',
            'level': 1,
            'topic': primary_topic,
            'original': question
        })
        return {
            "answer": CLARIFICATION_LEVELS[primary_topic]['level1'],
            "needs_clarification": True
        }
    
    # Rest of your existing function...
    return _process_question(qa_chain, question, session_id, file_urls)

def _process_question(qa_chain, question, session_id, file_urls):
    """Handle actual question processing after clarifications"""
    # Your existing RAG processing logic here
    rag_response = retrieve_with_retry(qa_chain, question)
    basic_answer = rag_response.get("result", "⚠️ No relevant information found.")
    try:
        # ✅ FAISS Retrieval with improved Retry logic
        rag_response = retrieve_with_retry(qa_chain, question)
        
        # Get the answer from RAG
        basic_answer = rag_response.get("result", "⚠️ No relevant information found.")
        source_docs = rag_response.get("source_documents", [])
        
        # If no relevant documents were found, inform the user
        if not source_docs:
            return "⚠️ I couldn't find any relevant documents that answer your question. Could you rephrase or provide more details?"
        # ✅ Improved URL mapping with better error handling
        source_strings = []
        seen_sources = set()
        for doc in source_docs:
            try:
                source = doc.metadata.get('source', 'Unknown source')
                filename = os.path.basename(source)
            # Handle file URL mapping with fallbacks
                if file_urls:
                    # Try exact match first
                    if filename in file_urls:
                        url = file_urls[filename]
                        if url not in seen_sources:
                            source_strings.append(url)
                            seen_sources.add(url)
                        continue
                    # Try partial matching if exact match fails
                    for key in file_urls:
                        if key in source or source in key:
                            url = file_urls[key]
                            if url not in seen_sources:
                                source_strings.append(url)
                                seen_sources.add(url)
                            break
                    else:  # No match found in the for-else pattern
                        cleaned_source = source.replace("\\", "/")
                        if cleaned_source not in seen_sources:
                            source_strings.append(cleaned_source)
                            seen_sources.add(cleaned_source)
            except Exception as e:
                logger.error(f"Error processing source document: {e}")
                if "Source unavailable" not in seen_sources:
                    source_strings.append("Source unavailable")
                    seen_sources.add("Source unavailable")
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
- If the question is **unrelated to EA** then respond with:  
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
        previous_interactions = "\n".join(session_memory[session_id][-30:]) if session_memory[session_id] else "No prior conversation."
        enhanced_response = enhance_chain.run(
            history=previous_interactions,
            answer=basic_answer,
            question=question,
            sources=sources_text
        ).strip() 
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
        session_memory[session_id].append(f"Q: {question}\nA: {final_answer}")  # Append to conversation list
        # Keep only last 30 interactions
        if len(session_memory[session_id]) > 30:
            session_memory[session_id] = session_memory[session_id][-30:]
        # Return either just answer or answer with sources
        if is_filtered_query_type(final_answer):
            return {"answer": final_answer}  # No sources key at all
        else:
            return {"answer": final_answer, "sources": source_strings}
    except Exception as e:
        logger.error(f"⚠️ Error processing query: {e}")
        return {"answer": f"⚠️ Sorry, I encountered an error while processing your question: {str(e)}"}
