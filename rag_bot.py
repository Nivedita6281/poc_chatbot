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
import time
import re
from collaborative_filtering import CollaborativeFilter

logger = logging.getLogger(__name__)
s3_client = boto3.client("s3")

# Session memory storage (Stores last 30 interactions per session)
session_memory = {}
# Add during bot initialization
collab_filter = CollaborativeFilter(
    similarity_threshold=0.6,  # Adjust based on your needs
    min_common_questions=1
)
class ConversationContext:
    def __init__(self, session_id):
        self.session_id = session_id
        self.clarification_state = None
        self.interactions = []  # Store full interaction objects instead of separate lists
        self.context_window = 30
        self.last_active = time.time()
        
    def add_interaction(self, question, answer, sources=None, needs_clarification=False):
        """Add a question-answer pair to the conversation history with metadata"""
        interaction = {
            'question': question,
            'answer': answer,
            'sources': sources or [],
            'timestamp': time.time(),
            'needs_clarification': needs_clarification
        }
        self.interactions.append(interaction)
        self.last_active = time.time()
        
        # Trim history if it exceeds context window
        if len(self.interactions) > self.context_window:
            self.interactions.pop(0)
    
    def get_context(self, max_interactions=3):
        """Generate conversation context string with most relevant interactions"""
        if not self.interactions:
            return "No previous conversation history."
            
        # Get last N interactions, prioritizing those with sources
        recent_with_sources = [i for i in self.interactions[-self.context_window:] if i.get('sources')]
        recent_general = [i for i in self.interactions[-self.context_window:] if not i.get('sources')]
        
        selected = (recent_with_sources + recent_general)[-max_interactions:]
        
        context = []
        for interaction in selected:
            ctx = f"Q: {interaction['question']}\nA: {interaction['answer']}"
            if interaction.get('sources'):
                ctx += f"\nSources: {', '.join(interaction['sources'])}"
            context.append(ctx)
            
        return "\n\n".join(context)
    def find_similar_question(self, current_question, threshold=85):
        """Modified version with better similarity detection"""
        # Preprocess question
        current_processed = re.sub(r'[^\w\s]', '', current_question.lower())
        for interaction in reversed(self.interactions):
            prev_processed = re.sub(r'[^\w\s]', '', interaction['question'].lower())
            # Try multiple similarity measures
            token_ratio = fuzz.token_set_ratio(current_processed, prev_processed)
            seq_ratio = fuzz.token_sort_ratio(current_processed, prev_processed)
            # Use the higher of the two scores
            similarity = max(token_ratio, seq_ratio)
            if similarity > threshold:
                logger.info(f"Found similar question with score {similarity}: {interaction['question']}")
                return interaction
        return None
    
    def set_clarification_state(self, state):
        """Set the current clarification state"""
        self.clarification_state = state
        
    def clear_clarification_state(self):
        """Clear the clarification state"""
        self.clarification_state = None

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
        combined.append(("bm25", idx, score * 0.7))  # BM25 weight: 60%
    for idx, doc in enumerate(faiss_docs):
        combined.append(("faiss", idx, 0.3))  # FAISS weight: 40%
    
    # Sort by combined score
    combined.sort(key=lambda x: x[2], reverse=True)
    return combined[:k]
def create_rag_bot(vector_store):
    """Creates a RAG bot using hybrid retrieval (BM25 + FAISS)."""
    if vector_store is None:
        raise ValueError("⚠️ FAISS vector store is not loaded. Please upload documents first.")
    retriever = vector_store.as_retriever(
        search_type="similarity",  # Use Max Marginal Relevance for better diversity
        search_kwargs={
            "k": 5 # Increase number of documents retrieved 
        }
    )
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# Static answers for frequently asked questions
static_answers = {
    "hi": "Hi! How can I assist you with EA-related questions today?",
    "hello": "Hello there! What EA-related questions can I help you with?",
    "hey": "Hey! Ready to help with any EA games or services questions.",
    "good morning": "Good morning! What EA-related assistance do you need today?",
    "good afternoon": "Good afternoon! How can I help with EA products or services?",
    "good evening": "Good evening! What EA questions can I answer for you?",
    "how are you":"I'm just a bot, but thanks for asking! How can I help with EA today?",
    "bye": "Goodbye! 👋 Have a great day!",
    "goodbye": "Take care! It was a pleasure assisting you.",
    "talk to you later": "Sure! Ping me anytime you need help.",
    "take care": "Take care! Wishing you well.",
    "until next time": "Until next time, stay safe!",
    "exit": "Session ended. Goodbye!",
    "quit": "Shutting down. See you!",
    "refund": "Find out how to get a refund for games that qualify under the Great Game Guarantee: https://help.ea.com/en-us/help/account/returns-and-cancellations/",
    "delete account": "To delete your EA account, visit: https://help.ea.com/en/help/account/close-ea-account",
    "reset password": "To reset your password, visit: https://ea.com/reset-password",
    "cross-play": "Cross-play in NHL® 25 lets you play with friends across PlayStation® and Xbox. Read more here: https://help.ea.com/in/cross-play-guide",
    "crafting metals": "Find more about Crafting Metals and Legend Tokens here: https://help.ea.com/in/solutions/?product=apex-legends&platform=&topic=metals-tokens-info",
}

ambiguous_phrases = [
    "can I turn it off?", "can I change it?", "how does it work?", "where can I find it?",
    "does it work online?", "is it enabled?", "is it better?", "do I need it?",
    "can I use this?", "will this affect my game?", "is it required?"
]

def check_document_relevance(doc, question, threshold=0.1):  # Reduced from 0.15
    lowered_q = question.lower()
    lowered_doc = doc.page_content.lower()

    # Always accept documents that contain important EA keywords
    important_keywords = [
        "ea sports", "electronic arts", "fifa", "madden", "nhl", 
        "apex legends", "battlefield", "sims", "nba", "fc 24","community","commentatory"
    ]
    if any(kw in lowered_doc for kw in important_keywords):
        return True

    # More lenient matching for EA-related questions
    question_words = set(lowered_q.split())
    overlap_count = sum(1 for word in question_words if word in lowered_doc)
    return overlap_count >= 2  # At least 2 matching words

def retrieve_with_retry(qa_chain, question, retries=6, delay=1):
    for attempt in range(retries):
        try:
            response = qa_chain({"query": question})
            # Add logging to track what's happening
            logger.info(f"Retrieved {len(response.get('source_documents', []))} documents")
            
            if response and response.get("source_documents"):
                relevant_docs = [doc for doc in response["source_documents"] 
                
                               if check_document_relevance(doc, question)]
                
                logger.info(f"Found {len(relevant_docs)} relevant documents")
                
                if relevant_docs:
                    response["source_documents"] = relevant_docs
                    return response
                else:
                    # Continue with less relevant docs rather than dropping them entirely
                    logger.warning("No highly relevant docs found, using all retrieved docs")
                    return response
            time.sleep(delay)
        except Exception as e:
            logger.error(f"Retrieval attempt {attempt+1} failed: {e}")
            time.sleep(delay)
    
    # Fall back to a gentler response
    return {
        "result": "I couldn't find specific information for your question in our knowledge base. Let me provide a general response based on what I know about EA.",
        "source_documents": []
    }
def is_filtered_query_type(answer):
    """Determine if the answer indicates a greeting, unrelated query, or need for clarification."""
    answer_lower = answer.lower()
    greeting_indicators = [
        "hi! how can i assist you", 
        "hello there!", 
        "hey! ready to help",
        "good morning! what ea-related",
        "good afternoon! how can i help",
        "good evening! what ea",
        "i'm just a bot, but thanks for asking"
    ]
    unrelated_indicators = [
        "not seem related to ea",
        "not related to ea",
        "please ask about ea-related topics",
        "not sure",
        "don't have",
        "sorry",
        "don't know"
    ]
    clarification_indicators = [
        "provide more context or details",
        "your question seems a bit unclear",
        "could you provide more specific details",
        "can you please provide more details"
    ]
    if any(indicator in answer_lower for indicator in greeting_indicators + unrelated_indicators + clarification_indicators):
        return True
    return False

def get_conversation_context(session_id):
    """Get or create conversation context for a session"""
    # Clean up old sessions (optional)
    current_time = time.time()
    expired_sessions = []
    for sid, context in session_memory.items():
        if current_time - context.last_active > 3600:  # 1 hour timeout
            expired_sessions.append(sid)
    
    for sid in expired_sessions:
        logger.info(f"Removing expired session {sid}")
        del session_memory[sid]
    # Create or return existing context
    if session_id not in session_memory:
        logger.info(f"Creating new conversation context for session {session_id}")
        session_memory[session_id] = ConversationContext(session_id)
    session_memory[session_id].last_active = current_time
    return session_memory[session_id]
def ask_question(qa_chain, question, session_id, file_urls):
    collab_filter.add_question(session_id, question)
    logger.info(f"Processing question for session {session_id}: {question}")
    context = get_conversation_context(session_id)
    # First check if this is a follow-up about previous answer
    if context.interactions and is_follow_up_question(question, context.interactions[-1]):
        return handle_follow_up_question(question, context.interactions[-1], context)
    # Check for static answers first
    best_match = process.extractOne(question.lower(), static_answers.keys(), scorer=fuzz.token_set_ratio)
    if best_match and best_match[1] > 75:  # Slightly increased threshold for better precision
        answer = static_answers[best_match[0]]
        response_type = "static" if best_match and best_match[1] > 75 else "dynamic"
        logger.info(f"Returning {response_type} response")
        context.add_interaction(question, answer)
        return {"answer": answer}

    # Handle clarification responses if in clarification state
    if context.clarification_state:
        return handle_clarification_response(qa_chain, question, context, file_urls)
    
    # Check pure ambiguity before any other processing
    is_ambiguous = any(phrase in question.lower() for phrase in ambiguous_phrases)
    if is_ambiguous:
        ambig_response = {
            "answer": "❓ Your question seems a bit unclear. Could you provide more specific details about what you're looking for?",
            "needs_clarification": True
        }
        return ambig_response
    
    # Detect if question needs clarification
    clarification_response = detect_clarification_needed(question, context)
    if clarification_response:
        return clarification_response
    
    # Process regular question
    return process_regular_question(qa_chain, question, context, file_urls)

def is_follow_up_question(current_question, last_interaction):
    """Improved follow-up detection with better heuristics"""
    question_lower = current_question.lower()
    last_question = last_interaction['question'].lower()
    follow_up_phrases = [
        # Direct reference phrases
        # Existing phrases
        "what are the key points mentioned above",
        "what are the keypoints mentioned above",
    
        # Add these new variations
        "what are the main points mentioned above",
        "can you summarize the above",
        "summarize the above",
        "recap the above",
        "what were the key takeaways",
        "what were the main takeaways",
        "can you recap what we discussed",
        "what were the highlights",
        "give me a summary",
        "break it down for me",
        
        # Clarification requests
        "can you clarify",
        "what do you mean by",
        "could you explain",
        "can you break down",
        "help me understand",
        
        # Specific reference phrases
        "about that",
        "concerning that",
        "in relation to that",
        "with respect to that",
        "on that note",
        "following up on",
        "building on that",
        "to add to that",
        
        # Question continuations
        "and how does that work",
        "what else about",
        "anything more on",
        "what was that about",
        "can we discuss more",
        
        # Pronoun references
        "it",  # e.g., "How does it work?"
        "that",  # e.g., "Tell me more about that"
        "this",  # e.g., "Explain this"
        "they",  # e.g., "What do they do?"
        
        # Implicit follow-ups
        "also",
        "too",
        "as well",
        "in addition",
        "furthermore",
        
        # Common conversational continuations
        "by the way",
        "speaking of",
        "while we're on the topic",
        "since you mentioned",
        
        # Short follow-up prompts
        "more details",
        "more info",
        "more explanation",
        "break it down",
        "give examples",
        "show me how",
        
        # Question fragments that imply continuation
        "how come",
        "why is that",
        "what makes",
        "what causes",
        "how would",
        
        # User confirmation phrases
        "are you saying",
        "does that mean",
        "so you're telling me",
        "if I understand correctly",
        
        # Comparative phrases
        "compared to",
        "versus",
        "difference between",
        "similar to"
    ]
    
    # Check if question contains follow-up phrases
    current_lower = current_question.lower()
    if any(phrase in current_lower for phrase in follow_up_phrases):
        return True
        
    # Check similarity to last question
    similarity = fuzz.token_set_ratio(current_lower, last_interaction['question'].lower())
    if similarity > 75:
        return True
    # Check for pronoun-only questions (very likely follow-ups)
    words = question_lower.split()
    if len(words) <= 5 and any(pronoun in words for pronoun in ["it", "this", "that", "they", "these", "those"]):
        logger.debug("Follow-up detected: pronoun-only question")
        return True
    # Check if question starts with a conjunction (common in follow-ups)
    conjunctions = ["and", "but", "so", "or", "yet", "for", "nor"]
    first_word = current_lower.split()[0] if current_lower.split() else ""
    if first_word in conjunctions:
        return True
        
    return False

def handle_follow_up_question(question, last_interaction, context):
    """Handle questions that reference previous answers"""
    question_lower = question.lower()
    
    # Get the focus of the follow-up question
    focus_words = set(question_lower.split()) - set(['it', 'the', 'in', 'what', 'are', 'is', 'about', 'how', 'can', 'does'])
    
    # For specific section requests like "strategic considerations"
    if any(word in question_lower for word in ["strategic", "considerations", "strategy"]):
        # Extract just the strategic considerations section from previous answer
        pattern = r"(?:\*\*Strategic Considerations\*\*:.*?(?=\n\n\d+\.|\Z))"
        match = re.search(pattern, last_interaction['answer'], re.DOTALL)
        if match:
            focused_answer = match.group(0)
            response = {
                "answer": f"Regarding strategic considerations in Madden 25 player ratings:\n\n{focused_answer}",
                "sources": last_interaction.get('sources', [])
            }
            context.add_interaction(question, response['answer'], response['sources'])
            return response
    # Handle summary requests
    if any(phrase in question_lower for phrase in ["key points", "keypoints", "summarize", "recap", "main points"]):
        if last_interaction:
            key_points = extract_key_points(last_interaction['answer'])
            if key_points:
                response = {
                    "answer": f"Here's a summary of our previous discussion:\n\n{key_points}",
                    "sources": last_interaction.get('sources', [])
                }
                context.add_interaction(question, response['answer'], response['sources'])
                return response
    
    # Handle other follow-ups by returning the last relevant answer
    # This is where we shouldn't just return the whole previous answer
    
    # Create a more specific prompt for handling follow-up questions
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
    follow_up_template = """
    Previous answer: {previous_answer}
    
    Follow-up question: {follow_up_question}
    
    Instructions:
    1. If the question is about EA, you MUST provide a detailed response with at least 3-5 key points
2. If information is incomplete, say "Based on available information:" then provide what you know
3. Never say "no specific mention" - instead provide related information
4. Structure responses with clear bullet points
5. Always try to connect to known EA products/features
    
    Focused answer:
    """
    
    follow_up_prompt = PromptTemplate(
        template=follow_up_template,
        input_variables=["previous_answer", "follow_up_question"]
    )
    follow_up_chain = LLMChain(llm=llm, prompt=follow_up_prompt)
    
    try:
        focused_answer = follow_up_chain.run(
            previous_answer=last_interaction['answer'],
            follow_up_question=question
        ).strip()
        
        # Generate suggestions for follow-ups too
        suggestions = generate_question_suggestions(context) if context.interactions else []
        
        response = {
            "answer": focused_answer,
            "sources": last_interaction.get('sources', []),
            "from_memory": True,
            "people_can_also_ask": suggestions if suggestions else []
        }
        context.add_interaction(question, response['answer'], response['sources'])
        return response
    except Exception as e:
        # Fall back to returning the previous answer if processing fails
        logger.error(f"Error processing follow-up: {e}")
        return {
            "answer": last_interaction['answer'],
            "sources": last_interaction.get('sources', []),
            "from_memory": True,
            "people_also_ask": generate_question_suggestions(context) or None
        }
def extract_key_points(answer_text):
    """Extract key points from a previous answer"""
    # First try to find numbered lists
    points = re.findall(r'\d+\.\s*(.*?)(?:\n|$)', answer_text)
    if not points:
        # Then try bullet points
        points = re.findall(r'[-•*]\s*(.*?)(?:\n|$)', answer_text)
    if not points:
        # Fallback to splitting sentences
        points = [s.strip() for s in re.split(r'[.!?]', answer_text) if s.strip()]
    
    # Filter and limit to 5 most relevant points
    points = [p for p in points if len(p.split()) > 3][:5]
    
    if points:
        return "\n".join(f"- {point.strip()}" for point in points)
    return "No key points could be extracted from the previous answer."

def handle_clarification_response(qa_chain, question, context, file_urls):
    """Handle user responses during clarification flow"""
    # Since we've removed CLARIFICATION_LEVELS, we'll simplify this function
    # Clear clarification state and process as regular question
    context.clear_clarification_state()
    return _process_question(qa_chain, question, context, file_urls)

def detect_clarification_needed(question, context):
    """Detect if question needs clarification and setup state if needed"""
    question_lower = question.lower()
    #is_ambiguous = any(phrase in question_lower for phrase in ambiguous_phrases)
    #if is_ambiguous and context.interactions:
        # Check if recent context clarifies the question
        #last_question = context.interactions[-1]['question'].lower()
        #if fuzz.token_set_ratio(question_lower, last_question) > 70:
           # return None
    # Check ambiguous phrases first
    is_ambiguous = any(phrase in question_lower for phrase in ambiguous_phrases)
    if is_ambiguous:
        return {
            "answer": "❓ Your question seems a bit unclear. Could you provide more specific details about what you're looking for?",
            "needs_clarification": True
        }
    
    return None

def process_regular_question(qa_chain, question, context, file_urls):
    """Process regular questions that don't need clarification"""
    # Check if this question was asked before in this session using the new interactions list
    similar_interaction = context.find_similar_question(question)
    if similar_interaction:
        return {
            "answer": similar_interaction['answer'],
            "sources": similar_interaction.get('sources', []),
            "from_memory": True,
            "people_also_ask": generate_question_suggestions(context) or None
        }
    
    # Process new question
    return _process_question(qa_chain, question, context, file_urls)

def _process_question(qa_chain, question, context, file_urls):
    try:
        # First check for similar questions in context
        similar_interaction = context.find_similar_question(question)
        if similar_interaction:
            response = {
                "answer": f"Regarding your similar previous question:\n\n{similar_interaction['answer']}",
                "sources": similar_interaction.get('sources', []),
                "from_memory": True
            }
            # Generate suggestions after adding the interaction
            context.add_interaction(question, response['answer'], response['sources'])
            suggestions = generate_question_suggestions(context)
            if suggestions:
                response["people_can_also_ask"] = suggestions
            return response
        
        # Process the new question
        rag_response = retrieve_with_retry(qa_chain, question)
        basic_answer = rag_response.get("result", "⚠️ No relevant information found.")
        source_docs = rag_response.get("source_documents", [])
        
        if not source_docs:
            answer = "⚠️ I couldn't find any relevant documents that answer your question. Could you rephrase or provide more details?"
            context.add_interaction(question, answer)
            return {"answer": answer}
        
        # Process sources
        source_strings = []
        seen_sources = set()
        for doc in source_docs:
            try:
                source = doc.metadata.get('source', 'Unknown source')
                filename = os.path.basename(source)
                
                if file_urls:
                    if filename in file_urls:
                        url = file_urls[filename]
                        if url not in seen_sources:
                            source_strings.append(url)
                            seen_sources.add(url)
                        continue
                    
                    for key in file_urls:
                        if key in source or source in key:
                            url = file_urls[key]
                            if url not in seen_sources:
                                source_strings.append(url)
                                seen_sources.add(url)
                            break
                    else:
                        cleaned_source = source.replace("\\", "/")
                        if cleaned_source not in seen_sources:
                            source_strings.append(cleaned_source)
                            seen_sources.add(cleaned_source)
            except Exception as e:
                logger.error(f"Error processing source document: {e}")
                if "Source unavailable" not in seen_sources:
                    source_strings.append("Source unavailable")
                    seen_sources.add("Source unavailable")

        # Enhance response with context
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)

        enhance_template = """You are an expert knowledge assistant for EA (Electronic Arts) questions ONLY. 

### IMPORTANT: FIRST determine if the question is about EA (Electronic Arts) games, services, or products:
1. If the question is about EA, you MUST provide a detailed response with at least 3-5 key points
2. If information is incomplete, say "Based on available information:" then provide what you know
3. Never say "no specific mention" - instead provide related information
4. Structure responses with clear bullet points
5. Always try to connect to known EA products/features
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
- Should retrieve sources even if the response is retrieving from session memory.
- Provide a detailed response with at least 5 numbered points (unless unnecessary for short, direct answers)
- Response should retrieve if question is related to EA and give at least 5 points.

Enhanced Answer:"""

        enhance_prompt = PromptTemplate(
        template=enhance_template,
        input_variables=["history", "answer", "question", "sources"]
        )
        enhance_chain = LLMChain(llm=llm, prompt=enhance_prompt)
        
        sources_text = "\n".join([f"- {src}" for src in source_strings]) if source_strings else "No specific sources retrieved."
        enhanced_response = enhance_chain.run(
            history=context.get_context(),
            answer=basic_answer,
            question=question,
            sources=sources_text
        ).strip()

        # Store the interaction FIRST
        context.add_interaction(question, enhanced_response, source_strings)
        
        # THEN generate suggestions based on the complete context
        suggestions = generate_question_suggestions(context)
        
        # Prepare response
        response = {
            "answer": enhanced_response,
            "sources": source_strings
        }
        
        # Only add suggestions if not a filtered query type and we have suggestions
        if not is_filtered_query_type(enhanced_response) and suggestions:
            response["people_can_also_ask"] = suggestions
            
        return response
            
    except Exception as e:
        logger.error(f"⚠️ Error processing query: {e}")
        error_msg = f"⚠️ Sorry, I encountered an error while processing your question: {str(e)}"
        context.add_interaction(question, error_msg)
        return {"answer": error_msg}
def generate_question_suggestions(context, max_suggestions=3):
    """Hybrid suggestion generator combining LLM and collaborative filtering"""
    if not context.interactions:
        return []
    
    current_interaction = context.interactions[-1]
    current_question = current_interaction['question']
    
    # Get suggestions from both sources
    llm_suggestions = generate_llm_suggestions(context, max_suggestions)
    cf_suggestions = collab_filter.get_suggestions(
        context.session_id,
        current_question,
        max_suggestions=max_suggestions
    )
    
    # Add logging for raw suggestions
    logger.info(f"Raw CF suggestions: {cf_suggestions}")
    logger.info(f"Raw LLM suggestions: {llm_suggestions}")
    
    # Filter CF suggestions by relevance to current topic
    filtered_cf_suggestions = [
        s for s in cf_suggestions 
        if is_relevant_to_current_topic(s, current_question)
    ]
    
    # Add the logging you asked about here
    current_topic = extract_main_topic(current_question)
    logger.info(f"Filtered CF suggestions (relevance > 50%): {filtered_cf_suggestions}")
    if filtered_cf_suggestions:
        suggestion_topic = extract_main_topic(filtered_cf_suggestions[0])
        logger.info(f"Current topic: {current_topic}, Suggested topic: {suggestion_topic}")
    
    # Combine and deduplicate suggestions
    combined = []
    seen = set()
    
    # Prioritize collaborative suggestions first
    for suggestion in filtered_cf_suggestions:
        if suggestion not in seen:
            combined.append(suggestion)
            seen.add(suggestion)
    
    # Add LLM suggestions if we need more
    for suggestion in llm_suggestions:
        if len(combined) >= max_suggestions:
            break
        if suggestion not in seen:
            combined.append(suggestion)
            seen.add(suggestion)
    
    return combined[:max_suggestions]
def is_relevant_to_current_topic(suggestion, current_question):
    """Check if suggestion is relevant to current question topic"""
    current_topic = extract_main_topic(current_question)
    suggestion_topic = extract_main_topic(suggestion)
    # Use multiple similarity measures
    token_ratio = fuzz.token_set_ratio(current_topic, suggestion_topic)
    seq_ratio = fuzz.token_sort_ratio(current_topic, suggestion_topic)
    
    # Consider the higher of the two scores
    return max(token_ratio, seq_ratio) > 50
def extract_main_topic(text):
    """Extract main topic from question text"""
    stop_words = {"what", "how", "why", "when", "where", "which", "are", "is", "do", "does", 
                 "can", "could", "would", "will", "the", "a", "an", "and", "or", "for"}
    words = [w for w in text.lower().split() if w not in stop_words]
    # Take more words or up to a certain length (e.g., 6-8 words)
    return " ".join(words[:8])   # First few meaningful words
def generate_llm_suggestions(context, max_suggestions):
    """Your original LLM-based suggestion generation"""
    current_interaction = context.interactions[-1]
    context_text = f"Q: {current_interaction['question']}\nA: {current_interaction['answer']}"
    
    prompt = f"""You are an intelligent assistant helping users with their queries. Based on the current and the context of previous interactions, suggest {max_suggestions} relevant follow-up questions the user might ask next. Make each question complete and specific.

Current Conversation:
{context_text}

Respond with exactly {max_suggestions} bullet point questions in this format:
- First suggested question
- Second suggested question
- Third suggested question

Suggested questions:"""
    
    try:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
        response = llm.invoke(prompt)
        
        suggestions = []
        for line in response.content.split('\n'):
            line = line.strip()
            if line.startswith('-') or line.startswith('*'):
                question = line[1:].strip()
                if question and len(suggestions) < max_suggestions:
                    suggestions.append(question)
        
        return suggestions[:max_suggestions] if suggestions else []
    except Exception as e:
        logger.error(f"Error generating LLM suggestions: {e}")
        return []
