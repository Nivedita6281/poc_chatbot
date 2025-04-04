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

logger = logging.getLogger(__name__)
s3_client = boto3.client("s3")

# Session memory storage (Stores last 30 interactions per session)
session_memory = {}

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
    
    def find_similar_question(self, current_question, threshold=80):
        """Find similar previous questions using fuzzy matching"""
        for interaction in reversed(self.interactions):
            similarity = fuzz.token_set_ratio(current_question.lower(), interaction['question'].lower())
            if similarity > threshold:
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

# Static answers for frequently asked questions
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

ambiguous_phrases = [
    "can I turn it off?", "can I change it?", "how does it work?", "where can I find it?",
    "does it work online?", "is it enabled?", "is it better?", "do I need it?",
    "can I use this?", "will this affect my game?", "is it required?"
]

def check_document_relevance(doc, question, threshold=0.3):
    if any(keyword in question.lower() for keyword in ["reaction", "feedback", "community","strategies","teams","Franchise Mode"]):
        threshold = 0.1  # Lower threshold for subjective queries
    doc_content = doc.page_content.lower()
    question_words = set(question.lower().split())
    overlap_count = sum(1 for word in question_words if word in doc_content)
    relevance_score = overlap_count / len(question_words) if question_words else 0
    return relevance_score >= threshold

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
    if session_id not in session_memory:
        session_memory[session_id] = ConversationContext(session_id)
    return session_memory[session_id]

def ask_question(qa_chain, question, session_id, file_urls):
    context = get_conversation_context(session_id)
    # First check if this is a follow-up about previous answer
    if context.interactions:
        last_interaction = context.interactions[-1]
        if is_follow_up_question(question, last_interaction):
            return handle_follow_up_question(question, last_interaction, context)
    # Check for static answers first
    best_match = process.extractOne(question.lower(), static_answers.keys(), scorer=fuzz.token_set_ratio)
    if best_match and best_match[1] > 75:  # Slightly increased threshold for better precision
        answer = static_answers[best_match[0]]
        context.add_interaction(question, answer)
        return {"answer": answer}
    
    # Handle clarification responses if in clarification state
    if context.clarification_state:
        return handle_clarification_response(qa_chain, question, context, file_urls)
    
    # Check pure ambiguity before any other processing
    is_ambiguous = any(phrase in question.lower() for phrase in ambiguous_phrases)
    if is_ambiguous and not any(term in question.lower() for topic in CLARIFICATION_LEVELS 
                             for term in CLARIFICATION_LEVELS[topic].get('triggers', [])):
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
    """Determine if the current question is a follow-up about the last answer"""
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
    if similarity > 60:
        return True
        
    # Check if question starts with a conjunction (common in follow-ups)
    conjunctions = ["and", "but", "so", "or", "yet", "for", "nor"]
    first_word = current_lower.split()[0] if current_lower.split() else ""
    if first_word in conjunctions:
        return True
        
    return False

def handle_follow_up_question(question, last_interaction, context):
    """Handle questions that reference previous answers"""
    # If asking about key points, extract them from previous answer
    if "keypoints" in question.lower() or "key points" in question.lower():
        key_points = extract_key_points(last_interaction['answer'])
        if key_points:
            response = {
                "answer": f"Here are the key points from our previous discussion:\n\n{key_points}",
                "sources": last_interaction.get('sources', [])
            }
            context.add_interaction(question, response['answer'], response['sources'])
            return response
    
    # Otherwise, try to answer based on previous context
    similar_question = context.find_similar_question(question)
    if similar_question:
        response = {
            "answer": f"Regarding your previous question about \"{similar_question['question']}\":\n\n{similar_question['answer']}",
            "sources": similar_question.get('sources', []),
            "from_memory": True
        }
        context.add_interaction(question, response['answer'], response['sources'])
        return response
    
    # Fall back to processing as new question
    return None

def extract_key_points(answer_text):
    """Extract key points from a previous answer"""
    # Look for numbered or bulleted points
    points = re.findall(r'\d+\.\s*(.*?)(?:\n|$)', answer_text)
    if not points:
        points = re.findall(r'-\s*(.*?)(?:\n|$)', answer_text)
    if not points:
        points = re.findall(r'\*\s*(.*?)(?:\n|$)', answer_text)
    
    if points:
        return "\n".join(f"- {point.strip()}" for point in points if point.strip())
    return None

def handle_clarification_response(qa_chain, question, context, file_urls):
    """Handle user responses during clarification flow"""
    last_state = context.clarification_state
    
    # Process level 1 response
    if last_state['level'] == 1:
        response_text = question.strip().lower()
        topic_options = list(CLARIFICATION_LEVELS[last_state['topic']]['level2'].keys())
        
        # Handle numeric responses (1, 2, 3, etc.)
        if response_text.isdigit():
            idx = int(response_text) - 1
            if 0 <= idx < len(topic_options):
                best_match = topic_options[idx]
                score = 100  # Perfect match if numeric option selected
            else:
                return {
                    "answer": f"Please select a valid option (1-{len(topic_options)}). Or describe your issue in more detail.",
                    "needs_clarification": True
                }
        else:
            # Use fuzzy matching for text responses
            best_match, score = process.extractOne(response_text, topic_options)
        
        if score > 60:  # If we have a good match
            context.clarification_state = {
                'level': 2,
                'topic': last_state['topic'],
                'subtopic': best_match,
                'original': last_state['original']
            }
            return {
                "answer": CLARIFICATION_LEVELS[last_state['topic']]['level2'][best_match],
                "needs_clarification": True
            }
        else:
            # Couldn't match response to any option, try to process as regular question
            context.clear_clarification_state()
            return _process_question(qa_chain, question, context, file_urls)
    
    # Process level 2 response
    elif last_state['level'] == 2:
        # Check if this is a numeric response for level 2
        if question.strip().isdigit():
            subtopic_detail = question.strip()
        else:
            subtopic_detail = question
        
        # Create a focused question combining all context
        focused_question = f"{last_state['original']} (Specifically about {last_state['topic']} - {last_state['subtopic']} - {subtopic_detail})"
        
        # Clear clarification state since we're done with clarification
        context.clear_clarification_state()
        
        # Process the focused question
        return _process_question(qa_chain, focused_question, context, file_urls)
    
    # If we get here, clarification wasn't successful
    context.clear_clarification_state()
    return _process_question(qa_chain, question, context, file_urls)

def detect_clarification_needed(question, context):
    """Detect if question needs clarification and setup state if needed"""
    question_lower = question.lower()
    
    # Check ambiguous phrases first - this was missing
    is_ambiguous = any(phrase in question_lower for phrase in ambiguous_phrases)
    if is_ambiguous and not any(topic_trigger in question_lower for topic in CLARIFICATION_LEVELS 
                              for topic_trigger in CLARIFICATION_LEVELS[topic].get('triggers', [])):
        return {
            "answer": "❓ Your question seems a bit unclear. Could you provide more specific details about what you're looking for?",
            "needs_clarification": True
        }
    
   # 1. Check for strong topic triggers (existing logic)
    detected_topics = []
    for topic, config in CLARIFICATION_LEVELS.items():
        if any(trigger in question_lower for trigger in config['triggers']):
            detected_topics.append(topic)
    
    # 3. Trigger clarification if ambiguous OR topic detected
    if is_ambiguous or detected_topics:
        primary_topic = detected_topics[0] if detected_topics else "game_features"  # Default to game_features
        context.set_clarification_state({
            'level': 1,
            'topic': primary_topic,
            'original': question
        })
        return {
            "answer": CLARIFICATION_LEVELS[primary_topic]['level1'],
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
            "from_memory": True
        }
    
    # Process new question
    return _process_question(qa_chain, question, context, file_urls)

def _process_question(qa_chain, question, context, file_urls):
    """Handle actual question processing after clarifications"""
    try:
        # First check for similar questions in context
        similar_interaction = context.find_similar_question(question)
        if similar_interaction:
            return {
                "answer": f"Regarding your similar previous question:\n\n{similar_interaction['answer']}",
                "sources": similar_interaction.get('sources', []),
                "from_memory": True
            }
        
        # Rest of your existing _process_question implementation...
        rag_response = retrieve_with_retry(qa_chain, question)
        basic_answer = rag_response.get("result", "⚠️ No relevant information found.")
        source_docs = rag_response.get("source_documents", [])
        
        if not source_docs:
            answer = "⚠️ I couldn't find any relevant documents that answer your question. Could you rephrase or provide more details?"
            context.add_interaction(question, answer)
            return {"answer": answer}
        
        # Process sources as before
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

        # Get relevant conversation history using the new interactions list
        recent_history = "\n\n".join([
            f"Q: {i['question']}\nA: {i['answer']}" 
            for i in context.interactions[-3:]
        ]) if context.interactions else ""

        # Enhance response with context
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
- Should retrieve sources even if the response is retrieving from session memory.
- Response should retrieve if question is related to EA and give at least 5 points.

Enhanced Answer:"""

        enhance_prompt = PromptTemplate(
            template=enhance_template,
            input_variables=["history", "answer", "question", "sources"]
        )
        enhance_chain = LLMChain(llm=llm, prompt=enhance_prompt)
        
        sources_text = "\n".join([f"- {src}" for src in source_strings]) if source_strings else "No specific sources retrieved."
        enhanced_response = enhance_chain.run(
            history=recent_history,  # Using recent history for better focus
            answer=basic_answer,
            question=question,
            sources=sources_text
        ).strip()
        
        final_answer = enhanced_response
        
        # Store the interaction
        context.add_interaction(question, final_answer, source_strings)
        
        # Return response
        if is_filtered_query_type(final_answer):
            return {"answer": final_answer}
        else:
            return {"answer": final_answer, "sources": source_strings}
            
    except Exception as e:
        logger.error(f"⚠️ Error processing query: {e}")
        error_msg = f"⚠️ Sorry, I encountered an error while processing your question: {str(e)}"
        context.add_interaction(question, error_msg)
        return {"answer": error_msg}
