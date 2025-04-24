import numpy as np
from collections import defaultdict
from fuzzywuzzy import fuzz
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class CollaborativeFilter:
    def __init__(self, similarity_threshold: float = 0.5, min_common_questions: int = 1):
        self.user_questions = defaultdict(list)
        self.question_users = defaultdict(set)
        self.similarity_threshold = similarity_threshold
        self.min_common_questions = min_common_questions
        self.debug_mode = True

    def _clean_question(self, question: str) -> str:
        """Clean and normalize the question text for better comparison"""
        cleaned = ' '.join(question.strip().lower().split())
        return cleaned

    def add_question(self, session_id: str, question: str) -> None:
        cleaned_question = self._clean_question(question)
        self.user_questions[session_id].append(cleaned_question)
        self.question_users[cleaned_question].add(session_id)
        
        if self.debug_mode:
            logger.debug(f"Added question to session {session_id[-5:]}: {cleaned_question[:50]}...")

    def get_suggestions(self, current_session: str, current_question: str, max_suggestions: int = 3) -> List[str]:
        if current_session not in self.user_questions:
            if self.debug_mode:
                logger.debug(f"No questions tracked for session {current_session[-5:]}")
            return []

        current_question = self._clean_question(current_question)
        similar_users = self._find_similar_users(current_session, current_question)
        
        if self.debug_mode:
            logger.debug(f"Found {len(similar_users)} similar users to session {current_session[-5:]}")
            for user, score in similar_users:
                logger.debug(f"User {user[-5:]} (score: {score:.2f}): {self.user_questions[user][-1][:50]}...")
        
        return self._generate_suggestions(current_session, current_question, similar_users, max_suggestions)

    def _find_similar_users(self, current_session: str, current_question: str) -> List[Tuple[str, float]]:
        candidate_sessions = [s for s in self.user_questions.keys() if s != current_session]
        
        if not candidate_sessions:
            return []

        current_qs = set(self.user_questions[current_session][-3:])
        similar_users = []
        
        for session in candidate_sessions:
            other_qs = set(self.user_questions[session][-3:])
            
            content_similarity = self._calculate_content_similarity(
                current_qs, other_qs, current_question
            )
            
            if content_similarity >= self.similarity_threshold:
                similar_users.append((session, content_similarity))
        
        return sorted(similar_users, key=lambda x: x[1], reverse=True)
    
    def _calculate_content_similarity(self, current_qs, other_qs, current_question) -> float:
        """Calculate weighted similarity between question sets"""
        if not current_qs or not other_qs:
            return 0.0
        
        # Content similarity
        max_similarities = []
        for q1 in current_qs:
            q_sims = [fuzz.token_set_ratio(q1, q2) / 100.0 for q2 in other_qs]
            max_similarities.append(max(q_sims) if q_sims else 0)
        
        content_sim = sum(max_similarities) / len(max_similarities) if max_similarities else 0.0
    
        # Topic similarity
        current_topic = self._extract_topic(current_question)
        other_topics = [self._extract_topic(q) for q in other_qs]
        topic_sim = max(
            fuzz.token_set_ratio(current_topic, topic) / 100.0 
            for topic in other_topics
        ) if other_topics else 0
    
        return (content_sim * 0.7 + topic_sim * 0.3)

    def _extract_topic(self, text: str) -> str:
        """Extract main topic from question text"""
        stop_words = {"what", "how", "why", "when", "where", "which", "are", "is", "do", "does",
                      "can", "could", "would", "will", "the", "a", "an", "and", "or", "for"}
        words = [w for w in text.lower().split() if w not in stop_words]
        return " ".join(words[:8])

    def _generate_suggestions(self, current_session: str, current_question: str, 
                            similar_users: List[Tuple[str, float]], max_suggestions: int) -> List[str]:
        if not similar_users:
            return []
            
        current_user_questions = set(self.user_questions[current_session])
        candidate_questions = []
        
        for user_id, similarity_score in similar_users:
            for question in reversed(self.user_questions[user_id][-5:]):
                if question not in current_user_questions:
                    question_similarity = fuzz.token_set_ratio(current_question, question) / 100.0
                    relevance_score = similarity_score * 0.7 + question_similarity * 0.3
                    candidate_questions.append((question, relevance_score))
        
        # Deduplicate and sort
        seen_questions = set()
        unique_questions = []
        
        for question, score in sorted(candidate_questions, key=lambda x: x[1], reverse=True):
            is_duplicate = any(fuzz.token_set_ratio(question, seen_q) > 85 for seen_q in seen_questions)
            if not is_duplicate:
                unique_questions.append(question)
                seen_questions.add(question)
                
            if len(unique_questions) >= max_suggestions:
                break
                
        return unique_questions[:max_suggestions]

    def debug_state(self):
        """Print current state for debugging"""
        logger.debug("\n=== Collaborative Filter Debug ===")
        logger.debug(f"Tracked Sessions: {len(self.user_questions)}")
        logger.debug(f"Tracked Questions: {sum(len(q) for q in self.user_questions.values())}")
        
        for session, questions in list(self.user_questions.items())[:3]:
            logger.debug(f"\nSession {session[-5:]} (last 3 questions):")
            for q in questions[-3:]:
                logger.debug(f" - {q[:60]}...")
