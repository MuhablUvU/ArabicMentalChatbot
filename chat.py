from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import sqlite3
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords
import json
import pickle
import os
import logging
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Arabic Chatbot API", description="API for Arabic NLP Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Download Arabic stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
arabic_stopwords = set(stopwords.words('arabic'))

class ChatRequest(BaseModel):
    user_id: str
    text: str

class ChatHistory(BaseModel):
    user_id: str
    text: str
    predicted_class: str
    confidence: float
    response: str
    timestamp: str

# Initialize SQLite database
def init_db():
    try:
        conn = sqlite3.connect("arabic_chat_history.db")
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                text TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                response TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        ''')
        conn.commit()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")
        raise
    finally:
        conn.close()

init_db()

# Save interaction to database
def save_interaction(user_id: str, text: str, predicted_class: str, confidence: float, response: str):
    try:
        conn = sqlite3.connect("arabic_chat_history.db")
        c = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute('''
            INSERT INTO chat_history (user_id, text, predicted_class, confidence, response, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, text, predicted_class, confidence, response, timestamp))
        conn.commit()
    except Exception as e:
        logger.error(f"Error saving interaction: {str(e)}")
        raise
    finally:
        conn.close()

# Retrieve user history
def get_user_history(user_id: str, limit: int = 10) -> list:
    try:
        conn = sqlite3.connect("arabic_chat_history.db")
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('''
            SELECT user_id, text, predicted_class, confidence, response, timestamp
            FROM chat_history 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (user_id, limit))
        return [dict(row) for row in c.fetchall()]
    except Exception as e:
        logger.error(f"Error retrieving history: {str(e)}")
        return []
    finally:
        conn.close()

# Arabic Chatbot Class
class ArabicChatbot:
    def __init__(self, 
                 model_path: str = "arabic_nlp_optimized.keras",
                 tokenizer_path: str = "tokenizer_optimized.pickle", 
                 encoder_path: str = "label_encoder_optimized.pickle",
                 responses_path: str = "chatbot_responses.json",
                 max_len: int = 150):
        
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.encoder_path = encoder_path
        self.responses_path = responses_path
        self.max_len = max_len
        self.model = None
        self.tokenizer = None
        self.encoder = None
        self.responses_database = {}
        self.lock = threading.Lock()
        
        self._load_components()
    
    def _load_components(self):
        """Load model, tokenizer, label encoder and responses"""
        try:
            # Load model
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                logger.info("✅ Model loaded successfully")
            else:
                logger.error(f"❌ Model file not found: {self.model_path}")
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Load tokenizer
            if os.path.exists(self.tokenizer_path):
                with open(self.tokenizer_path, "rb") as handle:
                    self.tokenizer = pickle.load(handle)
                logger.info("✅ Tokenizer loaded successfully")
            else:
                logger.error(f"❌ Tokenizer file not found: {self.tokenizer_path}")
                raise FileNotFoundError(f"Tokenizer file not found: {self.tokenizer_path}")
            
            # Load label encoder
            if os.path.exists(self.encoder_path):
                with open(self.encoder_path, "rb") as f:
                    self.encoder = pickle.load(f)
                logger.info(f"✅ Label encoder loaded. Classes: {self.encoder.classes_}")
            else:
                logger.error(f"❌ Label encoder file not found: {self.encoder_path}")
                raise FileNotFoundError(f"Label encoder file not found: {self.encoder_path}")
            
            # Load responses
            if os.path.exists(self.responses_path):
                with open(self.responses_path, "r", encoding="utf-8") as f:
                    self.responses_database = json.load(f)
                logger.info("✅ Responses loaded successfully")
            else:
                logger.error(f"❌ Responses file not found: {self.responses_path}")
                # Create default responses
                self.responses_database = {
                    "positive": ["شكراً لك! كيف يمكنني مساعدتك؟"],
                    "negative": ["أنا هنا للمساعدة، ما الذي تحتاجه؟"],
                    "neutral": ["كيف يمكنني مساعدتك؟"],
                    "question": ["هذا سؤال ممتاز!"],
                    "greeting": ["مرحباً! كيف يمكنني مساعدتك؟"],
                    "goodbye": ["مع السلامة! أراك لاحقاً."]
                }
                with open(self.responses_path, "w", encoding="utf-8") as f:
                    json.dump(self.responses_database, f, ensure_ascii=False, indent=2)
                logger.info("✅ Created default responses")
                
        except Exception as e:
            logger.error(f"❌ Error loading components: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """Enhanced Arabic text preprocessing"""
        if not isinstance(text, str):
            return ""
        
        # Remove URLs, emails, diacritics, and special characters
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[\u0617-\u061A\u064B-\u0652\u0670\u0640]', '', text)
        text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s]', ' ', text)
        
        # Normalize Arabic letters
        replacements = {
            'إ': 'ا', 'أ': 'ا', 'آ': 'ا', 'ى': 'ي', 
            'ؤ': 'و', 'ئ': 'ي', 'ة': 'ه', 'گ': 'ك'
        }
        for orig, repl in replacements.items():
            text = text.replace(orig, repl)
        
        # Remove stopwords and short words
        words = [w for w in text.split() 
                if w not in arabic_stopwords and len(w) > 2]
        return ' '.join(words)
    
    def predict_intent(self, text: str) -> tuple:
        """Predict intent with thread safety"""
        with self.lock:
            try:
                processed_text = self.preprocess_text(text)
                if not processed_text:
                    return "neutral", 0.5
                
                sequence = self.tokenizer.texts_to_sequences([processed_text])
                padded = pad_sequences(sequence, maxlen=self.max_len, padding="post", truncating="post")
                prediction = self.model.predict(padded, verbose=0)
                
                if len(self.encoder.classes_) == 2:
                    predicted_class = self.encoder.classes_[1] if prediction[0][0] > 0.5 else self.encoder.classes_[0]
                    confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
                else:
                    predicted_class_idx = np.argmax(prediction[0])
                    predicted_class = self.encoder.classes_[predicted_class_idx]
                    confidence = prediction[0][predicted_class_idx]
                
                return predicted_class, float(confidence)
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                return "neutral", 0.0
    
    def generate_response(self, predicted_class: str, confidence: float) -> str:
        """Generate response based on predicted class"""
        responses = self.responses_database.get(
            predicted_class, 
            self.responses_database.get("neutral", ["كيف يمكنني مساعدتك؟"])
        )
        
        if confidence > 0.8:
            return np.random.choice(responses)
        elif confidence > 0.6:
            return responses[0] if responses else "كيف يمكنني مساعدتك؟"
        else:
            return "عذراً، لم أفهم تماماً. هل يمكنك إعادة الصياغة؟"

# Initialize chatbot
try:
    chatbot = ArabicChatbot()
    logger.info("Chatbot initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize chatbot: {str(e)}")
    # Create a dummy chatbot to prevent complete failure
    class DummyChatbot:
        def predict_intent(self, text):
            return "neutral", 0.0
        def generate_response(self, predicted_class, confidence):
            return "عذراً، النظام غير متوفر حالياً"
    chatbot = DummyChatbot()

@app.post("/chat", response_model=dict)
async def chat(request: ChatRequest):
    try:
        # Handle empty input
        if not request.text.strip():
            response_text = "يرجى كتابة رسالة."
            save_interaction(
                request.user_id, 
                request.text, 
                "empty",
                0.0,
                response_text
            )
            return {"response": response_text}
        
        # Process input
        predicted_class, confidence = chatbot.predict_intent(request.text)
        response_text = chatbot.generate_response(predicted_class, confidence)
        
        # Save interaction
        save_interaction(
            request.user_id, 
            request.text, 
            predicted_class,
            confidence,
            response_text
        )
        
        return {
            "response": response_text,
            "predicted_class": predicted_class,
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/history/{user_id}", response_model=list)
async def get_history(user_id: str, limit: int = 10):
    try:
        history = get_user_history(user_id, limit)
        return history
    except Exception as e:
        logger.error(f"History retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": not isinstance(chatbot, DummyChatbot)}