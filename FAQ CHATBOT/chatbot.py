import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gradio as gr

# Automatically download required text processors
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ----------------------------------------------------------------------
# 1. THE BULLETPROOF DATASET (Covers almost all possible user inputs)
# ----------------------------------------------------------------------
faq_data = {
    "question": [
        # --- TOPIC 1: PLACING ORDERS ---
        "How do I place an order? What are the steps to order?",
        "How should I order? I want to buy something.",
        "How to place order online?",
        "Steps to order an item.",
        
        # --- TOPIC 2: REFUNDS ---
        "How do I get a refund for my return?",
        "When will I get my refund or money back?",
        "Refund process and status.",
        "I want my cashback/refund.",

        # --- TOPIC 3: RETURNS ---
        "What is your return policy?",
        "How to return an item? Hoe to return.",
        "Can I exchange a product?",
        "How do I open a return request?",

        # --- TOPIC 4: TRACKING ---
        "How can I track my order status?",
        "Where is my package right now?",
        "Track my delivery shipment.",

        # --- TOPIC 5: SHIPPING & DELIVERY ---
        "How long does shipping take?",
        "What are the delivery charges?",
        "Do you ship internationally?",

        # --- TOPIC 6: ORDER CANCELLATION / CHANGES ---
        "Can I change my delivery address after placing an order?",
        "How do I cancel my order?",

        # --- TOPIC 7: PAYMENTS & ACCOUNT ---
        "What payment methods do you accept? How can I pay?",
        "How do I reset my account password if I forgot it?"
    ],
    "answer": [
        # --- ANSWERS ---
        "To place an order: 1. Add items to your cart. 2. Click 'Proceed to Checkout'. 3. Enter your shipping address. 4. Choose a payment method and click 'Place Order'.",
        "To place an order: 1. Add items to your cart. 2. Click 'Proceed to Checkout'. 3. Enter your shipping address. 4. Choose a payment method and click 'Place Order'.",
        "To place an order: 1. Add items to your cart. 2. Click 'Proceed to Checkout'. 3. Enter your shipping address. 4. Choose a payment method and click 'Place Order'.",
        "To place an order: 1. Add items to your cart. 2. Click 'Proceed to Checkout'. 3. Enter your shipping address. 4. Choose a payment method and click 'Place Order'.",
        
        "Once we receive and inspect your returned item, your refund will be processed and automatically applied to your credit card or original payment method within 5-7 business days.",
        "Once we receive and inspect your returned item, your refund will be processed and automatically applied to your credit card or original payment method within 5-7 business days.",
        "Once we receive and inspect your returned item, your refund will be processed and automatically applied to your credit card or original payment method within 5-7 business days.",
        "Once we receive and inspect your returned item, your refund will be processed and automatically applied to your credit card or original payment method within 5-7 business days.",

        "We offer a 30-day return policy on all unused items in their original packaging.",
        "We offer a 30-day return policy on all unused items in their original packaging.",
        "Yes, we offer free exchanges within 15 days of delivery if the item is unused and in original condition.",
        "To start a return, go to your 'Order History' page, click on the order number, and select the 'Return Item' button.",

        "You can track your order using the real-time tracking link sent to your registered email address as soon as your package ships.",
        "You can track your order using the real-time tracking link sent to your registered email address as soon as your package ships.",
        "You can track your order using the real-time tracking link sent to your registered email address as soon as your package ships.",

        "Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days.",
        "Standard shipping is completely free for orders over $50! For smaller orders, shipping costs a flat fee of $4.99.",
        "Yes! We ship to over 50 countries worldwide. International delivery generally takes 7-14 business days.",

        "Address changes or corrections can only be made within 1 hour of placing the order by contacting our support desk instantly.",
        "You can cancel your order directly from your profile dashboard within 60 minutes of purchase before it enters processing.",

        "We accept Visa, MasterCard, PayPal, American Express, and Apple Pay.",
        "Click on the 'Forgot Password' link on our sign-in window and follow the secure link sent to your registered email address to change it."
    ]
}

df = pd.DataFrame(faq_data)

# ----------------------------------------------------------------------
# 2. NLP PREPROCESSING
# ----------------------------------------------------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

df['processed_question'] = df['question'].apply(preprocess_text)

# ----------------------------------------------------------------------
# 3. COSINE SIMILARITY ENGINE
# ----------------------------------------------------------------------
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['processed_question'])

def chatbot_engine(user_query, chat_history):
    processed_query = preprocess_text(user_query)
    query_vector = vectorizer.transform([processed_query])
    
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    best_match_idx = similarity_scores.argsort()[-1]
    highest_score = similarity_scores[best_match_idx]
    
    # Mathematical Threshold Optimization
    if highest_score > 0.25:
        return df['answer'].iloc[best_match_idx]
    else:
        return "🤖 I want to make sure I give you the perfect answer, but I can't find a exact match for that. Could you please rephrase your question, or email us at support@company.com?"

# ----------------------------------------------------------------------
# 4. LIVE INTERFACE DISPLAY
# ----------------------------------------------------------------------
demo = gr.ChatInterface(
    fn=chatbot_engine, 
    title="🤖 Enterprise FAQ Chatbot v2.0",
    description="Welcome to customer support! Ask me about ordering, refunds, returns, or shipping metrics."
)

demo.launch(share=True)