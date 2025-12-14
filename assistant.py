import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import json
import random
import datetime
import re 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os


# --- PHASE 1: Data Preparation and Preprocessing ---

lemmatizer = WordNetLemmatizer()
ignore_words = ['?', '!', '.', ',']

# 1. Load the Intents Data
try:
    with open('intents.json') as file:
        data = json.load(file)
except FileNotFoundError:
    print("Error: intents.json not found. Please ensure it is in the same directory.")
    exit()

patterns = []
tags = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

# 2. Encode the Labels (Tags)
le = LabelEncoder()
tags_encoded = le.fit_transform(tags)
classes = list(le.classes_)

# 3. Create the Bag of Words (BoW) Representation
# We use CountVectorizer to handle tokenization and BoW creation efficiently
vectorizer = CountVectorizer(tokenizer=nltk.word_tokenize, stop_words=ignore_words)
train_x = vectorizer.fit_transform(patterns)
# --- ADD THESE TWO LINES FOR DIAGNOSTICS ---
print(f"Total Patterns Loaded: {len(patterns)}")
print(f"Number of Unique Tags (Classes): {len(classes)}")
# -------------------------------------------

# --- PHASE 2: Build and Train the Scikit-learn Model ---

# Split data for basic validation (optional, but good practice)
# X_train, X_test, y_train, y_test = train_test_split(train_x, tags_encoded, test_size=0.2, random_state=42)

# Use Logistic Regression as a strong classifier for this task
model = LogisticRegression(solver='lbfgs', random_state=0, max_iter=200, C=10.0)

print("Starting model training with Logistic Regression...")
model.fit(train_x, tags_encoded)
print("Model training complete.")

# Save the necessary components (model and vectorizer)
with open('chatbot_model.pkl', 'wb') as file:
    pickle.dump(model, file)
with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)
with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(le, file)

# --- PHASE 3: Prediction and Chat Functions ---

# Helper function to get the BoW representation for a new sentence
def get_bow(sentence):
    return vectorizer.transform([sentence])

def predict_class(sentence):
    """Predict the intent class (tag) of the user's input."""
    bow = get_bow(sentence)
    
    # Get prediction probabilities from the model
    probabilities = model.predict_proba(bow)[0]
    
    # Get the index of the highest probability
    max_index = np.argmax(probabilities)
    
    # Set a confidence threshold
    ERROR_THRESHOLD = 0.70
    
    if probabilities[max_index] > ERROR_THRESHOLD:
        tag_index = max_index
        tag = le.inverse_transform([tag_index])[0]
        return [{"intent": tag, "probability": str(probabilities[max_index])}]
    else:
        # If confidence is low
        return []

def get_response(intents_list, intents_json, user_input):
    """Get a random response and execute dynamic actions."""
    if not intents_list:
        return "I'm sorry, I don't quite understand that. Can you rephrase?"
        
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    
    for i in list_of_intents:
        if i['tag'] == tag:
            # Handle special intents with dynamic actions
            if tag == 'check_time':
                current_time = datetime.datetime.now().strftime("%I:%M %p, %B %d, %Y")
                return random.choice(i['responses']) + f" **{current_time}**."
            
            elif tag == 'set_reminder':
                # Simple Entity Extraction using RegEx
                match = re.search(r'(\d{1,2}:\d{2}(?:am|pm)?|\d{1,2}(?:am|pm)|tomorrow|tonight)', user_input.lower())
                
                if match:
                    time_entity = match.group(0)
                    # Extract task by removing common reminder phrases
                    task_entity = re.sub(r'(remind me to|set a reminder for|i need a reminder about|schedule a reminder for)', '', user_input.lower()).strip()
                    task_entity = task_entity.replace(time_entity, '').strip()
                    
                    if task_entity:
                        return random.choice(i['responses']) + f" I've set a reminder for **{task_entity.title()}** at **{time_entity}**."
                    else:
                        return "I know the time, but what should I remind you about?"
                else:
                    return "I can set a reminder, but what time and for what task? Please specify both."
            
            # For all other intents, return a simple random response
            result = random.choice(i['responses'])
            return result
            
# --- PHASE 4: The Chat Loop ---

print("\n--- Simple Personal Assistant Chatbot is Ready! (Type 'bye' to exit) ---")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'bye':
        break
        
    # 1. Predict the Intent
    intents = predict_class(user_input)
    # 2. Get the Response
    response = get_response(intents, data, user_input)
    
    print(f"Assistant: {response}")

print("Chatbot terminated.")
