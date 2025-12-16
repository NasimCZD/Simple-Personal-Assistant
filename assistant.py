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

# Ensure this is loaded globally so get_response can see it
with open('intents.json', 'r') as f:
    intents = json.load(f)

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

# Change this: def get_response(user_input, intents_json):
# To this:
def get_response(user_message): 
    """Predicts intent and executes dynamic actions."""
    
    # 1. Vectorize the input message
    input_vector = vectorizer.transform([user_message])
    
    # 2. Predict the tag using your Logistic Regression model
    probabilities = model.predict_proba(input_vector)[0]
    max_index = np.argmax(probabilities)
    tag = classes[max_index]
    confidence = probabilities[max_index]

    # 3. Handle low confidence
    if confidence < 0.3:
        return "I'm sorry, I don't quite understand that. Could you try phrasing it differently?"

    # 4. Match the predicted tag with our intents JSON data
    # (Using the global variable 'intents' loaded at the top of your script)
    list_of_intents = intents['intents']
    
    for i in list_of_intents:
        if i['tag'] == tag:
            
            # --- ACTION: Check Time ---
            if tag == 'check_time':
                import datetime
                current_time = datetime.datetime.now().strftime("%I:%M %p, %B %d, %Y")
                return random.choice(i['responses']) + f" **{current_time}**."
            
            # --- ACTION: Set Reminder ---
            elif tag == 'set_reminder':
                import re
                # Look for a time/entity
                match = re.search(r'(\d{1,2}:\d{2}(?:am|pm)?|\d{1,2}(?:am|pm)|tomorrow|tonight)', user_message.lower())
                
                if match:
                    time_entity = match.group(0)
                    # Clean the task by removing the trigger phrases
                    task_entity = re.sub(r'(remind me to|set a reminder for|i need a reminder about|schedule a reminder for)', '', user_message.lower()).strip()
                    # task_entity = task_entity.replace(time_entity, '').strip()
                    # Changed the above line to below.This removes the time and the word 'at' if it's left at the end.
                    task_entity = re.sub(r'\s+at\s*$', '', task_entity.replace(time_entity, '').strip()) 
                    if task_entity:
                        return random.choice(i['responses']) + f" I've set a reminder for **{task_entity.title()}** at **{time_entity}**."
                    else:
                        return "I've got the time, but what exactly should I remind you about?"
                else:
                    return "I can set a reminder, but I need a time (like 5pm) and a task. Could you specify both?"

            # --- ACTION: Check Weather ---
            elif tag == 'check_weather':
                # Assuming you have the get_weather function defined above
                return get_weather("Amsterdam")

            # --- DEFAULT RESPONSE ---
            else:
                return random.choice(i['responses'])

    return "I'm not sure how to help with that yet."
            
# --- PHASE 4: The Chat Loop ---

print("\n--- Simple Personal Assistant Chatbot is Ready! (Type 'bye' to exit) ---")

# for the purpose of API
if __name__ == "__main__":
    print("--- Simple Personal Assistant Chatbot is Ready! (Type 'quit' to exit) ---")
    while True:
        message = input("You: ")
        if message.lower() == 'quit':
            break
        print(f"Assistant: {get_response(message)}")
        
    # 1. Predict the Intent
    intents = predict_class(user_input)
    # 2. Get the Response
    response = get_response(intents, data, user_input)
    

    print(f"Assistant: {response}")

print("Chatbot terminated.")
