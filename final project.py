# app.py
import os
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

app = Flask(__name__)

df_encoded = pd.read_csv('City_encoded.csv')
weights = {
    'Ratings': 1,
}

categories = [
    ('TouristSeason', ['Fall', 'Spring', 'Summer', 'Winter']),
    ('Accommodation', ['Ashrams', 'Camps', 'Guesthouses', 'Heritage Hotels', 'Homestays', 
                       'Hotels', 'Houseboats', 'Monasteries', 'Resorts']),
    ('DestinationType', ['Adventure', 'Backwaters', 'Beach', 'Coastal', 'Cultural', 'Desert', 
                         'Heritage', 'Hill Station', 'Island', 'Nightlife', 'Pilgrimage', 
                         'Romantic', 'Scenic', 'Skiing', 'Trekking', 'Urban', 'Wildlife']),
    ('Region', ['Center', 'East', 'North', 'South', 'West']),
    ('Budget', ['High', 'Low', 'Medium']),
]

for category, subcategories in categories:
    
    for subcategory in subcategories:
        weights[f"{category}_{subcategory}"] = 1 / len(subcategories)

# function to apply the weights to the values of dataframe
def apply_weights_to_columns(df, weights):
    for col in df.columns:
        if col in weights:
            df[col] *= weights[col]
            
# Function to convert descriptive inputs to model input format
def create_model_input(rating,seasons, accommodations, destination_type, regions, budget):
    # Initialize dictionary with zeros
    data = { 
        'Ratings': rating,
        'TouristSeason_Fall': 0,
        'TouristSeason_Spring': 0,
        'TouristSeason_Summer': 0,
        'TouristSeason_Winter': 0,        
        'Accommodation_Ashrams': 0,
        'Accommodation_Camps': 0,
        'Accommodation_Guesthouses': 0,
        'Accommodation_Heritage Hotels': 0,
        'Accommodation_Homestays': 0,
        'Accommodation_Hotels': 0,
        'Accommodation_Houseboats': 0,
        'Accommodation_Monasteries': 0,
        'Accommodation_Resorts': 0,        
        'DestinationType_Adventure': 0,
        'DestinationType_Backwaters': 0,
        'DestinationType_Beach': 0,
        'DestinationType_Coastal': 0,
        'DestinationType_Cultural': 0,
        'DestinationType_Desert': 0,
        'DestinationType_Heritage': 0,
        'DestinationType_Hill Station': 0,
        'DestinationType_Island': 0,
        'DestinationType_Nightlife': 0,
        'DestinationType_Pilgrimage': 0,
        'DestinationType_Romantic': 0,
        'DestinationType_Scenic': 0,
        'DestinationType_Skiing': 0,
        'DestinationType_Trekking': 0,
        'DestinationType_Urban': 0,
        'DestinationType_Wildlife': 0,        
        'Region_Center': 0,
        'Region_East': 0,
        'Region_North': 0,
        'Region_South': 0,
        'Region_West': 0,        
        'Budget_High': 0,
        'Budget_Low': 0,
        'Budget_Medium': 0
    }

    # Set the selected features to 1        
    for season in seasons.split(', '):
        data[f'TouristSeason_{season}'] = 1
        
    for accommodation in accommodations.split(', '):
        data[f'Accommodation_{accommodation}'] = 1
        
    for dest in destination_type.split(', '):
        data[f'DestinationType_{dest}'] = 1
        
    for region in regions.split(', '):
        data[f'Region_{region}'] = 1
        
    data[f'Budget_{budget}'] = 1

    return pd.DataFrame([data])




@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    region = data.get('region', 'East')
    seasons = data.get('seasons', 'Spring, Fall')
    accommodations = data.get('accommodations', 'Resorts, Guesthouses')
    destination_type = data.get('destination_type', 'Heritage')
    budget = data.get('budget', 'Low')
    rating = data.get('rating', 4.2)
    user_text = data.get('user_text', 'I like to go to the North of India. I would go there in Summer. I like Wildlife. I prefer heritage hotels in for historic charm. My budget is Medium')
    
    # input_data = create_model_input(region, seasons, accommodations, destination_type, budget, rating)
    # model_input_nparr = input_data.values.reshape(1, -1)
    # similarity_scores = cosine_similarity(X, model_input_nparr)
    # sorted_indices = similarity_scores.argsort(axis=0)[::-1].flatten()
    # top_n = 10
    # recommendations = [df.iloc[sorted_indices[i]]['City'] for i in range(top_n)]
    df_tabular = create_model_input(rating,seasons, accommodations, destination_type, region, budget)
    
        # Load and preprocess training data
    df = pd.read_csv("model_training_dataset.csv")

    # Preprocess data
    df['Region'] = df['Region'].str.lower()
    df['Season'] = df['Season'].str.lower()
    df['Accommodation'] = df['Accommodation'].str.lower()
    df['Destination'] = df['Destination'].str.lower()
    df['Budgets'] = df['Budgets'].str.lower()

    # Train models and vectorizers
    vectorizer1 = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_vec1 = vectorizer1.fit_transform(df['Region'])
    model1 = MultinomialNB()
    model1.fit(X_train_vec1, df['Region-Label'])

    vectorizer2 = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_vec2 = vectorizer2.fit_transform(df['Season'])
    model2 = SVC(kernel='linear', random_state=42)
    model2.fit(X_train_vec2, df['Season-Label'])

    vectorizer3 = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_vec3 = vectorizer3.fit_transform(df['Accommodation'])
    model3 = RandomForestClassifier(n_estimators=100, random_state=42)
    model3.fit(X_train_vec3, df['Accommodation-Label'])

    vectorizer4 = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_vec4 = vectorizer4.fit_transform(df['Destination'])
    model4 = RandomForestClassifier(n_estimators=100, random_state=42)
    model4.fit(X_train_vec4, df['Destination-Label'])

    vectorizer5 = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_vec5 = vectorizer5.fit_transform(df['Budgets'])
    model5 = GradientBoostingClassifier(random_state=42)
    model5.fit(X_train_vec5, df['Budget_Label'])

    # Convert new texts to numerical features
    #new_texts = ["I like to go to the Center of India. I would go there in Fall. I like Wildlife. I prefer heritage hotels in for historic charm. My budget is High"]

    # Dictionary to store final results
    final_results = {
        "Region": None,
        "Season": None,
        "Accommodation": None,
        "Destination Type": None,
        "Budget": None
    }

    # Mapping model names to their respective categories
    category_map = {
        "model1": "Region",
        "model2": "Season",
        "model3": "Accommodation",
        "model4": "Destination Type",
        "model5": "Budget"
    }

    # Make predictions using trained models
    text_vec1 = vectorizer1.transform([user_text])
    text_vec2 = vectorizer2.transform([user_text])
    text_vec3 = vectorizer3.transform([user_text])
    text_vec4 = vectorizer4.transform([user_text])
    text_vec5 = vectorizer5.transform([user_text])

    predictions1 = model1.predict(text_vec1)
    predictions2 = model2.predict(text_vec2)
    predictions3 = model3.predict(text_vec3)
    predictions4 = model4.predict(text_vec4)
    predictions5 = model5.predict(text_vec5)

    final_results["Region"] = predictions1[0]
    final_results["Season"] = predictions2[0]
    final_results["Accommodation"] = predictions3[0]
    final_results["Destination Type"] = predictions4[0]
    final_results["Budget"] = predictions5[0]

    # Extract the features
    rating = 5  # Assuming a rating value, adjust based on your input
    seasons = final_results["Season"]
    accommodations = final_results["Accommodation"]
    destination_type = final_results["Destination Type"]
    regions = final_results["Region"]
    budget = final_results["Budget"]

    # Create DataFrame using the extracted features
    df_text_mining = create_model_input(rating, seasons, accommodations, destination_type, regions, budget) 
    
    apply_weights_to_columns(df_encoded, weights)
    apply_weights_to_columns(df_tabular, weights)
    apply_weights_to_columns(df_text_mining, weights)
    
    combined_df = df_tabular.combine(df_text_mining, lambda s1, s2: s1.combine(s2, max))
    
    similarity_scores = cosine_similarity(combined_df, df_encoded.iloc[:, 1:])  # Assuming df has columns similar to input_df
    # Find the index of the most similar city
    similar_city_index = similarity_scores.argmax()

    # Retrieve the city name from the original dataset
    recommended_city = df_encoded.loc[similar_city_index, 'City']



    # Format for Dialogflow fulfillment
    return jsonify({
        "fulfillmentText": "Based on your preferences, here are some recommended cities:",
        "fulfillmentMessages": [
            {
                "text": {
                    "text": [recommended_city]
                }
            }
        ]
    })


import requests
import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))  # Default to 8080 if PORT not set
    app.run(host='localhost', port=port)
