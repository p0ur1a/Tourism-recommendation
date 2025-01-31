# Tourism Recommendation System

## Overview
The **Tourism Recommendation System** is an AI-powered travel assistant designed to provide personalized destination recommendations based on user preferences. This project leverages machine learning and natural language processing techniques to analyze tourism data and generate relevant suggestions.

## Features
- Personalized travel recommendations based on user input.
- Machine learning models trained on tourism data.
- Integration with Dialogflow for chatbot-based interaction.
- Flask-based backend for serving recommendations.
- Dataset containing information on 100 cities.

## Project Structure
```
Tourism-recommendation/
│── data/                 # Dataset for training and evaluation
│── models/               # Trained machine learning models
│── scripts/              # Data processing and model training scripts
│── app/                  # Flask application for serving recommendations
│── chatbot/              # Dialogflow integration
│── requirements.txt      # Dependencies
│── README.md             # Project documentation
```

## Installation
### Prerequisites
- Python 3.8+
- Flask
- Dialogflow API
- Required Python packages (listed in `requirements.txt`)

## Usage
1. Run the Flask application:
   ```sh
   python app/main.py
   ```
2. Access the chatbot through Dialogflow for travel recommendations.

## Future Improvements
- Expand dataset to include more cities and attractions.
- Improve recommendation algorithm using advanced NLP techniques.
- Develop a user-friendly frontend for enhanced interaction.

## License
This project is open-source and available under the MIT License.



