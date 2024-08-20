from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

# Define a route for the default URL, which loads the form
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for the action of the form, for example '/recommend/'
@app.route('/recommend/', methods=['POST'])
def recommend():
    # Get form inputs
    mood = request.form['mood']
    cuisine = request.form['cuisine']
    
    # Create a DataFrame for the model input
    input_data = pd.DataFrame([[mood, cuisine]], columns=['Mood', 'Cuisine'])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Get the recommended dish name
    recommended_dish = prediction[0]
    
    # Format the output
    prompt = f"I want a dish for when I am feeling {mood}"
    output = f"Prompt: {prompt}\nRecommended Dish: {recommended_dish}"
    
    return render_template('index.html', recommendation=output)

if __name__ == '__main__':
    app.run(debug=True)