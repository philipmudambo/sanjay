from flask import Flask, render_template, request
import torch
import numpy as np

# Initialize your Flask application
app = Flask(__name__)

# Load your pre-trained model
# Replace 'path_to_your_model.pth' with the actual path to your trained model file
model = torch.load('path_to_your_model.pth')
model.eval()  # Set the model to evaluation mode

# Define a route to render the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route to handle form submission and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    age = float(request.form['age'])
    gender = float(request.form['gender'])  # Assuming gender is encoded as a number (e.g., 0 for male, 1 for female)
    # Add more variables as needed

    # Preprocess the user input
    input_data = torch.tensor([[age, gender]], dtype=torch.float32)  # Create a tensor with the user input

    # Make predictions using your model
    with torch.no_grad():
        output = model(input_data)
        predicted_class = torch.argmax(output).item()  # Get the predicted class (assuming it's a classification task)

    # Display the predictions to the user
    prediction = "Prediction: {}".format(predicted_class)  # Adjust this based on your model's output format

    return render_template('result.html', prediction=prediction)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
