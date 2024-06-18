# Disease Prediction Web Application

This is a Flask web application that predicts diseases based on symptoms entered by the user.

## Overview

This web application allows users to input symptoms (comma-separated) and get predictions about possible diseases. It uses machine learning models trained on medical data to make predictions.

## Features

- Input symptoms and get disease predictions.
- Responsive web interface.
- Uses Random Forest, Naive Bayes, and Support Vector Machine models for prediction.

## Installation

To run this application locally, follow these steps:

1. Clone the repository:
git clone https://github.com/shilpi-13/Disease_prediction.git

2. Navigate into the project directory:
cd Disease_prediction

3. Set up a virtual environment (optional but recommended):
python -m venv venv

source venv/bin/activate # On Windows use venv\Scripts\activate
4. Install the dependencies:
pip install -r requirements.txt

## Usage

1. Start the Flask application:
python app.py

2. Open a web browser and go to `http://127.0.0.1:5000/`.

3. Enter symptoms (comma-separated) in the input form and click "Predict" to see the results.

## Technologies Used

- Python
- Flask
- HTML/CSS
- Machine Learning (scikit-learn)
- Git & GitHub

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.





