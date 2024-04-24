from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib
import os
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load your trained model and preprocessor
# Make sure to update the paths to where your model and preprocessor are stored
model = joblib.load('path_to_model/linear_regression_model.joblib')
preprocessor = joblib.load('path_to_preprocessor/preprocessor.joblib')

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == "POST":
        try:
            # Collecting input data from the form
            km_per_week = float(request.form['km4week'])
            speed_per_week = float(request.form['speed4week'])
            x_train = int(request.form['xtrain'])
    
            # Create a DataFrame to handle feature names and ensure correct transformations
            input_data = pd.DataFrame({
                'km4week': [km_per_week],
                'speed4week': [speed_per_week],
                'xtrain': [x_train],
            })
    
            # Apply the same preprocessing as was done for training the model
            input_data_transformed = preprocessor.transform(input_data)
            prediction = model.predict(input_data_transformed)
    
            output = round(prediction[0], 2)  # Round the prediction to 2 decimal places
            return render_template('front.html', prediction_text=f'Estimated Marathon Finish Time: {output} hours')
        
        except Exception as e:
            logging.error(f'Error occurred: {e}', exc_info=True)
            return jsonify({'error': 'There was an error processing your request.'})
        
    else:
        # For GET requests, just show the empty form
        return render_template("front.html", prediction_text='')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
