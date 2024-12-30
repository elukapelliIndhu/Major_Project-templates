from flask import Flask, jsonify, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)  # Corrected this line

# Load the SARIMA model
with open('fbprophet.pkl', 'rb') as f:
    model = pickle.load(f)

# Correct the dataset path (use raw string for Windows paths or escape backslashes)
dataset_path = ('rice production across different countries from 1961 to 2021.csv')

# Load the dataset and inspect its columns
try:
    # Load dataset
    data = pd.read_csv(dataset_path)

    # Inspect the first few rows of the dataset to check the data
    print(data.head())  # Debugging line - inspect the structure of the dataset
    
    # Ensure 'Year' and 'Value' columns exist and are properly formatted
    if 'Year' not in data.columns or 'Value' not in data.columns:
        raise ValueError("The dataset must contain 'Year' and 'Value' columns.")
    
    # Filter data for the range 1961-2021
    data = data[data['Year'].between(1961, 2021)]
    
    # Ensure 'Value' column is numeric
    data['Value'] = pd.to_numeric(data['Value'], errors='coerce')
    
    # Aggregate global production by year
    global_production = data.groupby('Year')['Value'].sum()

    # Print the aggregated production to ensure it's correct
    print(global_production)  # Debugging line - check aggregation result

except Exception as e:
    print(f"Error loading or processing the dataset: {str(e)}")
    global_production = pd.Series()

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html', prediction=None)

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get year input from the form
        year = int(request.form['year'])

        # Validate year and fetch data
        if 1961 <= year <= 2021:
            # Check if production data exists for the given year
            prediction = global_production.get(year, None)
            if prediction is not None:
                return render_template(
                    'prediction.html', 
                    prediction=f"{round(prediction, 2)} tonnes (Historical Data)", 
                    year=year
                )
            else:
                return render_template(
                    'prediction.html', 
                    prediction=None, 
                    year=year, 
                    error=f"No data available for the year {year}."
                )
        elif year > 2021:
            # Calculate steps to forecast for years beyond 2021
            steps_to_forecast = (year - 2021) * 12

            # Use SARIMA model to predict future production
            forecast = model.get_forecast(steps=steps_to_forecast)
            prediction = forecast.predicted_mean.iloc[-1]  # Get the predicted value

            return render_template(
                'prediction.html', 
                prediction=f"{round(prediction, 2)} tonnes (Predicted)", 
                year=year
            )
        else:
            return render_template(
                'prediction.html',
                prediction=None,
                error=f"Year {year} is outside the supported range (1961–2021 for historical data or after 2021 for prediction)."
            )

    except ValueError:
        # Handle invalid input (e.g., non-numeric year)
        return render_template(
            'prediction.html',
            prediction=None,
            error="Invalid input! Please enter a valid numeric year."
        )
    except Exception as e:
        # Catch other errors
        return render_template(
            'prediction.html',
            prediction=None,
            error=f"An error occurred: {str(e)}"
        )

if __name__ == '__main__':
    app.run(debug=True)
