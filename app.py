import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the trained Linear Regression model
try:
    with open('linear.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# HTML template with embedded Tailwind CSS for a professional layout
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Insurance Cost Prediction - Linear Regression</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gradient-to-br from-blue-50 via-white to-indigo-50 min-h-screen flex flex-col justify-between font-sans">
    <div class="container mx-auto px-4 py-8 max-w-3xl">
        <!-- Header -->
        <header class="text-center mb-8">
            <h1 class="text-3xl font-extrabold text-blue-900 tracking-tight">Insurance Cost Prediction</h1>
            <p class="text-gray-600 mt-2">Powered by Linear Regression & Flask[cite: 2]</p>
        </header>

        <!-- Main Card -->
        <div class="bg-white shadow-xl rounded-2xl overflow-hidden border border-gray-100 p-8">
            <form method="POST" action="/" class="space-y-6">
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    
                    <!-- Age -->
                    <div>
                        <label class="block text-sm font-semibold text-gray-700 mb-2">Age[cite: 2]</label>
                        <input type="number" name="age" min="18" max="100" required placeholder="e.g. 30" class="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none transition">
                    </div>

                    <!-- Sex -->
                    <div>
                        <label class="block text-sm font-semibold text-gray-700 mb-2">Sex[cite: 2]</label>
                        <select name="sex" required class="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none transition bg-white">
                            <option value="" disabled selected>Select Sex</option>
                            <option value="0">Female</option>
                            <option value="1">Male</option>
                        </select>
                    </div>

                    <!-- BMI -->
                    <div>
                        <label class="block text-sm font-semibold text-gray-700 mb-2">BMI[cite: 2]</label>
                        <input type="number" step="0.1" name="bmi" min="10" max="60" required placeholder="e.g. 25.5" class="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none transition">
                    </div>

                    <!-- Children -->
                    <div>
                        <label class="block text-sm font-semibold text-gray-700 mb-2">Children[cite: 2]</label>
                        <input type="number" name="children" min="0" max="10" required placeholder="e.g. 1" class="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none transition">
                    </div>

                    <!-- Smoker -->
                    <div>
                        <label class="block text-sm font-semibold text-gray-700 mb-2">Smoker[cite: 2]</label>
                        <select name="smoker" required class="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none transition bg-white">
                            <option value="" disabled selected>Select Option</option>
                            <option value="0">No</option>
                            <option value="1">Yes</option>
                        </select>
                    </div>

                    <!-- Region -->
                    <div>
                        <label class="block text-sm font-semibold text-gray-700 mb-2">Region[cite: 2]</label>
                        <select name="region" required class="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none transition bg-white">
                            <option value="" disabled selected>Select Region</option>
                            <option value="0">Northeast</option>
                            <option value="1">Northwest</option>
                            <option value="2">Southeast</option>
                            <option value="3">Southwest</option>
                        </select>
                    </div>

                </div>

                <!-- Submit Button -->
                <div>
                    <button type="submit" class="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg transition duration-200 shadow-lg hover:shadow-blue-500/30">
                        Predict Charges
                    </button>
                </div>
            </form>

            {% if prediction is not none %}
            <div class="mt-8 p-6 rounded-xl text-center bg-blue-50 border border-blue-200 text-blue-900">
                <h3 class="text-lg font-bold">Predicted Insurance Cost</h3>
                <p class="text-3xl mt-2 font-extrabold">${{ "%.2f"|format(prediction) }}</p>
            </div>
            {% endif %}
        </div>
    </div>

    <!-- Footer -->
    <footer class="text-center py-6 text-gray-500 text-sm">
        &copy; 2026 Model Deployment on Render. All rights reserved.
    </footer>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            age = float(request.form['age'])
            sex = int(request.form['sex'])
            bmi = float(request.form['bmi'])
            children = int(request.form['children'])
            smoker = int(request.form['smoker'])
            region = int(request.form['region'])

            features = np.array([[age, sex, bmi, children, smoker, region]])

            if model:
                pred = model.predict(features)
                prediction = float(pred[0])
        except Exception as e:
            print(f"Prediction error: {e}")

    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
