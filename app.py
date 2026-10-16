import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the trained Linear Regression model
try:
    with open('linear_2.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Attractive and Responsive HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Insurance Cost Estimator</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
    </style>
</head>
<body class="bg-gradient-to-tr from-slate-900 via-indigo-950 to-blue-900 min-h-screen flex flex-col justify-between text-gray-100">

    <!-- Navbar / Header -->
    <header class="py-6 px-6 text-center border-b border-indigo-900/50 backdrop-blur-md bg-slate-900/40">
        <h1 class="text-3xl font-extrabold tracking-tight text-white flex items-center justify-center gap-2">
            🏥 Smart Insurance Estimator
        </h1>
        <p class="text-indigo-300 text-sm mt-1">Linear Regression & Flask Powered Prediction System</p>
    </header>

    <!-- Main Container -->
    <main class="container mx-auto px-4 py-10 max-w-2xl flex-grow">
        <div class="bg-slate-900/80 backdrop-blur-xl shadow-2xl rounded-3xl border border-indigo-500/20 p-8 md:p-10">
            
            <form method="POST" action="/" class="space-y-6">
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    
                    <!-- Age -->
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-indigo-300 mb-2">Age</label>
                        <input type="number" name="age" min="18" max="100" required placeholder="e.g. 30" 
                            class="w-full px-4 py-3 bg-slate-800/60 border border-slate-700 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 transition">
                    </div>

                    <!-- Sex -->
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-indigo-300 mb-2">Sex</label>
                        <select name="sex" required 
                            class="w-full px-4 py-3 bg-slate-800/60 border border-slate-700 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 transition">
                            <option value="" disabled selected>Select Sex</option>
                            <option value="0">Female</option>
                            <option value="1">Male</option>
                        </select>
                    </div>

                    <!-- BMI -->
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-indigo-300 mb-2">BMI</label>
                        <input type="number" step="0.1" name="bmi" min="10" max="60" required placeholder="e.g. 25.5" 
                            class="w-full px-4 py-3 bg-slate-800/60 border border-slate-700 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 transition">
                    </div>

                    <!-- Children -->
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-indigo-300 mb-2">Children</label>
                        <input type="number" name="children" min="0" max="10" required placeholder="e.g. 1" 
                            class="w-full px-4 py-3 bg-slate-800/60 border border-slate-700 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 transition">
                    </div>

                    <!-- Smoker -->
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-indigo-300 mb-2">Smoker</label>
                        <select name="smoker" required 
                            class="w-full px-4 py-3 bg-slate-800/60 border border-slate-700 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 transition">
                            <option value="" disabled selected>Select Option</option>
                            <option value="0">No</option>
                            <option value="1">Yes</option>
                        </select>
                    </div>

                    <!-- Region -->
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-indigo-300 mb-2">Region</label>
                        <select name="region" required 
                            class="w-full px-4 py-3 bg-slate-800/60 border border-slate-700 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 transition">
                            <option value="" disabled selected>Select Region</option>
                            <option value="0">Northeast</option>
                            <option value="1">Northwest</option>
                            <option value="2">Southeast</option>
                            <option value="3">Southwest</option>
                        </select>
                    </div>

                </div>

                <!-- Submit Button -->
                <div class="pt-4">
                    <button type="submit" 
                        class="w-full bg-gradient-to-r from-indigo-600 to-blue-600 hover:from-indigo-500 hover:to-blue-500 text-white font-bold py-3.5 px-6 rounded-xl transition duration-300 shadow-lg shadow-indigo-500/25 active:scale-[0.99]">
                        Calculate Insurance Cost
                    </button>
                </div>
            </form>

            <!-- Prediction Result Box -->
            {% if prediction is not none %}
            <div class="mt-8 p-6 rounded-2xl bg-indigo-900/40 border border-indigo-500/30 text-center animate-pulse">
                <span class="text-xs uppercase tracking-widest text-indigo-300 font-semibold">Estimated Cost Result</span>
                <div class="text-4xl mt-2 font-extrabold text-white">
                    ${{ "%.2f"|format(prediction) }}
                </div>
            </div>
            {% endif %}
        </div>
    </main>

    <!-- Footer -->
    <footer class="text-center py-6 text-slate-500 text-xs border-t border-indigo-950">
        &copy; 2026 Prediction Model Deployment. All rights reserved.
    </footer>

</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Form values extraction
            age = float(request.form['age'])
            sex = int(request.form['sex'])
            bmi = float(request.form['bmi'])
            children = int(request.form['children'])
            smoker = int(request.form['smoker'])
            region = int(request.form['region'])

            # Array formatting matching the model input shape (6 features)
            features = np.array([[age, sex, bmi, children, smoker, region]])

            if model:
                pred = model.predict(features)
                prediction = float(pred[0])
        except Exception as e:
            print(f"Prediction error: {e}")

    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
