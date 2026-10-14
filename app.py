import pickle
from flask import Flask, render_template_string, request
import numpy as np

app = Flask(__name__)

# Load the trained GaussianNB model
with open('naive_bayes.pkl', 'rb') as f:
  model = pickle.load(f)

# Modern, responsive HTML template with embedded styling
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employee Prediction Portal</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-900 min-h-screen text-slate-100 flex items-center justify-center p-4">
    <div class="max-w-2xl w-full bg-slate-800/80 backdrop-blur-md rounded-2xl shadow-2xl border border-slate-700 p-8 my-8">
        <div class="text-center mb-8">
            <h1 class="text-3xl font-extrabold bg-gradient-to-r from-teal-400 to-indigo-400 bg-clip-text text-transparent">Employee Classification Portal</h1>
            <p class="text-slate-400 text-sm mt-2">Enter employee parameters to run the machine learning model[cite: 1].</p>
        </div>

        <form method="POST" class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <!-- Education -->
            <div>
                <label class="block text-sm font-medium text-slate-300 mb-2">Education</label>
                <select name="Education" required class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:ring-2 focus:ring-teal-500">
                    <option value="0">Bachelors</option>
                    <option value="1">Masters</option>
                    <option value="2">PHD</option>
                </select>
                <span class="text-xs text-slate-500 mt-1 block">Encoded: 0: Bachelors, 1: Masters, 2: PHD</span>
            </div>

            <!-- Joining Year -->
            <div>
                <label class="block text-sm font-medium text-slate-300 mb-2">Joining Year</label>
                <input type="number" name="JoiningYear" required placeholder="e.g., 2017" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:ring-2 focus:ring-teal-500">
            </div>

            <!-- City -->
            <div>
                <label class="block text-sm font-medium text-slate-300 mb-2">City</label>
                <select name="City" required class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:ring-2 focus:ring-teal-500">
                    <option value="0">Bangalore</option>
                    <option value="1">New Delhi</option>
                    <option value="2">Pune</option>
                </select>
                <span class="text-xs text-slate-500 mt-1 block">Encoded: 0: Bangalore, 1: New Delhi, 2: Pune</span>
            </div>

            <!-- Payment Tier -->
            <div>
                <label class="block text-sm font-medium text-slate-300 mb-2">Payment Tier</label>
                <select name="PaymentTier" required class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:ring-2 focus:ring-teal-500">
                    <option value="1">Tier 1</option>
                    <option value="2">Tier 2</option>
                    <option value="3">Tier 3</option>
                </select>
            </div>

            <!-- Age -->
            <div>
                <label class="block text-sm font-medium text-slate-300 mb-2">Age</label>
                <input type="number" name="Age" required placeholder="e.g., 28" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:ring-2 focus:ring-teal-500">
            </div>

            <!-- Gender -->
            <div>
                <label class="block text-sm font-medium text-slate-300 mb-2">Gender</label>
                <select name="Gender" required class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:ring-2 focus:ring-teal-500">
                    <option value="0">Female</option>
                    <option value="1">Male</option>
                </select>
            </div>

            <!-- Ever Benched -->
            <div>
                <label class="block text-sm font-medium text-slate-300 mb-2">Ever Benched</label>
                <select name="EverBenched" required class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:ring-2 focus:ring-teal-500">
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                </select>
            </div>

            <!-- Experience In Current Domain -->
            <div>
                <label class="block text-sm font-medium text-slate-300 mb-2">Experience in Current Domain</label>
                <input type="number" name="ExperienceInCurrentDomain" required placeholder="e.g., 3" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:ring-2 focus:ring-teal-500">
            </div>

            <div class="md:col-span-2 mt-4">
                <button type="submit" class="w-full bg-gradient-to-r from-teal-500 to-indigo-600 hover:from-teal-600 hover:to-indigo-700 text-white font-semibold py-3 px-6 rounded-lg transition duration-200 shadow-lg">
                    Predict Result
                </button>
            </div>
        </form>

        {% if prediction is not none %}
        <div class="mt-8 p-6 bg-slate-900/90 border border-teal-500/30 rounded-xl text-center">
            <h2 class="text-lg font-medium text-slate-400">Prediction Result</h2>
            <p class="text-3xl font-bold text-teal-400 mt-2">
                {% if prediction == 1 %}
                    Will Leave / Positive Class (1)
                {% else %}
                    Will Stay / Negative Class (0)
                {% endif %}
            </p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
  prediction = None
  if request.method == "POST":
    try:
      # Extract feature values in the exact order trained by the model
      features = [
          float(request.form["Education"]),
          float(request.form["JoiningYear"]),
          float(request.form["City"]),
          float(request.form["PaymentTier"]),
          float(request.form["Age"]),
          float(request.form["Gender"]),
          float(request.form["EverBenched"]),
          float(request.form["ExperienceInCurrentDomain"]),
      ]

      final_features = np.array([features])
      prediction = int(model.predict(final_features)[0])
    except Exception as e:
      prediction = f"Error: {str(e)}"

  return render_template_string(HTML_TEMPLATE, prediction=prediction)


if __name__ == "__main__":
  app.run(debug=True)
