from flask import Flask, request, render_template
import numpy as np
import pickle
import os

# Get the absolute path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the model
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'standscaler.pkl')
MINMAX_SCALER_PATH = os.path.join(BASE_DIR, 'minmaxscaler.pkl')

# Check if files exist
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"\u274C model.pkl not found in {BASE_DIR}")
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"\u274C standscaler.pkl not found in {BASE_DIR}")
if not os.path.exists(MINMAX_SCALER_PATH):
    print("\u26A0\uFE0F Warning: minmaxscaler.pkl not found! Check the file location.")
    mx = None  # Avoid using an undefined variable
else:
    with open(MINMAX_SCALER_PATH, "rb") as file:
        mx = pickle.load(file)
    print("\u2705 MinMaxScaler loaded successfully!")

# Load the model and standard scaler
model = pickle.load(open(MODEL_PATH, 'rb'))
sc = pickle.load(open(SCALER_PATH, 'rb'))
print("\u2705 Model and Standard Scaler loaded successfully!")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['pH']
    rainfall = request.form['Rainfall']

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list, dtype=float).reshape(1, -1)
    
    if mx is not None:
        mx_features = mx.transform(single_pred)
        sc_mx_features = sc.transform(mx_features)
        prediction = model.predict(sc_mx_features)
    else:
        sc_mx_features = sc.transform(single_pred)
        prediction = model.predict(sc_mx_features)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = f"{crop} is the best crop to be cultivated right there"
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    
    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)