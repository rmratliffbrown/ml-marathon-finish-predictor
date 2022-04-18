from flask import Flask, request, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main():
    
    if request.method == "POST":
        
        reg = joblib.load("reg.pkl")
        
        miles = request.form.get("miles")
        speed = request.form.get("speed")
        
        X = pd.DataFrame([[miles, speed]], columns = ["Miles", "Speed"])
        
        prediction = reg.predict(X)[0]
        
    else:
        prediction = ""
        
    return render_template("front.html", output = prediction)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)