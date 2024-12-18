from flask import Flask, render_template, request, redirect, url_for
from catboost import CatBoostClassifier
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model
model = CatBoostClassifier()
model.load_model("catboost_model.cbm")



app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

uploaded_df = None

@app.route("/", methods=["GET", "POST"])
def upload_file():
    global uploaded_df
    table_html = None  # Placeholder for the table to render
    
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            # Process the Excel file with pandas
            try:
                uploaded_df = pd.read_excel(file_path, nrows=100)  # Load first 20 rows
                table_html = uploaded_df.to_html(index=False, classes="excel-table", table_id="excelTable")
            except Exception as e:
                table_html = f"<p>Error reading Excel file: {e}</p>"
    
    return render_template("index.html", table_html=table_html)

@app.route("/predict", methods=["POST"])
def predict():
    global uploaded_df
    table_html = None
    
    if uploaded_df is not None:
        try:
            # Prepare features (assuming target is the last column, if present)
            features = uploaded_df.iloc[:, :-1]
            
            # Handle NaN values in categorical features
            features = features.fillna("sas")  # Replace NaN with "Unknown"
            
            # Ensure all categorical features are strings
            for col in features.columns:
                if features[col].dtype == 'object' or col in model.get_cat_feature_indices():
                    features[col] = features[col].astype(str)
            
            # Predict using CatBoost
            predictions = model.predict(features)
            
            # Add predictions to the DataFrame
            uploaded_df["Predicted"] = predictions
            
            # Define the highlighting function based on the legend
            def highlight(row):
                predicted = str(row["Predicted"]).strip().lower() 
                if predicted == "2":  # Patients with no record of readmission
                    return ['background-color: lightgray'] * len(row)
                elif predicted == "1":  # Patients readmitted after more than 30 days
                    return ['background-color: lightblue'] * len(row)
                elif predicted == "0":  # Patients readmitted within 30 days
                    return ['background-color: teal; color: white'] * len(row)
                return [''] * len(row)  # Default: No styling
            
          

            # Convert DataFrame with applied highlighting to HTML
            table_html = uploaded_df.style.apply(highlight, axis=1).to_html()
        
        except Exception as e:
            table_html = f"<p>Error during prediction: {e}</p>"
    
    return render_template("index.html", table_html=table_html)

if __name__ == "__main__":
    app.run(debug=True)