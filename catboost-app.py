from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import pandas as pd
import os
import time
from datetime import timedelta
import json
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['TEMP_FOLDER'] = 'temp'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)
app.secret_key = "supersecretkey"

ROWS_PER_PAGE = 50

def allowed_file(filename):
    """Check if the uploaded file is an Excel file."""
    return filename.lower().endswith(('.xls', '.xlsx'))

def color_rows(df):
    """Apply conditional formatting based on the last column."""
    if df.empty:
        return df

    def color_row(row):
        last_col = df.columns[-1]
        value = str(row[last_col]).strip().lower()

        if value == 'no':
            return ['background-color: #A49E9E'] * len(df.columns)
        elif value.startswith('>30'):
            return ['background-color: #A9EEFF'] * len(df.columns)
        elif value.startswith('<30'):
            return ['background-color: #60AFC2'] * len(df.columns)
        return [''] * len(df.columns)

    return df.style.apply(color_row, axis=1)

def process_file(file_path, page=1, initial_load=False):
    """Handle file processing and styling."""
    try:
        if initial_load:
            # Generate a unique ID for this upload
            session_id = str(uuid.uuid4())
            
            # Read Excel file
            df = pd.read_excel(file_path)
            
            if df.empty:
                flash("Uploaded file is empty.", "error")
                return None, None, 0
            
            # Replace missing values with '?'
            df = df.fillna('?')
            
            # Save metadata to session
            session['total_rows'] = len(df)
            session['columns'] = list(df.columns)
            session['session_id'] = session_id
            
            # Save DataFrame to temporary file
            temp_file = os.path.join(app.config['TEMP_FOLDER'], f"{session_id}.parquet")
            df.to_parquet(temp_file)
            
            # Process first page
            total_rows = len(df)
            total_pages = min((total_rows // ROWS_PER_PAGE) + (1 if total_rows % ROWS_PER_PAGE > 0 else 0), 10)
            
            df_page = df.iloc[0:ROWS_PER_PAGE]
            
            # Apply styling
            styled_df = df_page.style\
                .set_properties(**{
                    'font-size': '14px',
                    'font-family': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
                    'text-align': 'center',  # Center align all cells
                    'padding': '12px 16px',
                    'border-bottom': '1px solid #e0e0e0',
                    'white-space': 'nowrap',
                    'min-width': '120px'
                })\
                .set_table_styles([
                    {'selector': 'thead th', 'props': [
                        ('background-color', '#fff'),
                        ('font-weight', '500'),
                        ('color', '#000'),
                        ('border-bottom', '1px solid #e0e0e0'),
                        ('position', 'sticky'),
                        ('top', '0'),
                        ('z-index', '2'),
                        ('box-shadow', 'inset 0 -1px 0 #e0e0e0'),
                        ('padding', '16px 24px'),
                        ('text-align', 'center'),  # Center align headers
                        ('vertical-align', 'middle')
                    ]},
                    {'selector': 'tbody tr:hover', 'props': [
                        ('background-color', '#f8f9fa')
                    ]},
                    {'selector': 'td', 'props': [
                        ('text-align', 'center'),  # Center align cells
                        ('vertical-align', 'middle')
                    ]},
                    # Column-specific widths
                    {'selector': 'td:nth-child(1), th:nth-child(1)', 'props': [('width', '120px')]},  # race
                    {'selector': 'td:nth-child(2), th:nth-child(2)', 'props': [('width', '100px')]},  # gender
                    {'selector': 'td:nth-child(3), th:nth-child(3)', 'props': [('width', '80px')]},   # age
                    {'selector': 'td:nth-child(4), th:nth-child(4)', 'props': [('width', '150px')]},  # admission_type_id
                    {'selector': 'td:nth-child(5), th:nth-child(5)', 'props': [('width', '180px')]},  # discharge_disposition_id
                    {'selector': 'td:nth-child(6), th:nth-child(6)', 'props': [('width', '160px')]},  # admission_source_id
                    {'selector': 'td:nth-child(7), th:nth-child(7)', 'props': [('width', '120px')]}   # time_in_hospital
                ])
            
            table_html = styled_df.to_html(index=False, classes="excel-table", table_id="excelTable-page1")
            return table_html, None, total_pages
            
        else:
            # Check if we have a valid session
            if 'session_id' not in session:
                return None, None, 0
            
            # Load the specific page from the temporary file
            temp_file = os.path.join(app.config['TEMP_FOLDER'], f"{session['session_id']}.parquet")
            if not os.path.exists(temp_file):
                return None, None, 0
                
            # Read the data
            df = pd.read_parquet(temp_file)
            total_rows = session.get('total_rows', 0)
            start_idx = (page - 1) * ROWS_PER_PAGE
            end_idx = min(start_idx + ROWS_PER_PAGE, total_rows)
            
            df_page = df.iloc[start_idx:end_idx]
            if df_page.empty:
                return None, None, 0
                
            total_pages = min((total_rows // ROWS_PER_PAGE) + (1 if total_rows % ROWS_PER_PAGE > 0 else 0), 10)
            
            # Apply styling
            styled_df = df_page.style\
                .set_properties(**{
                    'font-size': '14px',
                    'font-family': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
                    'text-align': 'center',  # Center align all cells
                    'padding': '12px 16px',
                    'border-bottom': '1px solid #e0e0e0',
                    'white-space': 'nowrap',
                    'min-width': '120px'
                })\
                .set_table_styles([
                    {'selector': 'thead th', 'props': [
                        ('background-color', '#fff'),
                        ('font-weight', '500'),
                        ('color', '#000'),
                        ('border-bottom', '1px solid #e0e0e0'),
                        ('position', 'sticky'),
                        ('top', '0'),
                        ('z-index', '2'),
                        ('box-shadow', 'inset 0 -1px 0 #e0e0e0'),
                        ('padding', '16px 24px'),
                        ('text-align', 'center'),  # Center align headers
                        ('vertical-align', 'middle')
                    ]},
                    {'selector': 'tbody tr:hover', 'props': [
                        ('background-color', '#f8f9fa')
                    ]},
                    {'selector': 'td', 'props': [
                        ('text-align', 'center'),  # Center align cells
                        ('vertical-align', 'middle')
                    ]},
                    # Column-specific widths
                    {'selector': 'td:nth-child(1), th:nth-child(1)', 'props': [('width', '120px')]},  # race
                    {'selector': 'td:nth-child(2), th:nth-child(2)', 'props': [('width', '100px')]},  # gender
                    {'selector': 'td:nth-child(3), th:nth-child(3)', 'props': [('width', '80px')]},   # age
                    {'selector': 'td:nth-child(4), th:nth-child(4)', 'props': [('width', '150px')]},  # admission_type_id
                    {'selector': 'td:nth-child(5), th:nth-child(5)', 'props': [('width', '180px')]},  # discharge_disposition_id
                    {'selector': 'td:nth-child(6), th:nth-child(6)', 'props': [('width', '160px')]},  # admission_source_id
                    {'selector': 'td:nth-child(7), th:nth-child(7)', 'props': [('width', '120px')]}   # time_in_hospital
                ])
            
            table_html = styled_df.to_html(index=False, classes="excel-table", table_id=f"excelTable-page{page}")
            return table_html, None, total_pages

    except Exception as e:
        print(f"Exception occurred: {e}")
        return None, None, 0

@app.route("/upload-progress")
def upload_progress():
    """Return the current upload progress."""
    progress = session.get('upload_progress', 0)
    return jsonify({'progress': progress})

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "GET":
        session.clear()

    table_html = None
    total_pages = 0
    page = int(request.args.get('page', 1))

    if request.method == "POST":
        if 'file' in request.files:
            file = request.files.get("file")
            if not file or file.filename == "":
                flash("No file selected. Please choose a file to upload.", "error")
                return redirect(request.url)

            if allowed_file(file.filename):
                try:
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                    file.save(file_path)
                    session['uploaded_file'] = file_path
                    
                    # Initial load - read and cache the file
                    table_html, _, total_pages = process_file(file_path, page, initial_load=True)
                    
                    if not table_html:
                        raise Exception("Failed to process file")
                        
                except Exception as e:
                    flash(f"Error processing file: {str(e)}", "error")
                    return redirect(request.url)

    elif 'uploaded_file' in session:
        # Use cached data for subsequent requests
        table_html, _, total_pages = process_file(session['uploaded_file'], page)

    return render_template("index.html", 
                         table_html=table_html, 
                         total_pages=total_pages, 
                         current_page=page)

@app.route("/paginate")
def paginate():
    try:
        page = int(request.args.get('page', 1))

        if 'session_id' not in session or 'uploaded_file' not in session:
            return jsonify({'error': 'No data available'}), 400

        # Use cached data
        table_html, _, total_pages = process_file(session['uploaded_file'], page)
        
        if not table_html:
            return jsonify({'error': 'Error processing data'}), 500

        pagination_html = render_template(
            'pagination.html',
            total_pages=total_pages,
            current_page=page
        )

        return jsonify({
            'table_html': table_html,
            'pagination_html': pagination_html,
            'current_page': page,
            'total_pages': total_pages
        })

    except Exception as e:
        print(f"Error in pagination: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route("/clear-session")
def clear_session():
    if 'session_id' in session:
        # Remove temporary file
        temp_file = os.path.join(app.config['TEMP_FOLDER'], f"{session['session_id']}.parquet")
        if os.path.exists(temp_file):
            os.remove(temp_file)
    session.clear()
    return jsonify({"message": "Session cleared successfully."})

@app.route("/clear-cache")
def clear_cache():
    if 'session_id' in session:
        # Remove temporary file
        temp_file = os.path.join(app.config['TEMP_FOLDER'], f"{session['session_id']}.parquet")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        del session['session_id']
    return jsonify({"message": "Cache cleared successfully"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        algorithm = request.json.get('algorithm')
        if 'session_id' not in session:
            return jsonify({'error': 'No dataset loaded'}), 400

        # Load the data from the temporary file
        temp_file = os.path.join(app.config['TEMP_FOLDER'], f"{session['session_id']}.parquet")
        if not os.path.exists(temp_file):
            return jsonify({'error': 'Dataset not found'}), 400

        df = pd.read_parquet(temp_file)

        # Here you would implement your model prediction logic
        if algorithm == 'catboost':
            # Example implementation for Catboost
            # model = CatBoostClassifier()
            # predictions = model.predict(df)
            # accuracy = model.score(X_test, y_test)
            
            # Placeholder values (replace with actual model results)
            results = {
                'accuracy': 63.234,
                'total_patients': len(df),
                'readmitted': 90,  # Replace with actual count
                'not_readmitted': 50,  # Replace with actual count
                'misclassified': 20,  # Replace with actual count
                'misclassification_rate': 14  # Replace with actual rate
            }
        
        elif algorithm == 'bayesian':
            # Example implementation for Bayesian Optimized Catboost
            # model = BayesianOptimizedCatBoost()
            # predictions = model.predict(df)
            # accuracy = model.score(X_test, y_test)
            
            # Placeholder values (replace with actual model results)
            results = {
                'accuracy': 65.789,
                'total_patients': len(df),
                'readmitted': 85,  # Replace with actual count
                'not_readmitted': 55,  # Replace with actual count
                'misclassified': 18,  # Replace with actual count
                'misclassification_rate': 12  # Replace with actual rate
            }
        
        else:
            return jsonify({'error': 'Invalid algorithm selected'}), 400

        return jsonify(results)

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Clean up temporary files older than 1 hour
def cleanup_temp_files():
    current_time = time.time()
    for filename in os.listdir(app.config['TEMP_FOLDER']):
        file_path = os.path.join(app.config['TEMP_FOLDER'], filename)
        if os.path.getmtime(file_path) < current_time - 3600:  # 1 hour
            os.remove(file_path)

# Add cleanup to the session lifetime
app.permanent_session_lifetime = timedelta(hours=1)

if __name__ == "__main__":
    app.run(debug=True)
