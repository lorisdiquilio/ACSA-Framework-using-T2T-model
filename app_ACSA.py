from flask import Flask, request, render_template, redirect, url_for, session, jsonify, send_from_directory
import pandas as pd
import os
import plotly.express as px
import plotly.io as pio
import base64
import json
from flask_session import Session
from pyabsa_m.tasks import ABSAInstruction_2 as absa_instruction
from convertitori.converter.all_functions_converter import json_to_csv, convert_json_to_txt, writeSemEvalXML, writeCustomXML, writeSemEvalXML2, remove_opinion_from_jsonl, remove_aspects_opinions_and_check_duplicates, convert_jsonl_to_tsv, setPolarityOrCategory, writeSemEvalXML14, convert_xml14_to_json, convert_xml15_to_json, convert_xml16_to_json


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# to disable the GPU
import torch
torch.cuda.is_available = lambda: False

# Load the model
generator = absa_instruction.ABSAGenerator(
   checkpoint= "./checkpoint-1440"
)

# main page
@app.route('/')
def home():
    return render_template('index.html')


# available function
conversion_functions = {
    'JSON to CSV(ACOS)': json_to_csv,
    'CSV to TXT(ACSD)': convert_json_to_txt,
    'JSON to SemEvalXML15-Sub1': writeSemEvalXML,
    'JSON to SemEvalXML15-Sub1 (other version)': writeCustomXML,
    'JSON to SemEvalXML15-Sub2' : writeSemEvalXML2,
    'Remove Opinions from ACOS JSONL (ASTE)': remove_opinion_from_jsonl,
    'Remove Aspect-Opinions from ACOS JSONL (ACSA)': remove_aspects_opinions_and_check_duplicates,
    'ACSA JSONL to TSV(ACSA)': convert_jsonl_to_tsv,
    'JSON to SemEvalXML14 (ABSA)': writeSemEvalXML14,
    'SemEval14 to JSONL(ACSA)': convert_xml14_to_json,
    'SemEval15 to JSONL(ACSA)': convert_xml15_to_json,
    'SemEval16 to JSONL(ACSA)': convert_xml16_to_json

}


@app.route('/data_converter', methods=['GET', 'POST'])
def data_converter():
    if request.method == 'POST':
        conversion_type = request.form.get('conversion_type')
        input_file = request.files.get('input_file')
        output_folder = request.form.get('output_folder')
        
        if input_file and conversion_type in conversion_functions:
            input_file_path = os.path.join(app.config['UPLOAD_FOLDER'], input_file.filename)
            input_file.save(input_file_path)
            
            conversion_function = conversion_functions[conversion_type]
            conversion_function(input_file_path, output_folder)
            
            return 'Conversion completed successfully!'
    
    return render_template('data_converter.html', conversion_types=conversion_functions.keys())

@app.route('/formats/<path:filename>')
def custom_static(filename):
    return send_from_directory('formats', filename)

# load file
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        excel_file = request.files['excel_file']
        json_file = request.files.get('json_file')

        # save the loaded files
        excel_file_path = os.path.join(app.config['UPLOAD_FOLDER'], excel_file.filename)
        excel_file.save(excel_file_path)

        json_data = []
        json_file_path = None
        if json_file:
            json_file_path = os.path.join(app.config['UPLOAD_FOLDER'], json_file.filename)
            json_file.save(json_file_path)
            with open(json_file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

        # save the data of the session
        session['json_data'] = json_data
        session['filename'] = excel_file.filename
        session['json_file_path'] = json_file_path

        return 'Files loaded!'

    return render_template('upload.html')


# manual annotation
@app.route('/manual_annotation', methods=['GET', 'POST'])
def manual_annotation():
    filename = session.get('filename')
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_excel(file_path)
    json_data = session.get('json_data', [])
    
    if request.method == 'POST':
        for index, row in df.iterrows():
            text = row['text']
            labels = []
            annotation_count = len([key for key in request.form.keys() if key.startswith(f'polarity_{index}_')])
            for i in range(annotation_count):
                polarities = request.form.getlist(f'polarity_{index}_{i}')
                categories = request.form.getlist(f'category_{index}_{i}')
                custom_categories = request.form.getlist(f'custom_category_{index}_{i}')
                for polarity, category, custom_category in zip(polarities, categories, custom_categories):
                    category = custom_category if custom_category else category
                    if polarity and category:
                        labels.append({'polarity': polarity, 'category': category})
            if labels:
                json_data.append({'text': text, 'labels': labels})
        
        # save all annotation (old and new) in the original JSON
        json_file_path = session.get('json_file_path')
        if json_file_path:
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)
        
        return 'Annotations saved with success!'
    
    return render_template('manual_annotation.html', reviews=df.to_dict(orient='records'))

# semi-automatic annotations
@app.route('/semi_automatic_annotation', methods=['GET', 'POST'])
def semi_automatic_annotation():
    filename = session.get('filename')
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_excel(file_path)
    json_data = session.get('json_data', [])

    if request.method == 'POST':
        for index, row in df.iterrows():
            text = row['text']
            labels = []

            # get all annotation for this index
            annotation_count = len([key for key in request.form.keys() if key.startswith(f'polarity_{index}_')])
            for i in range(annotation_count):
                polarity_key = f'polarity_{index}_{i}'
                category_key = f'category_{index}_{i}'
                custom_category_key = f'custom_category_{index}_{i}'

                # get all instances of keys to handle the duplicates
                polarities = request.form.getlist(polarity_key)
                categories = request.form.getlist(category_key)
                custom_categories = request.form.getlist(custom_category_key)

                for polarity, category, custom_category in zip(polarities, categories, custom_categories):
                    category = custom_category if custom_category else category
                    if polarity and category:
                        labels.append({'polarity': polarity, 'category': category})

            if labels:
                json_data.append({'text': text, 'labels': labels})

        # save the annotations (old and new) in the original JSON
        json_file_path = session.get('json_file_path')
        if json_file_path:
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)

        return 'Annotazioni saved with success!'

    return render_template('semi_automatic_annotation.html', reviews=df.to_dict(orient='records'))

# Route per ottenere le previsioni del modello
@app.route('/get_predictions', methods=['POST'])
def get_predictions():
    filename = session.get('filename')
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_excel(file_path)
    predictions = []

    for index, row in df.iterrows():
        text = row['text']
        result = generator.predict(text)
        predictions.append({
            'text': text,
            'predictions': result['Tuples']  # Assuming 'Tuples' contains polarity and category predictions
        })

    return jsonify(predictions)


@app.route('/upload_reviews', methods=['GET', 'POST'])
def upload_reviews():
    if request.method == 'POST':
        excel_file = request.files['excel_file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], excel_file.filename)
        excel_file.save(file_path)

        df = pd.read_excel(file_path)
        predictions = []

        for index, row in df.iterrows():
            text = row['text']
            result = generator.predict(text)
            for quadruple in result['Tuples']:
                if 'category' in quadruple and 'polarity' in quadruple: 
                    predictions.append({
                        'text': text,
                        'category': quadruple['category'],
                        'polarity': quadruple['polarity']
                    })
                else:
                    print(f"Unexpected prediction format: {quadruple}")

        if predictions:
            pred_df = pd.DataFrame(predictions)

            # filtro fuori gli others
            filtered_df = pred_df[pred_df['category'] != 'Others']


            # colori per polarità
            color_map = {
                'Positive': '#28a745',  # verde
                'Negative': '#dc3545',  # rosso
                'Neutral': '#6c757d',    # grigio
                'Conflict': '#003399'   #blu
            }

            # istogramma
            fig = px.histogram(filtered_df, x="category", color="polarity", barmode="stack",
                               title="Aspect-Category Sentiment Analysis",
                               labels={"category": "Category", "polarity": "Polarity"},
                               color_discrete_map=color_map)

            graph_html = pio.to_html(fig, full_html=False)

            # sentiment index generale
            positive_count = len(filtered_df[filtered_df['polarity'] == 'Positive'])
            negative_count = len(filtered_df[filtered_df['polarity'] == 'Negative'])
            total_count = positive_count + negative_count

            if total_count == 0:
                sentiment_index_overall = 0  # gestione del caso in cui non ci sono recensioni
            else:
                sentiment_index_overall = (positive_count - negative_count) / total_count

            # Sentiment index per ciascuna categoria
            category_sentiment_index = filtered_df.groupby('category').apply(
                lambda x: ((len(x[x['polarity'] == 'Positive']) - len(x[x['polarity'] == 'Negative'])) /
                           (len(x[x['polarity'] == 'Positive']) + len(x[x['polarity'] == 'Negative']))) if (len(x[x['polarity'] == 'Positive']) + len(x[x['polarity'] == 'Negative'])) != 0 else 0
            ).reset_index(name='Sentiment Index')

            sentiment_index_html = f"<p>Overall Sentiment Index: {sentiment_index_overall:.2f}</p>"
            sentiment_index_html += "<h3>Sentiment Index by Category:</h3><ul>"
            for _, row in category_sentiment_index.iterrows():
                sentiment_index_html += f"<li>{row['category']}: {row['Sentiment Index']:.2f}</li>"
            sentiment_index_html += "</ul>"
        else:
            graph_html = "<p>No valid predictions were found.</p>"
            sentiment_index_html = ""

        return render_template('results_plot.html', graph_html=graph_html, sentiment_index_html=sentiment_index_html)

    return render_template('upload_reviews.html')



if __name__ == '__main__':
    app.run(debug=True)
