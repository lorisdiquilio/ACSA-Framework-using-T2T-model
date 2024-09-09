import json
import csv
import os
import xml.etree.ElementTree as ET


#questa funzione converte i dati dal jsonl del PyABSA che posso annotare con LabelStudio o manualmente in
#in un csv con le seguenti colonne: id, text, aspect, opinion, polarity, category
#se una frase ha più annotazioni, queste vengono ripetute per ognuna di esse

def json_to_csv(json_file_path, csv_file_path):
    data = []
    last_id = 0

    if os.path.exists(csv_file_path):
        with open(csv_file_path, 'r', newline='') as f:
            reader = csv.reader(f)
            existing_data = list(reader)
            if len(existing_data) > 1 and len(existing_data[1]) > 0:
                ids = [int(row[0]) for row in existing_data[1:]]
                last_id = max(ids)
            else:
                last_id = 0
        print(f"Existing data found, last_id: {last_id}")
    else:
        print("No existing CSV file found, starting with last_id: 0")

    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        items = json.load(json_file)
        for item in items:
            text = item['text']
            labels = item['labels']
            for label in labels:
                last_id += 1
                aspect = label['aspect']
                opinion = label['opinion']
                polarity = label['polarity']
                category = label['category']
                data.append([last_id, text, aspect, opinion, polarity, category])

    # scrivo i dati nel file CSV
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "text", "aspect", "opinion", "polarity", "category"])
        writer.writerows(data)
    print(f"CSV file created at: {csv_file_path}")
    print(f"Data written to CSV: {data}")

#json_to_csv('C:/Users/diquiliol/OneDrive - AptarGroup, Inc/Desktop/Aptar/AI - BA/Da portare in DOTTORATO/Dataset/converter/All_dataset_acos.jsonl', 'C:/Users/diquiliol/OneDrive - AptarGroup, Inc/Desktop/Aptar/AI - BA/Da portare in DOTTORATO/Dataset/converter/All_dataset_acos.csv')




#questa funzione converte i dati dal jsonl del PyABSA che posso annotare con LabelStudio o manualmente in
#in un txt con questo formato: "I'm giving it a one star rating for extremely clumsy, clunky and wasteful packaging..####[('packaging', 'GENERAL#SUSTAINABILITY', 'Negative'), ('packaging', 'GENERAL#USABILITY', 'Negative'), ('packaging', 'GENERAL#SUSTAINABILITY', 'Negative')]"
#usato per il tool che Generative-ABSA con il task tasd (paradigm extraction)

def convert_json_to_txt(json_file_path, txt_file_path):
    with open(json_file_path, 'r') as json_file, open(txt_file_path, 'w') as txt_file:
        data = json.load(json_file)
        for item in data:
            text = item['text']
            labels = item['labels']
            annotations = []
            for label in labels:
                aspect = label['aspect']
                category = label['category']
                polarity = label['polarity']
                annotations.append((aspect, category, polarity))
            txt_file.write(f"{text}.####{annotations}\n")

#convert_json_to_txt('/home/lorisdiquilio/Scrivania/Project/Framework/dataset/All_dataset_acos.jsonl', '/home/lorisdiquilio/Scrivania/Project/Dataset/all_tasd_dataset.txt')


#questa funzione converte il json estratto da LabelStudio (Subtask 1 - one aspect for each sentence) nel subtask1 del SemEval 2015
#ad esempio file di input questo: /home/lorisdiquilio/Scrivania/Project/Dataset/convertitori/data/input/train/train_label_studio_sub1.json
#sarebbe category e sentiment


def writeSemEvalXML(json_path, output_path):
    with open(json_path) as f:
        data = json.load(f)

    reviews = {}
    for sentence in data:
        review_number = sentence['Review_number']
        sentence_number = sentence['Sentence_number']
        if review_number not in reviews:
            reviews[review_number] = {}
        if sentence_number not in reviews[review_number]:
            reviews[review_number][sentence_number] = {'text': sentence['Sentence'], 'opinions': []}
        if 'Category' in sentence and 'Sentiment' in sentence:
            opinion = {'category': sentence['Category'], 'polarity': sentence['Sentiment']}
            reviews[review_number][sentence_number]['opinions'].append(opinion)

    root = ET.Element('Reviews')
    for review_number, sentences in reviews.items():
        review_elem = ET.SubElement(root, 'Review', {'rid': str(review_number)})
        sentences_elem = ET.SubElement(review_elem, 'sentences')
        for sentence_id, sentence_data in sentences.items():
            sentence_elem = ET.SubElement(sentences_elem, 'sentence', {'id': f"{review_number}:{sentence_id}"})
            text_elem = ET.SubElement(sentence_elem, 'text')
            text_elem.text = sentence_data['text']
            if sentence_data['opinions']:
                opinion_element = ET.SubElement(sentence_elem, 'Opinions')
                for opinion in sentence_data['opinions']:
                    opinion_sub_element = ET.SubElement(opinion_element, 'Opinion', opinion)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="\t", level=0)
    tree.write(output_path, encoding='UTF-8', xml_declaration=True, short_empty_elements=False)

#esempio di utilizzo
'''json_path = '/exportlabelstudio/dataset.json'
output_path = 'output-semeval.xml'
writeSemEvalXML(json_path, output_path)
print(f'Successfully converted {json_path} to {output_path}')'''




#questa funzione converte il json estratto da labelstudio (Subtask 1 - one aspect for each sentence) nel subtask1 del SemEval 2015 ma la versione con l'opinion targe
#ad esempio file di input questo: /home/lorisdiquilio/Scrivania/Project/Dataset/convertitori/data/input/train/train_label_studio_sub1.json
#sarebbe category e sentiment
def writeCustomXML(json_path, output_path):
    with open(json_path) as f:
        data = json.load(f)

    reviews = {}
    for sentence in data:
        review_number = sentence['Review_number']
        sentence_number = sentence['Sentence_number']
        if review_number not in reviews:
            reviews[review_number] = {}
        if sentence_number not in reviews[review_number]:
            reviews[review_number][sentence_number] = {'text': sentence['Sentence'], 'opinions': []}
        if 'Category' in sentence and 'Sentiment' in sentence:
            opinion = {
                'target':  'NULL',
                'category': sentence['Category'] if sentence['Category'] else 'NULL',
                'polarity': sentence['Sentiment'],
                'from': '0',
                'to': '0'
            }
            reviews[review_number][sentence_number]['opinions'].append(opinion)

    root = ET.Element('Reviews')
    for review_number, sentences in reviews.items():
        review_elem = ET.SubElement(root, 'Review', {'rid': str(review_number)})
        sentences_elem = ET.SubElement(review_elem, 'sentences')
        for sentence_id, sentence_data in sentences.items():
            sentence_elem = ET.SubElement(sentences_elem, 'sentence', {'id': f"{review_number}:{sentence_id}"})
            text_elem = ET.SubElement(sentence_elem, 'text')
            text_elem.text = sentence_data['text']
            if sentence_data['opinions']:
                opinion_element = ET.SubElement(sentence_elem, 'Opinions')
                for opinion in sentence_data['opinions']:
                    opinion_sub_element = ET.SubElement(opinion_element, 'Opinion', opinion)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="\t", level=0)
    tree.write(output_path, encoding='UTF-8', xml_declaration=True)

#esempio di utilizzo
'''json_path = '/home/lorisdiquilio/Scrivania/Project/Dataset/convertinxml/Converterinsb1/test.json'
output_path = 'output-custom.xml'
writeCustomXML(json_path, output_path)
print(f'Successfully converted {json_path} to {output_path}')'''



#queste due funzioni convertono il json di label studio nell'xml SemEval2015 per il subtask2, quello in cui le annotazioni sono fatte tutte alla fine di ogni frase
import xml.etree.ElementTree as ET
import json


def writeSemEvalXML2(json_path, output_path):
    with open(json_path) as f:
        data = json.load(f)

    reviews = {}
    for sentence in data:
        review_number = sentence['Review_number']
        if review_number not in reviews:
            reviews[review_number] = []
        reviews[review_number].append(sentence)

    root = ET.Element('Reviews')
    for review_number, sentences in reviews.items():
        review_elem = ET.SubElement(root, 'Review', {'rid': str(review_number)})
        sentences_elem = ET.SubElement(review_elem, 'sentences')
        opinions = []
        for sentence_index, sentence in enumerate(sentences):
            sentence_elem = ET.SubElement(sentences_elem, 'sentence', {'id': f"{review_number}:{sentence_index}"})
            text_elem = ET.SubElement(sentence_elem, 'text')
            text_elem.text = sentence['Sentence']
            sentiment = sentence.get('Sentiment')
            if sentiment:
                opinions.append({'category': sentence['Category'], 'polarity': sentiment})
        if opinions:
            opinions_elem = ET.SubElement(review_elem, 'Opinions')
            for opinion in opinions:
                ET.SubElement(opinions_elem, 'Opinion', opinion)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="\t", level=0)
    tree.write(output_path, encoding='UTF-8', xml_declaration=True, short_empty_elements=False)

'''
json_path = 'C:/Users/diquiliol/Downloads/convertitoreinsb2/train.json'
output_path = 'output-semeval2.xml'
writeSemEvalXML2(json_path, output_path)
print(f'Successfully converted {json_path} to {output_path}')
'''

    
#questa funzione serve a convertire il file con le annotazioni PyABSA, nello stesso, ma senza le opinion

import json

def remove_opinion_from_jsonl(jsonl_path, output_jsonl_path):
    with open(jsonl_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    # Rimuovere la chiave "opinion" da ogni record
    for entry in data:
        if 'opinion' in entry:
            del entry['opinion']

    with open(output_jsonl_path, 'w', encoding='utf-8') as output_json_file:
        json.dump(data, output_json_file, ensure_ascii=False, indent=2)

#esempio di utilizzo
if __name__ == "__main__":
    jsonl_path = "path/to/your/input.jsonl"
    output_jsonl_path = "path/to/your/output_no_opinion.jsonl"

    remove_opinion_from_jsonl(jsonl_path, output_jsonl_path)
    
    


#questa funzione serve per passare dal dataset PyABSA annotato allo stesso senza aspects e opinions. Non essendoci più questi due elementi, quando ci sono le stesse annotazioni all'interno di una frase queste vengono rimosse
import json

def remove_aspects_opinions_and_check_duplicates(jsonl_path, output_jsonl_path):
    with open(jsonl_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    unique_annotations_per_text = {}

    for entry in data:
        text = entry['text']
        if 'labels' in entry:
            unique_annotations = set()
            new_labels = []
            for label in entry['labels']:
                annotation_key = f"{label['category']}#{label['polarity']}"
                if annotation_key not in unique_annotations:
                    new_labels.append({"polarity": label["polarity"], "category": label["category"]})
                    unique_annotations.add(annotation_key)
            entry['labels'] = new_labels

            if text not in unique_annotations_per_text:
                unique_annotations_per_text[text] = set()

            existing_annotations = set(unique_annotations_per_text[text])
            new_annotations = set(annotation_key for label in entry['labels'] for annotation_key in [f"{label['category']}#{label['polarity']}"])

            if not existing_annotations.intersection(new_annotations):
                unique_annotations_per_text[text].update(new_annotations)
            else:
                # Rimuovi l'intero record se esiste un'annotazione duplicata
                entry['labels'] = []

    # Filtra i record senza etichette
    data = [entry for entry in data if 'labels' in entry and entry['labels']]

    with open(output_jsonl_path, 'w', encoding='utf-8') as output_json_file:
        json.dump(data, output_json_file, ensure_ascii=False, indent=2)


'''        
#esempio di utilizzo
if __name__ == "__main__":
    jsonl_path = "/home/lorisdiquilio/PyABSA/examples-v2/aspect_opinion_sentiment_category_extraction/tests.jsonl"
    output_jsonl_path = "/home/lorisdiquilio/Scrivania/Project/Dataset/convertitori/data/output/json/AC_PyABSA/test1_ap.jsonl"

    remove_aspects_opinions_and_check_duplicates(jsonl_path, output_jsonl_path)
'''
 
 
 
 
 #questa funzione serve per passare dal dataset PyABSA annotato solo con categorie e polarità (convertito con la funzione qui sopra) nel formato tabellare con category#1 o category#-1 per il lavoro ACSA-HGCN

def convert_jsonl_to_tsv(jsonl_path, tsv_path):
    with open(jsonl_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    with open(tsv_path, 'w', encoding='utf-8') as tsv_file:
        for entry in data:
            text = entry['text']
            labels = entry['labels']

            annotations_list = []

            for label in labels:
                category = label['category']
                polarity = label['polarity']

                if polarity == 'Positive':
                    sentiment = 1
                elif polarity == 'Negative':
                    sentiment = -1
                elif polarity == 'Neutral':
                    sentiment = 0
                elif polarity == 'Conflict':
                    sentiment = 2

                # creo una stringa per ogni annotazion
                annotation_str = f'{category}#{sentiment}'

                annotations_list.append(annotation_str)

            # unisco le stringhe (tabulazioni)
            annotations_combined = '\t'.join(annotations_list)

            tsv_file.write(f'{text}\t{annotations_combined}\n')




#queste due funzioni servono per trasformare il dataset JSON estratto da LabelStudio nel formato di SemEval2014 con gli aspect term

import json
import xml.etree.ElementTree as ET

def setPolarityOrCategory(label, ann):
    if (label == 'Positive') or (label == 'Negative') or (label == 'Neutral'):
        ann['polarity'] = label.lower()
    else:
        ann['category'] = label.lower()

def writeSemEvalXML14(json_path, output_path):
    with open(json_path) as f:
        data = json.load(f)

    sentences = []
    for i in data:
        s = {}
        s['id'] = i['id']
        s['text'] = i['data']['Sentences']
        s['annotations'] = []
        for a in i['annotations']:
            for r in a['result']:
                a2 = {}
                a2['start'] = r['value']['start']
                a2['end'] = r['value']['end']
                a2['term'] = r['value']['text']
                label = r['value']['labels'][0]
                sametextannfound = False
                for a3 in s['annotations']:
                    if (a2['start'] == a3['start']) and (a2['end'] == a3['end']):
                        setPolarityOrCategory(label, a3)
                        sametextannfound = True
                if not(sametextannfound): 
                    setPolarityOrCategory(label, a2)
                    s['annotations'].append(a2)            
        sentences.append(s);

    root = ET.Element('sentences')
    for s in sentences:
        sEl = ET.SubElement(root, 'sentence')
        sEl.attrib['id'] = str(s['id'])
        tEl = ET.SubElement(sEl, 'text')
        tEl.text = s['text']
        aspectTermsEl = ET.SubElement(sEl, 'aspectTerms')
        for a in s['annotations']:
            aTEl = ET.SubElement(aspectTermsEl, 'aspectTerm')
            aTEl.attrib['term'] = a['term']
            aTEl.attrib['polarity'] = a['polarity']
            aTEl.attrib['from'] = str(a['start'])
            aTEl.attrib['to'] = str(a['end'])
        aspectCategoriesEl = ET.SubElement(sEl, 'aspectCategories')
        for a in s['annotations']:
            aCEl = ET.SubElement(aspectCategoriesEl, 'aspectCategory')
            aCEl.attrib['category'] = a.get('category','unknowncategory')
            aCEl.attrib['polarity'] = a['polarity']
    tree = ET.ElementTree(root)
    ET.indent(tree, space="\t", level=0)
    tree.write(output_path, encoding="UTF-8", xml_declaration=True, short_empty_elements=False)
          


#questa funzione serve per trasformare il SemEval 2014 con gli aspect term al JSONL utile per il framework nostro (ACSA)

def convert_xml14_to_json(xml_file, json_file):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    sentences = []

    for sentence in root.findall('sentence'):
        text = sentence.find('text').text

        labels = []

        for aspectCategory in sentence.findall('aspectCategories/aspectCategory'):
            polarity = aspectCategory.get('polarity')
            category = aspectCategory.get('category')

            labels.append({'polarity': polarity, 'category': category})

        sentences.append({'text': text, 'labels': labels})

    with open(json_file, 'w') as f:
        json.dump(sentences, f, indent=4)



#questa funzione serve per passare dal formato SemEval2015 al JSONL utile per il nostro framework (ACSA)
def convert_xml15_to_json(xml_file, json_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    reviews = []

    for review in root.findall('Review'):
        for sentence in review.findall('sentences/sentence'):
            text = sentence.find('text').text

            labels = []

            for opinion in sentence.findall('Opinions/Opinion'):
                polarity = opinion.get('polarity')
                category = opinion.get('category')

                labels.append({'polarity': polarity, 'category': category})

            reviews.append({'text': text, 'labels': labels})

    with open(json_file, 'w') as f:
        json.dump(reviews, f, indent=4)


#questa funzione serve a trasformare i dati dal dataset SemEval2016 al JSONL utile per il nostro framework (ACSA)

def convert_xml16_to_json(xml_file_path, json_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    reviews = []

    for review in root.findall('Review'):
        review_text = ''

        for sentence in review.find('sentences').findall('sentence'):
            review_text += sentence.find('text').text + ' '

        labels = []

        for opinion in review.find('Opinions').findall('Opinion'):
            labels.append({
                'polarity': opinion.get('polarity').capitalize(),
                'category': opinion.get('category')
            })

        reviews.append({
            'text': review_text.strip(),
            'labels': labels
        })

    with open(json_file_path, 'w') as f:
        json.dump(reviews, f, ensure_ascii=False, indent=4)
