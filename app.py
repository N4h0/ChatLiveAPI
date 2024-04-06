from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flask_cors import CORS
from setfit import SetFitModel
import json

app = Flask(__name__)
CORS(app)

# Henter alle spørsmål og svar
questions = []
answers = []
with open('txtandCSV-files/Q&A.txt', 'r', encoding='utf-8') as file:
    for line in file:
        if line.startswith('Q:'):
            questions.append(line[3:].strip())
        if line.startswith('A') and not line.startswith('AF'):
            answers.append(line[3:].strip())

print(questions)

#Henter den encoda lista med spørsmål i json format
with open('txtandCSV-files/Q&A_embedded.json', 'r', encoding='utf-8') as file:
    loaded_list_as_lists = json.load(file)

#Konverterer den encoda lista til ei liste med arrays.
def convert_to_arrays(loaded_list_as_lists):
    return [np.array(sublist) for sublist in loaded_list_as_lists]

encoded_questions_list = [convert_to_arrays(sublist) for sublist in loaded_list_as_lists]

#Henter modellen
model_name = "NbAiLab/nb-sbert-base"
model = SentenceTransformer(model_name)
model = SetFitModel.from_pretrained("modeller/alpha2")

#Sjølve apien
@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    if not request.json or 'question' not in request.json:
        return jsonify({'error': 'Missing question in request'}), 400

    #Henter spørsmålet til bruker
    user_question = request.json['question']
    #Encoder spørsmålet til bruker
    encoded_user_question = model.encode([user_question])[0]
    similarity_scores = []
    #Gjer det på denne måten slik at einaste verdien som blir lagra er maksverdien av Q og alle AF til Q. 
    for sublist in encoded_questions_list:
        similarity_scores.append(max(cosine_similarity([encoded_user_question], sublist)[0]))
    most_similar_question_index = np.argmax(similarity_scores)
    
    nested_list = []

    for question, similarity_score in zip(questions, similarity_scores):
        sublist = [question, float(similarity_score)]
        nested_list.append(sublist)

    sorted_nested_list = sorted(nested_list, key=lambda x: x[1], reverse=True)


    most_similar_question = questions[most_similar_question_index]
    print("Returning output to user: ", most_similar_question)
    for item in sorted_nested_list:
        print(item)

    return jsonify(most_similar_question)

if __name__ == '__main__':
    app.run(debug=True)