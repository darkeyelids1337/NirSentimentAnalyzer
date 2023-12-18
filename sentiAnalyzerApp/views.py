from typing import List

import torch
from django.shortcuts import render
import json
from .apps import SentianalyzerappConfig
from django.http import JsonResponse
from . import dataloading
import re

# Home Page
def home(request):
    context = {
    }
    return render(request, 'sentiAnalyzerApp/home.html', context)
def predict_text(text: List[str]):
    """
    This function predicts the sentiment of a list of sentences using a pre-trained model.

    The sentences are first converted into embeddings using a custom data loading and embedding function.
    These embeddings are then passed through the model to generate logits. The model's predictions are
    then converted into human-readable labels and scores.

    Parameters
    ----------
    text : List[str]
        The list of sentences to classify.

    Returns
    -------
    results : List[dict]
        The list of dictionaries with each dictionary containing the text, predicted label, and corresponding score.
    """
    embeddings = torch.Tensor(dataloading.embed_text(text))
    logits = SentianalyzerappConfig.loaded_model(embeddings)
    preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
    scores = torch.softmax(logits, dim=1).detach().cpu().numpy()

    results = []
    for t, best_index, score_pair in zip(text, preds, scores):
        results.append({
            "text": t,
            "label": "positive" if best_index == 1 else "negative",
            "score": score_pair[best_index]
        })
    return results
# Predict Method
def predict(request):
    if request.method == "POST":
        data = request.POST['data']
        #response = SentianalyzerappConfig.loaded_model.predict(clear(data))
        response = predict_text([data])
        response[0]['score'] = str(response[0]['score'])
        print('res', response[0])
        print('type', type(response[0]))
        return JsonResponse(response[0], safe=False)

# Method to create clean text
# def clear(text):
#     tokenizer = SentianalyzerappConfig.tokenizer
#     cleanText = pad_sequences(
#     tokenizer.texts_to_sequences([preprocess_text(text)]),
#     padding='post',
#     maxlen=100
#     )
#     return cleanText
#
#
# # Method to do some preprocessing
# def preprocess_text(sen):
#     # Removing html tags
#     sentence = remove_tags(sen)
#     # Remove punctuations and numbers
#     sentence = re.sub('[^a-zA-Z]', ' ', sentence)
#     # Single character removal
#     sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
#     # Removing multiple spaces
#     sentence = re.sub(r'\s+', ' ', sentence)
#     return sentence
#
# #This code replaces anything which is enclosed withon <---> with spaces.
# TAG_RE = re.compile(r'<[^>]+>')
# def remove_tags(text):
#     return TAG_RE.sub('', text)