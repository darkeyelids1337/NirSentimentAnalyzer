from django.apps import AppConfig
# import joblib
# import os
# from django.conf import settings
# from keras.models import model_from_json
from . import modeling


class SentianalyzerappConfig(AppConfig):
    name = 'sentiAnalyzerApp'
    loaded_model = modeling.Model.load_from_checkpoint('sentiAnalyzerApp/models/yelp-sentiment-multilingual-epoch=13-val_loss=0.196.ckpt')
    # modelh5Path = os.path.join(settings.MODELS, 'model.h5')
    # modelJsonPath = os.path.join(settings.MODELS, 'model.json')
    # tokenizerPath = os.path.join(settings.MODELS, 'tokenizer.pkl')
    #
    # # Model Loading
    # with open(modelJsonPath, 'rb') as json_file:
    #     loaded_model_json = json_file.read()
    # with open(tokenizerPath, 'rb') as t:
    #     tokenizer = joblib.load(t)
    # t.close()
    # json_file.close()
    # loaded_model = model_from_json(loaded_model_json)
    # loaded_model.load_weights(modelh5Path)