
# raw_text="Im going home tomorrow, I'm going home 20 august, I went home yesterday, I went home 04/11/2021"
# named_entity_recognizer= NER(raw_text)
# for word in named_entity_recognizer.ents:
#     print(word.text,word.label_)


    
import spacy
# python -m spacy download en_core_web_sm

NER = spacy.load("en_core_web_sm")    
import random
import json
import pickle
import numpy as np
from pyowm import OWM
from pyowm.utils import config
from pyowm.utils import timestamps

import nltk
from nltk.stem import WordNetLemmatizer
# nltk.download('punkt')
# nltk.download('wordnet')
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('ai_chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i , r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []

    for r in results:
        return_list.append({'intent':classes[r[0]], 'probability': str(r[1])})
    
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    
    return result

print("AI Chatbot is starting. . . ")

while True:
    user_message = input("")
    ints = predict_class(user_message)
    if(ints[0]['intent']=="weather"):
        
        owm = OWM("b6c0e313b3cbe8933f23134327491f99")
        mgr = owm.weather_manager()

        city="Casablanca"
        # Search for current weather in London (Great Britain) and get details
        observation = mgr.weather_at_place(city)
        w = observation.weather
        

        w.detailed_status         # 'clouds'
        w.wind()                  # {'speed': 4.6, 'deg': 330}
        w.humidity                # 87
        w.temperature('celsius')  # {'temp_max': 10.5, 'temp': 9.7, 'temp_min': 9.0}
        w.rain                    # {}
        w.heat_index              # None
        w.clouds                  # 75

        # Will it be clear tomorrow at this time in Milan (Italy) 
        print("Weather in "+city+" can be descibed: "+w.detailed_status)
        print("Weather Information:")
        print("temperature now:",w.temperature('celsius')['feels_like'], ", Average:",w.temperature('celsius')['temp'],", Min:",w.temperature('celsius')['temp_min'],", Max:",w.temperature('celsius')['temp_max'])
        print("wind speed:",w.wind()['speed'],", wind degree: ",w.wind()['deg'])
        print("humidity:",w.humidity)
        if(len(w.rain)==0):
            print("no rain today")
        
        else:
            print("rain:",w.rain)
        
        named_entity_recognizer= NER(user_message)
        for entity in named_entity_recognizer.ents :
            if 'DATE' in entity.label_ :
                print("\n I understand that you're asking the weather for a specific date, this option will soon be implemented")
                break
    else:
        res = get_response(ints, intents)
        print(res)
        

 