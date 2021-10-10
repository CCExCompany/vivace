# Encoding https://pbpython.com/categorical-encoding.html
#https://dzone.com/articles/how-to-make-chatbots-more-intelligent-with-context
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import json
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD


intents = json.loads(open('intents.json').read())

#The patterns list contains every pattern in the json file mapped to the tag it belongs to.
patterns = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
    	patterns.append([pattern, intent['tag']])
patterns = np.array(patterns)




#Vectorize the patterns into tokens and create a bag of words
count = CountVectorizer()
# count2 = TfidfVectorizer() #assigns a weight for each word which becomes very low for frequently occurring words
bow = count.fit_transform(patterns[:,0])
# bow = count2.fit_transform(patterns[:,0])
feature_names = count.get_feature_names() #The tokens (every word in all the patterns)
classes = sorted(set(patterns[:,1])) #The classes or tags


# Transform the tokens and classes to byte stream to be used in chatbot2.py
pickle.dump(feature_names, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))




#Vew the bags of words as a dataframe
df_bow = pd.DataFrame((bow).toarray(), columns=feature_names)
df_bow["class"] = patterns[:,1]
#OneHot encode the class column
encoder = OneHotEncoder()
encoder_results = encoder.fit_transform(df_bow[["class"]])
df_bow = df_bow.join(pd.DataFrame(encoder_results.toarray(), columns=encoder.categories_))
df_bow.drop(labels=['class'], axis=1, inplace=True)
# df_bow.rename(columns={('stock market',):'stock market',('system information',):'system information',('goodbye',):"goodbye", ('greetings',):"greetings", ('weather',):"weather",('calendar',):"calendar" ,('name',):"name"}, inplace=True)
df_bow.rename(columns={('greetings',):'greetings'}, inplace=True)
df_bow.rename(columns={('goodbye',):'goodbye'}, inplace=True)
df_bow.rename(columns={('stock market',):'stock market'}, inplace=True)
df_bow.rename(columns={('name',):'name'}, inplace=True)
df_bow.rename(columns={('weather',):'weather'}, inplace=True)
df_bow.rename(columns={('search',):'search'}, inplace=True)
df_bow.rename(columns={('system information',):'system information'}, inplace=True)
print(df_bow)




#Transform features and taregt variables to numpy arrays
x = df_bow.iloc[:, 0:-7].values
y = df_bow.iloc[:, -7:].values



#Model
model = Sequential()
model.add(Dense(128, input_shape=(len(x[0]), ),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

ai_model = model.fit(x, y, epochs=200, batch_size=5, verbose=1)

model.save('ai_chatbot_model.h5', ai_model)
print("AI Chatbot Model Built")