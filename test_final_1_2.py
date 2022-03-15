#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[2]:


import streamlit as st
import joblib 
import pandas as pd
import string
from stop_words import get_stop_words
import spacy

from PIL import Image
import urllib.request
from PIL import Image
  


# In[3]:


nlp = spacy.load("en_core_web_sm")


# In[4]:


def trat_data_1(data_frame):
    if (data_frame["spoken_words"].str.split("-", n = 1, expand = True)[1][0] is None):
        data_frame["spoken_words"] = data_frame["spoken_words"].str.split("-", n = 1, expand = True)[0]
    else:
        data_frame["spoken_words"] = data_frame["spoken_words"].str.split("-", n = 1, expand = True)[1]
    return(data_frame)


# In[5]:


data = pd.read_csv('simpsons_train.csv')
data = trat_data_1(data)
data_2 = pd.read_csv("simpsons_test.csv")


# In[6]:


classifier = joblib.load("simpsons_balanced.pkl")


# In[7]:


def preprocessing(text):
    text = text.lower()
    punctuation = string.punctuation
    stop_words = get_stop_words('english')
    
    file = nlp(text)
    words = []
    for i in file:
        words.append(i.lemma_ )
    

    words_2 = []
    words_2 = [word for word in words if word  not in stop_words and word not in punctuation]
    words_2 = ' '.join([str(element) for element in words_2 if not element.isdigit()])

    return words_2


# In[8]:


def show(result):
        if (result == 'C. Montgomery Burns' ):
            urllib.request.urlretrieve(
  'https://static.wikia.nocookie.net/simpsons/images/6/6a/Mr_Burns_evil.gif/revision/latest/scale-to-width-down/481?cb=20100702150413',
   "gfg.png")
  
            img = Image.open("gfg.png")
        if(result =='Marge Simpson'):
             urllib.request.urlretrieve(
  'data:image/png;base64,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',
   "gfg.png")
  
             img = Image.open("gfg.png")
        if(result =='Lisa Simpson'):
             urllib.request.urlretrieve(
  'https://static1.purebreak.com.br/articles/5/98/80/5/@/393939-lisa-simpson-e-bissexual-em-os-simpsons-950x0-1.jpg',
                 "gfg.png")
  
             img = Image.open("gfg.png")
        if (result == "Homer Simpson"):
            urllib.request.urlretrieve(
  'https://img.elo7.com.br/product/original/32B11C6/homer-simpson-e-sua-cerveja-100-vetorizados-simpson.jpg',
                 "gfg.png")
            img = Image.open("gfg.png")
            
        if(result == "Milhouse Van Houten"):
            urllib.request.urlretrieve(
  'https://desenhos.band.uol.com.br/wp-content/uploads/2017/11/milhouse-van-houten-screammin.jpg',
                 "gfg.png")
            img = Image.open("gfg.png")
        if(result == "Moe Szyslak"):
            urllib.request.urlretrieve(
  'https://static.wikia.nocookie.net/simpsons/images/3/32/Moe_Szyslak_avat0.png/revision/latest?cb=20170314002144&path-prefix=pt',
                 "gfg.png")
            img = Image.open("gfg.png")
        if(result == "Bart Simpson"):
            urllib.request.urlretrieve(
  'https://i.pinimg.com/originals/98/b3/71/98b371cd43ead599d22bb40590df8287.jpg',
                 "gfg.png")
            img = Image.open("gfg.png")
        if(result == "Grampa Simpson"):
            urllib.request.urlretrieve(
  'https://i.pinimg.com/originals/a2/4a/cf/a24acf915859ab9a0ea58693d5dec083.jpg',
                 "gfg.png")
            img = Image.open("gfg.png")
        if(result == "Seymour Skinner"):
            urllib.request.urlretrieve(
  'data:image/jpeg;base64,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',
                 "gfg.png")
            img = Image.open("gfg.png")
        if(result == "Chief Wiggum"):
            urllib.request.urlretrieve("https://static.wikia.nocookie.net/simpsons/images/f/f0/Chief-wiggum-no-cap.png/revision/latest?cb=20210305053652",
  "gfg.png")
            img = Image.open("gfg.png")
        if(result == "Ned Flanders"):
            urllib.request.urlretrieve(
  'https://sm.ign.com/ign_pt/news/t/there-is-a/there-is-a-ned-flanders-themed-metal-band_gfun.jpg',
                 "gfg.png")
            img = Image.open("gfg.png")
        if(result == "Krusty the Clown"):
            urllib.request.urlretrieve(
  'https://i.pinimg.com/originals/b1/9b/c2/b19bc29fb72e2f4b9f11d8daa5290a86.jpg',
                 "gfg.png")
            img = Image.open("gfg.png")
            
            
    
        
        return(img)


# In[9]:


def pred_1(words,pipe):
    words = preprocessing(words)
    y_pred = pipe.predict([words]) 
    name = data.loc[(data.character_id == y_pred[0]),"character_name"].reset_index(drop= True)
    return(name[0])


# In[17]:


def repetidos_ou_pred(text):
    result= ""
    if(text in str (data["spoken_words"]) ):
        result = data.loc[(data.spoken_words == text),["character_name"]]
        
    if(text in str(data_2["spoken_words"])):
        result = data_2.loc[(data_2.spoken_words == text),["character_name"]]
        
    else: 
        prep = preprocessing(text)
        result = pred_1(prep,classifier)
    st.success('Who says?\n\n {}'.format(result))
    st.image(show(result))  
    return(result)
    


# In[18]:


def main():       
    st.header("Simpsons says")
    text = st.text_area("Type a text")
    result =""
    if st.button("Predict"): 
        repetidos_ou_pred(text)
        
if __name__=='__main__': 
    main()         


# In[19]:





# In[ ]:





# In[ ]:




