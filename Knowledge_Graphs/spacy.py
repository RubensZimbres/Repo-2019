#!python -m pip install --upgrade pip --user
#!python -m spacy download pt_core_news_sm --user
#!python -m spacy download pt
#!python -m spacy download en_core_web_sm --user

import spacy
nlp = spacy.load('pt_core_news_sm')

doc = nlp("Alice foi para a escola essa manhã")

print([(w.text, w.pos_) for w in doc])

from spacy import displacy
displacy.render(doc, style='dep')

[token.pos_ for token in doc]

[token.tag_ for token in doc]

[token.dep_ for token in doc]

[token.head.text for token in doc]

[(ent.text, ent.label_) for ent in doc.ents]

[sent.text for sent in doc.sents]

[chunk.text for chunk in doc.noun_chunks]

spacy.explain("ADP")

displacy.render(doc, style="ent")

doc.similarity(doc)

doc.vector

doc.vector_norm

from spacy.tokens import Doc, Token, Span
Token.set_extension("é_lugar", default=False)

doc[6]._.é_lugar = True

from spacy.matcher import Matcher

matcher = Matcher(nlp.vocab)

pattern = [{"LOWER": "new"}, {"LOWER": "york"}]
matcher.add('CITIES', None, pattern)
doc = nlp("I live in New York")
matches = matcher(doc)
for match_id, start, end in matches:
     span = doc[start:end]
     print(span.text)

'''Token Patterns'''

# "love cats", "loving cats", "loved cats"
pattern1 = [{"LEMMA": "love"}, {"LOWER": "cats"}]
# "10 people", "twenty people"
pattern2 = [{"LIKE_NUM": True}, {"TEXT": "people"}]
# "book", "a cat", "the sea" (noun + optional article)
pattern3 = [{"POS": "DET", "OP": "?"}, {"POS": "NOUN"}]



for token in doc:
  print(token.text, "...", token.dep_)
  
import re
import pandas as pd
import bs4
import requests
import spacy
from spacy import displacy
from spacy.matcher import Matcher 
from spacy.tokens import Span 
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

nlp = spacy.load('pt_core_news_sm')

sentences = pd.read_csv("Alice_pt.csv",sep=';')

sentences.columns=['index','sentence']
sentences=sentences.drop(['index'],axis=1).reset_index()
sentences

def extract_entities(sent):
    entity1 = ""
    entity2 = ""

    previous_token_dep = ""   
    previous_token_text = ""   

    prefix = ""
    modifier = ""

    for token in nlp(sent):
        if token.dep_ != "punct":
            if token.dep_ == "compound":
                prefix = token.text
                if previous_token_dep == "compound":
                    prefix = previous_token_text + " "+ token.text
          
        if token.dep_.endswith("mod") == True:
            modifier = token.text
            if previous_token_dep == "compound":
                modifier = previous_token_text + " "+ token.text
          
        if token.dep_.find("subj") == True:
            entity1 = modifier +" "+ prefix + " "+ token.text
            prefix = ""
            modifier = ""
            previous_token_dep = ""
            previous_token_text = ""      
    
        if token.dep_.find("obj") == True:
            entity2 = modifier +" "+ prefix +" "+ token.text
            
        previous_token_dep = token.dep_
        previous_token_text = token.text


    return [entity1.strip(), entity2.strip()]

extract_entities("Alice foi para a escola essa manhã")



entity_pairs = []

for i in tqdm(sentences["sentence"]):
    entity_pairs.append(extract_entities(i))

def get_relationship(sent):

    doc = nlp(sent)

    matcher = Matcher(nlp.vocab)

    pattern = [{'DEP':'ROOT'}, 
            {'DEP':'prep','OP':"?"},
            {'POS':'ADJ','OP':"?"},
            {'DEP':'agent','OP':"?"}] 

    matcher.add("matching_1", None, pattern) 

    matches = matcher(doc)
    limit = len(matches) - 1

    span = doc[matches[limit][1]:matches[limit][2]] 

    return(span.text)
  
relations = [get_relationship(i) for i in tqdm(sentences['sentence'])]


pd.Series(relations).value_counts()

source = [i[0] for i in entity_pairs]

target = [i[1] for i in entity_pairs]

dataframe = pd.DataFrame({'source':source, 'target':target, 'edge':relations})

GRAPH=nx.from_pandas_edgelist(dataframe, "source", "target", 
                          edge_attr=True, create_using=nx.MultiDiGraph())
plt.figure(figsize=(9,9))
pos = nx.spring_layout(GRAPH)
nx.draw(GRAPH, with_labels=True, node_color='lawngreen', edge_cmap=plt.cm.Blues, pos = pos)
plt.show()


GRAPH=nx.from_pandas_edgelist(dataframe[dataframe['source']=="Marcos"], "source", "target", 
                          edge_attr=True, create_using=nx.MultiDiGraph())
plt.figure(figsize=(9,9))
pos = nx.spring_layout(GRAPH, k = 0.5) 
nx.draw(GRAPH, with_labels=True, node_color='lawngreen', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)
plt.show()
