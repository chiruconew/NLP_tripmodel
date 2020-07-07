import pandas as pd
import gensim 
import pickle
from typing import List
import spacy

nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])
from src.features.tokenize import tokenize
from src.features.dictionary import create_dictionary
from src.data.prepare_data import read_sample
from src.features.clean import clean_up_text
from gensim.models import CoherenceModel


def lda_model(text: pd.DataFrame): # -> List[gensim.LdaModel,gensim.CoherenceModel,float]:
    doc       = clean_up_text(text)
    lemma     = tokenize(doc)
    id2word, corpus = create_dictionary(lemma)
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=id2word,num_topics=5, random_state=100,
                                    update_every=1,chunksize=100,passes=10,alpha='auto',per_word_topics=True)

    
    with open("models/model.pkl", "wb") as output_file:
        pickle.dump(lda_model, output_file)
    
    return(lda_model)


def load_doc() -> pd.DataFrame:
    return read_sample()