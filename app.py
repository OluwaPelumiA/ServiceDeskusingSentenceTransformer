#load necessary libraries
from flask import Flask, jsonify, request 
from flask_restful import Resource, Api 
import pandas as pd
import numpy as np
import re
import gensim 
from gensim.parsing.preprocessing import remove_stopwords
from gensim import corpora
import pprint
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import os
import sqlite3
import csv
import time
import datetime
import random
from sentence_transformers import LoggingHandler
from sentence_transformers import SentenceTransformer
import logging

  
# creating the flask app 
app = Flask(__name__) 

@app.route("/name", methods=["POST"])
# creating an API object 
#api = Api(app)

def ServiceDesk(): 
    if request.method=='POST':
        posted_data = request.get_json()
        original_question = posted_data['issue']
        #Sample Dataset gotten from the web
        txt = "./faq.txt"
        my_file = open(txt, "r")
        content = my_file. read()
        hold_lines = []
        holdLines2 = []
        with open(txt,'r') as text_file:
            for row in text_file:
                red= row
                if '?' in red:

                    hold_lines.append(red)
                else:
                    holdLines2.append(red)
        g = holdLines2[0:30]
        data ={"ISSUES":hold_lines,"Resolution":g}
        df = pd.DataFrame(data)
        new_f = df.replace('\\n',' ', regex=True)
        new_f.to_csv("newFile.csv", index=False)
        df=pd.read_csv("newFile.csv") # Convert to Dataframe


        #Create dummy Database
        conn = sqlite3.connect('knowledgeBases.db')
        c = conn.cursor()
        def createDB():
            c.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='knowledgeBass' ''')

            #if the count is 1, then table exists
            if c.fetchone()[0]==1 : 
                print('The table already exists.')
            else :
                c.execute("CREATE TABLE knowledgeBass (categoryOfIssues TEXT, priorityOfproblem TEXT, date INTEGER, issues TEXT, resolution TEXT, ticket_Id VARCHAR, issusesReportedDescription TEXT)")  

                print('just created a table.')
            conn.commit()
        createDB()
        #Data Creation to Populate database  
        ticket_number_lists = []
        def generate_ticket_id(no_of_people):
            dept = ['DS','SE','ACC','HR']
            i = 0
            for i in range(no_of_people):
                department = random.choice(dept)
                ticket_number_lists.append(department + str(random.randint(12000,99999)))
            return ticket_number_lists

        generate_ticket_id(30)
        def clean_words(sentence, stopwords=False):

            sentence = sentence.lower().strip()
            sentence = re.sub(r'[^a-z0-9\s]', '', sentence)

            if stopwords:
                 sentence = remove_stopwords(sentence)

            return sentence

        def get_cleaned_words(df,stopwords=False):    
            sents=df[["ISSUES"]];
            cleaned_word=[]

            for index,row in df.iterrows():
                #print(index,row)
                cleaned=clean_words(row["ISSUES"],stopwords)
                cleaned_word.append(cleaned)
            return cleaned_word

        cleaned_word=get_cleaned_words(df,stopwords=True)

        catOfIssues = ['Networking','Hardware','Operating System','Others']
        priOfIssues = ['High','Low','Medium']
        currentTime = time.time()
        dates = datetime.datetime.fromtimestamp(currentTime).strftime('%Y-%m-%d %H:%M:%S')
        issues = df['ISSUES']
        #print(len(issues))
        Resolution = df['Resolution']
        issusesReportedDescription = cleaned_word
        ticket_id = ticket_number_lists
        for each in range(len(df)):
            Issue = df['ISSUES'][each]
            Resolutn = df['Resolution'][each]
            priority = random.choice(priOfIssues)
            category = random.choice(catOfIssues)
            ticket = ticket_id[each]
            description = issusesReportedDescription[each]
            date=dates
            c.execute("INSERT INTO knowledgeBasess (categoryOfIssues, priorityOfproblem, date, issues, resolution, ticket_Id, issusesReportedDescription) VALUES (?,?,?,?,?,?,?)",
                                  (category, priority,date,Issue,Resolutn,ticket,description))
            conn.commit()

        #Check to see whether the data has been correctly inserted
        c.execute('''SELECT * fROM KnowledgeBasess;''')
        print(c.fetchone())

        #Export data into csv,although not necessary, you can skip this part


        print ("........Exporting sql data into CSV............")
        c.execute("SELECT * FROM KnowledgeBasess")
        with open("Services_DeskData.csv", "w") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter="\t")
            csv_writer.writerow([i[0] for i in c.description])
            csv_writer.writerows(c)

        dirpath = os.getcwd() + "/Services_DeskData.csv"
        print ("Data exported Successfully into {}".format(dirpath))

        #convert the sql data to dataframe for further use
        nw_df = pd.read_sql("SELECT * FROM KnowledgeBasess",conn)
        new_df = nw_df[['issues','resolution']]

        #Clean sentences and remove all stop words and tabs
        def clean_sentence(sentence, stopwords=False):

            sentence = sentence.lower().strip()
            sentence = re.sub(r'[^a-z0-9\s]', '', sentence)

            if stopwords:
                 sentence = remove_stopwords(sentence)

            return sentence

        def get_cleaned_sentences(new_df,stopwords=False):    
            sents=new_df[["issues"]];
            cleaned_sentences=[]

            for index,row in new_df.iterrows():
                #print(index,row)
                cleaned=clean_sentence(row["issues"],stopwords)
                cleaned_sentences.append(cleaned)
            return cleaned_sentences

        cleaned_sentences=get_cleaned_sentences(new_df,stopwords=True)
        #print(cleaned_sentences)

        print("\n")

        cleaned_sentences_with_stopwords=get_cleaned_sentences(new_df,stopwords=False)
        #print(cleaned_sentences_with_stopwords)

        original_question = original_question
        question=clean_sentence(original_question,stopwords=False)
        def retrieveAndPrintFAQAnswer(question_embedding,sentence_embeddings,FAQdf,sentences):
            max_sim=-1;
            index_sim=-1;
            for index,faq_embedding in enumerate(sentence_embeddings):

                sim=cosine_similarity(faq_embedding,question_embedding)[0][0];
                print(index, sim, sentences[index])
                if sim>max_sim:
                    max_sim=sim
                    index_sim=index

            print("\n")
            print("Question: ",question)
            print("\n");
            print("Retrieved: ",FAQdf.iloc[index_sim,0]) 
            print(FAQdf.iloc[index_sim,1])
            issues = question
            similar_query = FAQdf.iloc[index_sim,0] 
            suggested_resolution = FAQdf.iloc[index_sim,1]
            result = question+ "is:"+ suggested_resolution

            return result


        #### Just some code to print debug information to stdout
        np.set_printoptions(threshold=100)

        logging.basicConfig(format='%(asctime)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=logging.INFO,
                            handlers=[LoggingHandler()])
        #### /print debug information to stdout



        # Load pre-trained Sentence Transformer Model (based on DistilBERT). It will be downloaded automatically
        model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

        # Embed a list of sentencesh
        sentences = cleaned_sentences_with_stopwords

        sent_bertphrase_embedding=[];

        # The result is a list of sentence embeddings as numpy arrays
        for sent in sentences:
            sent_bertphrase_embedding.append(model.encode([sent]));


        question_embedding=model.encode([question]);

        Trial = retrieveAndPrintFAQAnswer(question_embedding,sent_bertphrase_embedding,new_df,sentences)
        return jsonify({'resolution': Trial}) 
#api.add_resource(ServiceDesk, '/home/<string:original_question>') 
  
  
# driver function 
if __name__ == '__main__': 
  
    app.run(host='0.0.0.0', port=8898,debug = True) 
