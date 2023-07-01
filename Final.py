
#all necessary library

from collections import defaultdict
import json
import csv
import re
import pandas as pd
import nltk
import string
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk import pos_tag
#from nltk.tokenize import word_tokenize
import math
from heapq import nlargest
import numpy as np
from collections import Counter



#all function

def remove_punctuation(df):
    df['title']=[re.sub('[^\w\s]+', '', s) for s in df['title'].tolist()]
    df['plot']=[re.sub('[^\w\s]+', '', s) for s in df['plot'].tolist()]
    return df

def tokenize(df):
    df['Tokenized_title']=df.apply(lambda row: nltk.word_tokenize(row['title']), axis=1)
    df['Tokenized_plot']=df.apply(lambda row: nltk.word_tokenize(row['plot']), axis=1)
    return df

def casefolding(token):
    len1=len(token)
    for i in range(0,len1):
        token[i]=token[i].casefold()
        
    return token


def stemming(token): 
    stemmer = PorterStemmer()
    len1=len(token)
    for i in range(0,len1):
        token[i]=stemmer.stem(token[i])
    return token


def Lemmatizer(token):
    len2=len(token)
    i=0
    lemmatizer = WordNetLemmatizer()
    for word, tag in pos_tag(token):
        wntag = tag[0].lower()
        if wntag=='j':
            wntag='a'
        wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
        if wntag is None:# not supply tag in case of None
            lemma = lemmatizer.lemmatize(word) 
        else:
            lemma = lemmatizer.lemmatize(word, pos=wntag)  
        token[i]=lemma
        i=i+1
    return(token)     


def detect_Stopword():
   
    df['title_plot']=df['clean_plot']+ df['clean_title']
    count=dict(df['title_plot'].explode().value_counts()[:27])
    df.drop(['title_plot'], axis=1)
    return  count
    




def delete_stopword_dataframe(list_stopwords,df_name,df):
    
    stopwords=list(list_stopwords.keys())
    list_new=[]
    for row in df[df_name]:
        list_new.append([x for x in row if x not in stopwords]) 
    df[df_name]=list_new
    
    
def delete_stopword_list(list_stopwords,List):
    stopwords=list(list_stopwords.keys())
    list_new=[]
    for element in List:
        if element not in stopwords:
            
            list_new.append(element)
            
    return list_new
    



def positionalindex(docs,d_i,positional_index):
    
    p_i=0 

    for row in docs: 

        for token in row:
            if token in positional_index:
                if d_i in positional_index[token]:
                    positional_index[token][d_i].append(p_i)
                else:
                    positional_index[token][d_i]=[]
                    positional_index[token]['freq']=positional_index[token]['freq']+1
                    positional_index[token][d_i].append(p_i)
  


            else:
                positional_index[token]=dict()
                positional_index[token]['freq']=1
                positional_index[token][d_i]=[]
                positional_index[token][d_i].append(p_i)
            p_i=p_i+1

        d_i= d_i + 1
        p_i=0
        
            
    

def insert_doc(doc,df,positional_index_t,positional_index_p): #doc contain title and plot
    list_temp=doc.split(',')
    title_new_doc=list_temp[0]
    len_temp=len(list_temp)
    for i in range(len_temp):
        plot_new_doc=list_temp[i]+plot_new_doc
    
    temp = {'title': [title_new_doc], 'plot': [plot_new_doc]}
    df_new = pd.DataFrame(data=temp)
    
    df_new=df_new.dropna(axis=0)
    df_new=remove_punctuation(df_new)
    
    df_new=tokenize(df_new)
   
    
    df_new['clean_title']=df_new['Tokenized_title'].apply(casefolding)
    df_new['clean_plot']=df_new['Tokenized_plot'].apply(casefolding)
   
    df_new['clean_title']=df_new['clean_title'].apply( stemming)
    df_new['clean_plot']=df_new['clean_plot'].apply(stemming)
   
    
    df_new['clean_title']=df_new['clean_title'].apply(Lemmatizer)
    df_new['clean_plot']=df_new['clean_plot'].apply(Lemmatizer)

    df_new.to_csv(file_loc,mode='a',columns=['title','plot'],index=False,header=False)

    remove_words=detect_Stopword()
    
    delete_stopword_dataframe(remove_words,'clean_plot',df_new)
    delete_stopword_dataframe(remove_words,'clean_title',df_new)


    last_index=df.index
    last_index=last_index[-1]
    positionalindex(df_new['clean_title'],last_index,positional_index_t)
    positionalindex(df_new['clean_plot'],last_index,positional_index_p)

    df=pd.concat([df, df_new])

    
    

def delete_from_indexing(docid,list_remove,positional_index_):
    for token in list_remove:
        if token in positional_index_:
            if docid in positional_index_[token]:
                del positional_index_[token][docid]
       
                positional_index_[token]['freq']=positional_index_[token]['freq']-1
                if positional_index_[token]['freq']==0:
                    del positional_index_[token]
  


def delete_doc(docid,df,positional_index_t,positional_index_p):
    #delete from positional index
    list_remove=df.iloc[docid]
    print(positional_index_t)
    # title
    delete_from_indexing(docid,list_remove['clean_title'],positional_index_t)
    #plot
    delete_from_indexing(docid,list_remove['clean_plot'],positional_index_p)   
    print(positional_index_t)
    #delete from df
   
    df=df.drop(docid)
    df=df.reset_index(drop=True)
    
    #delete from csvfile
    header = ['title', 'plot']
    df.to_csv(file_loc,columns=header,index=False, mode='w')
  

#little main
#dict_positionl_titles=dict(positionalindex('clean_title',0,))
#dict_positionl_plots=dict(positionalindex('clean_plot'))    
    

def TfIdf_for_document(term_freq):
    
    w = 1+math.log10(term_freq)
    return w



def TfIdf_for_Query(doc_freq ,term_freq ,N):
    
    idf = math.log10(N/doc_freq)
    tf = 1+math.log10(term_freq)
    w = tf*idf
    return w



def Ranking(dict_positionl_name,query,k_top):
    
    test=dict()
    score=dict()

    query=query.split()
    query={x:query.count(x) for x in query}
   

    for term in query:
        
        if term not in dict_positionl_name:
            continue
        test=dict_positionl_name[term]
       
        freq=test.pop('freq')
        
        
        for key in test:
            if key not in score:
                score[key] = score.get(key, 0)
            
            score[key]+=TfIdf_for_document(query[term])*TfIdf_for_Query(freq,len(test[key]),6000)
    
    return score
     
     

def Scoring_plot_title(ratio_title,query_title,query_plot,k_top):
    dict_title=dict()
    dict_plot=dict()
    scor_final=dict()
    dict_title = Ranking(positional_index_t,query_title,k_top)
    dict_plot = Ranking(positional_index_p,query_plot,k_top)
   
    
    
    list1=list(dict_title.keys())+list(dict_plot.keys())
    
    
 
    for doc in list1:
        if doc not in dict_plot:
            dict_plot[doc]=dict_plot.get(doc, 0)
            
        if doc not in dict_title:
            dict_title[doc]=dict_title.get(doc, 0)
        
        scor_final[doc]=dict_plot[doc]+(dict_title[doc]*ratio_title)
            

    # adding the values with common key
         
     
    result = nlargest(k_top, scor_final, key = scor_final.get)
    
    return result
    
def input_Query():
    print("Enter title Query :")
    Query_title=input() 
    print("Enter plot Query :")   
    Query_plot=input()
    print("title Ratio :")   
    ratio_title=float(input())
    print("K top :")  
    k_top=int(input())
    
    Query_title = Query_title.translate(str.maketrans('', '', string.punctuation))
    Query_title=Query_title.split()
    Query_title=casefolding(Query_title)
    Query_title=stemming(Query_title)
    Query_title=Lemmatizer(Query_title)
    Query_title=delete_stopword_list(remove_words,Query_title)
    query_title=" "
    query_title=query_title.join(Query_title)
    
   
    
        
    Query_plot = Query_plot.translate(str.maketrans('', '', string.punctuation))    
    Query_plot=Query_plot.split()
    Query_plot=casefolding(Query_plot)
    Query_plot=stemming(Query_plot)
    Query_plot=Lemmatizer(Query_plot)
    Query_plot=delete_stopword_list(remove_words,Query_plot)
    query_plot=" "
    query_plot=query_plot.join(Query_plot)
    
    
         
        
    
    return Scoring_plot_title(ratio_title,query_title,query_plot,k_top)
    


#main


file_loc ="D:\\College\\Term 6\\information retrieval\\origin.csv"
#file_loc = "E:\\University\\Term 6\\Information recovery\\test\\train.csv"


df = pd.read_csv(file_loc,usecols=['title','plot'])

# delete NAN
df=df.dropna(axis=0)

# remove punctuation
df['title']=[re.sub('[^\w\s]+', '', s) for s in df['title'].tolist()]
df['plot']=[re.sub('[^\w\s]+', '', s) for s in df['plot'].tolist()]

df.columns
# tokenize 
df['Tokenized_title']=df.apply(lambda row: nltk.word_tokenize(row['title']), axis=1)
df['Tokenized_plot']=df.apply(lambda row: nltk.word_tokenize(row['plot']), axis=1)



#case folding
df['clean_title']=df['Tokenized_title'].apply(casefolding)
df['clean_plot']=df['Tokenized_plot'].apply(casefolding)


# stemming title and plot 
df['clean_title']=df['clean_title'].apply(stemming)
df['clean_plot']=df['clean_plot'].apply(stemming)


# lemmatization title and plot
df['clean_title']=df['clean_title'].apply(Lemmatizer)
df['clean_plot']=df['clean_plot'].apply(Lemmatizer)


#stop word      
remove_words=detect_Stopword()
delete_stopword_dataframe(remove_words,'clean_plot',df)
delete_stopword_dataframe(remove_words,'clean_title',df)

    
#positional_index
positional_index_t=dict() 
positional_index_p=dict() 
positionalindex(df['clean_title'],0,positional_index_t)
positionalindex(df['clean_plot'],0,positional_index_p)


#menu
repeat=1
while(repeat):
    print("--------------------------------------------------------\n")
    print("------------ Information Retrieval System\n")
    print("------------ 1.Insert Document\n")
    print("------------ 2.Delet Document\n")
    print("------------ 3.Start Search\n")
    print("------------ 4.Exit\n")
    print("--------------------------------------------------------\n")
    ans=input()
    if ans=='1':
        new_doc=input("Please Enter Your Document:")
        insert_doc(new_doc,df,positional_index_t,positional_index_p)
    elif ans=='2':
        docid=int(input("Please Enter Document ID:"))
        delete_doc(docid,df,positional_index_t,positional_index_p)
    elif ans=='3':
        input_Query()
    elif ans=='4':
        repeat=0
    

        
        