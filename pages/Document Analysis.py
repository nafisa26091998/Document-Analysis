"""
!/usr/bin/python
-*- coding: utf-8 -*-
This file is subject to the terms and conditions defined in
file 'LICENSE.txt' which is part of this source code package.
__author__ = 'Nafisa Ali'

#################Module Information##############################################
#   Module Name         :   Document Analysis.py
#   Input Parameters    :   None
#   Output              :   None
#   Execution Steps     :   Streamlit application
#   Predecessor module  :   This module is a generic module
#   Successor module    :   NA
#   Last changed on     :   25th March 2023
#   Last changed by     :   Nafisa Ali
#   Reason for change   :   Code development
##################################################################################
"""

import streamlit as st
from annotated_text import annotated_text
from nltk.tokenize import sent_tokenize
import html
import argparse, json
from typing import Dict, List, Any
from tqdm import tqdm
from glob import glob
from time import sleep
from copy import deepcopy
import pandas as pd
import ast
from scipy.spatial import distance
from collections import namedtuple
import streamlit as st
from drug_named_entity_recognition import find_drugs
import time
import openai
import re
import Bio
import os
from Bio import Entrez
import sys
import json
from bs4 import BeautifulSoup
import datetime
from collections import namedtuple
from sentence_transformers import SentenceTransformer
sentence_trans = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
import json
import re
import traceback

Entrez.email = #Entrez email

page_bg_img = '''
<style>
.stApp {
background-image: url("https://st.depositphotos.com/4376739/60921/i/450/depositphotos_609213220-stock-photo-medical-abstract-background-health-care.jpg");
background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)
openai.api_key = #OpenAI key

article_metadata_tuple = namedtuple("Metadata", ['journal_title', 'subject_group', 'article_title', 'authors', 'editors', 'publication_date', 'abstract', 'keywords'])

article_data_tuple = namedtuple("ArticleData", ['sequence_no','section_title', 'section_text'])

citations_tuple = namedtuple("Citation", ["title", "authors", "year", "pubmed_id"], defaults=["", "", "", ""])

global list_of_PMCIDS

flag = 0

# '''
# Ada     - $0.0004/1000 tokens
# Babbage - $0.0005/1000 tokens
# Curie   - $0.002/1000 tokens
# Davinci - $0.02/1000 tokens
# '''

MODEL_COST_PER_TOKEN = {
    "text-ada-001": 0.0004 / 1000.0,
    "text-babbage-001": 0.0005 / 1000.0,
    "text-curie-001": 0.002 / 1000.0,
    "text-davinci-001": 0.02 / 1000.0,
    "text-davinci-002": 0.02 / 1000.0,
    "text-davinci-003": 0.02 / 1000.0
}

class ZeroShotPipeline():
    def __init__(self, prompt_template, domain_name, labels_list) -> None:
        self.load_template(prompt_template, domain_name, labels_list)
    
    def load_template(self, prompt_template, domain_name, labels_list):
        with open(r"Biomarker_pipeline\Biomarker_pipeline\prompts\{}.txt".format(prompt_template), 'r') as file:
            self.prompt_template = file.readlines()
            self.prompt_template = "".join(self.prompt_template)
        self.prompt_template = self.prompt_template.replace("[DOMAIN_NAME]", domain_name)
        self.prompt_template = self.prompt_template.replace("[LABELS_LIST]", json.dumps(labels_list))

    def generate_prompt(self, input_text):
        prompt_text = str(self.prompt_template)
        prompt_text = prompt_text.replace("[NEW_TEXT]", input_text)
        return prompt_text
    
    def generate_completion(self, prompt_text):
        try:
            response = openai.Completion.create(
                model = "text-davinci-003",
                prompt = prompt_text,
                max_tokens = 512,
                temperature = 0.0,
                top_p = 1,
                frequency_penalty = 0,
                presence_penalty = 0
            )
            completion_text = str(response['choices'][0]['text'].strip()) + "\n"
            estimated_cost = float(response['usage']['total_tokens'] * MODEL_COST_PER_TOKEN["text-davinci-003"])
        except Exception as error:
            print(f"Error thrown for prompt: \n {prompt_text} \n {error}")
            completion_text, estimated_cost = 'NA', 0.0
        return completion_text, estimated_cost
    
    def run_entity_extraction(self, paragraph):
        paragraph_biomakers = []
        inference_costs = []
#         for paragraph in tqdm(paragraphs, desc="Zeroshot GPT3 inference"):
        prompt_sample = self.generate_prompt(paragraph)
        completion_sample, completion_cost = self.generate_completion(prompt_sample)
        inference_costs.append(completion_cost)
        try:
            paragraph_biomakers.append(eval(completion_sample))
        except:
            paragraph_biomakers.append([completion_sample])

        return paragraph_biomakers, inference_costs


class FewShotPipeline():
    def __init__(self, prompt_template, domain_name, labels_list, example_inputs, example_outputs) -> None:
        
        self.load_template(prompt_template, domain_name, labels_list, example_inputs, example_outputs)
    
    def load_template(self, prompt_template, domain_name, labels_list, example_inputs, example_outputs):
        with open(r"Biomarker_pipeline\prompts\{}.txt".format(prompt_template), 'r') as file:
            self.prompt_template = file.readlines()
            self.prompt_template = "".join(self.prompt_template)

        self.prompt_template = self.prompt_template.replace("[DOMAIN_NAME]", domain_name)
        self.prompt_template = self.prompt_template.replace("[LABELS_LIST]", json.dumps(labels_list))

        example_placeholder = '''Input: [EXAMPLE_TEXT_NUMBER]\nOutput: [EXAMPLE_OUTPUT_NUMBER]'''
        example_datastring = ""
        for example_input, example_output in zip(example_inputs, example_outputs):
            example_datastring += f"Input: {example_input}\nOutput: {example_output}\n\n"
        self.prompt_template = self.prompt_template.replace(example_placeholder, example_datastring)

    def generate_prompt(self, input_text):
        prompt_text = str(self.prompt_template)
        prompt_text = prompt_text.replace("[NEW_TEXT]", input_text)
        return prompt_text
    
    def generate_completion(self, prompt_text):
        try:
            response = openai.Completion.create(
                model = "text-davinci-003",
                prompt = prompt_text,
                max_tokens = 512,
                temperature = 0.0,
                top_p = 1,
                frequency_penalty = 0,
                presence_penalty = 0
            )
            completion_text = str(response['choices'][0]['text'].strip()) + "\n"
            estimated_cost = float(response['usage']['total_tokens'] * MODEL_COST_PER_TOKEN["text-davinci-003"])
        except Exception as error:
            print(f"Error thrown for prompt: \n {prompt_text} \n {error}")
            completion_text, estimated_cost = 'NA', 0.0
        return completion_text, estimated_cost
    
    def run_entity_extraction(self, paragraphs):
        paragraph_biomakers = []
        inference_costs = []
#         for paragraph in tqdm(paragraphs, desc="Fewshot GPT3 inference"):
        prompt_sample = self.generate_prompt(paragraphs)
        completion_sample, completion_cost = self.generate_completion(prompt_sample)
        inference_costs.append(completion_cost)
        try:
            paragraph_biomakers.append(eval(completion_sample))
        except:
            paragraph_biomakers.append([completion_sample])
        
        return paragraph_biomakers, inference_costs
        

class Paragraph_Retrieval:
    def __init__(self, document_data, user_input):
        self.document_data = document_data
        self.load_model()
        self.query_embedding = self.get_embedding(user_input.strip())

    def load_model(self):
        self.model = sentence_trans
        self.model.max_seq_length = 512

    def process_document_data(self):

        data_dict = dict(zip(self.document_data.Section_Name, self.document_data.Content))
        return data_dict


    def get_embedding(self, text):
        embedding = self.model.encode(text, show_progress_bar=False)
        return embedding

    def similartiy_of_each_para(self, processed_para):

#         similarity_score_of_each_para = []
#         for i in processed_para:
#             if i != '':
#                 emb = self.get_embedding(i.strip())

#                 cosine_sim = 1 - distance.cosine(self.query_embedding, emb)
#                 similarity_score_of_each_para.append((i, cosine_sim))
#         return similarity_score_of_each_para
        if len(processed_para)!=0:
            emb = self.get_embedding(processed_para.strip())
            cosine_sim = 1 - distance.cosine(self.query_embedding, emb)
            return cosine_sim
#             print('First - ', cosine_sim)
            
        else:
            cosine_sim = 0        
            return cosine_sim

    def run_paragraph_retrieval(self):
  
        self.document_data['Cosine_score'] = [0]*len(self.document_data)
        self.document_data['Cosine_score'] = self.document_data['Section Text'].apply(lambda x:self.similartiy_of_each_para(x))
        final_df = self.document_data.sort_values(by=['Cosine_score'], ascending=False)
        final_df.reset_index(drop=True, inplace=True)
        return final_df        


class Reasoning:
    def __init__(self, assertion, data, num_paras):

        self.assertion = assertion
        self.data = data
       
        self.num_paras = num_paras
        self.final_output = {}
    
    def ready_paras(self):

        self.para_list = []
        context = "Content: \n"
        
        for i in range(self.num_paras):
            try:
                if len(context)/4 < 3000:
                    if len(self.data['Section Text'][i])!=0 or self.data['Section Text'][i]!= ' ':
                        context += '\npara-' + str(i) + ': ' + self.data['Content'][i]
                        self.para_list.append([self.data['Section Title'][i], self.data['Section Text'][i], self.data['Cosine_score'][i]])
            except:
                continue
        self.ranked_paras = self.para_list
        self.retrieved_paragraphs = context
    
    def reason_w_GPT3(self):

        self.ready_paras()
        curr_assertion = 'Refer to the Content and provide an answer to the above question not exceeding 2 lines'

        gpt3_prompt = self.retrieved_paragraphs + '\n' + 'Question: ' + self.assertion + '\n' + curr_assertion
#         print('total input tokens', len(gpt3_prompt)/4)
        
#         openai.api_key = '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'

        response1 = openai.Completion.create(
          model="text-davinci-003",
          prompt=gpt3_prompt,
          temperature=0,
          max_tokens=900,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0
        )
#         print(response1['choices'][0]['text'])
        self.gpt3_reasoning = response1['choices'][0]['text']
#         print('output tokens', len(self.gpt3_reasoning)/4)
        self.final_output['assertion'] = self.assertion
        self.final_output['retrieved_paras'] = self.ranked_paras
        self.final_output['reasoning'] = self.gpt3_reasoning
        return self.final_output    
    
  
    
    
def create_batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
            
def download_batch(batch):
    batch_response = Entrez.efetch(db="pmc", id=batch, retmode="xml")
    batch_xml= batch_response.read()
    soup = BeautifulSoup(batch_xml)
    for val in soup.select("pmc-articleset > article"):
        pmc_id = val.select_one("front > article-meta > article-id[pub-id-type='pmc']").text
        folder_path = f"../PMCDATA/PubMed/PMC{pmc_id}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        with open(folder_path + "/article.xml", 'w', encoding='utf-8') as f:
            f.write(str(val))
    


def extract_article_data(xml_file):

    with open(xml_file, 'r', encoding='utf-8') as f:
        xml_data = BeautifulSoup(f)
    pubmed_id = xml_data.select_one('article > front > article-meta > article-id[pub-id-type="pmid"]').text
    pmc_id = xml_data.select_one('article > front > article-meta > article-id[pub-id-type="pmc"]').text
    metadata = get_article_metadata(xml_data)
    data = get_article_data(xml_data)
    citations = get_article_citations(xml_data)
    ret_json = {}
    ret_json['metadata'] = metadata._asdict()
    citation_dict = [citation._asdict() for citation in citations]
    ret_json['citations'] = citation_dict
    xml_dir = os.path.dirname(xml_file)
    with open(os.path.join(xml_dir, "metadata.json"), 'w') as f:
        json.dump(ret_json, f, indent=4)

    with open(os.path.join(xml_dir, "data.json"), 'w') as f:
        data_dict = [section._asdict() for section in data]
        json.dump(data_dict, f, indent=4)



def get_article_metadata(xml_data):

    journal_title_tag = xml_data.select("article > front > journal-meta > journal-title-group > journal-title")
    journal_title = []
    for val in journal_title_tag:
        journal_title.append(val.text)
    #print("Journal Title: ", journal_title)

    article_metadata = xml_data.select_one("article > front > article-meta")

    subject_group = []
    subject_group_tag = article_metadata.select("article-categories > subj-group")
    #print(type(subject_group_tag))
    for val in subject_group_tag:
        for subtag in val.findAll():
            if not subtag.find() and subtag.text:
                subject_group.append(subtag.text)
    #print("Subject Groups : ", subject_group)

    article_title = article_metadata.select_one("title-group > article-title").text
    #print("Title: ", article_title)

    authors = []
    author_tags = article_metadata.select("contrib-group > contrib[contrib-type='author'] > name")
    for tag in author_tags:
        surname=tag.find("surname").text
        fname = tag.find("given-names").text
        #print("--> ", fname, surname)
        authors.append(f"{surname}, {fname}")
    #print("Authors : ", authors)

    editors = []
    editor_tags = article_metadata.select("contrib-group > contrib[contrib-type='editor'] > name")
    for tag in editor_tags:
        surname=tag.find("surname").text
        fname = tag.find("given-names").text
        #print("--> ", fname, surname)
        editors.append(f"{surname}, {fname}")
    #print("Editors : ", editors)

    pub_date_tag = article_metadata.select_one("article > front > article-meta > pub-date[pub-type='epub']")
    if not pub_date_tag:
        #pub_date_tag = article_metadata.select_one("article > front > article-meta > pub-date[pub-type='epub-ppub']")
        pub_date_tag = article_metadata.select_one("article > front > article-meta > pub-date")

    pub_date_day = pub_date_tag.find('day')
    pub_date_month = pub_date_tag.find('month')

    pub_date = f"{pub_date_tag.find('year').text}-{pub_date_month.text if pub_date_month else '01'}-{ pub_date_day.text if pub_date_day else '01' }"
    #print("Publication Date: ", pub_date)
    abstract = ""
    if article_metadata.find("abstract"):
        abstract = article_metadata.select_one("abstract").text
    #print("Abstract: ", abstract)
    keywords_tag = article_metadata.select("kwd-group > kwd")
    keywords = []
    for tag in keywords_tag:
        keywords.append(tag.text)
    #print("Keywords: ", keywords)
    return article_metadata_tuple(journal_title, subject_group, article_title, authors, editors, pub_date, abstract, keywords)


def get_article_data(xml_data):
    #Full Text:
    text_tags =  xml_data.select("article > body > sec")
    text_blocks = []
    article_metadata = xml_data.select_one("article > front > article-meta")
    article_title = article_metadata.select_one("title-group > article-title").text
    text_blocks.append(article_data_tuple(0, "Title", article_title))
    if article_metadata.find("abstract"):
        abstract = article_metadata.select_one("abstract").text
        text_blocks.append(article_data_tuple(0, "Abstract", abstract))

    for idx, tag in enumerate(text_tags):
        title = tag.find("title")
        paras = tag.find_all("p")

        section_text = "\n".join([para.text for para in paras])
        if title:
            section_title = title.text
        else:
            section_title = ""
        text_blocks.append(article_data_tuple(idx+1, section_title, section_text))

    return text_blocks

def get_article_citations(xml_data):
    # Citations
    citations = []
    citation_tags = xml_data.select("article > back > ref-list > ref")#"> ref > element-citation")
    for tag in citation_tags:
        try:
            select_tag = None
            if tag.find('element-citation'):
                select_tag = tag.select_one("element-citation")
            elif tag.find("mixed-citation"):
                select_tag = tag.select_one("mixed-citation")
            #print(select_tag)
            #print(select_tag)
            ct_title = select_tag.find("article-title").text

            author_tags = select_tag.select("person-group > name")
            ct_authors = []
            for author_tag in author_tags:
                ct_authors.append(f'{author_tag.find("surname").text}, {author_tag.find("given-names").text}')
            ct_year = None
            if select_tag.find("year"):
                ct_year = select_tag.find("year").text
            ct_pub_id = ""
            if select_tag.find("pub-id"):
                ct_pub_id_tag = select_tag.select_one("pub-id[pub-id-type='pmid']")
                if ct_pub_id_tag:
                    ct_pub_id = ct_pub_id_tag.text

            citation = citations_tuple(ct_title, ct_authors, ct_year, ct_pub_id)
            citations.append(citation)
        except:
            print(f"\t\tError while processing Citation")#{select_tag}")
            #traceback.print_exc()
    
    return citations    


def fetchPubmedData(therapy_area, min_date, max_date):
    
    drugs_to_search = therapy_area + ' '+ 'drugs'
    filters = {
    "term" : [drugs_to_search],
    "availability" : ["freetext"],
    "article_type" : ["case reports", 'clinical trial', 'clinical trial, phase i', 'clinical trial, phase ii','clinical trial, phase iii','clinical trial, phase iv', 'review', 'systematic review']}

    query = f"{' AND '.join(filters['term'])} AND ({ ' OR '.join([f'{v} [PTYP]' for v in filters['article_type']]) }) AND ({ ' OR '.join([ f'{v} [FILT]' for v in filters['availability']]) })"
    output = Entrez.esearch(db='pubmed', term=query, mindate=min_date, maxdate=max_date, retmode='json', retmax=10)
    # for record in output:
    #     print(record)
    d = json.load(output)
    matching_ids=d['esearchresult']['idlist']
    #Fetching Linked PMC Ids
    results = Entrez.read((Entrez.elink(
        dbfrom='pubmed', db='pmc', LinkName='pubmed_pmc', from_uid=matching_ids
    )))

    pubmed_pmc_map = {}
    matched_pmc_count = 0
    for result in results:
        #print(result)
        pmc_id = ''
        if result['LinkSetDb']:
            pmc_id = result['LinkSetDb'][0]['Link'][0]['Id']
            matched_pmc_count += 1
        pubmed_id = result['IdList'][0]
        pubmed_pmc_map[pubmed_id] = pmc_id
    
#     pmc_ids = [val for val in pubmed_pmc_map.values() if val]
#     pmc_article_folder = "../PMCDATA/PubMed/"
#     if not os.path.exists(pmc_article_folder):
#         os.makedirs(pmc_article_folder)
    
#     #data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # list of data 
#     batches = []
#     for batch in create_batch(pmc_ids, 10):
#         batches.append(batch)
        
#     for idx, batch in enumerate(batches[:1]):
#         download_batch(batch)            
    
#     for f in os.listdir(pmc_article_folder):
#         article_folder = os.path.join(pmc_article_folder, f)
#         if os.path.exists(article_folder) and os.path.isdir(article_folder):
#             xml_file = os.path.join(article_folder, "article.xml")
#             extract_article_data(xml_file)                    
        
#     all_files = os.listdir(r"C:\Users\na27078\PMCDATA\PubMed")
   
#     dict_data = {}
#     start_number_articles_read = 1
#     dir_loc = "C:\\Users\\na27078\\PMCDATA\\PubMed\\"

#     all_files = os.listdir("C:\\Users\\na27078\\PMCDATA\\PubMed\\")
#     all_files = all_files[start_number_articles_read:]
#     for i in all_files[start_number_articles_read:]:
#         dict_temp = {}
#         with open(dir_loc+i+"\\data.json") as f:
#             dict_temp["data"] = json.load(f)
#         with open(dir_loc+i+"\\metadata.json") as g:
#             dict_temp["metadata"] = json.load(g)
#         dict_data[i] = dict_temp
        
#     return dict_data
    #Nafisa making changes
    pmc_ids = ['PMC'+val for val in pubmed_pmc_map.values() if val]
    return pmc_ids
    
    
        

def get_article_data_(xml_data):

    article_data_tuple = namedtuple("ArticleData", ['sequence_no','section_title', 'section_text'])

    complete_article_text = []
    text_tags =  xml_data.select("sec ")
    text_blocks = []
    article_metadata = xml_data.select_one("front > article-meta")

    if article_metadata.find("abstract"):
        abstract = article_metadata.select_one("abstract").text
        text_blocks.append(article_data_tuple(0, "Abstract", abstract))
        complete_article_text.append(abstract)


    for idx in range(len(text_tags)):

        tag = text_tags[idx]

        str_tag = str(tag)

        sections = re.split('<sec(.*)>',str_tag)[1:]

        for sect_idx, section in enumerate(sections):

            section = BeautifulSoup(section)

            title = section.find("title")

            paras = section.find_all("p")

            section_text = "\n\n".join([para.text for para in paras])

            complete_article_text.append(section_text)

            if title:
                section_title = title.text
            else:
                section_title = ""

            text_blocks.append(article_data_tuple(sect_idx+1, section_title, section_text))

    section_num = []
    section_title = []
    section_text = []

    data_tuple = text_blocks.copy()

    for i in range(len(data_tuple)):
        section_num.append(data_tuple[i][0])
        section_title.append(data_tuple[i][1])
        section_text.append(data_tuple[i][2])

    df = pd.DataFrame({'Section Title':section_title,'Section Text':section_text})
    df = df[df['Section Text'].apply(lambda x: len(x)>50)]

    return df
    
def paper_aim(df):
    curr_assertion = 'What is the aim of the paper? If there are diseases mentioned, highlight it in the answer.'
    gpt3_prompt = 'Paper:' + '\n' +  df['Section Text'][0] + '\n' + df['Section Text'][1]+ '\n' + curr_assertion
    response1 = openai.Completion.create(
      model="text-davinci-003",
      prompt=gpt3_prompt,
      temperature=0,
      max_tokens=128,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    aim = response1['choices'][0]['text']
    return aim

def ad_dis(text, drug_name):
    curr_assertion = 'List the advantages and disadvantages of {} with respect to the Para, if it is mentioned in the Para above'.format(drug_name)
    gpt3_prompt = 'Para:' + '\n' +  text + '\n' + curr_assertion
    
    response1 = openai.Completion.create(
      model="text-davinci-003",
      prompt=gpt3_prompt,
      temperature=0,
      max_tokens=250,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    res = response1['choices'][0]['text']
    
    return res

def reasoning(text, drug_name):
    curr_assertion = 'Extract the sentence which describes the advantages and disadvantages of {} in Para'.format(drug_name)
    gpt3_prompt = 'Para:' + '\n' +  text + '\n' + curr_assertion
    
    response1 = openai.Completion.create(
      model="text-davinci-003",
      prompt=gpt3_prompt,
      temperature=0,
      max_tokens=250,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    reason = response1['choices'][0]['text']
    
    return reason

def compare_drugs(drug_list, text):
    t1 = 'Compare these drugs with respect to each other:'
    x = ''
    for drug in drug_list:
        x = x+ drug + ','
           
    
    curr_assertion = t1 + x + 'by referring to the above text. Based on the comparison, which drug is better?'
    gpt3_prompt = 'Paper:' + '\n' +  text + '\n' + curr_assertion
   
    response1 = openai.Completion.create(
      model="text-davinci-003",
      prompt=gpt3_prompt,
      temperature=0,
      max_tokens=1150,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    compare = response1['choices'][0]['text']
   
    return compare

def summarize_table(table_text):
    curr_assertion = 'Summarize the Table with complete analysis, including all values in natural language explaining it to a medical researcher. If there is an interesting observation, include it.'
    t = ''
    if len(table_text)<=3000:
        gpt3_prompt = 'Table:' + '\n' +  table_text + '\n' + curr_assertion
    else:
        try:
            table_text = table_text[:200]
        except:
            table_text = table_text
        gpt3_prompt = 'Table:' + '\n' +  table_text + '\n' + curr_assertion
    
    response1 = openai.Completion.create(
      model="text-davinci-003",
      prompt=gpt3_prompt,
      temperature=0,
      max_tokens=950,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    summary = response1['choices'][0]['text']
    
    return summary
    


def processData(therapy_area='Oncology', min_date=2016, max_date=2021):
#     data_dict = fetchPubmedData(therapy_area, min_date, max_date)
#     list_of_PMCIDS = list(data_dict.keys())
    list_of_PMCIDS = fetchPubmedData(therapy_area, min_date, max_date)
    x = ['-Select-']
    x.extend(list_of_PMCIDS)
    x = tuple(x)
    SelectedPMCID = st.selectbox(label = 'Select a PMCID to analyze..', options=x, index=0)
#     st.write('PMCID selected : ', SelectedPMCID)
    return SelectedPMCID

# with st.expander("Search parameters"):
st.title("Document Analysis")
st.caption('Know your document in minutes!')
empty1,col1,empty2,col2,empty3,col3 =st.columns([0.6,1.9,0.6,1.9,0.6, 1.9])

with empty1:

    st.empty()

with col1:
    st.write('Select a Therapy Area')
    SelectedTherapyArea = st.selectbox(
        ' ',
        ('-Select-', 'Oncology', 'HIV and AIDS', 'Neurodegenerative diseases', 'Immune-system diseases'),index =0,  label_visibility='collapsed')
    st.write(SelectedTherapyArea)

with empty2:

    st.empty()

with col2:
    st.write('Select a minimum date')
    SelectedMinDate = st.selectbox(
        ' ',
        ('-Select-', 2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010), index =0,  label_visibility='collapsed')
    st.write(SelectedMinDate)

with empty3:

    st.empty()

with col3:
    st.write('Select a Maximum date')
    SelectedMaxDate = col3.selectbox(
        ' ',
        ('-Select-', 2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2017, 2018, 2019, 2020, 2021), index =0,  label_visibility='collapsed')
    st.write(SelectedMaxDate)

# button = col2.button('Fetch Data!')

if SelectedTherapyArea!='-Select-' and SelectedMinDate!='-Select-' and SelectedMaxDate!='-Select-':
#         with st.spinner('Fetching relevant PUBMED articles..'): 
    SelectedPMCID = processData(SelectedTherapyArea, SelectedMinDate, SelectedMaxDate)

    if SelectedPMCID!='-Select-':
        pmc_request = Entrez.efetch(db='pmc', id=SelectedPMCID, retmode='xml')
        xml_data = BeautifulSoup(pmc_request.read())
        df = get_article_data_(xml_data)
        df = df.drop_duplicates()
        df.reset_index(drop=True, inplace =True)
        df['DrugNames'] = ['']*len(df)
        for i in range(len(df)):
            res = find_drugs(df['Section Text'][i].split(" "), is_ignore_case=True)
            drug_names = []
            for tup in res:
                drug_names.append(tup[0]['name'])
            df['DrugNames'][i] = list(set(drug_names))

        total_drug_names = []

        for i in range(len(df)):
            if len(df['DrugNames'][i])!=0:
                total_drug_names.extend(df['DrugNames'][i])

        paper_aim_ = paper_aim(df)
#         st.subheader('You have selected :')
        pmc_id_col, title_column = st.columns([2,2])
        with pmc_id_col:
            st.write('PMCID: {}'.format(SelectedPMCID))
            st.write('PUBMED Link: ' + 'https://www.ncbi.nlm.nih.gov/pmc/articles/{}/'.format(SelectedPMCID))
        
        with title_column:
            x = xml_data.select_one('article-title')
            st.write('Title: {}'.format(x.get_text()))
            
        
        st.markdown('**Aim of the paper:** {}'.format(paper_aim_))
        
        s = ''
        for i in list(set(total_drug_names)):
            s += "- " + i + "\n"
            
#         with st.container():
#             st.markdown('**Enter search query**')
#             query = st.text_input('Search parameter', '')
#             if len(query)!=0:
#                 passage_retrieve_obj = Paragraph_Retrieval(df, query)
#                 final_df = passage_retrieve_obj.run_paragraph_retrieval()
#                 r = Reasoning(query, final_df, 3) #retriever_param = 3
#                 final_output = r.reason_w_GPT3()

#                 answer = final_output['reasoning']
#                 to_use_ip_df = pd.DataFrame()
#                 to_use_ip_df['Section_Name'] = ['*']*len(final_output['retrieved_paras'])
#                 to_use_ip_df['Content'] = ['*']*len(final_output['retrieved_paras'])

#                 for _ in range(len(final_output['retrieved_paras'])):
#                     to_use_ip_df['Section_Name'][_] = final_output['retrieved_paras'][_][0]
#                     to_use_ip_df['Content'][_] = final_output['retrieved_paras'][_][1]

#                 to_use_ip_df = to_use_ip_df.groupby('Section_Name')['Content'].agg('\n'.join).reset_index()

#                 st.markdown('**Answer:**')
#                 st.markdown('\n')
#                 st.markdown(answer)
#                 st.markdown('**Referenced sections**')
#                 for i in range(len(to_use_ip_df)):
#                     st.markdown('**Section name - **'.format(to_use_ip_df['Section_Name'][i]))
#                     st.markdown('**Section content - **'.format(to_use_ip_df['Content'][i]))
#                     st.markdown('\n')
#             else:
#                 st.markdown('No queries asked..')
                
                
            
        with st.expander('Drug names mentioned in the paper'):
            st.markdown(s)
        
        univ_list = {}
        for i in range(len(df)):
            text = df['Section Text'][i]
            section = df['Section Title'][i]
            if i%5==0:
                time.sleep(10)
            if len(df['DrugNames'][i])!=0:
                univ_list[text] = {'drugs' : df['DrugNames'][i], 'section_name':section, 'meta': []} 
                for j in df['DrugNames'][i]:
                    res = ad_dis(text, j)
                    reason = reasoning(text,j)
#                     univ_list.append([section, text, j, res, reason])
                    univ_list[text]['meta'].append([j, res, reason])        
    
        with st.expander('Advantages and disadvantages of drugs with respect to the section content'):   
            for key, value in univ_list.items():
                text = key.replace('\n', '').strip()
                split_text = sent_tokenize(text)
                section = value['section_name']
                meta = value['meta']
                drug_names_list = []
                for m in meta:
                    drug_names_list.append(m[0])
                
                s = ''
                for i in drug_names_list:
                    s += "- " + i + "\n"
                
                st.markdown('**Section Name:** {}'.format(section))
                st.markdown('**Drug names mentioned in this section**')
                st.markdown(s)
                
                for m in meta:
                    note_indexes = []
                    drug_name = m[0]
                    res = m[1]
                    reason = m[2].replace('\n', '').strip()
                    split_reason = sent_tokenize(reason)
                    for sp in split_reason:
                        for i, v in enumerate(split_text):
                            if sp.lower() in v.lower() or sp.lower()==v.lower():
                                note_indexes.append(i)
                                break
                            elif v.lower() in sp.lower() or sp.lower()==v.lower():
                                note_indexes.append(i)
                                break
                    
                    
                    st.markdown('**Here is all you need to know about {}**'.format(drug_name))
                    st.write(res)
                    st.markdown('**Where was this answer picked from the section content?**')
                    tup = []
                    for i in split_reason:
                        annotated_text((i, 'Reason'))
                
                text_ = ''
                for index, value in enumerate(split_text):
                    if index in note_indexes:
                        x = '**:green['+value+']**'
                        text_ = text_ + x
                    else:
                        text_ = text_ + value
                
                
                st.markdown('\n')
#                 user_option = st.checkbox('Show section content?')
#                 if user_option:
                st.markdown('**Section Content**: {}'.format(text_))
    
    
        with st.expander('Comparing drugs with drugs'):
            for key, value in univ_list.items():
                text = key.replace('\n', '').strip()
                section = value['section_name']
                meta = value['meta']
                drug_names_list = []
                for m in meta:
                    drug_names_list.append(m[0])
                    
                if len(drug_names_list)>1:
                    compare = compare_drugs(drug_names_list, text)
                    
                    s = ''
                    for i in drug_names_list:
                        s += "- " + i + "\n"

                    st.markdown('**Section Name:** {}'.format(section))
                    st.markdown('**Drug names mentioned in this section**')
                    st.markdown(s)
                    st.markdown('\n')
                    st.markdown('**Drug comparison**: {}'.format(compare))
                    st.markdown('\n')
                    st.markdown('**Section Content**: {}'.format(text_))
                    
                else:
                    s = ''
                    for i in drug_names_list:
                        s += "- " + i + "\n"
                        
                    st.markdown('**Section Name:** {}'.format(section))
                    st.markdown('**Drug names mentioned in this section**')
                    st.markdown(s)
                    st.markdown('\n')
                    st.markdown('**Drug comparison**: There is no drug to compare.')
                    st.markdown('\n')
                    st.markdown('**Section Content**: {}'.format(text_))
                    
                
        with st.expander('Table analysis'):
            try:
                t = xml_data.prettify()
                html_data = html.unescape(t)
                df_list = pd.read_html(html_data)
                for i in range(len(df_list)):
                    table = df_list[i]
                    table_text = str(table.T.to_dict())
                    table_summary = summarize_table(table_text)
                    st.markdown('This is Table {}'.format(i))
                    st.dataframe(table)
                    st.markdown('\n')
                    st.markdown("Here's a brief summary of this table.")
                    st.markdown(table_summary)
            except:
                st.markdown("No Tables in this paper..")
            
                
        with st.expander('Biomarker detection'):
            
            #Zero shot
            
            zeroshot_prompter = ZeroShotPipeline(
                    prompt_template= "zeroshot_v1", 
                    domain_name="clinical", labels_list=["diagnostic biomarker", "prognostic biomarker"]
                    
            )
           
            fewshot_prompter = FewShotPipeline(
                    prompt_template="fewshot_v1", 
                    domain_name="clinical", labels_list=["diagnostic biomarker", "prognostic biomarker"], 
                    example_inputs=[
                        'A number of serum tumor markers have been studied in lung cancer, including carcino emryonic antigen (CEA), CA-125, CYFR A 21–21, chromogranin A, neuron-specific enolase (NSE), retinol-binding protein (RBP), α1-antitrypsin and squamous cell carcinoma antigen. However, no single blood test exists for lung cancer. CEA has been a widely studied tumor marker, and it has been reported that it is elevated in 0–38% of small cell lung cancer (SCLC) patients with limited disease and in 40–65% of those with extensive disease. It is estimated that CEA is elevated in 30–65% of patients with non-small-cell lung cancer (NSCLC). In a retrospective study of 153 NSCLC patients whose tumors were completely resected, Muley et al. found that patients who had an elevated CEA or CYFRA 21–21 level had lower overall survival rates than patients with normal levels [2]. In patients with stage IA disease, preoperative CEA levels above 5 ng/ml correlated with a poorer disease-free survival (22.2 vs 75%). Although this may identify a subset of early-stage patients who should be treated more aggressively, the number of patients who fall into this category is relatively small [3]. Both CEA and CA-125 appear to be lower in patients with early-stage disease compared with those with metastatic disease, and in a study that included 37 patients with advanced NSCLC, a decrease in these markers was found in those who had a documented radiologic response', 
                        'Mesenchymal stromal cells are non-hematopoietic multipotent cells that play an important role in MM development and progression via coordinating cellular migration and enhancing angiogenesis [42]. The stromal cells cultured with MM cell lines (U266/Lp-1) under hypoxic conditions were associated with a rise in α-smooth muscle actin, hypoxia-inducible factor (HIF)-2α and integrin-linked kinase proteins, indicating their role as potential angiogenic markers [43]. Interestingly, the inhibition of HIF-2α reduced both α-smooth muscle actin and integrin-linked kinase, resulting in attenuating angiogenesis in vitro. Mechanistically, the HIF-2α released by stromal cells promotes angiogenesis via increasing the attachment of Q-dot labeled cells and the excretion of angiogenic factors [43]. Along with the role of these angiogenic markers in the diagnosis/prognosis of MM, they represent possible drug targets.'
                    ], 
                    example_outputs=[
                        [{'T':'diagnostic biomarker', 'E':'CEA'}, {'T':'diagnostic biomarker', 'E':'CYFRA 21'}], 
                        [{'T':'diagnostic biomarker', 'E':'α-smooth muscle actin'}, {'T':'diagnostic biomarker', 'E':'hypoxia-inducible factor (HIF)-2α'}, {'T':'diagnostic biomarker', 'E':'integrin-linked kinase proteins'}]
                    ]
                    
            )
            df['Zero_shot_Biomarkers'] = ['']*len(df)
            df['Few_shot_Biomarkers'] = ['']*len(df)
            for i in range(len(df)):
                paragraphs = df['Section Text'][i]
                zero_shot_paragraph_biomakers, zero_shot_inference_costs = zeroshot_prompter.run_entity_extraction(paragraphs)
                df['Zero_shot_Biomarkers'][i] = zero_shot_paragraph_biomakers
            
            for i in range(len(df)):
                paragraphs = df['Section Text'][i]
                few_shot_paragraph_biomakers, few_shot_inference_costs = fewshot_prompter.run_entity_extraction(paragraphs)
                df['Few_shot_Biomarkers'][i] = zero_shot_paragraph_biomakers
                
                
            for i in range(len(df)):
                text = df['Section Text'][i]
                section = df['Section Title'][i]
                st.markdown('**Section Name:** {}'.format(section))
                st.markdown('**Biomarkers mentioned in this section**')
                st.markdown('**Zero shot results**')
                st.markdown(df['Zero_shot_Biomarkers'][i])
                st.markdown('\n')
                st.markdown('**Few shot results**')
                st.markdown(df['Few_shot_Biomarkers'][i])
                st.markdown('\n')
                st.markdown('**Section Content**: {}'.format(text))

                        
           
                
                         