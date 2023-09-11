#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install openai')
import fitz
import math
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import openai

openai.api_key = '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'

Lecture = input("강의명을 입력하세요: ")
Keyword = input("키워드: ")

prompt_text = "{}와 관련된 {}문제를 객관식으로 사용자가 선택할 수 있게 문제를 4지선다로 만들어주고 답도 알려줘.".format(Lecture, Keyword)

max_tokens = 3000

response = openai.Completion.create(
  engine="text-davinci-003",
  prompt=prompt_text,
  max_tokens=max_tokens,
)

generated_text = response.choices[0].text.strip()

if len(generated_text) < max_tokens:
    print(generated_text)
else:
    print(generated_text[:max_tokens])



# In[1]:


get_ipython().system('pip install rake-nltk')


# In[2]:


get_ipython().system('pip install PyMuPDF')


# In[8]:


def extract_text_from_pdf(file_path):
    """
    PDF 파일에서 각 페이지의 텍스트 추출
    :param file_path: PDF 파일 경로
    :return: 추출된 텍스트의 리스트, 각 원소는 (페이지 번호, 텍스트) 형태의 튜플
    """
    # PDF 파일 열기
    pdf_file = fitz.open(file_path)
    
    # 텍스트 추출
    extracted_text = []
    for page_number in range(pdf_file.page_count):
        page = pdf_file.load_page(page_number)
        text = page.get_text("text")
        extracted_text.append((page_number + 1, text))  # 페이지 번호는 0-based이므로 1을 더해줌
    
    # PDF 파일 닫기
    pdf_file.close()
    
    return extracted_text


# PDF 파일 경로 지정
pdf_file_path = "C:\네트워크.pdf"

# 텍스트 추출 함수 호출
text_list = extract_text_from_pdf(pdf_file_path)

# 각 페이지마다 텍스트 출력
for page_number, text in text_list:
    print("페이지 번호: ", page_number)
    print("텍스트: ", text)
    print("="*30)


# In[15]:


import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_keywords_from_pdf(file_path, num_keywords=5):
    """
    PDF 파일에서 각 페이지의 중요한 키워드를 TF-IDF를 사용하여 추출
    :param file_path: PDF 파일 경로
    :param num_keywords: 추출할 키워드의 수
    :return: 각 페이지의 키워드의 리스트
    """
    # 텍스트 추출 함수 사용
    text_list = extract_text_from_pdf(file_path)

    # 각 페이지의 텍스트만 추출하여 리스트로 만듬
    docs = [text for _, text in text_list]

    # TF-IDF 설정 및 계산
    vectorizer = TfidfVectorizer(max_df=0.85, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names()

    # 각 페이지의 키워드 추출
    keywords_list = []
    for doc_index, doc in enumerate(docs):
        feature_index = tfidf_matrix[doc_index, :].nonzero()[1]
        tfidf_scores = zip(feature_index, [tfidf_matrix[doc_index, x] for x in feature_index])
        sorted_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
        keywords = [feature_names[i] for i, _ in sorted_scores[:num_keywords]]
        keywords_list.append(keywords)

    return keywords_list

# PDF 파일 경로 지정
pdf_file_path = "C:\\네트워크.pdf"

# 키워드 추출 함수 호출
keywords_per_page = extract_keywords_from_pdf(pdf_file_path)

# 각 페이지마다 키워드 출력
for page_number, keywords in enumerate(keywords_per_page, start=1):
    print(f"페이지 번호: {page_number}")
    print("키워드:", ", ".join(keywords))
    print("="*30)

