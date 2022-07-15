import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter.scrolledtext import *
import tkinter.filedialog
import pytesseract
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('stopwords')

import time
timestr = time.strftime("%Y%m%d-%H%M%S")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def nltk_summarizer(raw_text):
    stopWords = set(stopwords.words("english"))
    word_frequencies = {}
    for word in nltk.word_tokenize(raw_text):
        if word not in stopWords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / maximum_frequncy)

    sentence_list = nltk.sent_tokenize(raw_text)
    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    percent_summary = 100
    summary_length = int(len(sentence_list) * (int(percent_summary) / 100))

    # print summary
    from heapq import nlargest

    original_summary = nlargest(summary_length, sentence_scores, key=sentence_scores.get)
    final_summary = [word for word in original_summary]
    summary = ' '.join(final_summary)
    return summary

##spacy_summarization
import en_core_web_sm
nlp = en_core_web_sm.load()
from spacy.lang.en.stop_words import STOP_WORDS
def text_summarizer(raw_docx):
    raw_text = raw_docx
    docx = nlp(raw_text)
    stopwords = list(STOP_WORDS)
    # Build Word Frequency # word.text is tokenization in spacy
    word_frequencies = {}
    for word in docx:
        if word.text not in stopwords:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1

    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / maximum_frequncy)
    # Sentence Tokens
    sentence_list = [sentence for sentence in docx.sents]

    # Sentence Scores
    sentence_scores = {}
    for sent in sentence_list:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if len(sent.text.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word.text.lower()]
                    else:
                        sentence_scores[sent] += word_frequencies[word.text.lower()]

    percent_summary = 100
    summary_length = int(len(sentence_list) * (int(percent_summary) / 100))
    from heapq import nlargest
    original_summary = nlargest(summary_length, sentence_scores, key=sentence_scores.get)
    final_summary = [word for word in original_summary]
    summary = ' '.join(final_summary)
    return summary

# NLP Pkgs
from spacy_summarization import text_summarizer
from nltk_summarization import nltk_summarizer
from bs4 import BeautifulSoup
from urllib.request import urlopen
#sumy package
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

#Sumy
def sumy_summary(docx):
    parser=PlaintextParser.from_string(docx,Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document,7)
    summary_list = [str(sentence) for sentence in summary]
    resultforsumy = ' '.join(summary_list)
    return resultforsumy


# Structure and Layout
window = Tk()
window.title("Summaryzer G"
             "UI")
window.geometry("700x400")
window.config(background='black')

style = ttk.Style(window)
style.configure('lefttab.TNotebook', tabposition='wn', )

# TAB LAYOUT
tab_control = ttk.Notebook(window, style='lefttab.TNotebook')

tab1 = ttk.Frame(tab_control)
tab2 = ttk.Frame(tab_control)
tab3 = ttk.Frame(tab_control)
tab4 = ttk.Frame(tab_control)


# ADD TABS TO NOTEBOOK
tab_control.add(tab1, text=f'{"Home":^20s}')
tab_control.add(tab2, text=f'{"File":^20s}')
tab_control.add(tab3, text=f'{"URL":^20s}')
tab_control.add(tab4, text=f'{"Comparer ":^20s}')


label1 = Label(tab1, text='Summaryzer', padx=5, pady=5)
label1.grid(column=0, row=0)

label2 = Label(tab2, text='File Processing', padx=5, pady=5)
label2.grid(column=0, row=0)

label3 = Label(tab3, text='URL', padx=5, pady=5)
label3.grid(column=0, row=0)

label3 = Label(tab4, text='Compare Summarizers', padx=5, pady=5)
label3.grid(column=0, row=0)



tab_control.pack(expand=1, fill='both')


# Functions
def get_summary():
    raw_text = str(entry.get('1.0', tk.END))
    final_text = text_summarizer(raw_text)
    print(final_text)
    result = '\nSummary:{}'.format(final_text)
    tab1_display.insert(tk.END, result)


def save_summary():
    raw_text = str(entry.get('1.0', tk.END))
    final_text = text_summarizer(raw_text)
    file_name = 'yoursummary' + timestr + '.txt'
    with open(file_name, 'w') as f:
        f.write(final_text)
    result = '\nName of File: {} ,\nSummary:{}'.format(file_name,final_text)
    tab1_display.insert(tk.END,result)


# Clear entry widget
def clear_text():
    entry.delete('1.0', END)
def clear_display_result():
    tab1_display.delete('1.0', END)
# Clear Text  with position 1.0
def clear_text_file():
    displayed_file.delete('1.0', END)
# Clear Result of Functions
def clear_text_result():
    tab2_display_text.delete('1.0', END)
# Clear For URL
def clear_url_entry():
    url_entry.delete(0, END)
def clear_url_display():
    tab3_display_text.delete('1.0', END)
# Clear entry widget
def clear_compare_text():
    entry1.delete('1.0', END)
def clear_compare_display_result():
    tab4_display.delete('1.0', END)


# Functions for TAB 2 FILE PROCESSER
# Open File to Read and Process
def openfiles():
    file1 = tkinter.filedialog.askopenfilename(filetype=(("Text Files", ".txt"), ("All files", "*")))
    read_text = open(file1).read()
    displayed_file.insert(tk.END,read_text)

def imagefiles():
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
    a = pytesseract.image_to_string(tkinter.filedialog.askopenfilename(filetype=(("Image Files", ".png"), ("All files", "*"))))
    read_text = a
    displayed_file.insert(tk.END,read_text)

def get_file_summary():
    raw_text = displayed_file.get('1.0', tk.END)
    final_text = text_summarizer(raw_text)
    result = '\nSummary:{}'.format(final_text)
    tab2_display_text.insert(tk.END, result)

# Fetch Text From Url
def get_text():
    raw_text = str(url_entry.get())
    page = urlopen(raw_text)
    soup = BeautifulSoup(page)
    fetched_text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
    url_display.insert(tk.END, fetched_text)


def get_url_summary():
    raw_text = url_display.get('1.0', tk.END)
    final_text = text_summarizer(raw_text)
    result = '\nSummary:{}'.format(final_text)
    tab3_display_text.insert(tk.END, result)

# COMPARER FUNCTIONS

def use_spacy():
    raw_text = str(entry1.get('1.0', tk.END))
    final_text = text_summarizer(raw_text)
    print(final_text)
    result = '\nSpacy Summary:{}\n'.format(final_text)
    tab4_display.insert(tk.END, result)

def use_nltk():
    raw_text = str(entry1.get('1.0', tk.END))
    final_text = nltk_summarizer(raw_text)
    print(final_text)
    result = '\nNLTK Summary:{}\n'.format(final_text)
    tab4_display.insert(tk.END, result)

def process(text):
    tokens = word_tokenize(text)
    words = [w.lower() for w in tokens]

    porter = nltk.PorterStemmer()
    stemmed_tokens = [porter.stem(t) for t in words]

    # removing stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in stemmed_tokens if not w in stop_words]

    # count words
    count = nltk.defaultdict(int)
    for word in filtered_tokens:
        count[word] += 1
    return count;


def cos_sim(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


def getSimilarity(dict1, dict2):
    all_words_list = []
    for key in dict1:
        all_words_list.append(key)
    for key in dict2:
        all_words_list.append(key)
    all_words_list_size = len(all_words_list)

    v1 = np.zeros(all_words_list_size, dtype=np.int)
    v2 = np.zeros(all_words_list_size, dtype=np.int)
    i = 0
    for (key) in all_words_list:
        v1[i] = dict1.get(key, 0)
        v2[i] = dict2.get(key, 0)
        i += 1
    return cos_sim(v1, v2);


def check_similarity():
    raw_text = str(entry1.get('1.0', tk.END))
    dict1 = process(nltk_summarizer(raw_text))
    dict2 = process(sumy_summary(raw_text))
    dict3 = process(text_summarizer(raw_text))
    #print(final_text)
    print("Similarity between nltk and sumy is : ", getSimilarity(dict1, dict2))
    print("Similarity between nltk and spacy is : ", getSimilarity(dict1, dict3))
    print("Similarity between sumy and spacy is : ", getSimilarity(dict2, dict3))
    result1 = '\nSimilarity between nltk and sumy is :{}\n'.format(getSimilarity(dict1, dict2))
    result2 = '\nSimilarity between nltk and spacy is:{}\n'.format(getSimilarity(dict1, dict3))
    result3 = '\nSimilarity between sumy and spacy is:{}\n'.format(getSimilarity(dict2, dict3))
    tab4_display1.insert(tk.END, result1, result2, result3)

def use_sumy():
    raw_text = str(entry1.get('1.0', tk.END))
    final_text = sumy_summary(raw_text)
    print(final_text)
    result = '\nSumy Summary:{}\n'.format(final_text)
    tab4_display.insert(tk.END, result)



# MAIN NLP TAB
l1 = Label(tab1, text="Enter Text To Summarize")
l1.grid(row=1, column=0)

entry = ScrolledText(tab1, height=10, width=100)
entry.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

# BUTTONS home
button1 = Button(tab1, text="Reset", command=clear_text, width=12, bg='#03A9F4', fg='#fff')
button1.grid(row=4, column=0, padx=10, pady=10)

button2 = Button(tab1, text="Summarize", command=get_summary, width=12, bg='#03A9F4', fg='#fff')
button2.grid(row=4, column=1, padx=10, pady=10)

button3 = Button(tab1, text="Clear Result", command=clear_display_result, width=12, bg='#03A9F4', fg='#fff')
button3.grid(row=5, column=0, padx=10, pady=10)

button4 = Button(tab1, text="Save", command=save_summary, width=12, bg='#03A9F4', fg='#fff')
button4.grid(row=5, column=1, padx=10, pady=10)

# Display Screen For Result
tab1_display = ScrolledText(tab1, height=10, width=100)
tab1_display.grid(row=7, column=0, columnspan=3, padx=5, pady=5)

# FILE PROCESSING TAB
l1 = Label(tab2, text="Open File To Summarize")
l1.grid(row=1, column=1)

displayed_file = ScrolledText(tab2, height=10, width=100)  # Initial was Text(tab2)
displayed_file.grid(row=2, column=0, columnspan=3, padx=5, pady=3)

# BUTTONS FOR SECOND TAB/FILE READING TAB
b0 = Button(tab2, text="Open File", width=12, command=openfiles, bg='#c5cae9')
b0.grid(row=3, column=0, padx=10, pady=10)

b1 = Button(tab2, text="Reset ", width=12, command=clear_text_file, bg="#b9f6ca")
b1.grid(row=3, column=1, padx=10, pady=10)

b2 = Button(tab2, text="Summarize", width=12, command=get_file_summary, bg='blue', fg='#fff')
b2.grid(row=3, column=2, padx=10, pady=10)

b3 = Button(tab2, text="Clear Result", width=12, command=clear_text_result)
b3.grid(row=5, column=1, padx=10, pady=10)

b4 = Button(tab2, text="Close", width=12, command=window.destroy)
b4.grid(row=5, column=2, padx=10, pady=10)

b5 = Button(tab2, text="Image File", width=12, command=imagefiles, bg='blue', fg='#fff')
b5.grid(row=5, column=0, padx=10, pady=10)

# Display Screen
# tab2_display_text = Text(tab2)
tab2_display_text = ScrolledText(tab2, height=10, width=100)
tab2_display_text.grid(row=7, column=0, columnspan=3, padx=5, pady=5)

# Allows you to edit
tab2_display_text.config(state=NORMAL)

# URL TAB
l1 = Label(tab3, text="Enter URL To Summarize")
l1.grid(row=1, column=0)

raw_entry = StringVar()
url_entry = Entry(tab3, textvariable=raw_entry, width=50)
url_entry.grid(row=1, column=1)

# BUTTONS url tab
button1 = Button(tab3, text="Reset", command=clear_url_entry, width=12, bg='#03A9F4', fg='#fff')
button1.grid(row=4, column=0, padx=10, pady=10)

button2 = Button(tab3, text="Get Text", command=get_text, width=12, bg='#03A9F4', fg='#fff')
button2.grid(row=4, column=1, padx=10, pady=10)

button3 = Button(tab3, text="Clear Result", command=clear_url_display, width=12, bg='#03A9F4', fg='#fff')
button3.grid(row=5, column=0, padx=10, pady=10)

button4 = Button(tab3, text="Summarize", command=get_url_summary, width=12, bg='#03A9F4', fg='#fff')
button4.grid(row=5, column=1, padx=10, pady=10)

# Display Screen For Result
url_display = ScrolledText(tab3, height=10, width=100)
url_display.grid(row=7, column=0, columnspan=3, padx=5, pady=5)

tab3_display_text = ScrolledText(tab3, height=10, width=100)
tab3_display_text.grid(row=10, column=0, columnspan=3, padx=5, pady=5)

# COMPARER TAB
l1 = Label(tab4, text="Enter Text To Summarize")
l1.grid(row=1, column=0)

entry1 = ScrolledText(tab4, height=10, width=100)
entry1.grid(row=2, column=0, columnspan=3, padx=5, pady=3)

# BUTTONS
button1 = Button(tab4, text="Reset", command=clear_compare_text, width=12, bg='#03A9F4', fg='#fff')
button1.grid(row=4, column=0, padx=10, pady=10)

button2 = Button(tab4, text="SpaCy", command=use_spacy, width=12, bg='#03A9F4', fg='#fff')
button2.grid(row=4, column=1, padx=10, pady=10)

button3 = Button(tab4, text="Clear Result", command=clear_compare_display_result, width=12, bg='#03A9F4', fg='#fff')
button3.grid(row=5, column=0, padx=10, pady=10)

button4 = Button(tab4, text="NLTK", command=use_nltk, width=12, bg='#03A9F4', fg='#fff')
button4.grid(row=4, column=2, padx=10, pady=10)

button4 = Button(tab4, text="Similarity", command=check_similarity, width=12, bg='#03A9F4', fg='#fff')
button4.grid(row=8, column=1, padx=10, pady=10)

button4 = Button(tab4, text="Sumy", command=use_sumy, width=12, bg='#03A9F4', fg='#fff')
button4.grid(row=5, column=2, padx=10, pady=10)

variable = StringVar()
variable.set("SpaCy")

# Display Screen For Result
tab4_display = ScrolledText(tab4, height=15, width=100)
tab4_display.grid(row=7, column=0, columnspan=3, padx=5, pady=5)

tab4_display1 = ScrolledText(tab4, height=15, width=100)
tab4_display1.grid(row=9, column=0, columnspan=3, padx=5, pady=5)


window.mainloop()


