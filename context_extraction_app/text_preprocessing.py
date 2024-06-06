#######################################################
#### text_preprocessing
#### This module divides the pdf text into sentences
#### and process text for further analysis
#######################################################

# Import Packages
import spacy
import re
from langchain_text_splitters.character import RecursiveCharacterTextSplitter


def pdfreader_text_extraction(path: str, package = "pypdfium2") -> str:
    '''
    pdfreader_text_extraction extract all the text from the PDF using different packages 
    Reference: there is a PDF extractor benchmark by py-pdf (https://github.com/py-pdf/benchmarks?tab=readme-ov-file)

    Parameter:
        path (str) : the file address of the file
        package (str) : the name of the package to use. Choices are 'pypdf2' of the PdfReader.


    Return:
        text (str) : the text extracted from the pdf file
    '''
    # Read the document
    document = open(path, 'rb')
    text = ""

    if package == "pypdf2": 

        # Import PyPDF2
        from PyPDF2 import PdfReader

        # Read pdf
        pdf = PdfReader(document)

        # Extract Text
        for page in pdf.pages:
            text += page.extract_text()
            
    if package == "pypdfium2":

        # Import PDFium
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(document)

        for i in range(len(pdf)):
            page = pdf.get_page(i)
            text_per_page = page.get_textpage()
            text += text_per_page.get_text_range()
            text += "\n"

            [f.close() for f in (text_per_page, page)]

    # Close Document
    document.close()

    return text

def text_cleaning(text:str) -> str:
    '''
    Remove escape pattern (\n, \xa0, \r, \s) from the text

    Parameter:
        text (str): the original text to clean
    
    Return:
        text (str) : the text after removing the escape pattern
    '''

    while re.search("\\n", text):
        # Replace all line change into space to prevent it from distracting sentence recognition.
        text = re.sub("\\n", " ", text)

    while re.search("\\xa0", text):
        # Replace all the \xa0 (No Breakspace) to a space
        text = re.sub("\\xa0", " ", text)

    while re.search("\\r", text):
        # Remove all \r
        text = re.sub("\\r", "", text)

    text = re.sub("\s{2,}", " ", text)
    
    return text

def sentence_division(text: str, return_str = True) -> list:
    '''
    sentence_division applies SpaCy pipeline to recognize the sentence and return the result.

    Parameter:
        text (str): the text that you want to identify the sentence from
        return_str (boolean): If True, each element of the list will contain a string that represents a sentence. 
                              If False, the element of the list will have a slice of the document object.

    Return:
        sentences(list) :
    '''
    # Load spaCy's en_core_web_sm, English pipeline optimized for CPU
    # https://spacy.io/models/en/
    nlp = spacy.load('en_core_web_sm')
    
    # Apply the spacy pipeline into the text
    spacy_text = nlp(text)

    if return_str:
        sentences = list(spacy_text.sents)
        sentences = [str(sent) for sent in sentences]
    else:
        sentences = list(spacy_text.sents)

    return sentences

def address_identification(text:str, add_dictionary = []) -> bool:
    '''
    Identify text that may be an address using a dictionary method. The original dictionary is the address_dictionary list. The dictionary can be edited.

    Parameter:
        text (str): original text
        add_dictionary (list): additional words that needed to be excluded

    Return
        identifier (true) : returns true when the text is an address.
    '''
    identifier = False
    address_dictionary = ["AVRDC-The World Vegetable Center", "P.O", "AVRDC", "written", "tel", "fax", "email", "printed", "CIAT"]
    address_dictionary += add_dictionary

    lower_case_address_dictionary = [address.lower() for address in address_dictionary]
    lower_case_text = text.lower()

    for condition in lower_case_address_dictionary:
        if re.search(condition, lower_case_text) is not None:
            identifier = True
            break
    
    return identifier

def text_splitter (text: str, chunk_size = 1024, chunk_overlap = 100) -> list:
    '''
    Divide the text using the RecursiveCharacterTextSplitter
    Parameter:
        text (str) : the original string to split
        chunk_size (int): maximum size of chunks to return 
        chunk_overlap (int): overlap in characters between chunks
    Return :
        splitted_text (list) : list of strings that is splitted based on the conditions
    '''
    text_spliter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    splitted_text = text_spliter.split_text(text)

    return splitted_text