from dotenv import load_dotenv
import os
import google.generativeai as genai
import argparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema.document import Document
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader


load_dotenv()

CHROMA = "Chroma"
html_path = "content/htmls"
pdf_path = "content/pdfs"

api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=api_key)
llm_model = genai.GenerativeModel('gemini-1.5-flash')

def data_extraction():
    docs = []
    html_files = [f for f in os.listdir(html_path) if f.endswith('.html')]
    
    for html_file in html_files:
        file_path = os.path.join(html_path, html_file)
        try:
            # Create a loader for each HTML file
            loader = UnstructuredHTMLLoader(file_path)
            # Load the data from the file
            data = loader.load()
            for doc in data:
                doc.metadata['name'] = html_file
            # Append the data to the list
            docs.extend(data)
        except Exception as e:
            print(f"Error loading {html_file}: {e}")
    print("html pages loaded")
    
    loader = PyPDFDirectoryLoader(pdf_path)
    pdf_docs = loader.load()

    print("pdf files loaded")
    return docs, pdf_docs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Option to save to files.")
    parser.add_argument('--save', action='store_true', help="Save the extracted text to .txt files.")

    args = parser.parse_args()

    docs, pdf_docs = data_extraction()

    if args.save:
        for doc in docs:
            filename = "content/data/" + doc.metadata['name'] + ".txt"
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(doc.page_content)
        for doc in pdf_docs:
            filename = "content/data/" + "MINI Cars Brochure" + ".txt"
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(doc.page_content)

    docs += pdf_docs

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=api_key)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap= 50)
    texts = text_splitter.split_documents(docs)
    for text in texts:
        text.page_content = "Source: " + text.metadata['source'] + " Content: "+ text.page_content
    vectorstore = Chroma.from_documents(persist_directory=CHROMA, documents=texts, embedding=embeddings)