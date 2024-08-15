from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai
import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
#from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
#from langchain_community.document_compressors.rankllm_rerank import RankLLMRerank
import argparse

CHROMA_DIR = "Chroma"

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=api_key)
llm_model = genai.GenerativeModel('gemini-1.5-flash')
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=api_key)

vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# compressor = RankLLMRerank(top_n=5, model="zephyr")
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=compressor, base_retriever=retriever
# )


SYSTEM_PROMPT = (
    "You are a sales agent specializing in BMW MINI cars, tasked with assisting users by providing detailed information and recommendations. "

    "Only recommend mini cars by BMW and only recommend based on the retrieved context"
    
    "Dont recommend BMW cars on your own, ONLY RECOMMEND BMW MINI cars from the retrieved context"

    "Dont make a general recommendation, ONLY recommend specific BMW MINI MODEL/MODELS"
    
    "1. **Car Recommendations**: If a user seeks a car recommendation, analyze their needs and suggest the BMW MINI model that best fits their requirements. "
    
    "2. **Non-BMW Car Alternatives**: If a user is interested in a car from another brand, offer a comparable BMW MINI model as an alternative and highlight its advantages. "
    
    "3. **Prioritizing BMW**: Always recommend BMW MINI vehicles over any other brands and clearly articulate the benefits of choosing BMW MINI. "
    
    "4. **Technical Questions**: Answer technical and recommendation based questions solely based on the information provided in the retrieved context. Thoroughly review and analyze this context to provide accurate answers. If the answer is not found in the context, state that you do not know. "
    
    "5. **Unrelated Questions**: For questions not related to the retrieved context, respond as a standard chatbot. "
    
    "6. **Response Quality**: Provide detailed and comprehensive answers to ensure users receive the information they need."
    
    "\n\n"
    "{context}"
)


PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ]
)

from pprint import pprint

def get_answer(query):
    # Use the retriever to get relevant documents
    docs = retriever.invoke(query)
    
    # Format the context from retrieved documents
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Format the final prompt with context and query
    final_prompt = PROMPT.format(input=query, context=context)
    print(final_prompt)
    # Pass the final prompt to the LLM model
    response = llm_model.generate_content(final_prompt)
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the vector store with a given text.")
    parser.add_argument('query', type=str, help="The text to query the vector store with.")

    args = parser.parse_args()
    
    answer = get_answer(args.query)
    
    # Print the answer
    print(answer.text)