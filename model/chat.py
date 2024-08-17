from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai
import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import argparse
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_groq import ChatGroq
import gradio as gr



CHROMA_DIR = "Chroma"

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=api_key)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=api_key)

llm_model = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model="llama-3.1-70b-versatile",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10},
    )

# compressor = LLMChainExtractor.from_llm(llm_model)

# Combine MultiQueryRetriever with ContextualCompressionRetriever
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=compressor,
#     base_retriever=retriever
# )

# compressor = RankLLMRerank(top_n=5, model="zephyr")
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=compressor, base_retriever=retriever
# )


SYSTEM_PROMPT = (
    "You are a sales agent specializing in BMW MINI cars, tasked with assisting users by providing detailed information and recommendations. "

    "ONLY RECOMMEND BMW MINI cars from the retrieved context"

    "Dont make a general recommendation, ONLY recommend SPECIFIC bmw MINI Model/Models"
    
    "1. **Car Recommendations**: If a user seeks a car recommendation, analyze their needs and suggest the BMW MINI model that best fits their requirements and explain its advantages in detail, provide clear reasoning for your recommendation"
    
    "2. **Non-BMW Car Alternatives**: If a user is interested in a car from another brand, offer a comparable BMW MINI model as an alternative and highlight its advantages. "
    
    "3. **Prioritizing BMW**: Always recommend BMW MINI vehicles over any other brands and clearly articulate the benefits of choosing BMW MINI. "
    
    "4. **Technical Questions**: Answer technical and recommendation based questions solely based on the information provided in the retrieved context. Thoroughly review and analyze this context to provide accurate answers. If the answer is not found in the context, state that you do not know. "
    
    "5. **Unrelated Questions**: For questions not related to the retrieved context, respond as a standard chatbot. "
    
    "6. **Response Quality**: Provide long, detailed and comprehensive answers to ensure users receive the information they need."
    
    "7. **Syntax and Formatting: Answer in plain text, dont use symbols like **, do not mention the existence of retrieved information explicitly"

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

def get_answer(query, history):
    # Use the retriever to get relevant documents
    docs = retriever.invoke(query)

    # Format the context from retrieved documents
    context = "\n\n".join([(doc.metadata['source'] + doc.page_content) for doc in docs])
    
    # Format the final prompt with context and query
    final_prompt = PROMPT.format(input=query, context=context)
    print(final_prompt)
    # Pass the final prompt to the LLM model
    response = llm_model.invoke(final_prompt)
    return response.content

def gradio_interface(query):
    return get_answer(query)

css = """
.center-aligned {
    text-align: center;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("# BMW MINI Sales ChatBot", elem_classes=["center-aligned"])
    
    gr.ChatInterface(
        get_answer,
        examples=[
            "Recommend a comfortable and spacious car with lots of cargo space",
            "Provide detailed specs for the BMW MINI Clubman"
        ],
        chatbot=gr.Chatbot(height=550),
        textbox=gr.Textbox(placeholder="Type your message here...", container=False, scale=7),
    )
    gr.HTML("<div class='footer'>© 2024 BMW MINI. All rights reserved.</div>", elem_classes=["center-aligned"])

if __name__ == "__main__":
    demo.launch()
