from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai
import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import argparse
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
import gradio as gr
import uuid

CHROMA_DIR = "Chroma"
MAX_RETRIES = 3
RETRY_DELAY = 1

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')

if not api_key or not groq_api_key:
    raise ValueError("API keys not found in environment variables")

genai.configure(api_key=api_key)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=api_key)

llm_model = ChatGroq(
    api_key=groq_api_key,
    model="llama-3.1-70b-versatile",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=MAX_RETRIES
)

try:
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
except Exception as e:
    print(f'Vector Database Error {e}')
    raise

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10},
    )

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm_model, retriever, contextualize_q_prompt
)

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
    
    "7. **Syntax and Formatting: Do NOT mention the existence of retrieved information explicitly"

    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm_model, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

def get_answer(query: str, history: list) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            response = conversational_rag_chain.invoke(
                {"input": query},
                config={"configurable": {"session_id": session_id}}
            )
            return response.get("answer", "I'm sorry, I couldn't find an answer to your question.")
        except Exception as e:
            print(f"Error in attempt {attempt + 1}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                return "I'm sorry, there was an error processing your request. Please try again later."

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
        textbox=gr.Textbox(placeholder="Type your message here...", container=False, scale=7)
    )
    gr.HTML("<div class='footer'>© 2024 BMW MINI. All rights reserved.</div>", elem_classes=["center-aligned"])

if __name__ == "__main__":
    session_id =  str(uuid.uuid4())
    try:
        demo.launch()
    except Exception as e:
        print(f"Failed to launch Gradio interface: {e}")
