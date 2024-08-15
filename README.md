RAG based sales agent chatbot that can answer queries about BMW cars and provide suitable recommendations based on personal info.
create a .env file and place gemini api key "GEMINI_API_KEY = {key}" before calling python scripts, u can get one for free at https://aistudio.google.com/app/apikey
python database.py. to build the chroma database
python database.py --save. to save langchain documents as .txt files (optional)
python chat.py "{query}" to call