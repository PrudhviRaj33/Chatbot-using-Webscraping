from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup
from gradio_client import Client
from langchain.memory import ConversationBufferMemory
from concurrent.futures import ThreadPoolExecutor, CancelledError

app = Flask(__name__)

# Initialize LangChain memory
memory = ConversationBufferMemory()

# LLM Client Setup
llm_client = Client("PrudhviRajGandrothu/llama-3.1")

def retrieve_web_content(query):
    """
    Retrieves content from multiple websites based on the query and returns extracted text.
    """
    websites = [
        f"https://www.google.com/search?q={query}",
        f"https://www.bing.com/search?q={query}",
        f"https://duckduckgo.com/?q={query}"
    ]
    
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    
    def fetch_content(site):
        try:
            response = requests.get(site, headers=headers, timeout=5)
            soup = BeautifulSoup(response.text, "html.parser")
            result_divs = soup.find_all("div", class_="BNeawe s3v9rd AP7Wnd")
            return " ".join([result.get_text() for result in result_divs[:3]])
        except requests.exceptions.RequestException as e:
            print(f"Error fetching from {site}: {e}")
            return ""

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_content, websites))
    
    return " ".join(results)

def generate_response_from_llm(content, query):
    """
    Sends content to the LLM and retrieves a response, with memory using LangChain.
    """
    try:
        # Retrieve conversation memory so far
        past_conversations = memory.load_memory_variables({})

        # Generate response from LLM
        result = llm_client.predict(
            message=f"User query: {query}\nWeb content: {content}\nMemory: {past_conversations.get('history', '')}",
            api_name="/chat"
        )
        
        # Clean the response: removing formatting characters like asterisks
        clean_result = result.replace("*", "").strip()

        # Add both user query and LLM response to memory
        memory.chat_memory.add_user_message(query)
        memory.chat_memory.add_ai_message(clean_result)

        return clean_result
    
    except CancelledError:
        print("Task was cancelled")
        return "Sorry, your request was cancelled due to a timeout or conflict. Please try again."
    except Exception as e:
        print(f"Error generating LLM response: {e}")
        return "Sorry, an error occurred. Please try again later."


@app.route("/", methods=["GET", "POST"])
def index():
    """
    This route renders the chatbot interface and displays LLM results with memory.
    """
    if request.method == "POST":
        user_query = request.form.get("query")
        if user_query:
            # Retrieve web content based on the user query
            web_content = retrieve_web_content(user_query)
            
            # Generate response from the LLM based on retrieved web content and memory
            llm_response = generate_response_from_llm(web_content, user_query)
            
            # Get the entire conversation history from memory
            chat_history = memory.chat_memory.messages
            
            return render_template("index.html", query=user_query, response=llm_response, chat_history=chat_history)
    
    # On GET request, display an empty chat interface
    return render_template("index.html", query="", response="", chat_history=[])

if __name__ == "__main__":
    app.run(debug=True)