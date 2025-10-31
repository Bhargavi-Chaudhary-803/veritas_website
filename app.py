import os
from flask import Flask, request, Response, stream_with_context, jsonify
from flask_cors import CORS
import requests
import json
import logging
import uuid
from supabase.client import create_client, Client # Import Supabase client libraries

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("✅ CORS configured for all origins.")

# --- CONFIGURATION ---

# Shivaay API configuration
SHIVAAY_API_URL = "https://api.futurixai.com/api/lara/v1/chat/completions"
SHIVAAY_AUTH_TOKEN = os.environ.get("SHIVAAY_AUTH_TOKEN") 

if not SHIVAAY_AUTH_TOKEN:
    logging.error("SHIVAAY_AUTH_TOKEN environment variable not set. Please configure it on Render.")
    # In a real deployment, the app would crash or fail to generate responses.

# Supabase configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
CHAT_TABLE = "chat_sessions" # Name of the table in Supabase

# Initialize Supabase client
supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logging.info("🌐 Supabase client initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize Supabase client: {e}")
        supabase = None # Set to None if initialization failed

SYSTEM_INSTRUCTION = (
    "You are Veritas, an empathetic and professional AI assistant specializing in Pre-Consultation Clinical History Collection. "
    "Your primary function is to guide the user through a structured, interactive conversation to gather their clinical history, focusing on: "
    "1. Current Symptoms/Chief Complaint. "
    "2. Onset, location, duration, character, aggravating/alleviating factors (OPQRST analysis). "
    "3. Past Medical History (including known conditions, surgeries, and allergies). "
    "4. Family History (relevant illnesses). "
    "5. Lifestyle factors (smoking, alcohol, diet, stress). "
    "Maintain a calm, non-judgemental, and empathetic tone throughout the conversation. "
    "**Language Note:** Use English by default. "
    "**Language Note:** Seamlessly support and use **Hinglish (Hindi/English code-switching)**, ensuring clarity and empathy. Furthermore, you must **support and respond in any of the 22 Scheduled Indian Languages** (Assamese, Bengali, Bodo, Dogri, Gujarati, Hindi, Kannada, Kashmiri, Konkani, Maithili, Malayalam, Manipuri, Marathi, Nepali, Odia, Punjabi, Sanskrit, Santali, Sindhi, Tamil, Telugu, Urdu) if the user initiates the conversation in that language or code-switches into it. If the user sticks to one language, follow their lead. "
    "CRITICAL CONSTRAINT 1 (Privacy): Prioritize data privacy. Collect only necessary clinical information and do NOT ask for or store personally identifiable information (PII) such as full name, date of birth, address, or financial details. "
    "CRITICAL CONSTRAINT 2 (Brevity): Your responses MUST be concise, short (typically a single sentence or precise question), and focused to efficiently guide the patient to the next piece of required information. Only provide detailed explanations or long answers when explicitly asked or if the patient's safety/privacy is at risk."

)

logging.info("🤖 Veritas Clinical History Assistant started using Shivaay API (Streaming Enabled)")

# --- HELPER FUNCTIONS ---

def _convert_history(conversation_history, system_instruction):
    """
    Converts the chat history (Gemini style from frontend) to the required Shivaay/OpenAI format.
    Prepends the system instruction.
    """
    shivaay_messages = [
        {"role": "system", "content": system_instruction}
    ]
    
    # Ensure history is iterable and handle potential malformed entries defensively
    if not isinstance(conversation_history, list):
        conversation_history = []

    for message in conversation_history:
        # Map client roles ('user', 'model') to Shivaay roles ('user', 'assistant')
        role = message.get('role')
        if role == 'model':
            role = 'assistant'
        
        # Extract content (assuming frontend sends parts or text keys)
        content = message.get('parts', [{}])[0].get('text', '') or message.get('text', '')
        
        if role and content:
            shivaay_messages.append({"role": role, "content": content})
            
    return shivaay_messages

def _load_chat_history(user_id):
    """Loads chat history from Supabase for a given user_id."""
    if not supabase:
        logging.warning("Supabase not initialized. Cannot load history.")
        return []
    
    try:
        # .single() will return a 406 error if zero rows are found, which is intended here 
        # to distinguish a new user from a true database error.
        response = supabase.table(CHAT_TABLE).select("history").eq("user_id", user_id).single().execute()
        
        # Check if data is present and is a dictionary containing the 'history' key
        if response.data and isinstance(response.data, dict) and response.data.get('history'):
            logging.info(f"Loaded history for user {user_id}.")
            return response.data['history']
        
        return []
    except Exception as e:
        # Catch 406 (No rows found) for new users; log other critical errors.
        if "406" not in str(e) and "The resource was not found" not in str(e):
             logging.error(f"Error loading history for {user_id}: {e}")
        return []

def _save_full_turn(user_id, updated_history):
    """Saves the entire updated history array back to Supabase."""
    if not supabase:
        logging.warning("Supabase not initialized. Cannot save history.")
        return
        
    try:
        # Use upsert to insert if user_id doesn't exist, or update if it does.
        data_to_save = {
            "user_id": user_id,
            "history": updated_history
        }
        
        # The 'on_conflict="user_id"' is crucial for the upsert operation
        response = supabase.table(CHAT_TABLE).upsert(data_to_save, on_conflict="user_id").execute()
        
        if response.data:
            logging.info(f"Successfully saved history for user {user_id}.")
        else:
            logging.error(f"Supabase upsert failed for user {user_id}. Response: {response}")

    except Exception as e:
        logging.error(f"CRITICAL: Failed to save history to Supabase for {user_id}: {e}")

def generate_response_stream(conversation_history):
    """
    Generates content by calling the external Shivaay API endpoint and yielding chunks.
    """
    try:
        # 1. Convert client history to Shivaay API format
        shivaay_messages = _convert_history(conversation_history, SYSTEM_INSTRUCTION)
        
        # 2. Construct the API payload
        payload = {
            "model": "shivaay",
            "messages": shivaay_messages,
            "temperature": 0.7,
            "top_p": 0.8,
            "repetition_penalty": 1.05,
            "max_tokens": 512,
            "stream": True
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {SHIVAAY_AUTH_TOKEN}" 
        }

        # 3. Make the streaming API request
        response = requests.post(
            SHIVAAY_API_URL, 
            headers=headers, 
            json=payload, 
            stream=True
        )

        response.raise_for_status()

        # 4. Process the stream
        for line in response.iter_lines():
            if line:
                try:
                    chunk_data = line.decode('utf-8')
                    if chunk_data.startswith('data: '):
                        chunk_data = chunk_data[6:].strip()
                    
                    if chunk_data and chunk_data != '[DONE]':
                        chunk = json.loads(chunk_data)
                        content = chunk.get('choices', [{}])[0].get('delta', {}).get('content')
                        
                        if content:
                            yield content
                            
                except json.JSONDecodeError:
                    logging.warning(f"Failed to decode JSON chunk: {line}")
                except Exception as e:
                    logging.error(f"Error processing stream chunk: {e}")
                    yield f"[STREAM ERROR] Processing error: {e}"
                    break
            
    except requests.exceptions.HTTPError as e:
        logging.error(f"API HTTP Error: {e.response.status_code} - {e.response.text}")
        yield f"\n[CRITICAL ERROR] API request failed with status code {e.response.status_code}. Response: {e.response.text.strip()}"
    except requests.exceptions.ConnectionError as e:
        logging.error(f"API Connection Error: {e}")
        yield f"\n[CRITICAL ERROR] Could not connect to the Shivaay API: {e}"
    except Exception as e:
        logging.error(f"Unforeseen error during streaming: {e}")
        yield f"\n[CRITICAL ERROR] Unforeseen Server Error: {e}"


# --- API ENDPOINTS ---
# --- NEW API ENDPOINT FOR NEW/INITIAL SESSIONS ---
@app.route('/new_session', methods=['POST'])
def new_session_handler():
    """Endpoint to start a brand new chat session and return the initial bot message."""
    try:
        data = request.get_json()
        user_id = data.get('user_id')

        if not user_id:
            return jsonify({"error": "User ID is required."}), 400

        # Define the initial bot message
        initial_message_text = "Welcome to Veritas. I'm here to securely collect your clinical history for your doctor. What is the main symptom or reason for your consultation today?"
        
        history = [{
            "role": "model",
            "parts": [{"text": initial_message_text}]
        }]
        
        # Save the initial message to Supabase for the new session
        _save_full_turn(user_id, history)
        
        # Return the message text to the frontend (as a string, not a stream)
        return jsonify({"message": initial_message_text}), 200

    except Exception as e:
        logging.error(f"Server-side error during new_session request: {e}")
        return jsonify({"error": "Internal Server Error during new session processing."}), 500

@app.route('/history', methods=['POST'])
def history_handler():
    """Endpoint to fetch existing chat history for a user."""
    try:
        data = request.get_json()
        user_id = data.get('user_id')

        if not user_id:
            return jsonify({"error": "User ID is required."}), 400

        history = _load_chat_history(user_id)
        
        # NOTE: Removed the logic here to generate/save the initial message.
        # This is now handled by the /new_session endpoint, which the frontend
        # must call if /history returns an empty array.
        
        return jsonify({"history": history}), 200

    except Exception as e:
        logging.error(f"Server-side error during history request: {e}")
        return jsonify({"error": "Internal Server Error during history processing."}), 500

@app.route('/chat', methods=['POST'])
def chat_handler():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        user_message = data.get('message')
        
        if not user_id or not user_message:
            return jsonify({"error": "User ID and message are required."}), 400

        # 1. Load existing history and append the new user message
        conversation_history = _load_chat_history(user_id)
        
        # Format the new user message to match the history structure (Gemini style)
        user_turn = {"role": "user", "parts": [{"text": user_message}]}
        conversation_history.append(user_turn)

        # 2. Start streaming the LLM response
        full_bot_reply = ""
        def streaming_wrapper(history):
            nonlocal full_bot_reply
            for chunk in generate_response_stream(history):
                full_bot_reply += chunk
                yield chunk
            
            # 3. After the stream completes, save the full turn to Supabase
            if full_bot_reply:
                bot_turn = {"role": "model", "parts": [{"text": full_bot_reply}]}
                history.append(bot_turn)
                _save_full_turn(user_id, history)

        # Send the streamed response back to the client
        response = Response(
            stream_with_context(streaming_wrapper(conversation_history)),
            mimetype='text/plain' 
        )
        
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response

    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        error_text = e.response.text.strip()
        logging.error(f"API HTTP Error: {status_code} - {error_text}")
        return Response(f"\n[CRITICAL ERROR] API request failed with status code {status_code}. Response: {error_text}", mimetype='text/plain', status=500)
    
    except Exception as e:
        logging.error(f"Server-side error during chat request: {e}")
        return jsonify({"error": "Internal Server Error during processing."}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT', 5000))

