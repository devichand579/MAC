import sys
import os

# Get root directory (MAC) - assume app.py is run from root directory
# If running from root, use current directory; otherwise go up from userstudy
if os.path.basename(os.getcwd()) == 'MAC' or os.path.exists(os.path.join(os.getcwd(), 'MAC')):
    ROOT_DIR = os.getcwd()
    if os.path.basename(ROOT_DIR) != 'MAC':
        # If we're inside MAC, find the MAC directory
        current = ROOT_DIR
        while current and os.path.basename(current) != 'MAC':
            current = os.path.dirname(current)
        if current:
            ROOT_DIR = current
else:
    # Fallback: assume we're in userstudy directory, go up one level to MAC
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add paths for imports
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'TAC_models', 'qb'))  # For QueryBlazer
sys.path.append(os.path.join(ROOT_DIR, 'ms-swift-chatas'))  # For swift module
sys.path.append(os.path.join(ROOT_DIR, 'userstudy', 'models', 'mpc'))

from flask import Flask, render_template, request, jsonify, session, send_from_directory
import random
import uuid
import csv
from datetime import datetime
import json
import time
import pandas as pd
import pickle
import string

# Import QueryBlazer
try:
    from queryblazer import QueryBlazer, Config
    QB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: QueryBlazer not available: {e}")
    QB_AVAILABLE = False
    QueryBlazer = None
    Config = None

# Import conv module (should be in userstudy directory)
USERSTUDY_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, USERSTUDY_DIR)
from conv import CONTEXT_POOL

sys.setrecursionlimit(10000)


class BaseModel:
    slow_model = False
    def complete(self, text, chat_history):
        raise NotImplementedError

class MPC_DDC_Model(BaseModel):

    def load_trie_from_file(self, file_path):
        if not os.path.exists(file_path):
            return None
        try:
            with open(file_path, 'rb') as file:
                return pickle.load(file)
        except Exception as e:
            print(f"Error loading trie from {file_path}: {e}")
            return None

    def __init__(self):
        # Paths relative to root directory
        mpc_main_path = os.path.join(ROOT_DIR, "userstudy", "ckpt", "mpc_suff", "main.mpc")
        mpc_suffix_path = os.path.join(ROOT_DIR, "userstudy", "ckpt", "mpc_suff", "suffix.mpc")
        self.main_trie = self.load_trie_from_file(mpc_main_path)
        self.suffix_trie = None
        self.suffix_trie = self.load_trie_from_file(mpc_suffix_path)
    
    def get_completions(self, prefix, k_completions=1, suffix_context=2):
        main_trie = self.main_trie
        suffix_trie = self.suffix_trie
        res = []
        main_completions = main_trie.find_completions(prefix)

        main_completions_dict = {}
        for z in main_completions:
            if z[0] in main_completions_dict:
                continue
            main_completions_dict[z[0]]=1
            # adding prefix "MT" to denote main trie completions
            res.append([z[0], "MT:%s" % str(z[1])])

        if suffix_trie is not None and len(res)<k_completions:
            backfill = k_completions - len(res)
            prefix_tokens = prefix.strip().split(' ')
            ends_with_space = " " if prefix[-1]==" " else ""
            suffix_completions = []
            suffix_completions_dict = {}
            
            # minimum words to consider during suffix match
            for idx in range(suffix_context, len(prefix_tokens)):
                suffix = " ".join(prefix_tokens[-idx:]) + ends_with_space
                partial_prefix = " ".join(prefix_tokens[:len(prefix_tokens)-idx])
                for temp_completions in suffix_trie.find_completions(suffix):
                    full_completion = partial_prefix + " " + temp_completions[0]
                    if full_completion in main_completions_dict or full_completion in suffix_completions_dict:
                        continue
                    suffix_completions_dict[full_completion] = 1
                    suffix_completions.append((full_completion, temp_completions[1]))
                # print("suffix completions : %d" % len(suffix_completions))
            suffix_completions = sorted(suffix_completions, key=lambda x: x[1], reverse=True)[:backfill]
            if len(suffix_completions)>0:
                for z in suffix_completions:
                    # adding prefix "ST" to denote suffix trie completions
                    res.append([z[0], "ST:%s" % str(z[1])])
        return [r[0] for r in res[:k_completions]]


    def complete(self, text, chat_history):
        if not text:
            return ""
        text_lower = text.lower()
        suggestions = self.get_completions(text_lower, k_completions=1)
        if not suggestions or len(suggestions) == 0:
            return ""
        return suggestions[0]


class QB_MMDD_Model(BaseModel):
    slow_model = True
    def __init__(self):
        if not QB_AVAILABLE:
            raise ImportError("QueryBlazer is not available. Please ensure TAC_models/qb is in the path.")
        
        self.branch_factor = 30
        self.beam_size = 30
        self.topk = 10
        self.length_limit = 100
        
        # Use QB_ckpts from root directory (has precomputed.bin)
        # Fallback to userstudy/ckpt/qb if needed
        qb_ckpt_dir = os.path.join(ROOT_DIR, "QB_ckpts", "QB_MMDD")
        if not os.path.exists(os.path.join(qb_ckpt_dir, "precomputed.bin")):
            # Try userstudy/ckpt/qb
            qb_ckpt_dir = os.path.join(ROOT_DIR, "userstudy", "ckpt", "qb")
        
        self.encoder = os.path.join(qb_ckpt_dir, "encoder.fst")
        self.model = os.path.join(qb_ckpt_dir, "ngram.fst")
        self.precomputed = os.path.join(qb_ckpt_dir, "precomputed.bin")
        
        # Check if files exist
        if not os.path.exists(self.encoder):
            raise FileNotFoundError(f"Encoder not found: {self.encoder}")
        if not os.path.exists(self.model):
            raise FileNotFoundError(f"Model not found: {self.model}")
        if not os.path.exists(self.precomputed):
            raise FileNotFoundError(f"Precomputed file not found: {self.precomputed}")
        
        self.config = Config(branch_factor=self.branch_factor, beam_size=self.beam_size, topk=self.topk, length_limit=self.length_limit)
        self.qbz = QueryBlazer(encoder=self.encoder, model=self.model, config=self.config)
        assert self.qbz.LoadPrecomputed(self.precomputed), "Failed to load precomputed data"

    def complete(self, text, chat_history) -> str:
        if not text:
            return ""
        text_lower = text.lower()
        output = self.qbz.Complete(text_lower)
        if not output or not output[0] or not output[0][0]:
            return ""
        pred, cost = output[0][0][0]
        subword_len = output[0][0][1]
        return pred


class T5cOASST_Model(BaseModel):
    slow_model = True
    def __init__(self):
        # self.model = T5Model(model_name="t5-base", context=True, ckpt="ckpt/t5/t5-base-boozy-cerulean-shark-epoch_39.pth")
        self.slow_model = True

    def complete(self, text, chat_history) -> str:
        if not text:
            return ""
        input = " <|EOU|> ".join([x['message'] for x in chat_history]) + " <|EOU|> " + text
        t5_model_client = T5ModelClient(api_url="https://f35e2828febd.ngrok-free.app")
        prediction = t5_model_client.predict(input).predicted_text
        prediction = prediction.split("<|EOU|>")[-1].lstrip().lower()
        return prediction


class MiniCPM_Model(BaseModel):
    slow_model = True
    def __init__(self):
        from swift.llm import PtEngine
        # Initialize the MiniCPM model with the checkpoint
        self.model_name = "openbmb/MiniCPM-V-2_6"
        # Use absolute path for adapter
        adapter_relative_path = os.path.join("ckpts", "Imagechat_ckpts", "MiniCPM_V")
        self.adapter_path = os.path.join(ROOT_DIR, adapter_relative_path)
        if not os.path.exists(self.adapter_path):
            raise FileNotFoundError(f"Adapter path not found: {self.adapter_path}")
        self.engine = PtEngine(self.model_name, max_batch_size=1, adapters=[self.adapter_path], device_map='cuda:0')
        self.slow_model = True

    def complete(self, text, chat_history, image_path=None) -> str:
        if not text:
            return ""
        
        # Format the query similar to infer_chatas.py
        query = ""
        for utterance in chat_history:
            if 'image_path' in utterance and utterance['image_path']:
                query += f"{utterance['speaker']}:<|IMAGE|>\n"
            else:
                query += f"{utterance['speaker']}:{utterance['message']}\n"
        
        query += f"You:{text}"
        
        # Create the request message
        if image_path:
            message = {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": query},
                ]
            }
        else:
            message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                ]
            }
        
        # Create inference request
        from swift.llm import InferRequest, RequestConfig
        request = InferRequest(messages=[message])
        config = RequestConfig(max_tokens=50, temperature=0)
        
        # Get prediction
        try:
            response = self.engine.infer([request], config)[0]
            prediction = response.choices[0].message.content
            # Clean up prediction to match format
            prediction = prediction.lower()
            return text + prediction

        except Exception as e:
            print(f"Error in MiniCPM prediction: {e}")
            return ""

import string

class SimpleModel(BaseModel):
    def complete(self, text, chat_history) -> str:
        print("chat_history:", chat_history)
        if not text:
            return ""
        return text + ''.join(random.choices(string.ascii_lowercase, k=5))

app = Flask(__name__, static_folder='static', template_folder='static')
app.secret_key = 'super secret key'
# Initialize models
models = {}
try:
    if QB_AVAILABLE:
        models["QB_MMDD_Model"] = QB_MMDD_Model()
    else:
        models["QB_MMDD_Model"] = SimpleModel()
except Exception as e:
    print(f"Warning: Could not initialize QB_MMDD_Model: {e}")
    models["QB_MMDD_Model"] = SimpleModel()

try:
    models["MPC_DDC_Model"] = MPC_DDC_Model()
except Exception as e:
    print(f"Warning: Could not initialize MPC_DDC_Model: {e}")
    models["MPC_DDC_Model"] = SimpleModel()

models["SimpleModel"] = SimpleModel()
models["T5cOASST_Model"] = T5cOASST_Model()
# MiniCPM_Model is lazy-loaded - only initialize when needed
# models["MiniCPM_Model"] = MiniCPM_Model()

import json


# Correctly locate the log file next to the app.py script
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(APP_ROOT, 'session_logs.csv')
CONTEXT_LOG_FILE = os.path.join(APP_ROOT, 'context_log.csv')


# Initialize log file with headers if it doesn't exist
try:
    with open(LOG_FILE, 'x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['session_id', 'timestamp', 'event_type', 'details', 'event_id', 'parent_event_id'])
except FileExistsError:
    pass

try:
    with open(CONTEXT_LOG_FILE, 'x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['session_id', 'context'])
except FileExistsError:
    pass

@app.route('/')
def index():
    # Generate a new session ID if it's a new example or session doesn't exist
    if request.args.get('new_example') == 'true' or 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('index.html')

@app.route('/get_context', methods=['GET'])
def get_context():
    # Check if we should use image-based conversations
    use_image_chat = True
    
    if use_image_chat:
        # Load conversation from CSV file (path relative to userstudy directory)
        csv_path = os.path.join(USERSTUDY_DIR, 'imagechat_samples', 'test_samples.csv')
        try:
            # Let's print the first few lines of the file to debug
            print("Checking CSV file content:")
            with open(csv_path, 'r') as f:
                first_line = f.readline().strip()
                print(f"First line: {first_line}")
            
            # Use a more robust approach to read and parse the CSV file
            import csv
            import json
            import re
            
            # Read all conversations from the CSV file
            conversations = []
            with open(csv_path, 'r') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    if len(row) >= 2:
                        conversations.append((row[0], row[1]))
            
            # Select a random conversation
            conversation_id, raw_json_str = random.choice(conversations)
            print(f"Selected conversation ID: {conversation_id}")
            
            # Clean and fix the JSON string
            # 1. Remove outer quotes if present
            if raw_json_str.startswith('"') and raw_json_str.endswith('"'):
                raw_json_str = raw_json_str[1:-1]
            
            # 2. Fix unescaped quotes within the JSON string
            # This regex finds quotes inside utterance values that aren't escaped
            fixed_json_str = re.sub(r'(\"utterance\": \")([^\"]*)(\"\",)', 
                                   lambda m: m.group(1) + m.group(2).replace('"', '\"') + m.group(3), 
                                   raw_json_str)
            
            # 3. Try to parse with json first (safer than ast.literal_eval)
            try:
                print(f"Attempting to parse JSON: {fixed_json_str[:100]}...")
                conversation_data = json.loads(fixed_json_str.replace("'", "\""))
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {e}, trying ast.literal_eval as fallback")
                # Fallback to ast.literal_eval with additional safety
                import ast
                try:
                    # Replace single quotes with double quotes for JSON compatibility
                    conversation_data = ast.literal_eval(raw_json_str)
                except Exception as e:
                    print(f"Failed to parse conversation data with ast.literal_eval: {e}")
                    # Last resort: try to manually fix common issues and parse again
                    # Replace problematic characters and try again
                    sanitized_str = raw_json_str.replace('\\"', '"').replace('"', '\\"')
                    try:
                        conversation_data = json.loads(sanitized_str)
                    except Exception as e2:
                        print(f"All parsing attempts failed: {e2}")
                        raise e2
            
            # Format the conversation for the frontend
            context = []
            image_path = None
            
            for utterance in conversation_data:
                # Create a consistent structure for the frontend
                item = {
                    'speaker': utterance['speaker'],
                    'message': utterance['utterance'],  # Use 'message' key for frontend consistency
                    'utterance': utterance['utterance']  # Keep original 'utterance' key as well
                }
                
                # If this utterance has an image, save the path
                # Check if image_hash exists and is valid (should be 32 characters for MD5)
                if 'image_hash' in utterance and utterance['image_hash'] and len(utterance['image_hash']) > 0:
                    # Use absolute paths for file operations
                    images_dir = os.path.join(USERSTUDY_DIR, 'imagechat_samples', 'images')
                    image_file_path = os.path.join(images_dir, f"{utterance['image_hash']}.jpg")
                    
                    if os.path.exists(image_file_path):
                        # Use URL path for frontend (Flask route)
                        item['image_path'] = f"/imagechat_samples/images/{utterance['image_hash']}.jpg"
                        # Store the first image path for the model (use file path)
                        if not image_path:
                            image_path = image_file_path
                    else:
                        print(f"Warning: Image file not found: {image_file_path}")
                        # Try to find a matching image with a partial hash
                        if len(utterance['image_hash']) >= 8:  # At least 8 chars to avoid false matches
                            if os.path.exists(images_dir):
                                for img_file in os.listdir(images_dir):
                                    if img_file.startswith(utterance['image_hash']) and img_file.endswith('.jpg'):
                                        print(f"Found matching image with partial hash: {img_file}")
                                        # Use URL path for frontend
                                        item['image_path'] = f"/imagechat_samples/images/{img_file}"
                                        # Store the first image path for the model (use file path)
                                        if not image_path:
                                            image_path = os.path.join(images_dir, img_file)
                                        break
                
                context.append(item)
            
            # Store image path in session for the model to use
            session['image_path'] = image_path
            session['conversation_id'] = conversation_id
            
            # Log the selected conversation
            with open(CONTEXT_LOG_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([session.get('session_id'), conversation_id, json.dumps(context)])
                
            return jsonify(context)
            
        except Exception as e:
            print(f"Error loading image chat data: {e}")
            # Fall back to regular context pool
            use_image_chat = False
    
    # Original context pool logic as fallback
    context = random.choice(CONTEXT_POOL)
    x = random.randint(1, len(context))
    context = context[:x]
    with open(CONTEXT_LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([session.get('session_id'), json.dumps(context)])
    return jsonify(context)

@app.route('/set_model', methods=['POST'])
def set_model():
    model_name = "QB_MMDD_Model"  # Changed to QueryBlazer
    
    # Lazy-load MiniCPM if requested (but we're using QB_MMDD_Model)
    if model_name == 'MiniCPM_Model' and model_name not in models:
        try:
            models["MiniCPM_Model"] = MiniCPM_Model()
        except Exception as e:
            print(f"Error initializing MiniCPM_Model: {e}")
            model_name = 'QB_MMDD_Model'  # Fallback to QueryBlazer
    
    model = models[model_name]
    session['model_name'] = model_name
    return jsonify({'slow_model': model.slow_model, 'model_name': model_name})

@app.route('/complete', methods=['POST'])
def complete():
    data = request.json
    text = data.get('text', '')
    chat_history = data.get('chat_history', [])
    
    # Use model from session if set, otherwise default to QB_MMDD_Model
    model_name = session.get('model_name', 'QB_MMDD_Model')
    if model_name not in models:
        # Lazy-load MiniCPM if requested
        if model_name == 'MiniCPM_Model':
            try:
                models["MiniCPM_Model"] = MiniCPM_Model()
            except Exception as e:
                print(f"Error initializing MiniCPM_Model: {e}")
                model_name = 'QB_MMDD_Model'  # Fallback to QueryBlazer
        else:
            model_name = 'QB_MMDD_Model'  # Fallback to QueryBlazer
    
    model = models[model_name]
    
    # MiniCPM supports image_path, other models don't
    if model_name == 'MiniCPM_Model' and session.get('image_path'):
        suggestion = model.complete(text, chat_history, session.get('image_path'))
    else:
        # For QueryBlazer and other models that don't support image_path
        suggestion = model.complete(text, chat_history)
    
    # Store only the prediction part, not the text + prediction
    # The frontend will handle appending it to the user's text
    return jsonify({'suggestion': suggestion})

@app.route('/log_event', methods=['POST'])
def log_event():
    data = request.get_json()
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            session.get('session_id'),
            datetime.utcnow().isoformat(),
            data.get('event_type'),
            json.dumps(data.get('details')),
            data.get('event_id'),
            data.get('parent_event_id')
        ])
    return jsonify({'status': 'ok'})

@app.route('/imagechat_samples/images/<filename>')
def serve_imagechat_image(filename):
    images_dir = os.path.join(USERSTUDY_DIR, 'imagechat_samples', 'images')
    return send_from_directory(images_dir, filename)

@app.route('/images/<path:filename>')
def serve_image(filename):
    images_dir = os.path.join(USERSTUDY_DIR, 'imagechat_samples', 'images')
    return send_from_directory(images_dir, filename)

if __name__ == '__main__':
    port = 5000
    
    print(f" * Running on http://0.0.0.0:{port}/ (Press CTRL+C to quit)")
    print(f" * To access this from your laptop, use SSH port forwarding:")
    print(f"   ssh -L {port}:localhost:{port} username@remote_server_ip")
    
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)
