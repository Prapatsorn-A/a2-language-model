import torch
import torchtext
import pickle
from flask import Flask, render_template, request, jsonify
import math

# Import the LSTM model from model.py
from model import LSTMLanguageModel

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
model = LSTMLanguageModel(vocab_size=3449, emb_dim=1024, hid_dim=1024, num_layers=2, dropout_rate=0.65).to(device)
model.load_state_dict(torch.load('best-val-lstm_lm.pt', map_location=device))

# Load vocabulary
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

# Initialize a tokenizer
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

# Text generation function
def generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    tokens = tokenizer(prompt)
    indices = [vocab[t] for t in tokens]
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)
            prediction_idx = torch.multinomial(probs, num_samples=1).item()

            while prediction_idx == vocab['<unk>']:
                prediction_idx = torch.multinomial(probs, num_samples=1).item()

            if prediction_idx == vocab['<eos>']:
                break

            indices.append(prediction_idx)

    itos = vocab.get_itos()
    generated_text = ' '.join([itos[i] for i in indices])
    return generated_text

# Flask app setup
app = Flask(__name__)

# Home route, which renders the HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Generate route, which handles the POST request for text generation
@app.route('/generate', methods=['POST'])
def generate_text():
    # Retrieve prompt and temperature from the form data
    prompt = request.form['prompt']
    temperature = float(request.form['temperature'])

    # Generate the text using the provided prompt and temperature
    generated_text = generate(prompt, max_seq_len=100, temperature=temperature, model=model, tokenizer=tokenizer, vocab=vocab, device=device)
    
    # Return the generated text as a JSON response
    return jsonify({'generated_text': generated_text})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
