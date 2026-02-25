import os
import torch
import torch.nn as nn
import symusic
from miditok import REMI, TokenizerConfig

# 1. ΡΥΘΜΙΣΕΙΣ (CONFIG) ΚΑΙ TOKENIZER
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SEQ_LEN = 2048

TOKENIZER_PARAMS = {
    "pitch_range": (21, 109), 
    "beat_res": {(0, 4): 8}, 
    "num_velocities": 32,
    "special_tokens": ["PAD", "BOS", "EOS", "MASK"], 
    "use_chords": True, 
    "use_rests": True,
    "use_tempos": True, 
    "use_time_signatures": True, 
    "use_programs": True
}

# Αρχικοποίηση του REMI Tokenizer
tokenizer = REMI(TokenizerConfig(**TOKENIZER_PARAMS))


class MidiTransformerSmall(nn.Module):
    def __init__(self, vocab_size, pad_token_id):
        super().__init__()
        self.pad_token_id = pad_token_id
        
        # Ενσωματώσεις (Embeddings) για τα Tokens και τη Θέση (Positional)
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=pad_token_id)
        self.pos_embedding = nn.Embedding(2049, 128)
        
        # Το ειδικό token ταξινόμησης (παρόμοιο με το [CLS] του BERT)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 128))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.dropout = nn.Dropout(0.1)
        # Τελικό επίπεδο για την παραγωγή ενός μόνο αριθμού (Regression)
        self.fc_out = nn.Linear(128, 1)

    def forward(self, x):
        bsz, seq_len = x.size()
        
        tok_emb = self.embedding(x)
        positions = torch.arange(1, seq_len + 1, device=x.device).unsqueeze(0).repeat(bsz, 1)
        pos_emb = self.pos_embedding(positions)
        
        cls_tokens = self.cls_token.repeat(bsz, 1, 1)
        cls_pos = self.pos_embedding(torch.zeros((bsz, 1), dtype=torch.long, device=x.device))
        
        # Συνένωση του [CLS] token με την υπόλοιπη ακολουθία
        x_emb = torch.cat([cls_tokens + cls_pos, tok_emb + pos_emb], dim=1)
        
        # Δημιουργία μάσκας για να αγνοηθούν τα PAD tokens
        pad_mask = torch.cat([torch.zeros((bsz, 1), dtype=torch.bool, device=x.device), (x == self.pad_token_id)], dim=1)
        
        x_trans = self.transformer(x_emb, src_key_padding_mask=pad_mask)
        
        # Εξάγουμε μόνο το output που αντιστοιχεί στο [CLS] token (θέση 0)
        return self.fc_out(self.dropout(x_trans[:, 0, :])).squeeze(-1)


# 3. ΦΟΡΤΩΣΗ ΜΟΝΤΕΛΟΥ
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "difficulty_model.pth")

# Global μεταβλητή για να φορτωθεί το μοντέλο μόνο μία φορά στη μνήμη
_global_model = None

def load_model():
    global _global_model
    if _global_model is not None:
        return _global_model
        
    try:
        if os.path.exists(MODEL_PATH):
            model = MidiTransformerSmall(vocab_size=tokenizer.vocab_size, pad_token_id=tokenizer.pad_token_id).to(device)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device)['model_state_dict'])
            model.eval() # Θέτουμε το μοντέλο σε κατάσταση αξιολόγησης (κλείνει το Dropout)
            _global_model = model
            print("Transformer Model for difficulty estimation was loaded")
            return model
        else:
            print(f"Model {MODEL_PATH} was not found")
            return None
    except Exception as e:
        print(f"Error loading difficulty model: {e}")
        return None


# 4. ΒΟΗΘΗΤΙΚΕΣ ΣΥΝΑΡΤΗΣΕΙΣ (INFERENCE)
def get_difficulty_label(score):
    """Μετατρέπει το συνεχές σκορ (0-10) σε κατηγορηματική ετικέτα."""
    if score < 2.5: return "Beginner"
    elif score < 4.5: return "Easy"
    elif score < 6.5: return "Intermediate"
    elif score < 8.5: return "Advanced"
    return "Expert"

def predict_difficulty(midi_path):
    """
    Η κύρια συνάρτηση (API) που θα καλείται από το GUI (app.py).
    Διαβάζει το MIDI, το μετατρέπει σε Tokens και το περνάει από το Transformer.
    """
    model = load_model()
    
    if model is None: 
        return 0.0, "Model is missing"
        
    try:
        # 1. Φόρτωση και Tokenization
        midi = symusic.Score(midi_path)
        
        # Εξαγωγή των IDs από την ακολουθία των tokens
        tok_seq = tokenizer(midi)
        tokens = (tok_seq.ids if hasattr(tok_seq, 'ids') else tok_seq)[:MAX_SEQ_LEN]
        
        # 2. Padding (Γέμισμα με μηδενικά για να φτάσει το MAX_SEQ_LEN)
        tokens += [tokenizer.pad_token_id] * (MAX_SEQ_LEN - len(tokens))
        
        # 3. Δημιουργία Tensor και μεταφορά στη συσκευή (CPU/GPU)
        x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
        
        # 4. Πρόβλεψη (Inference)
        with torch.no_grad():
            raw_output = model(x).item()
            # Κλιμάκωση (Scaling) της εξόδου του δικτύου στο διάστημα [1, 10]
            score = max(0.0, min(10.0, raw_output * 9.0 + 1.0))
            
        final_score = round(score, 2)
        label = get_difficulty_label(final_score)
        return final_score, label
        
    except Exception as e:
        print(f"Error: {e}")
        return "0.0", {"Error": str(e)}