import os
import sys
import subprocess
import pretty_midi
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from midi2audio import FluidSynth

# ==============================================================================
# 1. ΔΙΑΧΕΙΡΙΣΗ ΔΙΑΔΡΟΜΩΝ (PATH MANAGEMENT)
# ==============================================================================
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'src'))

LOGO_PATH = os.path.join(project_root, 'fig', 'PlayWise.png')
UPLOAD_FOLDER = os.path.join(project_root, 'data', 'uploads')
SOUNDFONT_PATH = os.path.join(project_root, 'data', 'FluidR3_GM.sf2')
from src.dsp_transcriber import transcribe_audio_dsp
from src.dl_transcriber import transcribe_audio_dl
from src.difficulty_eval import predict_difficulty
# ΜΕΤΑΓΡΑΦΉ ΜΕ TRANSKUN: True, ΜΕΤΑΓΡΑΦΉ ΜΕ DSP: False
transcriber_flag = True 

# ==============================================================================
# 2. ΡΥΘΜΙΣΕΙΣ FLASK (CONFIGURATION)
# ==============================================================================
app = Flask(__name__)
CORS(app)
# Δημιουργία του φακέλου uploads αν δεν υπάρχει
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app_state = {
    'wav_filepath': None,
    'midi_filepath': None,
    'current_base_name': 'transcribed'
}

# ==============================================================================
# 3. ΒΟΗΘΗΤΙΚΕΣ ΣΥΝΑΡΤΗΣΕΙΣ (HELPER FUNCTIONS)
# ==============================================================================
def parse_midi_data(filepath):
    try:
        midi_data = pretty_midi.PrettyMIDI(filepath)
        all_notes = [{'start': note.start, 'end': note.end, 'pitch': note.pitch, 'velocity': note.velocity}
                     for instrument in midi_data.instruments if not instrument.is_drum for note in instrument.notes]
        return sorted(all_notes, key=lambda x: x['start']), midi_data.get_end_time()
    except Exception as e:
        print(f"MIDI Parsing Error: {e}")
        return None, None


def process_midi_to_audio(midi_path, output_name):
    try:
        wav_path = os.path.join(UPLOAD_FOLDER, f"{output_name}.wav")
        
        command = [
            'fluidsynth',
            '-ni',
            '-T', 'wav',
            '-F', wav_path,
            SOUNDFONT_PATH, 
            midi_path
        ]

        subprocess.run(command, capture_output=True, text=True, shell=True)

        if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
            return wav_path
        return None
    except:
        return None
# ==============================================================================
# 4. ΔΡΟΜΟΛΟΓΗΣΕΙΣ FLASK (API ROUTES)
# ==============================================================================

@app.route('/')
def index():
    """Σερβίρει το index.html από τον φάκελο templates"""
    return render_template('index.html')

@app.route('/api/logo')
def get_logo():
    """Σερβίρει το λογότυπο με path check"""
    if os.path.exists(LOGO_PATH):
        return send_file(LOGO_PATH, mimetype='image/png')
    return ('Logo not found', 404)

@app.route('/api/upload_audio', methods=['POST'])
def upload_audio():
    if 'file' not in request.files: return jsonify({'success': False, 'error': 'No file'})
    file = request.files['file']

    original_name = secure_filename(file.filename)
    if not original_name: original_name = "audio_upload"
    base_name = os.path.splitext(original_name)[0]

    audio_path = os.path.join(UPLOAD_FOLDER, original_name)
    midi_filename = f"{base_name}_MIDI.mid"
    midi_path = os.path.join(UPLOAD_FOLDER, midi_filename)
    file.save(audio_path)

    # Παρακάτω επιλέγεται το είδος του μεταγραφέα
    transcriber = transcribe_audio_dl if transcriber_flag else transcribe_audio_dsp
    success = transcriber(audio_path, midi_path)
    if not success:
        return jsonify({'success': False, 'error': 'Transkun transcription failed'})

    all_notes, duration = parse_midi_data(midi_path)
    if not all_notes: return jsonify({'success': False, 'error': 'Failed to parse MIDI'})

    wav_path = process_midi_to_audio(midi_path, f"{base_name}_MIDI")
    app_state.update({'wav_filepath': wav_path, 'midi_filepath': midi_path, 'current_base_name': base_name})

    score, label = predict_difficulty(midi_path)

    return jsonify({
        'success': True,
        'filename': midi_filename,
        'notes': all_notes,
        'duration': duration,
        'difficulty_score': score,
        'difficulty_label': label
    })

@app.route('/api/upload_midi', methods=['POST'])
def upload_midi():
    if 'file' not in request.files: return jsonify({'success': False, 'error': 'No file'})
    file = request.files['file']

    original_name = secure_filename(file.filename)
    if not original_name: original_name = "midi_upload"
    base_name = os.path.splitext(original_name)[0]

    midi_filename = f"{base_name}_MIDI.mid"
    midi_path = os.path.join(UPLOAD_FOLDER, midi_filename)
    file.save(midi_path)

    all_notes, duration = parse_midi_data(midi_path)
    if not all_notes: return jsonify({'success': False, 'error': 'Failed to parse'})

    wav_path = process_midi_to_audio(midi_path, f"{base_name}_MIDI")
    app_state.update({'wav_filepath': wav_path, 'midi_filepath': midi_path, 'current_base_name': base_name})

    score, label = predict_difficulty(midi_path)
    return jsonify({
        'success': True,
        'filename': midi_filename,
        'notes': all_notes,
        'duration': duration,
        'difficulty_score': score,
        'difficulty_label': label
    })

@app.route('/api/audio')
def get_audio():
    wav_path = app_state.get('wav_filepath')
    if wav_path and os.path.exists(wav_path):
        return send_file(wav_path, mimetype='audio/wav')
    return ('', 404)

@app.route('/api/download_midi')
def download_midi():
    midi_path = app_state.get('midi_filepath')
    base_name = app_state.get('current_base_name', 'transcribed')
    download_name = f"{base_name}_MIDI.mid"
    return send_file(midi_path, as_attachment=True, download_name=download_name) if midi_path and os.path.exists(midi_path) else ('', 404)

# ==============================================================================
# ΕΚΚΙΝΗΣΗ ΔΙΑΚΟΜΙΣΤΗ (SERVER STARTUP)
# ==============================================================================
if __name__ == '__main__':
    print("="*60)
    print(" 🎹 PlayWise AI - Εκκίνηση Τοπικού Διακομιστή (Flask)")
    print(" 👉 Ανοίξτε τον browser σας στη διεύθυνση: http://127.0.0.1:5000")
    print("="*60)
    
    app.run(host='0.0.0.0', port=5000)