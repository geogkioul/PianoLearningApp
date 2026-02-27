# 🎹 PlayWise: Piano Learning Application with Automatic Transcription and Piece Difficulty Estimation

Welcome to the PlayWise repository. This software is a complete system (Full-Stack Web Application) designed to bridge the gap between acoustic music performance (Audio) and digital sheet music (MIDI).
The application was developed as part of coursework for the Sound and Image Technology course at the ECE Department of the Aristotle University of Thessaloniki for the winter semester of 2025-26.

It provides Automatic Music Transcription capabilities using Deep Learning models, while also evaluating the technical difficulty level of the generated piece.

## 🏛️ System Architecture

The application follows the industry-standard MVC (Model-View-Controller) pattern:

- **Front-End (View):** An interactive user interface built with HTML/CSS/JavaScript. It communicates with the server through asynchronous calls (AJAX/Fetch API) and visualizes music information (Piano Roll) using dynamic graphics (Canvas API).
- **Back-End (Controller):** A Python Flask-based server, responsible for request routing, file management, and audio synthesis (via FluidSynth).
- **Core Logic (Model):** The modules in the `src/` folder that implement heavy mathematical and algorithmic processing (PyTorch, Librosa, SciPy).

## ⚙️ 1. Prerequisites

Before you begin, you must install two operating system-level dependencies. These are not Python packages, but standalone programs that our system calls in the background:

- **FFmpeg:** Required for the librosa library. It decodes compressed audio files (such as `.mp3`) into clean numeric sequences (raw arrays) that our algorithms can read.
	- **Windows:** Download it from `gyan.dev` and add it to the Windows PATH.
	- **Ubuntu/Debian:** `sudo apt-get install ffmpeg`
	- **macOS:** `brew install ffmpeg`

- **FluidSynth:** A software synthesizer. It converts MIDI symbols into real audio waves (`.wav`) so you can listen to them through your web browser.
	- **Windows/macOS/Linux:** See the detailed guide in the section below.

## 🔊 Installing & Configuring FluidSynth

The application uses FluidSynth to convert MIDI files into playable audio (`.wav`) inside the browser.

### Windows

1. Go to the FluidSynth [**GitHub Releases**](https://github.com/FluidSynth/fluidsynth/releases) page and download the latest `.zip` package for Windows.
2. Extract the archive to a stable folder (e.g. `C:\tools\fluidsynth`).
3. Locate the `bin` folder inside the extracted directory (e.g. `C:\tools\fluidsynth\bin`).
4. Add that path to **System Environment Variables** → **Path**.
5. Restart your terminal (or VS Code terminal) so the updated PATH is loaded.

### macOS

Install FluidSynth with Homebrew:

```bash
brew install fluidsynth
```

### Linux (Ubuntu/Debian)

Install FluidSynth with `apt`:

```bash
sudo apt install fluidsynth
```

### Project SoundFont Requirement

PlayWise requires a SoundFont file named `FluidR3_GM.sf2` to be located in the `data/` directory:

```text
data/FluidR3_GM.sf2
```

Without this file, MIDI-to-audio conversion will fail.

### Installation Verification

After installation (and after restarting terminal on Windows), verify that FluidSynth is available:

```bash
fluidsynth --version
```

If the command outputs a version number, FluidSynth is installed correctly and available in PATH.

## 🚀 2. Installation Guide

### Step 1: Clone Repository
Open your terminal and run:

```bash
git clone https://github.com/geogkioul/PianoLearningApp.git
cd playwise
```

### Step 2: Create a Virtual Environment
It is best practice to isolate the project libraries to avoid conflicts with other programs on your computer:

```bash
# Create an environment named 'venv'
python -m venv venv

# Activation
# For Windows:
venv\Scripts\activate
# For Linux/macOS:
source venv/bin/activate
```

### Step 3: Install Python Packages
With the environment activated, install the `requirements.txt` dependencies:

```bash
pip install -r requirements.txt
```
### Step 4: Manual Download of Required Assets
Due to GitHub file size limits (>100MB) and respect for third-party copyrights, the sound bank is not included in the repository. You must add it manually:
1. You must download a SoundFont file (recommended: [FluidR3_GM.sf2](https://member.keymusician.com/Member/FluidR3_GM/))
2. After downloading, **move the file into the project's data/ folder**.
3. Make sure the file is named exactly `FluidR3_GM.sf2`

Your final folder structure should look roughly like this:

```text
playwise/
├── data/
│   ├── uploads/          (Auto-generated temporary storage folder)
│   └── FluidR3_GM.sf2    
├── models/
│   └── transformer.pth   
├── src/
│   ├── dl_transcriber.py
│   ├── dsp_transcriber.py
│   └── difficulty_eval.py
├── gui/
│   ├── app.py
│   └── templates/
│       └── index.html
├── requirements.txt
└── README.md
```

## 🏃 3. Running the Application

Make sure you are in the project root folder and that `venv` is activated. Run the Flask server:

```bash
python app.py
```

In the terminal, you will see a message:
`Open your browser at: http://127.0.0.1:5000`

Open a web browser (Chrome, Firefox, Safari) and go to that address. The application's graphical interface will appear.

## 💡 4. Usage Instructions

### A. Audio Transcription (Audio to MIDI)

Click "Choose File" and upload a clean piano piece in `.mp3` or `.wav` format.

Press "Transcribe".
Note: If you use the Transkun Deep Learning transcription model without a Graphics Card (GPU), the process may take several minutes depending on the piece duration.

### B. Analyze an Existing MIDI

If you already have a `.mid` file (e.g. from recording software of a digital piano) and you only want to estimate its difficulty or visualize it:

Use the right panel "Load Existing MIDI".

Select your file and click "Analyze MIDI". The process is almost instantaneous, since the computationally heavy acoustic transcription step is skipped.

### C. Results Overview

Once the analysis is complete:

- The Difficulty Level (e.g. Beginner, Expert) will be displayed along with a score from 1-10 (produced by the Custom Transformer model).
- You can play back the piece inside the browser (the system has converted the MIDI into audio using your SoundFont).
- The Piano Roll will interactively display (visual blocks) which notes are played, helping the student visually understand timing values and intervals.
- Press "Download MIDI" to save the final score to your computer and open it in any program (e.g. MuseScore, Synthesia, DAW).

## 🛠️ 5. Troubleshooting

- **FileNotFoundError: [WinError 2] The system cannot find the file specified when calling Transkun:**
	Make sure the transkun library was installed successfully via pip, and that Python Scripts paths (e.g. `C:\Python39\Scripts`) are included in your system Environment Variables.

- **The piece plays with no sound / FluidSynth throws an error:**
	Make sure you have installed fluidsynth on your operating system and that the file `FluidR3_GM.sf2` exists inside the `data/` folder.

- **Very slow transcription:**
	PyTorch automatically checks whether CUDA technology is available. If you run the code on a standard CPU, neural network processing can take several minutes. Using an NVIDIA GPU significantly reduces transcription time.
