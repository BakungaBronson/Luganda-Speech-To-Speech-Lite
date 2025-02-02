# Luganda Speech-to-Speech Lite

A lightweight Luganda speech-to-speech system with support for Apple Silicon GPU acceleration and OpenAI-compatible APIs.

## Features

- Speech-to-Text and Text-to-Speech for Luganda language
- GPU acceleration support for Apple Silicon (M1/M2) via PyTorch MPS
- OpenAI-compatible API endpoints
- Modern React frontend with Shadcn UI
- Docker support with multi-architecture builds

## Models

This project uses the following open-source models:

### Speech-to-Text
- Model: [allandclive/whisper-tiny-luganda-v2](https://huggingface.co/allandclive/whisper-tiny-luganda-v2)
- Type: Fine-tuned Whisper model optimized for Luganda speech recognition

### Text-to-Speech
- Text to Mel-Spectrogram: [Sunbird/sunbird-lug-tts](https://huggingface.co/Sunbird/sunbird-lug-tts)
  - Type: Tacotron2 model trained for Luganda speech synthesis
- Vocoder: [speechbrain/tts-hifigan-ljspeech](https://huggingface.co/speechbrain/tts-hifigan-ljspeech)
  - Type: HiFiGAN vocoder for high-quality waveform generation

## Prerequisites

- Python 3.9+ (3.10+ recommended for Apple Silicon)
- Node.js 18+
- Docker and Docker Compose (optional)
- FFmpeg

## Installation

### Using Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd luganda-speech-to-speech-lite
   ```

2. Start the services using Docker Compose:
   ```bash
   # For Apple Silicon users (enables GPU acceleration)
   USE_MPS=true docker-compose up --build

   # For other platforms
   docker-compose up --build
   ```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### Manual Installation with Python Virtual Environment

1. Create and activate a Python virtual environment:
   ```bash
   # Create a new virtual environment
   python -m venv venv

   # Activate the virtual environment
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

2. Set up the backend:
   ```bash
   # Install Python dependencies
   pip install -r requirements.txt

   # Optional: Enable GPU acceleration on Apple Silicon
   export USE_MPS=true

   # Start the FastAPI server
   cd backend
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

3. Set up the frontend:
   ```bash
   # Install Node.js dependencies
   cd frontend
   npm install

   # Start the development server
   npm run dev
   ```

4. Deactivate the virtual environment when done:
   ```bash
   deactivate
   ```

Note: When running without Docker, make sure to manage your Python virtual environment properly:
- Always activate the virtual environment before running the backend server
- Use `pip list` to verify installed packages
- If you install new packages, update requirements.txt:
  ```bash
  pip freeze > requirements.txt
  ```
- To recreate the environment on another machine:
  ```bash
  python -m venv venv
  source venv/bin/activate  # or .\venv\Scripts\activate on Windows
  pip install -r requirements.txt
  ```

## API Endpoints

### Standard Endpoints

- `POST /api/v1/transcribe`: Convert speech to text
- `POST /api/v1/synthesize`: Convert text to speech

### OpenAI-Compatible Endpoint

- `POST /v1/chat/completions`: OpenAI-style chat completions endpoint

## Environment Variables

### Backend

- `USE_MPS`: Enable GPU acceleration on Apple Silicon (default: false)

### Frontend

- `VITE_API_URL`: Backend API URL (default: http://localhost:8000)

## Development

### Backend Development

The backend is built with FastAPI and uses PyTorch for model inference. Key files:

- `backend/main.py`: Main FastAPI application
- `requirements.txt`: Python dependencies

### Frontend Development

The frontend is built with React, Vite, and Shadcn UI. Key files:

- `frontend/src/App.tsx`: Main application component
- `frontend/src/components/`: UI components
- `frontend/src/lib/`: Utility functions

## Docker Support

The project includes Docker support with multi-architecture builds:

- `Dockerfile`: Backend service configuration
- `frontend/Dockerfile`: Frontend service configuration
- `docker-compose.yml`: Service orchestration
- `frontend/nginx.conf`: Nginx configuration for the frontend

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
