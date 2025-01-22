import warnings
from fastapi import FastAPI, UploadFile, HTTPException, Form, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Optional
import magic
import torch
import librosa
import logging
import os
import subprocess
from dotenv import load_dotenv
from functools import lru_cache
from model_manager import ModelManager
import torchaudio
import io
import soundfile as sf
import tempfile
import json
import openai

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure default OpenAI key from environment
DEFAULT_OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# Ignore warnings from finetuned model
warnings.filterwarnings('ignore')

app = FastAPI(title="Luganda Speech-to-Speech API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_MIMETYPES = (
    "audio/aac",
    "audio/mpeg",
    "audio/ogg",
    "audio/webm",
    "video/webm",
    "audio/mp4",
    "audio/wav",
    "audio/x-wav"
)

SYSTEM_PROMPT = """You are a knowledgeable Luganda teacher with knowledge of both English and Luganda. 
The user will send you text in Luganda and you will respond in Luganda as well. 
The Luganda given comes from a speech to text output and will not be exact to the word, 
so use the nearest word to it to generate a response. Your responses must be concise."""

def convert_audio_to_wav(input_bytes):
    """Convert audio bytes to WAV format using ffmpeg"""
    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as input_file:
        input_file.write(input_bytes)
        input_path = input_file.name

    output_path = input_path + '.wav'
    try:
        subprocess.run([
            'ffmpeg', '-y',
            '-i', input_path,
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            output_path
        ], check=True, capture_output=True)

        with open(output_path, 'rb') as f:
            wav_bytes = f.read()

        return wav_bytes

    finally:
        # Clean up temporary files
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)

@app.get("/")
async def root():
    """Get API information and status"""
    model_manager = ModelManager.get_instance()
    return {
        "status": "ok",
        "device": str(model_manager.device),
        "endpoints": {
            "speech_to_text": "/api/v1/transcribe",
            "text_to_speech": "/api/v1/synthesize",
            "chat": "/api/v1/chat"
        }
    }

@app.post("/api/v1/transcribe")
async def transcribe_audio(
    file: UploadFile,
    config: str = Form(None)
):
    """Transcribe audio to text"""
    if not file:
        raise HTTPException(status_code=400, detail="No audio file provided")
    
    try:
        # Read audio file
        content = await file.read()
        
        try:
            # Convert to WAV first
            wav_content = convert_audio_to_wav(content)
            audio_stream = io.BytesIO(wav_content)
            
            # Load audio using librosa
            audio_input, sr = librosa.load(audio_stream, sr=16000)
            
            if len(audio_input) == 0:
                raise HTTPException(status_code=400, detail="Empty audio file")
                
            # Get model manager instance
            model_manager = ModelManager.get_instance()
            
            # Transcribe audio
            transcription = model_manager.transcribe_audio(
                audio_input,
                sample_rate=sr
            )
            
            return {
                "text": transcription,
                "language": "lg"
            }
                
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg conversion failed: {e.stderr.decode()}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to convert audio: {e.stderr.decode()}"
            )
            
        except Exception as e:
            logger.error(f"Audio processing error: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to process audio: {str(e)}"
            )
            
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/chat")
async def chat(
    request: Request,
    x_openai_key: Optional[str] = Header(None, alias="X-OpenAI-Key")
):
    """Process text with OpenAI API"""
    try:
        # Parse request body
        data = await request.json()
        logger.info(f"Chat request data: {data}")
        
        text = data.get('text')
        if not text:
            logger.error("No text provided in request")
            raise HTTPException(status_code=400, detail="No text provided")
        
        try:
            # Use provided API key or fall back to default
            api_key = x_openai_key or DEFAULT_OPENAI_KEY
            if not api_key:
                raise HTTPException(
                    status_code=400,
                    detail="OpenAI API key not provided and no default key configured"
                )
            
            # Configure OpenAI with the API key
            openai.api_key = api_key
            
            # Call OpenAI API
            logger.info(f"Sending to OpenAI: {text}")
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                temperature=1,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            
            response_text = response['choices'][0]['message']['content']
            logger.info(f"OpenAI response: {response_text}")
            
            return {
                "text": response_text
            }
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            logger.error("Error details:", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process with OpenAI: {str(e)}"
            )
            
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON in request body: {str(e)}"
        )
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/synthesize")
async def synthesize_speech(request: Request):
    """Convert text to speech"""
    try:
        # Log raw request body
        body = await request.body()
        logger.info(f"Raw request body: {body.decode()}")
        
        # Parse request body
        data = await request.json()
        logger.info(f"Parsed request data: {data}")
        
        text = data.get('text')
        if not text:
            logger.error("No text provided in request")
            raise HTTPException(status_code=400, detail="No text provided")
        
        logger.info(f"Synthesizing speech for text: {text}")
        
        try:
            # Get model manager instance
            model_manager = ModelManager.get_instance()
            
            # Generate speech
            logger.info("Starting speech synthesis")
            waveforms = model_manager.synthesize_speech(text)
            logger.info(f"Generated waveforms shape: {waveforms.shape}")
            
            # Convert to bytes
            buffer = io.BytesIO()
            logger.info("Converting waveforms to WAV format")
            
            # Save as WAV file (waveforms is already 2D after squeeze in model_manager)
            torchaudio.save(buffer, waveforms, 22050, format='wav')
            buffer.seek(0)
            
            logger.info("Successfully generated speech audio")
            return StreamingResponse(
                buffer,
                media_type="audio/wav",
                headers={
                    'Content-Disposition': 'attachment; filename="speech.wav"'
                }
            )
            
        except Exception as e:
            logger.error(f"Speech synthesis error: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error("Error traceback:", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to synthesize speech: {str(e)}"
            )
            
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON in request body: {str(e)}"
        )
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)