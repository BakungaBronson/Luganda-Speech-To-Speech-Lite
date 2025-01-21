import torch
from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import io
import json
import librosa
import torchaudio
import logging
from model_manager import ModelManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Luganda Speech-to-Speech API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranscriptionConfig(BaseModel):
    beam_size: int = Field(default=5, ge=1, le=10)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)

class SynthesisConfig(BaseModel):
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    pitch: float = Field(default=1.0, ge=0.5, le=2.0)

class AudioTranscriptionResponse(BaseModel):
    text: str
    language: str = "lg"
    confidence: float

@app.on_event("startup")
async def startup_event():
    """Initialize the model manager on startup"""
    try:
        ModelManager.get_instance()
        logger.info(f"Models loaded successfully. Using device: {ModelManager.get_instance().device}")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

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
            "openai_compatible": "/v1/chat/completions"
        }
    }

@app.post("/api/v1/transcribe", response_model=AudioTranscriptionResponse)
async def transcribe_audio(
    file: UploadFile,
    config: Optional[str] = Form(None)
):
    """Transcribe audio to text with configurable parameters"""
    if not file:
        raise HTTPException(status_code=400, detail="No audio file provided")
    
    try:
        # Parse config if provided
        transcription_config = TranscriptionConfig()
        if config:
            try:
                config_dict = json.loads(config)
                transcription_config = TranscriptionConfig(**config_dict)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid config JSON")
            except Exception as e:
                raise HTTPException(status_code=422, detail=str(e))

        from pydub import AudioSegment
        import tempfile
        import shutil

        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        input_path = os.path.join(temp_dir, "input.webm")
        output_path = os.path.join(temp_dir, "output.wav")
        
        try:
            # Save uploaded file temporarily
            content = await file.read()
            if len(content) == 0:
                raise HTTPException(status_code=400, detail="Empty audio file received")
                
            with open(input_path, "wb") as temp_file:
                temp_file.write(content)
            
            try:
                # Convert WebM to WAV using pydub
                audio = AudioSegment.from_file(input_path)
                if len(audio) == 0:
                    raise HTTPException(status_code=400, detail="Invalid audio file: zero duration")
                
                audio = audio.set_frame_rate(16000)  # Set sample rate to 16kHz
                audio.export(output_path, format="wav")
                
                # Load audio using librosa
                audio_input, sr = librosa.load(output_path, sr=16000)
                if len(audio_input) == 0:
                    raise HTTPException(status_code=400, detail="Failed to process audio: empty signal")
                
            except Exception as e:
                logger.error(f"Audio processing error: {str(e)}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to process audio file: {str(e)}"
                )
                
        except Exception as e:
            logger.error(f"File handling error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing upload: {str(e)}"
            )
            
        finally:
            # Clean up temporary files
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.error(f"Error cleaning up temporary files: {str(e)}")
        
        # Get model manager instance
        model_manager = ModelManager.get_instance()
        
        # Transcribe audio with config parameters
        transcription = model_manager.transcribe_audio(
            audio_input,
            sample_rate=sr,
            beam_size=transcription_config.beam_size,
            temperature=transcription_config.temperature
        )
        
        # Calculate simple confidence score (placeholder)
        confidence = 0.95  # In a real implementation, this would come from the model
        
        return {
            "text": transcription,
            "language": "lg",
            "confidence": confidence
        }
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/synthesize")
async def synthesize_speech(
    background_tasks: BackgroundTasks,
    text: str = None,
    config: Optional[SynthesisConfig] = None
):
    """Convert text to speech with configurable parameters"""
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    try:
        # Get model manager instance
        model_manager = ModelManager.get_instance()
        
        # Generate speech
        waveforms = model_manager.synthesize_speech(text)
        
        # Apply speed and pitch modifications if config provided
        if config:
            if config.speed != 1.0:
                # Simple speed modification using interpolation
                old_length = waveforms.shape[0]
                new_length = int(old_length / config.speed)
                waveforms = torch.nn.functional.interpolate(
                    waveforms.unsqueeze(0).unsqueeze(0),
                    size=new_length,
                    mode='linear',
                    align_corners=False
                ).squeeze()
        
        # Convert to bytes
        buffer = io.BytesIO()
        torchaudio.save(buffer, waveforms.unsqueeze(0), 22050, format='wav')
        buffer.seek(0)
        
        # Clean up background tasks
        background_tasks.add_task(buffer.close)
        
        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={
                'Content-Disposition': 'attachment; filename="speech.wav"'
            }
        )
        
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)