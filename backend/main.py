import warnings
import asyncio
from fastapi import FastAPI, UploadFile, HTTPException, Form, Request, Header, Depends, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Union, Any, Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
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
import time
from database import init_db, get_session, create_conversation, add_message, get_conversation, list_conversations, delete_conversation, update_last_assistant_message
from sqlalchemy.ext.asyncio import AsyncSession

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI-compatible API models
class ChatMessage(BaseModel):
    role: str = Field(..., description="The role of the message sender (system/user/assistant)")
    content: str = Field(..., description="The content of the message")
    name: str = None

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="The model to use for completion")
    messages: List[ChatMessage] = Field(..., description="The messages to generate a response for")
    temperature: float = Field(1.0, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(256, gt=0, description="Maximum number of tokens to generate")
    top_p: float = Field(1.0, ge=0.0, le=1.0, description="Nucleus sampling threshold")
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0, description="Presence penalty")
    
class ErrorResponse(BaseModel):
    error: Dict[str, Any]

class Usage(BaseModel):
    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)

class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Usage

# OpenAI API error handler
async def openai_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    error_msg = str(exc)
    if isinstance(exc, HTTPException):
        status_code = exc.status_code
        error_type = "invalid_request_error" if status_code == 400 else "api_error"
    else:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        error_type = "server_error"
    
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": error_msg,
                "type": error_type,
                "code": status_code,
                "param": None
            }
        }
    )

# Configure environment and API settings
DEFAULT_OPENAI_KEY = os.getenv("OPENAI_API_KEY")
API_HOST = os.getenv("API_HOST", "http://localhost:8000")

# Configure audio storage and URLs
AUDIO_DIR = "audio_files"
TEMP_AUDIO_DIR = os.path.join(AUDIO_DIR, "temp")
AUDIO_BASE_URL = f"{API_HOST}/audio"
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

# Configure temporary file cleanup (files older than 1 hour)
TEMP_FILE_MAX_AGE = 3600  # 1 hour in seconds

def cleanup_temp_audio():
    """Clean up old temporary audio files"""
    now = time.time()
    for file in os.listdir(TEMP_AUDIO_DIR):
        file_path = os.path.join(TEMP_AUDIO_DIR, file)
        if os.path.getmtime(file_path) < now - TEMP_FILE_MAX_AGE:
            try:
                os.remove(file_path)
            except Exception as e:
                logger.error(f"Failed to remove temp file {file_path}: {str(e)}")

def get_audio_url(audio_path: str) -> str:
    """Convert audio file path to full URL"""
    if not audio_path:
        return None
    return f"{AUDIO_BASE_URL}/{audio_path}"

# Ignore warnings from finetuned model
warnings.filterwarnings('ignore')

app = FastAPI(title="Luganda Speech-to-Speech API")

# Add OpenAI-compatible error handler
app.add_exception_handler(Exception, openai_exception_handler)

# Mount audio files directory
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

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
so use the nearest word to it to generate a response."""

def save_audio_file(audio_data: bytes, conversation_id: Union[int, str]) -> str:
    """Save audio file and return its path"""
    # Handle temporary files
    if conversation_id == "temp":
        cleanup_temp_audio()  # Clean up old temp files
        filename = f"temp_{int(time.time())}.wav"
        filepath = os.path.join(TEMP_AUDIO_DIR, filename)
        with open(filepath, 'wb') as f:
            f.write(audio_data)
        return os.path.join("temp", filename)
    
    # Handle conversation files
    conv_dir = os.path.join(AUDIO_DIR, str(conversation_id))
    os.makedirs(conv_dir, exist_ok=True)
    
    # Generate unique filename
    filename = f"{int(time.time())}.wav"
    filepath = os.path.join(conv_dir, filename)
    
    # Save file
    with open(filepath, 'wb') as f:
        f.write(audio_data)
    
    # Return relative path from AUDIO_DIR
    return os.path.join(str(conversation_id), filename)

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    await init_db()

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
            "chat": "/api/v1/chat",
            "conversations": "/api/v1/conversations",
            "openai_compatible": {
                "chat": "/v1/chat/completions"
            }
        }
    }

@app.post("/api/v1/conversations")
async def start_conversation(session: AsyncSession = Depends(get_session)):
    """Start a new conversation"""
    conversation = await create_conversation()
    return {"conversation_id": conversation.id}

@app.get("/api/v1/conversations")
async def get_conversations(
    skip: int = 0,
    limit: int = 10,
    session: AsyncSession = Depends(get_session)
):
    """List all conversations"""
    return await list_conversations(skip=skip, limit=limit)

@app.get("/api/v1/conversations/{conversation_id}")
async def get_conversation_by_id(
    conversation_id: int,
    session: AsyncSession = Depends(get_session)
):
    """Get a specific conversation with full URLs"""
    conversation = await get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Add full URLs to the conversation response
    return {
        "id": conversation["id"],
        "started_at": conversation["started_at"],
        "url": f"{API_HOST}/chat/{conversation['id']}",
        "messages": [
            {
                "id": msg["id"],
                "role": msg["role"],
                "content": msg["content"],
                "audio_url": get_audio_url(msg["audio_path"]) if msg["audio_path"] else None,
                "created_at": msg["created_at"]
            }
            for msg in conversation["messages"]
        ]
    }

@app.delete("/api/v1/conversations/{conversation_id}")
async def delete_conversation_by_id(
    conversation_id: int,
    session: AsyncSession = Depends(get_session)
):
    """Delete a conversation"""
    # Delete audio files
    conv_dir = os.path.join(AUDIO_DIR, str(conversation_id))
    if os.path.exists(conv_dir):
        for file in os.listdir(conv_dir):
            os.remove(os.path.join(conv_dir, file))
        os.rmdir(conv_dir)
    
    await delete_conversation(conversation_id)
    return {"status": "success"}

@app.post("/api/v1/transcribe")
async def transcribe_audio(
    file: UploadFile,
    config: str = Form(None),
    conversation_id: Optional[int] = Form(None),
    session: AsyncSession = Depends(get_session)
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
            
            # Save message if conversation_id provided
            if conversation_id:
                await add_message(
                    conversation_id=conversation_id,
                    role="user",
                    content=transcription
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

@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    x_openai_key: Optional[str] = Header(None, alias="X-OpenAI-Key")
):
    """OpenAI-compatible chat completion endpoint"""
    try:
        # Use provided API key or fall back to default
        api_key = x_openai_key or DEFAULT_OPENAI_KEY
        if not api_key:
            raise HTTPException(
                status_code=401,
                detail="Authentication failed. Please provide a valid API key."
            )
        
        # Configure OpenAI with the API key
        openai.api_key = api_key
        
        # Get the last user message
        user_message = next(
            (msg for msg in reversed(request.messages) if msg.role == "user"),
            None
        )
        if not user_message:
            raise HTTPException(
                status_code=400,
                detail="No user message found in the conversation"
            )
        
        # Call OpenAI API with the provided parameters
        response = openai.ChatCompletion.create(
            model=request.model,
            messages=[{
                "role": "system",
                "content": SYSTEM_PROMPT
            }] + [{"role": m.role, "content": m.content} for m in request.messages],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty
        )
        
        # Convert OpenAI response to our format
        chat_response = ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatChoice(
                    index=idx,
                    message=ChatMessage(
                        role="assistant",
                        content=choice["message"]["content"]
                    ),
                    finish_reason=choice.get("finish_reason", "stop")
                )
                for idx, choice in enumerate(response["choices"])
            ],
            usage=Usage(
                prompt_tokens=response["usage"]["prompt_tokens"],
                completion_tokens=response["usage"]["completion_tokens"],
                total_tokens=response["usage"]["total_tokens"]
            )
        )
        
        return chat_response
        
    except Exception as e:
        logger.error(f"Chat completion error: {str(e)}")
        if isinstance(e, openai.error.OpenAIError):
            status_code = getattr(e, "http_status", 500)
            raise HTTPException(status_code=status_code, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/chat")
async def chat(
    request: Request,
    x_openai_key: Optional[str] = Header(None, alias="X-OpenAI-Key"),
    session: AsyncSession = Depends(get_session)
):
    """Process text with OpenAI API"""
    try:
        # Parse request body
        data = await request.json()
        logger.info(f"Chat request data: {data}")
        
        text = data.get('text')
        conversation_id = data.get('conversation_id')
        
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
                model="gpt-4o",  # Using standard GPT-4 model name
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
            
            response_data = {
                "text": response_text
            }

            # Generate audio for the response
            model_manager = ModelManager.get_instance()
            waveforms = model_manager.synthesize_speech(response_text)
            
            # Convert to WAV bytes
            buffer = io.BytesIO()
            torchaudio.save(buffer, waveforms, 22050, format='wav')
            buffer.seek(0)
            wav_bytes = buffer.read()
            
            # Save message and audio if conversation_id provided
            if conversation_id:
                # Save the audio file and get its path
                audio_path = save_audio_file(wav_bytes, str(conversation_id))
                
                # Save the message with audio path
                await add_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=response_text,
                    audio_path=audio_path
                )
                
                # Get updated conversation
                conversation = await get_conversation(conversation_id)
                
                # Add conversation data to response
                response_data["conversation"] = {
                    "id": conversation_id,
                    "url": f"{API_HOST}/chat/{conversation_id}",
                    "messages": [
                        {
                            "id": msg["id"],
                            "role": msg["role"],
                            "content": msg["content"],
                            "audio_url": get_audio_url(msg["audio_path"]) if msg["audio_path"] else None,
                            "created_at": msg["created_at"]
                        }
                        for msg in conversation["messages"]
                    ]
                }
            else:
                # Save to temp directory for immediate playback
                audio_path = save_audio_file(wav_bytes, "temp")

            # Add audio URL to response
            response_data["audio_url"] = get_audio_url(audio_path)
            return response_data
            
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
async def synthesize_speech(
    request: Request,
    session: AsyncSession = Depends(get_session)
):
    """Convert text to speech"""
    try:
        # Log raw request body
        body = await request.body()
        logger.info(f"Raw request body: {body.decode()}")
        
        # Parse request body
        data = await request.json()
        logger.info(f"Parsed request data: {data}")
        
        text = data.get('text')
        conversation_id = data.get('conversation_id')
        
        if not text:
            logger.error("No text provided in request")
            raise HTTPException(status_code=400, detail="No text provided")
        
        logger.info(f"Synthesizing speech for text: {text}")
        
        try:
            # Get model manager instance
            model_manager = ModelManager.get_instance()
            
            # Get synthesis parameters
            speed = float(data.get('speed', 1.0))
            pitch = float(data.get('pitch', 1.0))
            
            # Generate speech with parameters
            logger.info(f"Starting speech synthesis with speed={speed}, pitch={pitch}")
            waveforms = model_manager.synthesize_speech(text, speed=speed, pitch=pitch)
            logger.info(f"Generated waveforms shape: {waveforms.shape}")
            
            # Convert to bytes
            buffer = io.BytesIO()
            logger.info("Converting waveforms to WAV format")
            
            # Save as WAV file
            torchaudio.save(buffer, waveforms, 22050, format='wav')
            buffer.seek(0)
            wav_bytes = buffer.read()
            
            # Save audio file if conversation_id provided
            audio_path = None
            if conversation_id:
                try:
                    audio_path = save_audio_file(wav_bytes, conversation_id)
                    # Update the last assistant message with the audio path
                    if audio_path:
                        await update_last_assistant_message(conversation_id, audio_path)
                except Exception as e:
                    logger.error(f"Error saving audio or updating message: {str(e)}")
                    # Continue even if saving audio fails - we'll still return the audio stream
            
            # Return audio stream
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