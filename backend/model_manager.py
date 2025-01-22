import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from speechbrain.inference import Tacotron2, HIFIGAN
import os
from typing import Optional
from functools import lru_cache
import torchaudio.transforms as T
import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    _instance: Optional['ModelManager'] = None
    
    def __init__(self):
        self._device = self._get_device()
        self._load_models()
    
    @staticmethod
    def _get_device():
        """
        Determine the best available device for model inference.
        Handles MPS (Apple Silicon), CUDA, and CPU with proper error checking.
        """
        try:
            # Check for MPS (Apple Silicon)
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                try:
                    # Test MPS device creation
                    device = torch.device("mps")
                    # Verify MPS is working with a small tensor operation
                    test_tensor = torch.ones(1, device=device)
                    del test_tensor
                    print("Using MPS (Apple Silicon) device")
                    return device
                except Exception as e:
                    print(f"MPS available but error occurred: {str(e)}. Falling back to next option.")

            # Check for CUDA
            if torch.cuda.is_available():
                try:
                    device = torch.device("cuda")
                    # Verify CUDA is working
                    test_tensor = torch.ones(1, device=device)
                    del test_tensor
                    print("Using CUDA device")
                    return device
                except Exception as e:
                    print(f"CUDA available but error occurred: {str(e)}. Falling back to CPU.")

            # Fallback to CPU
            print("Using CPU device")
            return torch.device("cpu")

        except Exception as e:
            print(f"Error during device selection: {str(e)}. Defaulting to CPU.")
            return torch.device("cpu")
    
    def _load_models(self):
        """Load all required models with proper error handling and device management"""
        os.makedirs("model_cache", exist_ok=True)
        
        # Configure environment variables for model caching
        os.environ["HF_HOME"] = os.getenv("MODEL_CACHE_DIR", "/app/model_cache")
        os.environ["HUGGINGFACE_HUB_CACHE"] = os.getenv("MODEL_CACHE_DIR", "/app/model_cache")
        os.environ["SPEECHBRAIN_CACHE"] = os.getenv("MODEL_CACHE_DIR", "/app/model_cache")

        try:
            logger.info("Loading Whisper processor...")
            self._processor = WhisperProcessor.from_pretrained(
                "allandclive/whisper-tiny-luganda-v2",
                cache_dir="model_cache",
            )
            
            logger.info(f"Loading Whisper model to {self._device}...")
            self._model = WhisperForConditionalGeneration.from_pretrained(
                "allandclive/whisper-tiny-luganda-v2",
                cache_dir="model_cache",
            )
            
            # Enable memory efficient attention if available
            if hasattr(self._model.config, "use_attention_mask"):
                self._model.config.use_attention_mask = True
            
            # Move model to device and optimize memory
            self._model.to(self._device)
            if self._device.type in ["cuda", "mps"]:
                logger.info("Converting model to half precision for GPU/MPS optimization")
                self._model = self._model.half()
            
            logger.info("Loading Tacotron2 model...")
            try:
                self._tacotron = Tacotron2.from_hparams(
                    source="Sunbird/sunbird-lug-tts",
                    savedir="luganda_tts",
                    run_opts={"device": str(self._device)}
                )
            except Exception as e:
                logger.error(f"Error loading Tacotron2: {str(e)}. Retrying without device specification...")
                # Fallback: Try loading without device specification
                self._tacotron = Tacotron2.from_hparams(
                    source="Sunbird/sunbird-lug-tts",
                    savedir="luganda_tts"
                )
            
            logger.info("Loading HiFiGAN vocoder...")
            try:
                self._gan = HIFIGAN.from_hparams(
                    source="speechbrain/tts-hifigan-ljspeech",
                    savedir="vocoder",
                    run_opts={"device": str(self._device)}
                )
            except Exception as e:
                logger.error(f"Error loading HiFiGAN: {str(e)}. Retrying without device specification...")
                # Fallback: Try loading without device specification
                self._gan = HIFIGAN.from_hparams(
                    source="speechbrain/tts-hifigan-ljspeech",
                    savedir="vocoder"
                )
            
            logger.info("All models loaded successfully!")
            
        except Exception as e:
            error_msg = f"Critical error loading models: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    @classmethod
    @lru_cache(maxsize=1)
    def get_instance(cls) -> 'ModelManager':
        """Get or create singleton instance with caching"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @property
    def device(self):
        return self._device
    
    @property
    def processor(self):
        return self._processor
    
    @property
    def model(self):
        return self._model
    
    @property
    def tacotron(self):
        return self._tacotron
    
    @property
    def gan(self):
        return self._gan
    
    def transcribe_audio(self, audio_input, sample_rate=16000):
        """Transcribe audio using Whisper model"""
        # Process audio input
        input_features = self.processor(
            audio_input,
            sampling_rate=sample_rate,
            return_tensors="pt"
        ).input_features.to(self.device)
        
        # Generate token ids without specifying language
        generated_ids = self.model.generate(
            input_features,
            attention_mask=torch.ones_like(input_features),
            forced_decoder_ids=None
        )
        
        # Decode token ids to text
        transcription = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        
        return transcription
    
    def synthesize_speech(self, text, speed=1.0, pitch=1.0):
        """Convert text to speech using Tacotron2 and HiFiGAN with speed and pitch control"""
        # Generate mel spectrogram
        mel_output, mel_length, alignment = self.tacotron.encode_text(text)
        
        # Generate waveform
        waveforms = self.gan.decode_batch(mel_output)
        
        # Squeeze to get correct dimensions (batch, time)
        waveforms = waveforms.squeeze(1)  # Remove the channel dimension
        
        # Apply speed modification
        if speed != 1.0:
            effect = T.Speed(speed)
            waveforms = effect(waveforms)
        
        # Apply pitch modification
        if pitch != 1.0:
            # Calculate semitone shift using logarithmic scale
            n_steps = 12 * math.log2(pitch)
            effect = T.PitchShift(sample_rate=22050, n_steps=n_steps)
            waveforms = effect(waveforms)
        
        return waveforms
