import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from speechbrain.inference import Tacotron2, HIFIGAN
import os
from typing import Optional
from functools import lru_cache

class ModelManager:
    _instance: Optional['ModelManager'] = None
    
    def __init__(self):
        self._device = self._get_device()
        self._load_models()
    
    @staticmethod
    def _get_device():
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    
    def _load_models(self):
        # Load Whisper model for STT
        self._processor = WhisperProcessor.from_pretrained("allandclive/whisper-tiny-luganda-v2")
        self._model = WhisperForConditionalGeneration.from_pretrained(
            "allandclive/whisper-tiny-luganda-v2"
        ).to(self._device)
        
        # Load TTS models
        self._tacotron = Tacotron2.from_hparams(
            source="Sunbird/sunbird-lug-tts",
            savedir="luganda_tts"
        )
        self._gan = HIFIGAN.from_hparams(
            source="speechbrain/tts-hifigan-ljspeech",
            savedir="vocoder"
        )
    
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
    
    def transcribe_audio(self, audio_input, sample_rate=16000, beam_size=5, temperature=0.0):
        """Transcribe audio using Whisper model with configurable parameters"""
        input_features = self.processor(
            audio_input,
            sampling_rate=sample_rate,
            return_tensors="pt"
        ).input_features.to(self.device)
        
        # Create attention mask
        attention_mask = torch.ones_like(input_features).to(self.device)
        
        # Configure generation parameters
        generate_kwargs = {
            "attention_mask": attention_mask,
            "num_beams": beam_size,
            "temperature": temperature if temperature > 0 else None,
            "do_sample": temperature > 0,
            "language": "lg",  # Luganda language code
            "task": "transcribe"
        }
        
        generated_ids = self.model.generate(
            input_features,
            **generate_kwargs
        )
        
        transcription = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        
        return transcription
    
    def synthesize_speech(self, text, speed=1.0, pitch=1.0):
        """Convert text to speech with configurable parameters"""
        # Generate mel spectrogram
        mel_output, mel_length, alignment = self.tacotron.encode_text(text)
        
        # Apply pitch modification if needed
        if pitch != 1.0:
            mel_output = mel_output * pitch
        
        # Generate waveform
        waveforms = self.gan.decode_batch(mel_output)
        
        # Apply speed modification if needed
        if speed != 1.0:
            waveform = waveforms.squeeze(1)
            old_length = waveform.shape[0]
            new_length = int(old_length / speed)
            waveform = torch.nn.functional.interpolate(
                waveform.unsqueeze(0).unsqueeze(0),
                size=new_length,
                mode='linear',
                align_corners=False
            ).squeeze()
            return waveform
        
        return waveforms.squeeze(1)