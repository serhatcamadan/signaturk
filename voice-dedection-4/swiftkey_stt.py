"""
SwiftKey-Style Real-Time Speech-to-Text
========================================
A streaming STT implementation that displays text as you speak,
mimicking the real-time typing behavior of mobile keyboards like SwiftKey.

Uses RealtimeSTT (faster_whisper wrapper) with CUDA acceleration.
"""

import sys
import io
import os
import logging

# === ADIM 1: Windows Konsolunu UTF-8'e zorla ===
# Windows PowerShell/CMD varsayılan olarak CP1252 kullanır
# Türkçe karakterler (ş, ı, ğ, ü, ö, ç) için UTF-8 gerekli
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# === ADIM 2: TÜM logging'i sustur (IMPORT'TAN ÖNCE!) ===
# Root logger'ı da susturuyoruz - bu sayede hiçbir log çıkmaz
logging.disable(logging.CRITICAL)  # TÜM logları devre dışı bırak

# Alternatif: Belirli logger'ları sustur
for logger_name in ["RealtimeSTT", "faster_whisper", "pyaudio", "urllib3", "filelock"]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)
    logging.getLogger(logger_name).disabled = True

from RealtimeSTT import AudioToTextRecorder


class SwiftKeyStyleSTT:
    """
    Real-time Speech-to-Text with streaming output.
    
    This class provides instant text feedback as you speak, similar to how
    SwiftKey shows predictions while typing. It uses two callback mechanisms:
    - Realtime callback: Fires continuously with intermediate results (the streaming effect)
    - Final callback: Fires when speech segment is complete
    """
    
    def __init__(
        self,
        model: str = "small",
        language: str = "tr",
        device: str = "cuda",
        compute_type: str = "int8",
    ):
        """
        Initialize the SwiftKey-style STT recorder.
        
        Args:
            model: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            language: Target language code (e.g., 'tr' for Turkish)
            device: Compute device ('cuda' for GPU, 'cpu' for CPU)
            compute_type: Precision type ('int8' for speed, 'float16' for balance)
        """
        self.model = model
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.recorder = None
        self._last_text_length = 0
        
    def _on_realtime_update(self, text: str) -> None:
        """
        Callback for intermediate/streaming results.
        
        This is the KEY to the SwiftKey effect - it fires continuously
        as the user speaks, providing partial transcriptions that update
        in real-time, creating the "typing" visual effect.
        
        Args:
            text: Current intermediate transcription (may change as more audio comes in)
        """
        if text:
            # Clear the current line and print updated text
            # \r moves cursor to start, end='' prevents newline
            # We pad with spaces to clear any leftover characters from longer previous text
            padding = " " * max(0, self._last_text_length - len(text))
            sys.stdout.write(f"\r🎤 {text}{padding}")
            sys.stdout.flush()
            self._last_text_length = len(text)
    
    def _on_final_result(self, text: str) -> None:
        """
        Callback for final transcription when speech segment ends.
        
        This fires when the user pauses or stops speaking, providing
        the complete, finalized transcription for that segment.
        
        Args:
            text: Final transcription of the speech segment
        """
        if text:
            # Move to new line after final result
            sys.stdout.write(f"\r✅ {text}                    \n")
            sys.stdout.flush()
            self._last_text_length = 0
    
    def start(self) -> None:
        """
        Start the real-time speech-to-text recorder.
        
        Configures AudioToTextRecorder with:
        - enable_realtime_transcription=True: Critical for streaming effect
        - realtime_processing_pause: Short pause for responsive updates
        - on_realtime_transcription_update: Callback for intermediate results
        """
        print("=" * 60)
        print("🎙️  SwiftKey-Style Real-Time Speech-to-Text")
        print("=" * 60)
        print(f"📋 Model: {self.model} | Language: {self.language}")
        print(f"💻 Device: {self.device} | Precision: {self.compute_type}")
        print("")
        print("⏳ Model yükleniyor... (ilk sefer 20-40 sn sürebilir)")
        sys.stdout.flush()
        
        # Configure the recorder for real-time streaming
        self.recorder = AudioToTextRecorder(
            # === Core Settings ===
            model=self.model,
            language=self.language,
            device=self.device,
            compute_type=self.compute_type,
            
            # === Real-Time Streaming (The SwiftKey Effect) ===
            # This is CRITICAL - enables intermediate results while speaking
            enable_realtime_transcription=True,
            
            # How often to process audio for realtime updates (lower = more responsive)
            realtime_processing_pause=0.1,
            
            # Model for realtime (can use smaller for speed, larger for accuracy)
            realtime_model_type=self.model,
            
            # === Callbacks ===
            # Fires continuously with intermediate text (streaming effect)
            on_realtime_transcription_update=self._on_realtime_update,
            
            # === Voice Activity Detection (VAD) ===
            # Silero VAD is fast and accurate for detecting speech
            silero_sensitivity=0.4,
            
            # === UI Settings ===
            spinner=False,  # No visual noise
            
            # === Performance Tuning ===
            # Shorter buffer for lower latency
            buffer_size=512,
            
            # 2 saniye sessizlik olana kadar dinlemeye devam et
            post_speech_silence_duration=2.0,
            
            # Minimum audio length to process
            min_length_of_recording=0.3,
        )
        
        # === MODEL HAZIR ===
        print("✅ Model yüklendi!")
        print("-" * 60)
        print("🔊 Konuşmaya başlayın... (Ctrl+C ile çıkış)")
        print("-" * 60)
        sys.stdout.flush()
        
        try:
            # Main loop: continuously transcribe speech
            while True:
                # text() blocks until speech is detected and finalized
                # During speech, realtime callback fires with updates
                final_text = self.recorder.text(self._on_final_result)
                
        except KeyboardInterrupt:
            print("\n" + "=" * 60)
            print("👋 Stopping... Goodbye!")
            print("=" * 60)
        finally:
            self.stop()
    
    def stop(self) -> None:
        """Stop the recorder and cleanup resources."""
        if self.recorder:
            self.recorder.stop()
            self.recorder = None


def main():
    """Entry point for the SwiftKey-style STT demo."""
    # Create STT instance optimized for RTX 3060
    stt = SwiftKeyStyleSTT(
        model="small",      # Good balance of speed/accuracy for Turkish
        language="tr",      # Turkish language
        device="cuda",      # Use NVIDIA GPU
        compute_type="int8" # Fastest inference with minimal quality loss
    )
    
    # Start real-time transcription
    stt.start()


if __name__ == "__main__":
    main()

