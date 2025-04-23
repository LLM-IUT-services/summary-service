""" # STT wrapper class
This class is a wrapper for Vosk STT library.
It can transcribe audio files in Persian and English.
It uses Vosk models for Persian and English.
It can also apply some basic audio enhancements.
## Usage:
>>> persian_stt = SpeechToText(lang="fa")  # or "en" for english
>>> with open("audio.mp3", "rb") as f:
>>>       text = persian_stt.transcribe_mp3_fileobj(f)
>>> print(text)
"""

import os
import wave
import json
from io import BytesIO
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer, SetLogLevel

SetLogLevel(-1)  # no log for vosk


class SpeechToText:
    def __init__(self, lang: str, model_size: str = 'small'):
        model_dir = 'models/' + lang
        if lang == 'fa':
            model_dir += '/vosk-model-small-fa-0.42'
        elif lang == 'en':
            model_dir += '/vosk-model-small-en-us-0.15'
        if not os.path.exists(model_dir):
            raise FileNotFoundError(
                f"Model directory '{model_dir}' not found.")

        self.model = Model(model_dir)

    def transcribe_mp3_fileobj(self, file_obj, max_alternatives: int = 0) -> str:
        # Convert MP3 file-like object to WAV in memory
        audio = AudioSegment.from_file(file_obj, format="mp3")
        audio = audio.set_channels(1).set_frame_rate(16000)

        # Optional: Apply audio enhancements (normalization, noise reduction)
        audio = audio.normalize()
        audio = audio.low_pass_filter(4000)  # Reduce high-frequency noise

        wav_io = BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)

        # Read WAV data
        wf = wave.open(wav_io, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
            raise ValueError("Audio must be mono PCM WAV at 16kHz.")

        # Enable more tolerant settings
        rec = KaldiRecognizer(self.model, wf.getframerate())
        rec.SetWords(True)  # Enable word-level details
        rec.SetMaxAlternatives(max_alternatives)  # Get alternatives if > 0

        result_text = ""
        while True:
            data = wf.readframes(4000)
            if not data:
                break
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                result_text += result.get("text", "") + " "

        final_result = json.loads(rec.FinalResult())
        result_text += final_result.get("text", "")

        # Optional: Post-process text (spell check, etc.)
        # result_text = self._post_process(result_text)

        return result_text.strip()
