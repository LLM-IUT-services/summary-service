import speech_recognition as sr
import whisper
import torch

recognizer = sr.Recognizer()
device = "cuda" if torch.cuda.is_available() else "cpu"


def support_speech_to_text():
    model = whisper.load_model("large").to(device)  #"small" / "large"
    result = model.transcribe("audio.mp3")
    print(result["text"])
    return result["text"] 

def speech_to_text():
    # خواندن فایل صوتی
    with sr.AudioFile("audio.wav") as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio, language="fa-IR")
        print("text:", text)
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return support_speech_to_text()
