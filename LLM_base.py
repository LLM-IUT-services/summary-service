""" # LLM wrapper classes
This module contains the **wrapper classes for the LLMs**.
The classes are:
1. **EnPipeline** : Both summarization and QA in English
2. **FaSummarizationPipeline** : Only summarization in Persian
3. **FaQA_Pipeline** : Only QA in Persian
"""

import speech_recognition as sr
from transformers import (
    AutoModelForQuestionAnswering,
    T5ForConditionalGeneration,
    AutoModelForSeq2SeqLM,
    T5Tokenizer,
    AutoTokenizer,
    pipeline,
)


class FaSummarizationPipeline:
    """
    This class is a wrapper for MT5.
    Corrently it only does summarization in Persian.
    Example:
    >>> pipe = FaSummarizationPipeline()
    >>> text = "..."
    >>> output = pipe.summarize(text)
    """
    def __init__(self, model_name='HooshvareLab/pn-summary-mt5-small'):
        self.model_name = model_name
        # Load the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    """Tokenizes the input text, Generates the output, Decodes the output."""
    def _generate(self, input_text, input_max_length, model_max_length, num_beams, length_penalty):
        # Tokenize the input text
        input_ids = self.tokenizer(
            input_text, return_tensors="pt", max_length=input_max_length, truncation=True).input_ids
        # Generate the output
        outputs = self.model.generate(
            input_ids,
            max_length=model_max_length,  # max length of output
            length_penalty=length_penalty,  # more penalty means less length
            num_beams=num_beams  # more beams takes more time but better output
        )
        # Decode the output
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def summarize(self, input_text: str, input_max_length=256,
                  model_max_length=256,
                  num_beams=5,
                  length_penalty=1.0) -> str:
        pass


class FaQA_Pipeline:
    """
    Wrapper class for parsbert-persian-QA.
    It only does QA in Persian.
    Example:
    >>> pipe = FaQA_Pipeline()
    >>> context = "..."
    >>> question = "..."
    >>> output = pipe.QA(context, question)
    """
    def __init__(self, model_name='mansoorhamidzadeh/parsbert-persian-QA'):
        self.model_name = model_name
        # Load the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)

        # Create a QA pipeline
        self.qa_pipeline = pipeline(
            "question-answering", model=self.model, tokenizer=self.tokenizer)

    def QA(self, context: str, question: str) -> str:
        pass


class EnPipeline:
    """
    Wrapper class for flan-t5.
    It can do both summarization and QA in English.
    Example 1:
    >>> pipe = EnPipeline()
    >>> text = "..."
    >>> output1 = pipe.summarize(text)
    Example 2:
    >>> pipe = EnPipeline()
    >>> question , context = "..." , "..."
    >>> output2 = pipe.QA(context, question)
    """
    def __init__(self, model_name='google/flan-t5-small'):
        self.model_name = model_name
        # Load the model and tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    """Tokenizes the input text, Generates the output, Decodes the output."""
    def _generate(self, input_text, input_max_length, model_max_length, num_beams, length_penalty):
        # Tokenize the input text
        input_ids = self.tokenizer(
            input_text, return_tensors="pt", max_length=input_max_length, truncation=True).input_ids
        # Generate the output
        outputs = self.model.generate(
            input_ids,
            max_length=model_max_length,  # max length of output
            length_penalty=length_penalty,  # more penalty means less length
            num_beams=num_beams  # more beams takes more time but better output
        )
        # Decode the output
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def summarize(self, input_text: str, input_max_length=256,
                  model_max_length=256,
                  num_beams=5,
                  length_penalty=1.0) -> str:
        pass

    def QA(self, context: str, question: str)->str:
        pass


r = sr.Recognizer()
r.pause_threshold = 0.5  # Reduce latency between conversations

def speech_to_text():
    with sr.Microphone() as source2:  # open mic
        r.adjust_for_ambient_noise(source2, duration=1)  # Noise adjustment with sufficient time

        while True:
            try:
                print("Listening...")  
                audio2 = r.listen(source2)

            # persion test
                try:
                    mytext = r.recognize_google(audio2, language="fa-IR")
                    return mytext
                    print("Recognized (FA):", mytext)
                    continue
                except sr.UnknownValueError:
                    pass 

            # english test
                try:
                    mytext = r.recognize_google(audio2, language="en-US")
                    return mytext
                    print("Recognized (EN):", mytext)
                    continue
                except sr.UnknownValueError:
                    print("Could not recognize speech in any language")
        
            except sr.RequestError as e:
                print("Could not request results; {0}".format(e))
