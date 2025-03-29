""" # LLM wrapper classes
This module contains the **wrapper classes for the LLMs**.
The classes are:
1. **EnPipeline** : Both summarization and QA in English
2. **FaSummarizationPipeline** : Only summarization in Persian
3. **FaQA_Pipeline** : Only QA in Persian
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Goal: Disable oneDNN optimizations in TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Goal: Reduce the level of TensorFlow logs and warnings

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
    def _generate(self, input_text, input_max_length, model_max_length, num_beams, length_penalty, repetition_penalty,min_length):
        # Tokenize the input text
        input_ids = self.tokenizer(
            input_text, return_tensors="pt", max_length=input_max_length, truncation=True).input_ids
        # Generate the output
        outputs = self.model.generate(
            input_ids,
            max_length=model_max_length,  # max length of output
            length_penalty=length_penalty,  # more penalty means less length
            num_beams=num_beams,  
            repetition_penalty=repetition_penalty,  # prevent repetition
            min_length=min_length
        )
        # Decode the output
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def summarize(self, input_text: str, input_max_length=512,  
                  model_max_length=256,
                  num_beams=10,
                  length_penalty=1.0,
                  repetition_penalty=2.0,
                  min_length=30) -> str:  
        # Adding a more explicit summarization prompt
        input_text = f"Summarize the following text, focusing on the main points and highlight the imporatant details:{input_text}"  
        return self._generate(input_text, input_max_length, model_max_length, num_beams, length_penalty, repetition_penalty,min_length)

class FaQA_Pipeline:
    def __init__(self, model_name='mansoorhamidzadeh/parsbert-persian-QA'):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.qa_pipeline = pipeline(
            "question-answering", 
            model=self.model, 
            tokenizer=self.tokenizer
        )

    def QA(self, context: str, question: str) -> str:
        result = self.qa_pipeline(question=question, context=context)
        return result['answer']


class EnPipeline:
    def __init__(self, model_name='google/flan-t5-small'):
        self.model_name = model_name
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False) 
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    """Tokenizes the input text, Generates the output, Decodes the output."""
    def _generate(self, input_text, input_max_length=512,model_max_length=200, num_beams=5, length_penalty=0.7,repetition_penalty=1.5,no_repeat_ngram_size=2,min_length=20):
        # Tokenize the input text
        input_ids = self.tokenizer(
            input_text, return_tensors="pt", max_length=input_max_length, truncation=True).input_ids
        input_ids = input_ids.to(self.model.device)  
        # Generate the output
        outputs = self.model.generate(
            input_ids,
            max_length=model_max_length,  # max length of output
            length_penalty=length_penalty,  # more penalty means less length
            num_beams=num_beams,  # more beams takes more time but better output
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            min_length=min_length
    )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def summarize(self, input_text: str, input_max_length=256,min_length=30,
              model_max_length=512,
              num_beams=10,
              length_penalty=1,
              repetition_penalty=1.5,
              no_repeat_ngram_size=2) -> str:
      input_text = f"Summarize the following text, focusing on the main points and highlight the imporatant details:{input_text}" 
      return self._generate(input_text, input_max_length, model_max_length, num_beams, length_penalty, repetition_penalty, no_repeat_ngram_size, min_length)
    
    def QA(self, context: str, question: str) -> str:
        input_text = f"question: {question} context: {context}"
        return self._generate(input_text)

# if __name__ == "__main__":
#     # Test Persian QA
#     print("\nTesting Persian QA:")
#     fa_qa = FaQA_Pipeline()
#     persian_context = """
#     شرکت گوگل در سال ۱۹۹۸ توسط لری پیج و سرگئی برین تأسیس شد. 
#     این شرکت در ابتدا به عنوان یک موتور جستجو شروع به کار کرد 
#     اما امروزه محصولات متنوعی از جمله سیستم عامل اندروید، 
#     مرورگر کروم و سرویس ابری گوگل درایو را ارائه می‌دهد.
#     """
#     persian_question = "بنیانگذاران گوگل چه کسانی هستند؟"
#     print(f"Answer: {fa_qa.QA(persian_context, persian_question)}")
#     # output = لری پیج و سرگئی برین

#     # Test English QA
#     print("\nTesting English QA:")
#     en_qa = EnPipeline()
#     english_context = """
#     The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. 
#     It is named after the engineer Gustave Eiffel, whose company designed and built the tower.
#     Constructed in 1889, it was initially criticized by some of France's leading artists 
#     and intellectuals for its design, but it has become a global cultural icon of France.
#     """
#     english_question = "Who designed the Eiffel Tower?"
#     print(f"Answer: {en_qa.QA(english_context, english_question)}")
#     # output = Gustave Eiffel
