""" # LLM wrapper classes
This module contains the **wrapper classes for the LLMs**.
The classes are:
1. **EnPipeline** : Both summarization and QA in English
2. **FaSummarizationPipeline** : Only summarization in Persian
3. **FaQA_Pipeline** : Only QA in Persian
"""


from transformers import (
    AutoModelForQuestionAnswering,
    T5ForConditionalGeneration,
    AutoModelForSeq2SeqLM,
    T5Tokenizer,
    AutoTokenizer,
    pipeline,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import os
import torch
from datasets import load_dataset

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
    def _generate(self, input_text, input_max_length,model_max_length, num_beams, length_penalty,repetition_penalty,no_repeat_ngram_size,min_length):
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
    
    def _preprocess_function(examples, tokenizer, max_input_length=512, max_target_length=128):
        """Tokenize the input text (article) and target text (summary)."""
        model_inputs = tokenizer(
            examples["article"], 
            max_length=max_input_length, 
            truncation=True, 
            padding="max_length"
        )
        labels = tokenizer(
            examples["highlights"], 
            max_length=max_target_length, 
            truncation=True, 
            padding="max_length"
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
        
    def _trainDataForSummarizing(self):
        dataset = load_dataset("cnn_dailymail", "3.0.0")
        # Split the dataset into train and validation sets
        train_dataset = dataset["train"].train_test_split(test_size=0.9)["train"]
        validation_dataset = dataset["train"].train_test_split(test_size=0.5)["test"]  
        # Tokenize the datasets
        tokenized_datasets = train_dataset.map(
            lambda x: self._preprocess_function(x), 
            batched=True
        )
        tokenized_validation_dataset = validation_dataset.map(
            lambda x: self._preprocess_function(x),
            batched=True
        )
        # Define data collator
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, self.model)

        # Training arguments
        training_args = Seq2SeqTrainingArguments(
        output_dir="./results", 
        logging_strategy="steps",             
        learning_rate=3e-5,                    
        gradient_accumulation_steps=4,         
        weight_decay=0.01,                     
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,                     
        warmup_steps=1000,                                               
        predict_with_generate=True,            
        generation_max_length=128,             
        generation_num_beams=5,                
        push_to_hub=False,
        num_train_epochs=2,  
        per_device_train_batch_size=16,  
        per_device_eval_batch_size=16,
        fp16=True, 
        save_strategy="epoch",  
        save_total_limit=1, 
        evaluation_strategy="epoch", 
        )


        # Trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets, # Use the tokenized train dataset
            eval_dataset=tokenized_validation_dataset, # Use the tokenized validation dataset
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        # Train the model
        trainer.train()
        # Save model 
        model_dir = "models/"
        model_path = os.path.join(model_dir, "trained_summarize_model.pth")

        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)

        # Save model properly
        trainer.save_model(model_dir) 
        self.tokenizer.save_pretrained(model_dir)

        # Save model state dict separately
        torch.save(self.model.state_dict(), model_path)  # Save only weights


<<<<<<< HEAD
    def summarize(self, input_text: str, input_max_length=256,
                  model_max_length=256,
                  num_beams=5,
                  length_penalty=1.0) -> str:
            model_dir="models/"
            model_path = os.path.join(model_dir, "trained_summaeize_model.pth")
            # Ensure model directory exists
            os.makedirs(model_dir, exist_ok=True)
            #Load model if available, otherwise train
            if os.path.exists(model_path):
                print("Loading trained model from disk...")
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)  # Initialize model
                self.model.load_state_dict(torch.load(model_path, map_location="cpu"))  # Load weights
            else:
                print("No trained model found. Initializing new model...")
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
                self._trainDataForSummarizing()

            return self._generate(input_text, input_max_length, model_max_length, num_beams, length_penalty)
=======
    def summarize(self, input_text: str, input_max_length=256,min_length=30,
              model_max_length=512,
              num_beams=10,
              length_penalty=1,
              repetition_penalty=1.5,
              no_repeat_ngram_size=2) -> str:
      input_text = f"Summarize the following text, focusing on the main points and highlight the imporatant details:{input_text}" 
      return self._generate(input_text, input_max_length, model_max_length, num_beams, length_penalty, repetition_penalty, no_repeat_ngram_size, min_length)
>>>>>>> f407b39 (complete summarize method in EnPipeline class)



    def QA(self, context: str, question: str)->str:
        pass
