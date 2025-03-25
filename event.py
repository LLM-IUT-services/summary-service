import pandas as pd
import os
import LLM_base as lm


# Load summarizer model
summarizer = lm.FaSummarizationPipeline()

def is_file_too_large(file_path, max_size_mb=5):
    """Check if file size is larger than max_size_mb"""
    return os.path.getsize(file_path) / (1024 * 1024) > max_size_mb

def read_file(file_path):
    """Read CSV or XLSX file"""
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            return pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Only CSV and XLSX are supported.")
    except Exception as e:
        return f"Error in reading file: {e}"

def generate_summary(text):
    """Generate summary for given text"""
    try:
        summary = summarizer.summarize(text,input_max_length=1024,model_max_length=256)
        return summary
    except Exception as e:
        return f"Error in generating summary: {e}"

def process_event_file(file_path):
    """Process event file and return summary"""
    if is_file_too_large(file_path):
        return "File size is too large. Maximum allowed size is 5 MB."

    df = read_file(file_path)
    if isinstance(df, str):  # If error occurred
        return df  

    # Convert dataframe to text
    text = " ".join(df.astype(str).values.flatten())

    # Generate summary
    summary = generate_summary(text)
    return summary

# Usage example
# file_path = "OfficebazWorldcup2022.xlsx"  
# result = process_event_file(file_path)
# print(result)



#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