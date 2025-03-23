import pandas as pd
import os
from transformers import pipeline

# بارگذاری مدل خلاصه‌سازی
summarizer = pipeline("summarization", model="google/flan-t5-base")

def is_file_too_large(file_path, max_size_mb=5):
    """بررسی اندازه فایل"""
    return os.path.getsize(file_path) / (1024 * 1024) > max_size_mb

def read_file(file_path):
    """خواندن فایل CSV یا XLSX"""
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            return pd.read_excel(file_path)
        else:
            raise ValueError("فقط فرمت‌های CSV و XLSX پشتیبانی می‌شوند.")
    except Exception as e:
        return f"خطا در خواندن فایل: {e}"

def generate_summary(text):
    """تولید خلاصه از متن با استفاده از مدل"""
    try:
        summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"خطا در خلاصه‌سازی: {e}"

def process_event_file(file_path):
    """پردازش فایل و تولید خلاصه گزارش"""
    if is_file_too_large(file_path):
        return "حجم فایل بیش از حد مجاز است."

    df = read_file(file_path)
    if isinstance(df, str):  # اگر خواندن فایل با خطا مواجه شد
        return df  

    # تبدیل داده‌ها به متن برای خلاصه‌سازی
    text = " ".join(df.astype(str).values.flatten())

    # تولید خلاصه
    summary = generate_summary(text)
    return summary

# مثال استفاده
file_path = "OfficebazWorldcup2022.xlsx"  # مسیر فایل موردنظر
result = process_event_file(file_path)
print(result)



#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