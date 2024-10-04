from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 基于bert的解码器
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
# 基于bert的base大小的忽略大小的解码器
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

words =['I love you', 'I hate you']

# 返回pytorch向量,并且截断
inputs = tokenizer(words, return_tensors='pt',truncation=True, padding=True)

with torch.no_grad():
    # inputs 是一个字典，相当于inputs_ids = inputs['input_ids'], attention_mask=inputs['attention_mask']
    outputs = model(**inputs)

logits = outputs.logits
# 在类别维度寻找
predictions = torch.argmax(logits, dim=-1)

for word, prediction in zip(words, predictions):
    if prediction >= 2:
        print(f'"{word}" is positive')
    else:
        print(f'"{word}" is negative')
