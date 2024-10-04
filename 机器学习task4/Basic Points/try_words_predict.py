from transformers import  AutoTokenizer, AutoModelForMaskedLM
import torch


# 基于bert的解码器
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased",ignore_mismatched_sizes=True)
model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased",ignore_mismatched_sizes=True)

words =['I have [MASK] bag,and I [MASK] it', 'I [MASK] you']

# 返回pytorch向量,并且截断
inputs = tokenizer(words, return_tensors='pt',padding=True, truncation=True)

with torch.no_grad():
    # inputs 是一个字典，相当于inputs_ids = inputs['input_ids'], attention_mask=inputs['attention_mask']
    outputs = model(**inputs)

# 类别预测概率
logits = outputs.logits
# 得到mask标记的位置的结果,返回一元组，第一个代表第几个句子，第二个代表位置
mask_position = torch.where(inputs.input_ids == tokenizer.mask_token_id)


for i in range(len(mask_position[0])):
    sentence_index = mask_position[0][i].item()
    mask_index = mask_position[1][i].item()

    mask_num = (mask_position[0][:i] == sentence_index).sum().item()
    now_predict = mask_num + 1

    masked_probability= logits[sentence_index, mask_index, :]
    predict = torch.argmax(masked_probability,dim=-1).item()
    predict_word = tokenizer.convert_ids_to_tokens([predict])[0]

    print(f"Words:{words[sentence_index]},Mask_num:{now_predict}, Predict_word:{predict_word}")
