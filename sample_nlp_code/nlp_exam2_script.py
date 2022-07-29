'''
Joshua Ting
NLP Final Exam
Text Classification Code: Binary Classification
'''
import pandas as pd
import numpy as np
from transformers import pipeline
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from LanguageModel import CustomTextDatasetMultiClass
import torch, gc
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from datasets import load_metric
import random

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

'''
1. Load Data
'''
dataset_path = './datasets/exam2/'
train_file = 'Train.csv'
valid_file = None
test_file = 'Test_submission_netid.csv'

# encoding='latin-1'
train_df = pd.read_csv(dataset_path+train_file, encoding='ISO-8859-1')
train_df, valid_df = train_test_split(train_df, test_size=0.10, random_state=seed)
train_df = train_df.reset_index()
valid_df = valid_df.reset_index()
test_df = pd.read_csv(dataset_path+test_file, encoding='ISO-8859-1')
test_df = test_df.reset_index()

# For testing script
# train_df = train_df.head(100)
# valid_df = valid_df.head(100)
# test_df = test_df.head(100)

print(train_df.head())
print(valid_df.head())
print(test_df.head())

# Check labels
print('\n\nLabels:', train_df['label'].unique()) #5 
print('\n\nLabels:', valid_df['label'].unique()) #5 
print('\n\nLabels:', test_df['label'].unique()) #5 
num_labels = len(train_df['label'].unique())

train_sentences = train_df['text'].tolist()
train_labels = train_df['label']
valid_sentences = valid_df['text'].tolist()
valid_labels = valid_df['label']
test_sentences = test_df['text'].tolist()
test_df['label'] = valid_labels[0]
test_labels = test_df['label']

print(len(train_sentences), len(train_labels))
print(len(valid_sentences), len(valid_labels))

train_labels_encoded = torch.tensor(train_labels, dtype = torch.float32)
valid_labels_encoded = torch.tensor(valid_labels, dtype = torch.float32)
test_labels_encoded = torch.tensor(test_labels, dtype = torch.float32)

train_one_hot_label = torch.nn.functional.one_hot(train_labels_encoded.to(torch.int64), num_labels)
valid_one_hot_label = torch.nn.functional.one_hot(valid_labels_encoded.to(torch.int64), num_labels)
test_one_hot_label = torch.nn.functional.one_hot(test_labels_encoded.to(torch.int64), num_labels)

train_one_hot_label = train_one_hot_label.float()
valid_one_hot_label = valid_one_hot_label.float()
test_one_hot_label = test_one_hot_label.float()

'''
2. Load Pretrained Models:

 1. howey/electra-small-mnli (2 epochs, 13M, 60mins)
 2. monologg/koelectra-small-finetuned-nsmc (2 epochs, 14M, 60mins)
 3. cross-encoder/ms-marco-MiniLM-L-12-v2 (5 epochs, 15M, 21mins)
 4. cross-encoder/ms-marco-TinyBERT-L-2-v2 (10 epochs, 4M, 7mins)
 5. mrm8488/bert-tiny-finetuned-sms-spam-detection (12 epochs, 4M, 7mins)
'''
model1_name = 'howey/electra-small-mnli'
model2_name = 'monologg/koelectra-small-finetuned-nsmc'
model3_name = 'cross-encoder/ms-marco-MiniLM-L-2-v2'
model4_name = 'cross-encoder/ms-marco-TinyBERT-L-2-v2'
model5_name = 'mrm8488/bert-tiny-finetuned-sms-spam-detection'

# 1. model1
model1_tokenizer = AutoTokenizer.from_pretrained(model1_name)
model1 = AutoModelForSequenceClassification.from_pretrained(model1_name, num_labels=num_labels, ignore_mismatched_sizes=True)
print(f'\n\n{model1_name} number of parameters:', model1.num_parameters())

model1_train_dataloader = CustomTextDatasetMultiClass(train_sentences, train_one_hot_label, model1_tokenizer, max_length=512)
model1_valid_dataloader = CustomTextDatasetMultiClass(valid_sentences, valid_one_hot_label, model1_tokenizer, max_length=512)
model1_test_dataloader = CustomTextDatasetMultiClass(test_sentences, test_one_hot_label, model1_tokenizer, max_length=512)

model1_data_collator = DataCollatorWithPadding(tokenizer=model1_tokenizer)

# # 2. model2
# model2_tokenizer = AutoTokenizer.from_pretrained(model2_name)
# model2 = AutoModelForSequenceClassification.from_pretrained(model2_name, num_labels=num_labels, ignore_mismatched_sizes=True)
# print(f'\n\n{model2_name} number of parameters:', model2.num_parameters())

# model2_train_dataloader = CustomTextDatasetMultiClass(train_sentences, train_one_hot_label, model2_tokenizer, max_length=512)
# model2_valid_dataloader = CustomTextDatasetMultiClass(valid_sentences, valid_one_hot_label, model2_tokenizer, max_length=512)
# model2_test_dataloader = CustomTextDatasetMultiClass(test_sentences, test_one_hot_label, model2_tokenizer, max_length=512)

# model2_data_collator = DataCollatorWithPadding(tokenizer=model2_tokenizer)

# # 3. model3
# model3_tokenizer = AutoTokenizer.from_pretrained(model3_name)
# model3 = AutoModelForSequenceClassification.from_pretrained(model3_name, num_labels=num_labels, ignore_mismatched_sizes=True)
# print(f'\n\n{model3_name} number of parameters:', model3.num_parameters())

# model3_train_dataloader = CustomTextDatasetMultiClass(train_sentences, train_one_hot_label, model3_tokenizer, max_length=512)
# model3_valid_dataloader = CustomTextDatasetMultiClass(valid_sentences, valid_one_hot_label, model3_tokenizer, max_length=512)
# model3_test_dataloader = CustomTextDatasetMultiClass(test_sentences, test_one_hot_label, model3_tokenizer, max_length=512)

# model3_data_collator = DataCollatorWithPadding(tokenizer=model3_tokenizer)

# # 4. model4
# model4_tokenizer = AutoTokenizer.from_pretrained(model4_name)
# model4 = AutoModelForSequenceClassification.from_pretrained(model4_name, num_labels=num_labels, ignore_mismatched_sizes=True)
# print(f'\n\n{model4_name} number of parameters:', model4.num_parameters())

# model4_train_dataloader = CustomTextDatasetMultiClass(train_sentences, train_one_hot_label, model4_tokenizer, max_length=512)
# model4_valid_dataloader = CustomTextDatasetMultiClass(valid_sentences, valid_one_hot_label, model4_tokenizer, max_length=512)
# model4_test_dataloader = CustomTextDatasetMultiClass(test_sentences, test_one_hot_label, model4_tokenizer, max_length=512)

# model4_data_collator = DataCollatorWithPadding(tokenizer=model4_tokenizer)

# # 5. model5
# model5_tokenizer = AutoTokenizer.from_pretrained(model5_name)
# model5 = AutoModelForSequenceClassification.from_pretrained(model5_name, num_labels=num_labels, ignore_mismatched_sizes=True)
# print(f'\n\n{model5_name} number of parameters:', model5.num_parameters())

# model5_train_dataloader = CustomTextDatasetMultiClass(train_sentences, train_one_hot_label, model5_tokenizer, max_length=512)
# model5_valid_dataloader = CustomTextDatasetMultiClass(valid_sentences, valid_one_hot_label, model5_tokenizer, max_length=512)
# model5_test_dataloader = CustomTextDatasetMultiClass(test_sentences, test_one_hot_label, model5_tokenizer, max_length=512)

# model5_data_collator = DataCollatorWithPadding(tokenizer=model5_tokenizer)

'''
3. Train
'''
def compute_metrics(eval_preds):
    metric = load_metric('f1','accuracy')
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    labels_argmax = np.argmax(labels, axis=-1)
    results = metric.compute(predictions=predictions, references=labels_argmax, average='macro')
    print(results)
    return results

training_args = TrainingArguments(
    output_dir="./results_binary",
    learning_rate=2e-5,
    per_device_train_batch_size=24,
    per_device_eval_batch_size=24,
    # gradient_accumulation_steps=1,
    # eval_accumulation_steps=1,
    num_train_epochs=1,
    weight_decay=0.01,
    seed=seed,
)

# 1. model1
model1_trainer = Trainer(
    model=model1,
    args=training_args,
    train_dataset=model1_train_dataloader,
    eval_dataset=model1_valid_dataloader,
    tokenizer=model1_tokenizer,
    data_collator=model1_data_collator,
    compute_metrics=compute_metrics,
)

model1_trainer.train()
model1_trainer.save_model('./results_binary/' + 'model1')
model1_results = model1_trainer.evaluate()

with open('./results_binary/model1/model1_results.txt', 'w') as f:
    print(model1_results, file=f)

del model1_trainer
gc.collect()
torch.cuda.empty_cache()

# # # 2. model2
# # model2_trainer = Trainer(
# #     model=model2,
# #     args=training_args,
# #     train_dataset=model2_train_dataloader,
# #     eval_dataset=model2_valid_dataloader,
# #     tokenizer=model2_tokenizer,
# #     data_collator=model2_data_collator,
# #     compute_metrics=compute_metrics,
# # )

# # model2_trainer.train()
# # model2_trainer.save_model('./results_binary/' + 'model2')
# # model2_results = model2_trainer.evaluate()

# # with open('./results_binary/model2/model2_results.txt', 'w') as f:
# #     print(model2_results, file=f)

# # del model2_trainer
# # gc.collect()
# # torch.cuda.empty_cache()

# # # 3. model3
# # model3_trainer = Trainer(
# #     model=model3,
# #     args=training_args,
# #     train_dataset=model3_train_dataloader,
# #     eval_dataset=model3_valid_dataloader,
# #     tokenizer=model3_tokenizer,
# #     data_collator=model3_data_collator,
# #     compute_metrics=compute_metrics,
# # )

# # model3_trainer.train()
# # model3_trainer.save_model('./results_binary/' + 'model3')
# # model3_results = model3_trainer.evaluate()

# # with open('./results_binary/model3/model3_results.txt', 'w') as f:
# #     print(model3_results, file=f)

# # del model3_trainer
# # gc.collect()
# # torch.cuda.empty_cache()

# # # 4. model4
# # model4_trainer = Trainer(
# #     model=model4,
# #     args=training_args,
# #     train_dataset=model4_train_dataloader,
# #     eval_dataset=model4_valid_dataloader,
# #     tokenizer=model4_tokenizer,
# #     data_collator=model4_data_collator,
# #     compute_metrics=compute_metrics,
# # )

# # model4_trainer.train()
# # model4_trainer.save_model('./results_binary/' + 'model4')
# # model4_results = model4_trainer.evaluate()

# # with open('./results_binary/model4/model4_results.txt', 'w') as f:
# #     print(model4_results, file=f)

# # del model4_trainer
# # gc.collect()
# # torch.cuda.empty_cache()

# # 5. model5
# model5_trainer = Trainer(
#     model=model5,
#     args=training_args,
#     train_dataset=model5_train_dataloader,
#     eval_dataset=model5_valid_dataloader,
#     tokenizer=model5_tokenizer,
#     data_collator=model5_data_collator,
#     compute_metrics=compute_metrics,
# )

# model5_trainer.train()
# model5_trainer.save_model('./results_binary/' + 'model5')
# model5_results = model5_trainer.evaluate()

# with open('./results_binary/model5/model5_results.txt', 'w') as f:
#     print(model5_results, file=f)

# del model5_trainer
# gc.collect()
# torch.cuda.empty_cache()

'''
4. Generate Test Predictions
'''
# 1. model1_ft
model1_ft = AutoModelForSequenceClassification.from_pretrained("./results_binary/model1", num_labels=num_labels)
model1_trainer_ft = Trainer(
    model=model1_ft,
    args=training_args,
    train_dataset=model1_train_dataloader,
    eval_dataset=model1_valid_dataloader,
    tokenizer=model1_tokenizer,
    data_collator=model1_data_collator,
    compute_metrics=compute_metrics,
)
model1_predictions = np.argmax(model1_trainer_ft.predict(model1_test_dataloader).predictions, axis=-1)
test_df_model1 = test_df.copy()
test_df_model1['label'] = model1_predictions

print('model1 Test DF:', test_df_model1.head())
test_df_model1.to_csv('./results_binary/model1/test_df_model1.csv', index=False)

# # # 2. model2_ft
# # model2_ft = AutoModelForSequenceClassification.from_pretrained("./results_binary/model2", num_labels=num_labels)
# # model2_trainer_ft = Trainer(
# #     model=model2_ft,
# #     args=training_args,
# #     train_dataset=model2_train_dataloader,
# #     eval_dataset=model2_valid_dataloader,
# #     tokenizer=model2_tokenizer,
# #     data_collator=model2_data_collator,
# #     compute_metrics=compute_metrics,
# # )
# # model2_predictions = np.argmax(model2_trainer_ft.predict(model2_test_dataloader).predictions, axis=-1)
# # test_df_model2 = test_df.copy()
# # test_df_model2['label'] = model2_predictions

# # print('model2 Test DF:', test_df_model2.head())
# # test_df_model2.to_csv('./results_binary/model2/test_df_model2.csv', index=False)

# # # 3. model3_ft
# # model3_ft = AutoModelForSequenceClassification.from_pretrained("./results_binary/model3", num_labels=num_labels)
# # model3_trainer_ft = Trainer(
# #     model=model3_ft,
# #     args=training_args,
# #     train_dataset=model3_train_dataloader,
# #     eval_dataset=model3_valid_dataloader,
# #     tokenizer=model3_tokenizer,
# #     data_collator=model3_data_collator,
# #     compute_metrics=compute_metrics,
# # )
# # model3_predictions = np.argmax(model3_trainer_ft.predict(model3_test_dataloader).predictions, axis=-1)
# # test_df_model3 = test_df.copy()
# # test_df_model3['label'] = model3_predictions

# # print('model3 Test DF:', test_df_model3.head())
# # test_df_model3.to_csv('./results_binary/model3/test_df_model3.csv', index=False)

# # # 4. model4_ft
# # model4_ft = AutoModelForSequenceClassification.from_pretrained("./results_binary/model4", num_labels=num_labels)
# # model4_trainer_ft = Trainer(
# #     model=model4_ft,
# #     args=training_args,
# #     train_dataset=model4_train_dataloader,
# #     eval_dataset=model4_valid_dataloader,
# #     tokenizer=model4_tokenizer,
# #     data_collator=model4_data_collator,
# #     compute_metrics=compute_metrics,
# # )
# # model4_predictions = np.argmax(model4_trainer_ft.predict(model4_test_dataloader).predictions, axis=-1)
# # test_df_model4 = test_df.copy()
# # test_df_model4['label'] = model4_predictions

# # print('model4 Test DF:', test_df_model4.head())
# # test_df_model4.to_csv('./results_binary/model4/test_df_model4.csv', index=False)

# # 5. model5_ft
# model5_ft = AutoModelForSequenceClassification.from_pretrained("./results_binary/model5", num_labels=num_labels)
# model5_trainer_ft = Trainer(
#     model=model5_ft,
#     args=training_args,
#     train_dataset=model5_train_dataloader,
#     eval_dataset=model5_valid_dataloader,
#     tokenizer=model5_tokenizer,
#     data_collator=model5_data_collator,
#     compute_metrics=compute_metrics,
# )
# model5_predictions = np.argmax(model5_trainer_ft.predict(model5_test_dataloader).predictions, axis=-1)
# test_df_model5 = test_df.copy()
# test_df_model5['label'] = model5_predictions

# print('model5 Test DF:', test_df_model5.head())
# test_df_model5.to_csv('./results_binary/model5/test_df_model5.csv', index=False)


'''
5. Show Best Model
'''
with open('./results_binary/model1/model1_results.txt', 'r') as f:
    print('model1 Results:', f.read())

# # with open('./results_binary/model2/model2_results.txt', 'r') as f:
# #     print('model2 Results:', f.read())

# # with open('./results_binary/model3/model3_results.txt', 'r') as f:
# #     print('model3 Results:', f.read())

# # with open('./results_binary/model4/model4_results.txt', 'r') as f:
# #     print('model4 Results:', f.read())

# with open('./results_binary/model5/model5_results.txt', 'r') as f:
#     print('model5 Results:', f.read())
