import sys
import torch

sys.path.insert(0, '/home/labeconomia/ldiquilio/impossibile_2')

from pyabsa_rest14_hard.tasks import ABSAInstruction_2 as absa_instruction

torch.cuda.is_available = lambda: False

#torch.cuda.is_available()

import os
import warnings
import json

import findfile
from pyabsa_m.tasks import ABSAInstruction_2 as absa_instruction


warnings.filterwarnings("ignore")
import pandas as pd


from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-xl')
model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-xl')


import json

with open('/home/labeconomia/ldiquilio/impossibile_2/data/Split/SemEval-141516-LARGE-REST-HARD/train_dataset_141516.jsonl', 'rb') as f:
    data = f.read().decode('utf-8', errors='ignore')
    id_train_file_path = json.loads(data)

with open('/home/labeconomia/ldiquilio/impossibile_2/data/Split/SemEval-141516-LARGE-REST-HARD/test_dataset_141516.jsonl', 'rb') as f2:
    data2 = f2.read().decode('utf-8', errors='ignore')
    id_test_file_path = json.loads(data2)

'''
with open('/home/labeconomia/ldiquilio/impossibile_2/data/Split/beauty/random_restr_42/train_dataset_beauty.jsonl', 'rb') as f:
    data = f.read().decode('utf-8', errors='ignore')
    id_train_file_path = json.loads(data)

with open('/home/labeconomia/ldiquilio/impossibile_2/data/Split/beauty/random_restr_42/test_dataset_beauty.jsonl', 'rb') as f2:
    data2 = f2.read().decode('utf-8', errors='ignore')
    id_test_file_path = json.loads(data2)
'''


id_tr_df =pd.json_normalize(id_train_file_path, 'labels', ['text'])


id_te_df =pd.json_normalize(id_test_file_path, 'labels', ['text'])


id_tr_df = pd.DataFrame(id_train_file_path)
id_te_df = pd.DataFrame(id_test_file_path)


task_name = "multitask"
experiment_name = "instruction"
# model_checkpoint = 'allenai/tk-instruct-base-def-pos'
# model_checkpoint = "kevinscaria/ate_tk-instruct-base-def-pos-neg-neut-combined"
# model_checkpoint = 'allenai/tk-instruct-large-def-pos'
# model_checkpoint = 'allenai/tk-instruct-3b-def-pos'
model_checkpoint = "google/flan-t5-xl"

print("Experiment Name: ", experiment_name)
model_out_path = "checkpoints"
model_out_path = os.path.join(
    model_out_path, task_name, f"{model_checkpoint.replace('/', '')}-{experiment_name}"
)
print("Model output path: ", model_out_path)


loader = absa_instruction.data_utils_2.InstructDatasetLoader(id_tr_df, id_te_df)

if loader.train_df_id is not None:
    loader.train_df_id = loader.prepare_instruction_dataloader_c(loader.train_df_id)
if loader.test_df_id is not None:
    loader.test_df_id = loader.prepare_instruction_dataloader_c(loader.test_df_id)
if loader.train_df_ood is not None:
    loader.train_df_ood = loader.prepare_instruction_dataloader_c(loader.train_df_ood)
if loader.test_df_ood is not None:
    loader.test_df_ood = loader.prepare_instruction_dataloader_c(loader.test_df_ood)

    # Create T5 utils object
t5_exp = absa_instruction.model.T5Classifier(model_checkpoint)

# Tokenize Dataset
id_ds, id_tokenized_ds, ood_ds, ood_tokenzed_ds = loader.create_datasets(
    t5_exp.tokenize_function_inputs
)

# Training arguments
training_args = {
    "output_dir": model_out_path,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "learning_rate": 5e-5,
    "per_device_train_batch_size": 6,
    "per_device_eval_batch_size": 16,
    "num_train_epochs": 10,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "load_best_model_at_end": True,
    "push_to_hub": False,
    "eval_accumulation_steps": 1,
    "predict_with_generate": True,
    "logging_steps": 1000000000,
    "use_mps_device": False,
    # 'fp16': True,
    "fp16": False,
}

# Train model
model_trainer = t5_exp.train(id_tokenized_ds, **training_args)


'''
# Get prediction labels - Training set
id_tr_pred_labels = t5_exp.get_labels(
    predictor=model_trainer,
    tokenized_dataset=id_tokenized_ds,
    sample_set="train",
    batch_size=16,
)
id_tr_labels = [i.strip() for i in id_ds["train"]["labels"]]

# Get prediction labels - Testing set
id_te_pred_labels = t5_exp.get_labels(
    predictor=model_trainer,
    tokenized_dataset=id_tokenized_ds,
    sample_set="test",
    batch_size=16,
)
id_te_labels = [i.strip() for i in id_ds["test"]["labels"]]

# # Compute Metrics
# metrics = t5_exp.get_metrics(id_tr_labels, id_tr_pred_labels)
# print('----------------------- Training Set Metrics -----------------------')
# print(metrics)
#
# metrics = t5_exp.get_metrics(id_te_labels, id_te_pred_labels)
# print('----------------------- Testing Set Metrics -----------------------')
# print(metrics)
'''
