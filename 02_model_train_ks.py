import torch
from torch import Tensor

import os
import time
import datetime

from dataset.load_embedding import *
from dataset.custom_emb_dataloader import *
from model.downstream_model import *
from trainer.model_trainer import *
from evaluator.model_evaluator import *
from utils.constant_mapping import *

def save_to_txt(content, filename):
    with open(filename, 'a') as f:
        print(content, file=f)
        
def return_current_datetime():
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    return_text = f"Current date and time: {formatted_datetime}"
    return return_text

def set_device(device_index=None):
    if device_index is not None and torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        if num_devices > device_index:
            torch.cuda.set_device(device_index)
            device = torch.device("cuda")
            print(f"Using GPU {torch.cuda.current_device()}")
            return device
        else:
            torch.cuda.set_device(0)
            device = torch.device("cuda:0")
            print("Specified GPU index is out of range. Using the first GPU.")
            return device
    else:
        device = torch.device("cpu")
        print("CUDA is not available or GPU index is not specified. Using CPU.")
        return device

device = set_device(0)
upstream_model_type = "wavlm_large"
upstream_model_variation = upstream_model_type.split("_")[-1]
no_of_encoders = 12 if upstream_model_variation == "base" else 24

task_type = "ks"
dataset_code = "speechcommand"
dataset_name = DatasetKeywordMapping.get_data_name(dataset_code)
label_mapping = LabelKeywordMapping.get_label_mapping(dataset_code)[0]

data_loading_percentage = 100
frame_pooling_type = "mean"
layer_pooling_type = "mean"

batch_size = 512
num_epochs = 100
learning_rate = 2.5e-3
weight_decay = 5e-8
saved_checkpoint_count = 1
patience = 1
factor = 0.5

l1_lambda = 0
l2_lambda = 0

input_dim = 768 if upstream_model_variation == "base" else 1024
embedding_dim = 1500
output_dim = len(label_mapping)

dropout_prob1 = 0
dropout_prob2 = 0
dropout_prob_array = [dropout_prob1, dropout_prob2]

root_path = "/home/braveenan/voice_dataset"
root_data = os.path.join(root_path, dataset_name)

root_emb_path = root_path.replace("/voice_dataset", f"/embedding_old/{upstream_model_type}/{frame_pooling_type}")
root_emb_data = os.path.join(root_emb_path, dataset_name)

current_timestamp = str(int(time.time()))
result_folder_path = f"result/{task_type}/{frame_pooling_type}_{layer_pooling_type}/{current_timestamp}"
checkpoint_folder_path = f"checkpoint/{task_type}/{frame_pooling_type}_{layer_pooling_type}/{current_timestamp}"

def create_file_path(upstream_model_type, task_type, folder_path, file_format):
    os.makedirs(folder_path, exist_ok=True)
    file_name = f"{upstream_model_type}_{task_type}{file_format}"
    file_path = os.path.join(folder_path, file_name)
    return file_path

result_text_path = create_file_path(upstream_model_type, task_type, result_folder_path, ".txt")
result_plot_path = create_file_path(upstream_model_type, task_type, result_folder_path, ".png")
model_checkpoint_path = create_file_path(upstream_model_type, task_type, checkpoint_folder_path, ".pth")
data_count_path = create_file_path(upstream_model_type, "data_count", result_folder_path, ".txt")

current_datetime = return_current_datetime()
print(current_datetime)
save_to_txt(current_datetime, result_text_path)

task_name = TaskKeywordMapping.get_task_name(task_type)
task_text = f"Task name: {task_name}"
print(task_text)
save_to_txt(task_text, result_text_path)

upstreammodel_text = f"Upstream model type: {upstream_model_type}"
print(upstreammodel_text)
save_to_txt(upstreammodel_text, result_text_path)

frame_pooling_text = f"Frame pooling type: {frame_pooling_type}"
print(frame_pooling_text)
save_to_txt(frame_pooling_text, result_text_path)

layer_pooling_text = f"Layer pooling type: {layer_pooling_type}"
print(layer_pooling_text)
save_to_txt(layer_pooling_text, result_text_path)

loader = LoadEmbeddingSingle(
    upstream_model_type=upstream_model_type,
    frame_pooling_type = frame_pooling_type,
    device=device
)

dataset_text = f"Dataset name: {dataset_name}"
print(dataset_text)
save_to_txt(dataset_text, result_text_path)

training_data, validation_data, testing_data = loader.load_embedding(dataset_code, root_data, root_emb_data, label_mapping, data_loading_percentage)
dataset_length_text = f"No of training data samples: {len(training_data)} \nNo of validation data samples: {len(validation_data)}"
print(dataset_length_text)
save_to_txt(dataset_length_text, result_text_path)

train_dataloader = CustomEmbDataLoaderSingle(training_data, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
val_dataloader = CustomEmbDataLoaderSingle(validation_data, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)

model = DownstreamSingleTaskModel(input_dim, embedding_dim, output_dim, dropout_prob_array, layer_pooling_type)
model.to(device)
print(model)

optimizer_parameters = {
    "learning_rate": learning_rate,
    "weight_decay": weight_decay
}
scheduler_parameters = {
    "patience": patience, 
    "factor": factor
}

trainer = SingleTaskModelTrainer(model, optimizer_parameters, scheduler_parameters, device, num_epochs, saved_checkpoint_count, l1_lambda, l2_lambda, task_type)
trainer.train_dataloader = train_dataloader
trainer.test_dataloader = val_dataloader
trainer.data_count_path = data_count_path
trainer.result_text_path = result_text_path
trainer.result_plot_path = result_plot_path
trainer.model_checkpoint_path = model_checkpoint_path
trainer.plot_title = task_name

# Train the model
trainer.train()

# Plot metrics separately when needed
trainer.plot_metrics()

current_datetime = return_current_datetime()
print(current_datetime)
save_to_txt(current_datetime, result_text_path)

best_checkpoint_path = model_checkpoint_path.replace(".pth", "_best.pth")
print(best_checkpoint_path)

best_file_name = best_checkpoint_path.replace(".pth", "")
best_file_name = best_file_name.replace("checkpoint/", "result/")
best_text_path = f"{best_file_name}_eval.txt"

current_datetime = return_current_datetime()
print(current_datetime)
save_to_txt(current_datetime, best_text_path)

save_to_txt(task_text, best_text_path)
save_to_txt(upstreammodel_text, best_text_path)
save_to_txt(frame_pooling_text, best_text_path)
save_to_txt(layer_pooling_text, best_text_path)

print(dataset_text)
save_to_txt(dataset_text, best_text_path)

dataset_length_text = f"No of testing data samples: {len(testing_data)}"
print(dataset_length_text)
save_to_txt(dataset_length_text, best_text_path)

evaluator = SingleTaskModelEvaluator(model, best_checkpoint_path, testing_data, device, task_type)
evaluator.get_loss_and_accuracy(best_text_path)
evaluator.get_labels_and_predictions(best_file_name)

current_datetime = return_current_datetime()
print(current_datetime)
save_to_txt(current_datetime, best_text_path)