import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataset.custom_emb_dataloader import *
import pandas as pd
import csv
       
class SingleTaskEvaluatorOutput:
    def __init__(self, avg_loss_all=None, accuracy_all=None):
        self.avg_loss_all = avg_loss_all
        self.accuracy_all = accuracy_all

        
class SingleTaskBatchProcessOutput:
    def __init__(self, loss_all=None, batch_all=None, correct_count_all=None, samples_count_all=None, labels_task1=None, predictions_task1=None):
        self.loss_all = loss_all
        self.batch_all = batch_all
        self.correct_count_all = correct_count_all
        self.samples_count_all = samples_count_all
        self.labels_task1 = labels_task1
        self.predictions_task1 = predictions_task1
        
           
class SingleTaskModelEvaluator:
    def __init__(self, model, model_checkpoint_path, dataset, device, task_type):
        self.device = device
        self.task_type = task_type
        
        self.dataloader = CustomEmbDataLoaderSingle(dataset, batch_size=1, shuffle=False, pin_memory=True)
        
        if model_checkpoint_path is not None:
            checkpoint = torch.load(model_checkpoint_path, map_location=torch.device(device))
            model.load_state_dict(checkpoint)
        self.model = model

        self.loss_fn = nn.CrossEntropyLoss()
        
    def _get_predicted_and_count(self, prediction, labels):
        correct_count = (prediction == labels).sum().item()
        samples_count = labels.size(0)
        return labels, prediction, correct_count, samples_count

    def _calculate_total_loss(self, loss_task, loss_all, loss_weight):
        batch_task = int(loss_task is not None)
        if loss_task is None:
            loss_task = torch.tensor(0)
            loss_task.to(self.device)
        else:
            loss_all += loss_task*loss_weight
        return loss_task, loss_all, batch_task

    def _calculate_batch_loss(self, logits, labels, loss_fn):
        loss = None
        if labels is not None:
            if labels.size(0) != 0:
                loss = loss_fn(logits, labels)
        return loss

    def _process_batch(self, batch):
        input_seq = batch[0].to(self.device)
        labels_task1 = batch[1].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_seq=input_seq)
            logits_task1 = outputs.logits
            prediction_task1 = outputs.prediction
            
            loss_all = 0
            batch_all = 1
            loss_weight1 = 1.0 
                
            loss_task1 = self._calculate_batch_loss(logits_task1, labels_task1, self.loss_fn)
        
            loss_task1, loss_all, batch_task1 = self._calculate_total_loss(loss_task1, loss_all, loss_weight1)
        
        labels_task1, predictions_task1, correct_count_task1, samples_count_task1 = self._get_predicted_and_count(prediction_task1, labels_task1)
        
        samples_count_all = samples_count_task1 
        correct_count_all = correct_count_task1 

        return SingleTaskBatchProcessOutput(
            loss_all=loss_all.item(),
            batch_all=batch_all,
            correct_count_all=correct_count_all,
            samples_count_all=samples_count_all,
            labels_task1=labels_task1,
            predictions_task1=predictions_task1,
        )
    
    def _save_to_txt(self, content, filename):
        with open(filename, 'a') as f:
            print(content, file=f)

    def _calculate_accuracy(self, total_correct, total_samples):
        if total_samples == 0:
            return 0.0
        else:
            return total_correct / total_samples
    
    def _calculate_average_loss(self, total_loss, total_batches):
        if total_batches == 0:
            return 0.0
        else:
            return total_loss / total_batches

    def _process_data_loader(self, dataloader):
        self.model.eval()
        total_loss_all = 0.0
        total_batches_all = 0
        total_correct_all = 0
        total_samples_all = 0

        with torch.no_grad():
            for batch in dataloader:
                batch_output = self._process_batch(batch)
                total_loss_all += batch_output.loss_all
                total_batches_all += batch_output.batch_all
                total_correct_all += batch_output.correct_count_all
                total_samples_all += batch_output.samples_count_all

        accuracy_all = self._calculate_accuracy(total_correct_all, total_samples_all)
        avg_loss_all = self._calculate_average_loss(total_loss_all, total_batches_all)

        return SingleTaskEvaluatorOutput(
            avg_loss_all=avg_loss_all,
            accuracy_all=accuracy_all,
        )
        
    def _generate_output_info(self, loss, accuracy, task_name, result_text_path):
        output_info = f"Loss {task_name}: {loss:.4f}, Accuracy {task_name}: {accuracy:.4f}"
        print(output_info)
        if result_text_path is not None:
            self._save_to_txt(output_info, result_text_path)

    def _generate_text_array(self, input_integer):
        text_array = []
        for i in range(1, input_integer + 1):
            text_array.append(f"feature_{i}")
        return text_array
    
    def get_loss_and_accuracy(self, result_text_path=None):
        eval_output = self._process_data_loader(self.dataloader)
        
        self._generate_output_info(eval_output.avg_loss_all, eval_output.accuracy_all, self.task_type, result_text_path)
        
    def get_labels_and_predictions(self, output_file_name):
        prediction_csv_file_task1 = f"{output_file_name}_prediction_{self.task_type}.csv"
       
        with open(prediction_csv_file_task1, mode='w', newline='') as file1:
            writer1 = csv.writer(file1)
            writer1.writerow([f'True Labels-{self.task_type}', f'Predicted Labels-{self.task_type}'])
            
            for batch in self.dataloader:
                batch_output = self._process_batch(batch)
                
                labels_task1 = batch_output.labels_task1.cpu()
                predictions_task1 = batch_output.predictions_task1.cpu()
                
                writer1.writerow([labels_task1.item(), predictions_task1.item()])