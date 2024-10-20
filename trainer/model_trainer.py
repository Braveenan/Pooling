import os
import torch
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
from torch.optim import AdamW
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from matplotlib.ticker import LogLocator

class SingleTaskTrainerOutput:
    def __init__(self, avg_loss_all=None, accuracy_all=None):
        self.avg_loss_all = avg_loss_all
        self.accuracy_all = accuracy_all
    
    
class SingleTaskBatchProcessOutput:
    def __init__(self, loss_all=None, batch_all=None, correct_count_all=None, samples_count_all=None):
        self.loss_all = loss_all
        self.batch_all = batch_all
        self.correct_count_all = correct_count_all
        self.samples_count_all = samples_count_all
   
    
class SingleTaskModelTrainer:
    def __init__(self, model, optimizer_parameters, scheduler_parameters, device, num_epochs, saved_checkpoint_count, l1_lambda, l2_lambda, task_type):
        self.model = model
        self.device = device
        self.num_epochs = num_epochs
        self.saved_checkpoint_count = saved_checkpoint_count
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.task_type = task_type

        self.optimizer = AdamW(model.parameters(), lr=optimizer_parameters['learning_rate'], weight_decay=optimizer_parameters['weight_decay'])
        self.loss_fn = nn.CrossEntropyLoss()
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=scheduler_parameters['patience'], factor=scheduler_parameters['factor'], verbose=True)
        
        self.test_accuracy_all_threshold = 0.0
        
        self.train_losses_all = []
        self.train_accuracies_all = []
        self.test_losses_all = []
        self.test_accuracies_all = []

        self.learning_rate_array = []
        
    def _get_count_batch(self, prediction, labels):
        correct_count = (prediction == labels).sum().item()
        samples_count = labels.size(0)
        return correct_count, samples_count
    
    def _calculate_total_loss(self, loss_task, loss_all):
        batch_task = int(loss_task is not None)
        loss_all += loss_task
        return loss_task, loss_all, batch_task

    def _calculate_batch_loss(self, logits, labels, loss_fn):
        loss = None
        data_count = None
    
        loss = loss_fn(logits, labels)
        data_count = labels.size(0)
        
        return loss, data_count
        
    def _process_batch(self, batch, train_mode=True):
        input_seq = batch[0].to(self.device)
        labels_task1 = batch[1].to(self.device)

        self.optimizer.zero_grad()

        outputs = self.model(input_seq=input_seq)
        logits_task1 = outputs.logits
        prediction_task1 = outputs.prediction
                    
        loss_all = 0
        batch_all = 1

        loss_task1, data_count1 = self._calculate_batch_loss(logits_task1, labels_task1, self.loss_fn)
        loss_task1, loss_all, batch_task1 = self._calculate_total_loss(loss_task1, loss_all)
        
        # Get the model parameters
        model_params = [param for param in self.model.parameters()]
        
        # Calculate L1 regularization term
        l1_regularization = torch.tensor(0., device=self.device)
        for param in model_params:
            l1_regularization += torch.sum(torch.abs(param))

        # Add L1 regularization term to the total loss
        loss_all += self.l1_lambda * l1_regularization

        # Calculate L2 regularization term
        l2_regularization = torch.tensor(0., device=self.device)
        for param in model_params:
            l2_regularization += torch.sum(torch.square(param))

        # Add L2 regularization term to the total loss
        loss_all += self.l2_lambda * l2_regularization
            
        if train_mode:
            loss_all.backward()
            self.optimizer.step()
            
        correct_count_task1, samples_count_task1 = self._get_count_batch(prediction_task1, labels_task1)
        
        samples_count_all = samples_count_task1 
        correct_count_all = correct_count_task1 

        if self.data_count_path is not None:
            operation = "train" if train_mode else "val"
            data_count_info = f"{operation} {data_count1}"
            self._save_to_txt(data_count_info, self.data_count_path)

        return SingleTaskBatchProcessOutput(
            loss_all=loss_all.item(),
            batch_all=batch_all,
            correct_count_all=correct_count_all,
            samples_count_all=samples_count_all,
        )
    
    def _process_data_loader(self, data_loader, train_mode=True):
        self.model.train() if train_mode else self.model.eval()
        total_loss_all = 0.0
        total_batches_all = 0
        total_correct_all = 0
        total_samples_all = 0

        with torch.set_grad_enabled(train_mode):
            for batch in data_loader:
                batch_output = self._process_batch(batch, train_mode)
                total_loss_all += batch_output.loss_all
                total_batches_all += batch_output.batch_all
                total_correct_all += batch_output.correct_count_all
                total_samples_all += batch_output.samples_count_all

        accuracy_all = self._calculate_accuracy(total_correct_all, total_samples_all)
        avg_loss_all = self._calculate_average_loss(total_loss_all, total_batches_all)

        return SingleTaskTrainerOutput(
            avg_loss_all=avg_loss_all,
            accuracy_all=accuracy_all,
        )
    
    def _train_epoch(self, data_loader):
        train_output = self._process_data_loader(data_loader, train_mode=True)

        return train_output

    def _test_epoch(self, data_loader):
        test_output = self._process_data_loader(data_loader, train_mode=False)

        return test_output
    
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

    def _save_to_txt(self, content, filename):
        with open(filename, 'a') as f:
            print(content, file=f)

    def _plot_metric(self, num_epochs, train_data, test_data, plot_path, plot_title, metric_name):
        # Create a figure
        plt.figure(figsize=(16, 8))
    
        # Plot Train and Test data
        plt.plot(range(1, num_epochs + 1), train_data, label=f'Train {metric_name}', marker='o')
        plt.plot(range(1, num_epochs + 1), test_data, label=f'Test {metric_name}', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.legend()
        plt.xticks(range(1, num_epochs + 1))  # Set x-axis ticks to discrete values
    
        # Set title and save the figure
        plt.title(plot_title)
        plt.tight_layout()
    
        # Modify the plot_path based on metric_name
        modified_plot_path = plot_path.replace(".png", f"_{metric_name}.png")
    
        plt.savefig(modified_plot_path, transparent=True)
        plt.close()
        plt.clf()

    def _plot_lr(self, num_epochs, lr_array, plot_path, plot_title):
        # Create a figure
        plt.figure(figsize=(16, 8))
    
        # Plot Train and Test data
        plt.plot(range(1, num_epochs + 1), lr_array, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Learning rate')
        plt.legend()
        plt.xticks(range(1, num_epochs + 1))  # Set x-axis ticks to discrete values

        plt.yscale('log')

        plt.gca().yaxis.set_major_locator(LogLocator())
    
        # Set title and save the figure
        plt.title(plot_title)
        plt.tight_layout()
    
        # Modify the plot_path based on metric_name
        modified_plot_path = plot_path.replace(".png", f"_lr.png")
    
        plt.savefig(modified_plot_path, transparent=True)
        plt.close()
        plt.clf()
    
    def _plot_function(self, num_epochs, train_accuracies, test_accuracies, train_losses, test_losses, plot_path, plot_title):
        self._plot_metric(num_epochs, train_accuracies, test_accuracies, plot_path, plot_title, "Accuracy")
        self._plot_metric(num_epochs, train_losses, test_losses, plot_path, plot_title, "Loss")
        
    def _save_model_checkpoint(self, model_checkpoint_path, epoch, test_accuracy, saved_checkpoint_count):
        model_epoch_checkpoint_path = model_checkpoint_path.replace(".pth", f"_epoch{epoch}.pth")
        torch.save(self.model.state_dict(), model_epoch_checkpoint_path)
        
        if epoch > saved_checkpoint_count:
            model_epoch_checkpoint_to_remove = model_checkpoint_path.replace(".pth", f"_epoch{epoch-saved_checkpoint_count}.pth")
            if os.path.exists(model_epoch_checkpoint_to_remove):
                os.remove(model_epoch_checkpoint_to_remove)

        if test_accuracy > self.test_accuracy_all_threshold:
            self.test_accuracy_all_threshold = test_accuracy
            model_best_checkpoint_path = model_checkpoint_path.replace(".pth", f"_best.pth")
            torch.save(self.model.state_dict(), model_best_checkpoint_path)
        
    def _generate_epoch_info(self, epoch, num_epochs, train_loss, train_accuracy, test_loss, test_accuracy, task_name, result_text_path):
        epoch_info = f"Epoch {epoch+1}/{num_epochs} - Train Loss {task_name}: {train_loss:.4f}, Train Accuracy {task_name}: {train_accuracy:.4f}, Test Loss {task_name}: {test_loss:.4f}, Test Accuracy {task_name}: {test_accuracy:.4f}"
        print(epoch_info)
        if result_text_path is not None:
            self._save_to_txt(epoch_info, result_text_path)
        
    def train(self):
        for epoch in range(self.num_epochs):
            train_output = self._train_epoch(self.train_dataloader)
            test_output = self._test_epoch(self.test_dataloader)

            train_loss_all, train_accuracy_all = train_output.avg_loss_all, train_output.accuracy_all
            test_loss_all, test_accuracy_all = test_output.avg_loss_all, test_output.accuracy_all

            self.train_losses_all.append(train_loss_all)
            self.train_accuracies_all.append(train_accuracy_all)
            self.test_losses_all.append(test_loss_all)
            self.test_accuracies_all.append(test_accuracy_all)
            
            self._generate_epoch_info(epoch, self.num_epochs, train_loss_all, train_accuracy_all, test_loss_all, test_accuracy_all, self.task_type, self.result_text_path)               
                
            self._save_model_checkpoint(self.model_checkpoint_path, epoch+1, test_accuracy_all, self.saved_checkpoint_count)
            self.scheduler.step(test_loss_all)
            learning_rate_value = self.optimizer.param_groups[0]["lr"]
            self.learning_rate_array.append(learning_rate_value)

        best_accuracy_text_all = f"Best model accuracy: {self.test_accuracy_all_threshold}"
        print(best_accuracy_text_all)
        self._save_to_txt(best_accuracy_text_all, self.result_text_path)
            
    def plot_metrics(self):
        self._plot_function(self.num_epochs, self.train_accuracies_all, self.test_accuracies_all, self.train_losses_all, self.test_losses_all, self.result_plot_path, self.plot_title)

        self. _plot_lr(self.num_epochs, self.learning_rate_array, self.result_plot_path, self.plot_title)
    
    
