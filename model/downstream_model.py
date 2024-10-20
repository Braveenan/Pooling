import torch
import torch.nn as nn
from pooling.pooling import *

class SingleClassifierOutput:
    def __init__(self, logits=None, prediction=None):
        self.logits = logits
        self.prediction = prediction

class DownstreamSingleTaskModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, output_dim, dropout_prob_array, pooling_type):
        super(DownstreamSingleTaskModel, self).__init__()

        # Initialize the pooling layer
        self.pooling = Pooling(pooling_type)
            
        self.dropout1 = nn.Dropout(p=dropout_prob_array[0])
        self.hidden_layer = nn.Linear(input_dim, embedding_dim)
        self.dropout2 = nn.Dropout(p=dropout_prob_array[1])
        self.classifier_layer = nn.Linear(embedding_dim, output_dim)
        
    def forward(self, input_seq):
        # Apply pooling to input sequence
        pooled_input = self.pooling.get_vector_after_pooling(input_seq, dim=1)  
        
        embedding = self.dropout1(pooled_input)
        embedding = self.hidden_layer(embedding)
        embedding = self.dropout2(embedding)
        logits = self.classifier_layer(embedding)
        
        # Get predictions
        _, prediction = torch.max(logits, dim=1)
        
        # Return the output as a SingleClassifierOutput object
        return SingleClassifierOutput(logits=logits, prediction=prediction)
    
    def get_all_embeddings(self, input_seq):
        pooled_input = self.pooling.get_vector_after_pooling(input_seq, dim=1)  # Same pooling step
        embedding = self.hidden_layer(pooled_input)
        
        return embedding
