import socket
import os
import daemon, sys, time
import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, SequentialSampler
from transformers import BertTokenizer
from transformers import BertForSequenceClassification

import nltk
# from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 


def toxic(text, model):

    ts = time.time()
    test=np.ravel([text])

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', 
                                              do_lower_case=True)

    encoded_data_test = tokenizer.batch_encode_plus(
        test, 
        add_special_tokens=True, 
        return_attention_mask=True, 
        pad_to_max_length=True, 
        max_length=256, 
        return_tensors='pt'
    )


    input_ids_test = encoded_data_test['input_ids']
    attention_masks_test = encoded_data_test['attention_mask']

    dataset_test = TensorDataset(input_ids_test, attention_masks_test)
    dataloader_test = DataLoader(dataset_test, 
                           sampler=SequentialSampler(dataset_test), 
                           batch_size=4)

    device='cpu'
    predictions = predicti(dataloader_test)
    print('predict Ok')
    x=np.argmax(predictions, axis=1).flatten()
    print('toxic return',x[0])
    print('cpu time:',time.time()-ts)
    return x[0] 


def predicti(dataloader_test):

    model.eval()
    
    loss_val_total = 0
    predictions = []
    
    for batch in dataloader_test:
        
        batch = tuple(b.to('cpu') for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
        # loss = outputs[0]
        logits = outputs[0]
        # loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        # label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        # true_vals.append(label_ids)
    
    # loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    # true_vals = np.concatenate(true_vals, axis=0)
            
    return predictions


# filename = f"/tmp/{sys.argv[1]}.txt" 
# print(filename)
# with open( filename, 'r') as f:
#     text = f.read()

# print(text)




model_data = torch.load('/model/finetuned_BERT_epoch_1.model',  map_location=torch.device('cpu'))

model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                          num_labels=2,
                                                          output_attentions=False,
                                                          output_hidden_states=False)

lemmatizer = WordNetLemmatizer()
model.load_state_dict(model_data)

print('-' * 20)
print('Model loaded')
print('-' * 20)

server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind("/tmp/python_unix_sockets_example")
server.listen()
while True:
    conn, _addr = server.accept()
    data = conn.recv(1024)
    res = toxic(data.encode() , model)
    conn.send( str(res).decode() )
    conn.close()        



