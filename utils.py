#!/usr/bin/env python

"""File containing function to reading, processing and loading the report."""

import torch
import torch.nn
import numpy as np
from torch.utils.data import Dataset
import re


global PAD_IDX, UNK_IDX
UNK_IDX = 0
PAD_IDX = 1

# https://github.com/Lightning-AI/lightning/issues/2644
from pytorch_lightning.callbacks import EarlyStopping

class EarlyStoppingWithWarmup(EarlyStopping):
    """
    EarlyStopping, except don't watch the first `warmup` epochs.
    """
    def __init__(self, warmup=10, **kwargs):
        super().__init__(**kwargs)
        self.warmup = warmup
    
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch < self.warmup:
            return
        else:
            self._run_early_stopping_check(trainer)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch < self.warmup:
            return
        else:
            self._run_early_stopping_check(trainer)

def fin_imp_place(lines):
    """
    Find where FINDINGS and IMPRESSION start and end in a report

    Parameter:
    ----------
    lines: list
        A raw radiology report

    Return:
    -------
    fin_start, fin_end, imp_start, imp_end: int, int, int, int
        Integers indicating the start and end index in the report
    """
    fin_start, fin_end, imp_start, imp_end = None, None, None, None
    section_order_dict = {}
    for i in range(len(lines)):
        line = lines[i]
        if 'FINDINGS' in line:
            section_order_dict['fin'] = i
            fin_start = i + 2
        if 'IMPRESSION' in line:
            section_order_dict['imp'] = i
            imp_start = i + 2
        if 'EXAMINATION' in line:
            section_order_dict['exam'] = i
        if 'INDICATION' in line:
            section_order_dict['ind'] = i
        if 'TECHNIQUE' in line:
            section_order_dict['tech'] = i
        if 'COMPARISON' in line:
            section_order_dict['comp'] = i

    #Now find where FINDINGS and IMPRESSION end in the report

    section_order = list(section_order_dict.keys())  #The list indicating the order of sections e.g., ['exam','ind', 'tech','comp','fin','imp']

    #Figure out the index of FINDINGS in the list e.g., finding_idx = 4 means it is in the 5th place in the report
    finding_idx = section_order.index('fin')

    if finding_idx == len(section_order) - 1: #FINDINGS is the very last section
        fin_end = len(lines)
    else: #if FINDINGS is *not* the last section, find where FINDINGS ends by looking at the next section after FINDINGS
        finding_next = finding_idx + 1
        next_section = section_order[finding_next] #get the name of the next section
        fin_end = section_order_dict[next_section] - 1 #get where the next section starts in the report

    #Same thing for IMPRESSION
    impression_idx = section_order.index('imp')

    if impression_idx == len(section_order) - 1: #IMPRESSION is the very last section
        imp_end = len(lines)
    else: #if IMPRSSION is *not* the last section, find where IMPRSSION ends by looking at the next section after IMPRESSION
        impression_next = impression_idx + 1
        next_section = section_order[impression_next]
        imp_end = section_order_dict[next_section] - 1

    return fin_start, fin_end, imp_start, imp_end

def tokenized_session(session):
    """
    Remove unwanted chars and tokenize the session

    Parameter:
    ----------
    session: list
        A selected / sliced session from a report. i.e. it can be either FINDINGS or IMPRESSION

    Returns:
    -------
    tk_session: list
        The tokenzied session
    """
    tk_session = ''

    #Remove the unwanted char
    for line in session:
        line = re.sub(r'[\n,.]', '', line)
        line = re.sub(r'^ ', '', line)
        tk_session += line

    #Tokenize
    #tk_session = tk_session.split(' ')

    return tk_session


def findings_impression(data_path):
    """
    Slice FINDINDS and IMPRESSION from a radiology report

    Parameter:
    ----------
    data_path: file path
        A file path to ONE radiology report in .txt format

    Returns:
    -------
    tk_findings, tk_impressions: list, list
        Tokenzied FINDINGS and IMPRESSION
    """
    #Read a radiology report
    with open(data_path) as f:
        lines = f.readlines()

    #Find where FINDINGS and IMPRESSIONS start in a report
    f, imp = None, None
    for i in range(len(lines)):
        line = lines[i]
        if 'FINDINGS' in line:
            f = i
        if 'IMPRESSION' in line:
            imp = i

    #Slice FINDINGS and IMPRESSIONS from the report
    findings = lines[f+2: imp-1]
    impression = lines[imp+2:]

    return tokenized_session(findings), tokenized_session(impression)


def build_vocab(f_tokens, imp_tokens, max_vocab_size=512): 
    """
    Build a vocabulary dictionary

    Parameters:
    ----------
    f_tokens: list
        A list of tokens, where f_token[i] returns the i_th tokenized FINDINGS 

    imp_tokens: list
        A list of tokens, where imp_tokens[i] returns the i_th tokenized IMPRESSION

    max_vocab_size: int
        The maximum number of vocabularies stored in id2token and token2id

    Returns:
    --------
    id2token: list
        A list of tokens, where id2token[i] returns token that corresponds to token i

    token2id: dictionary 
        A dictionary where keys represent tokens and corresponding values represent indices
    """      
    id2token = []    
    token2id, token_freq = {}, {}

    for tokens in f_tokens:        
        for token in tokens:            
            if token in token_freq:                
                token_freq[token] += 1            
            else:                
                token_freq[token] = 1    

    for tokens in imp_tokens:        
        for token in tokens:            
            if token in token_freq:                
                token_freq[token] += 1            
            else:               
                token_freq[token] = 1        
            most = sorted(token_freq, key = token_freq.get, reverse = True)    
            id2token = most[:max_vocab_size]       
            id2token.insert(0,'<PAD>')    
            id2token.insert(0,'<UNK>')        

    for token in id2token:        
        token2id[token] = 0       
        token2id[token] += id2token.index(token)       

    return token2id, id2token


def token2index(tokens_data):
    """
    Convert token to id in the dataset

    Parameters:
    ----------
    tokens_data: list
        Tokenized data. tokens_data[i] returns the i_th tokenized data.

    Returns:
    --------
    indices_data: list
        A list of index_list (index list for each sentence)
    """      
    indices_data = []
    for tokens in tokens_data:
        index_list = []
        for token in tokens:
            if token in token2id: 
                index_list.append(token2id[token]) 
            else: index_list.append(0)
        indices_data.append(index_list)

    return indices_data


def cxr_collate_func(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all
    data have the same length
    """
    tokens = []
    len_tokens = []

    for datum in batch:
        len_tokens.append(len(datum))
    # padding
    for datum in batch:
        padded_vec = np.pad(np.array(datum), pad_width=((0,max_sentence_length-len(datum))), mode="constant", constant_values=PAD_IDX)
        tokens.append(padded_vec)
      
    return [torch.from_numpy(np.array(tokens)), torch.LongTensor(len_tokens)]


class CXRDataset(Dataset):
    """
    Class that represents a FINDINGS or IMPRESSION data that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """
    def __init__(self, indexed_tokens, max_sentence_length):
        """
        Parameters:
        -----------
        f_tokens: list
            A list of FINDINGS tokens

        imp_tokens: list 
            A listof prem tokens

        max_sentence_length: int
            A fixed length of all sentence
        """
        self.indexed_tokens = indexed_tokens
        self.max_sentence_length = max_sentence_length
        
    def __len__(self):
        return len(self.indexed_tokens)

    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        token_idx = self.indexed_tokens[key][:self.max_sentence_length]

        return token_idx

    

def process_chexpert(data):
    """
    Processed CheXpert dataset

    Parameters:
    ----------
    data: DataFrame
        Train / Valid CheXpert dataset

    Returns:
    --------
    data: DataFrame
        Processed train/valid CheXpert dataset
    """      
    #Filter categories the original paper used
    data = data[['Path', 'No Finding','Atelectasis', 'Cardiomegaly', 'Edema', 'Fracture', 'Pleural Effusion', 'Pneumonia', 'Pneumothorax']]

    #Fill NA in 'No Finding' Column - 1: abnormality detected; 0: no abnormality
    data['No Finding'].fillna(0, inplace=True)

    categories = ['No Finding','Atelectasis', 'Cardiomegaly', 'Edema', 'Fracture', 'Pleural Effusion', 'Pneumonia', 'Pneumothorax']

    label_map = {}
    i = 0
    for category in categories:
        label_map[category] = i
        i += 1
    
    def categorize(data):
        for category in categories:
            if data[category] == np.float64(1):
                return label_map[category]

    data['Class'] = data.apply(lambda row: categorize(row), axis=1)

    data.dropna(subset=['Class'], inplace=True)

    return data

