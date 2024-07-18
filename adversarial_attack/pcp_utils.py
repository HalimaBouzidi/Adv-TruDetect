import re
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from typing import List, Any
from copy import copy, deepcopy

def get_all_text_labels(dataloader: DataLoader) -> List[str]:
    text_labels = []
    for batch in dataloader:
        text_label = batch[2]  
        text_labels.extend(text_label)
    return list(set(text_labels))  # Remove duplicates

def get_samples_by_text_label(dataloader: DataLoader, target_text: str) -> List[Any]:
    matching_samples = []
    for batch in dataloader:
        data, class_label, text_label = batch 
        for i, label in enumerate(text_label):
            if label == target_text:
                matching_samples.append((data[i], class_label[i]))
    return matching_samples

def get_cmp_by_emb(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None  

def get_emb_by_cmp(dictionary, value):
    for key, val in dictionary.items():
        if key == value:
            return val
    return None 

def get_all_embeddings(HTnn_net, approx_list_pcp):
    embds = []
    for i in range(len(approx_list_pcp)):
        n_array = [get_emb_by_cmp(HTnn_net.val_data.word2vec_dict, elem) for elem in approx_list_pcp[i]]
        embd = torch.unsqueeze(torch.from_numpy(np.array(n_array)), 0)
        embds.append(embd)
    return embds


def approximation_error(orig_pcp_list, approx_pcp_list):
    error_maps = {  'i':   {'i': 0, 'and':.75, 'nnd': .25, 'or': .75, 'nor':.25, 'xor':.5, 'xnr':.5}, 
                    'and': {'i':.75, 'and':0, 'nnd': 1., 'or': .5, 'nor':.5, 'xor':.75, 'xnr':.25 }, 
                    'nnd': {'and':1., 'nnd':0, 'i': .25, 'or': .5, 'nor':.5, 'xor':.25, 'xnr':.75 }, 
                    'or':  {'and':.5, 'nnd': .5, 'i': 0.75, 'or':0, 'nor':1., 'xor':.25, 'xnr':.75 }, 
                    'nor':  {'and':.5, 'nnd': .5, 'i': 0.25, 'or':1., 'nor':0, 'xor':.75, 'xnr':.25 }, 
                    'xor':  {'and':.75, 'nnd': .25, 'i': 0.5, 'or':.25, 'nor':.75, 'xor':0, 'xnr':1. }, 
                    'xnr':  {'and':.25, 'nnd': .75, 'i': 0.5, 'or':.75, 'nor':.25, 'xor':1., 'xnr':0 }   }
    
    final_error = 0 
    for i in range(len(orig_pcp_list)):
        orig_pcp, approx_pcp = orig_pcp_list[i], approx_pcp_list[i]
        error = 0
        for j in range(len(orig_pcp)):
            orig, approx = orig_pcp[j], approx_pcp[j]
            orig_op, _= separate_letters_numbers(orig.split('_')[1])
            approx_op, _= separate_letters_numbers(approx.split('_')[1])
            if orig_op in ['i', 'and', 'nnd', 'or', 'nor', 'xor', 'xnor']:  
                error += error_maps[orig_op][approx_op]
        
        final_error += error/len(orig_pcp)
    
    return final_error/len(orig_pcp_list)


def detect_score(HTnn_net, approx_pcp_list):
    input_data = torch.stack(approx_pcp_list, dim=0).to(HTnn_net.device)
    out = HTnn_net.model(input_data)
    _, pred = torch.max(out, 1) 
    return pred.sum().item()/len(approx_pcp_list)

def separate_letters_numbers(input_string):
    letters = ''.join(re.findall(r'[a-zA-Z]', input_string))
    numbers = ''.join(re.findall(r'\d', input_string))
    return letters, numbers


def mutate_pcp_word(pcp_word, dict):
    mutate_list = {'i':   ['nnd', 'nor', 'xor', 'xnr'],
                   'and': ['or', 'nor', 'xnr'],
                   'nnd': ['or', 'nor', 'xor'],
                   'or':  ['and', 'nnd', 'xor'],
                   'nor': ['and', 'nnd', 'xnr'],
                   'xor': ['nnd', 'or'],
                   'xnr': ['and', 'nor']}

    splits = pcp_word.split('_')
    op, num = separate_letters_numbers(splits[1])
    if op in ['i', 'and', 'nnd', 'or', 'nor', 'xor', 'xnor']:
        new_word = splits[0]+'_'+random.choice(mutate_list[op])+str(num)+'_'+splits[2]
        if get_emb_by_cmp(dict, new_word) != None:
            return new_word
    return pcp_word

def mutate_pcp_list(HTnn_net, orig_list, n_changes=1, p=0.5):
    dict = HTnn_net.val_data.word2vec_dict
    orig_pcp_copy = deepcopy(orig_list)
    for i in range(len(orig_pcp_copy)):
        if random.random() > p:
            pcp_word = orig_pcp_copy[i][random.randint(0, len(orig_pcp_copy[i])-1)]
            new_pcp_word = mutate_pcp_word(pcp_word, dict)
            
            for j in range(len(orig_pcp_copy)): # Apply changes to all pcps that share the word
                for k in range(len(orig_pcp_copy[j])):
                    if orig_pcp_copy[j][k].split('_')[1] == pcp_word.split('_')[1]:
                        orig_pcp_copy[j][k] = orig_pcp_copy[j][k].split('_')[0]+'_'+new_pcp_word.split('_')[1]+'_'+orig_pcp_copy[j][k].split('_')[2]

            n_changes -= 1
        if n_changes == 0:
            break
    return orig_pcp_copy


