import torch
import pickle
import random
import argparse
from adversarial_attack.pcp_utils import *

def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def genetic_search(HTnn_net, orig_pcp_list, population_size, generations):
    population = [mutate_pcp_list(HTnn_net, deepcopy(orig_pcp_list), n_changes=random.randint(1, 2)) for _ in range(population_size)]  
    
    best_solutions = []
    
    for _ in range(generations):
        fitness_scores, appx_errs, dect_errs = [], [], []
        for i in range(len(population)):
            appx_err = approximation_error(deepcopy(orig_pcp_list), deepcopy(population[i]))
            dect_err = detect_score(HTnn_net, get_all_embeddings(HTnn_net, deepcopy(population[i])))
            appx_errs.append(round(appx_err, 2))
            dect_errs.append(round(dect_err, 2))
            fitness_scores.append(appx_err+dect_err)
                
        # Update best solutions
        for i, fitness in enumerate(fitness_scores):
            best_solutions.append((deepcopy(population[i]), fitness, appx_errs[i], dect_errs[i]))
        best_solutions.sort(key=lambda x: x[1])  # Sort by fitness score
        best_solutions = best_solutions[:1]  # Keep only top 10
        
        parents = random.choices(population, weights=fitness_scores, k=population_size)   
        new_population = []
        for i in range(0, population_size, 2):
            parent1, parent2 = parents[i], parents[i+1]    
            child1 = mutate_pcp_list(HTnn_net, deepcopy(parent1))
            child2 = mutate_pcp_list(HTnn_net, deepcopy(parent2))
            new_population.extend([child1, child2])
        
        population = new_population
    
    return best_solutions

parser = argparse.ArgumentParser(description='Adversarial Attacks on PCP Traces')
parser.add_argument('--seed', default=1, type=int, help='Value of the random seed.')
parser.add_argument('--model', default='cnn', type=str, choices=['cnn', 'lstm'])


if __name__ == '__main__':

    run_args = parser.parse_args()

    set_random_seeds(random_seed=run_args.seed)

    with open('save/source_config.pkl', 'rb') as pickle_file:
        source_config_copy = pickle.load(pickle_file)

    if run_args.model == 'cnn':
        from dnn_model.CNN_netlist_softma_save_resluts import Classifier_Netlist
        path = './weights/CNN_model_pretrained.pth'
        HTnn_net = Classifier_Netlist(group_id=str(2), base_path='json_temp_file', source_config=source_config_copy, pretrained=path)   

    elif run_args.model == 'lstm':
        from dnn_model.LSTM_netlist_softmax_save_results import Classifier_Netlist
        path = './weights/LSTM_model_pretrained.pth'
        HTnn_net = Classifier_Netlist(group_id=str(2), base_path='json_temp_file', source_config=source_config_copy, pretrained=path)   


    print("*********** Get all the HT circuits names")
    all_text_labels = get_all_text_labels(HTnn_net.val_dataloader)
    trojan_comps_labels = []
    for elem in all_text_labels:
        if elem.startswith("t"):
            trojan_comps_labels.append(elem)

    print("*********** Get all the PCPs and embeddings from each HT circuit")
    for idx, circuit in enumerate(trojan_comps_labels):
        
        print('******* Adversarial Attack on PCP for HT circuit: ', circuit)
        trojan_comp = trojan_comps_labels[idx]
        pcp_embs = get_samples_by_text_label(HTnn_net.val_dataloader, trojan_comp)

        all_embds, all_cmps, all_labels = [], [], []
        
        for pcp_emb in pcp_embs:
            p_emb, label = pcp_emb
            full_pcp_cmp = []
            
            for i in range(5):
                name = get_cmp_by_emb(HTnn_net.val_data.word2vec_dict, list(np.float32(p_emb[i])))
                full_pcp_cmp.append(name)
            
            all_labels.append(label.item())
            all_embds.append(p_emb)
            all_cmps.append(full_pcp_cmp)

        best_solution = genetic_search(HTnn_net, deepcopy(all_cmps), population_size=128, generations=10)
        
        print(best_solution[0][0])
        print('********* Average Approximation Error', best_solution[0][2])
        print('********* HT Detection Error', best_solution[0][3])
