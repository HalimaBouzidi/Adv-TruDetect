import torch
import pickle
import random
import argparse
from adversarial_attack.pcp_utils import *
from adversarial_attack.fgsm import FGSM
from adversarial_attack.pgd import PGD
from adversarial_attack.bim import BIM
from adversarial_attack.utils import compute_accuracy, compute_confusion_matrix

def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

parser = argparse.ArgumentParser(description='Test AttentiveNas Models')
parser.add_argument('--seed', default=1, type=int, help='Value of the random seed.')
parser.add_argument('--model', default='cnn', type=str, choices=['cnn', 'lstm'])
parser.add_argument('--attack', default='fgsm', type=str, choices=['fgsm', 'pgd', 'bim'])
parser.add_argument('--eps', default=8, type=int, help='Epsilon for the adversarial attack')


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

        print('******* Adversarial Attack on E-PCP for HT circuit: ', circuit)
        trojan_comp = trojan_comps_labels[idx]
        pcp_embs = get_samples_by_text_label(HTnn_net.val_dataloader, trojan_comp)

        all_embds, all_cmps, all_labels = [], [], []
        for pcp_emb in pcp_embs:
            p_emb, label = pcp_emb
            all_labels.append(label.item())
            all_embds.append(torch.unsqueeze(p_emb, 0))

        clean_data = torch.stack(all_embds, dim=0).to(HTnn_net.device)
        clean_labels = torch.tensor(all_labels).to(HTnn_net.device)

        if run_args.attack == 'fgsm':

            print('******* FGSM Adversarial attack *******')

            adv_attack = FGSM(HTnn_net.model, eps=float(run_args.eps8/255))
            adv_data = adv_attack(clean_data, clean_labels)

            robust_acc = compute_accuracy(HTnn_net.model, adv_data, clean_labels, batch_size=32, device=HTnn_net.device)
            print("Robust Accuracy", robust_acc) 

            cm = compute_confusion_matrix(HTnn_net.model, clean_data, clean_labels, HTnn_net.device)
            tn, fp, fn, tp = cm.ravel()
            print("Before Adv Attack -- TN {} | FP {} | FN {} | TP {}".format(tn, fp, fn, tp))

            cm = compute_confusion_matrix(HTnn_net.model, adv_data, clean_labels, HTnn_net.device)
            tn, fp, fn, tp = cm.ravel()
            print("After Adv Attack -- TN {} | FP {} | FN {} | TP {}".format(tn, fp, fn, tp))


        elif run_args.attack == 'pgd':

            print('******* PGD Adversarial attack *******')

            adv_attack = PGD(HTnn_net.model, eps=float(run_args.eps8/255))
            adv_data = adv_attack(clean_data, clean_labels)

            robust_acc = compute_accuracy(HTnn_net.model, adv_data, clean_labels, batch_size=32, device=HTnn_net.device)
            print("Robust Accuracy", robust_acc) 

            cm = compute_confusion_matrix(HTnn_net.model, clean_data, clean_labels, HTnn_net.device)
            tn, fp, fn, tp = cm.ravel()
            print("Before Adv Attack -- TN {} | FP {} | FN {} | TP {}".format(tn, fp, fn, tp))

            cm = compute_confusion_matrix(HTnn_net.model, adv_data, clean_labels, HTnn_net.device)
            tn, fp, fn, tp = cm.ravel()
            print("After Adv Attack -- TN {} | FP {} | FN {} | TP {}".format(tn, fp, fn, tp))


        elif run_args.attack == 'bim':

            print('******* BiM Adversarial attack *******')

            adv_attack = BIM(HTnn_net.model, eps=float(run_args.eps8/255))
            adv_data = adv_attack(clean_data, clean_labels)

            robust_acc = compute_accuracy(HTnn_net.model, adv_data, clean_labels, batch_size=32, device=HTnn_net.device)
            print("Robust Accuracy", robust_acc) 

            cm = compute_confusion_matrix(HTnn_net.model, clean_data, clean_labels, HTnn_net.device)
            tn, fp, fn, tp = cm.ravel()
            print("Before Adv Attack -- TN {} | FP {} | FN {} | TP {}".format(tn, fp, fn, tp))

            cm = compute_confusion_matrix(HTnn_net.model, adv_data, clean_labels, HTnn_net.device)
            tn, fp, fn, tp = cm.ravel()
            print("After Adv Attack -- TN {} | FP {} | FN {} | TP {}".format(tn, fp, fn, tp))

        else:

            raise NotImplementedError
    