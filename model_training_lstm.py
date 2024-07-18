######sci-kit######################
import time
import torch
import gensim

######pytorch######################
from word2vec_netlist.trainer_net import Word2VecTrainer_Netlist
from netlist_scraper import NetlistDataScraper

from dnn_model.LSTM_netlist_softmax_save_results import Classifier_Netlist

from logger import setup_logger
from numpy import dot
from numpy.linalg import norm

######python######################
import sys
import os
import yaml
import copy
from functools import reduce
from numpy import linalg as LA


import psutil
count = psutil.cpu_count()
print(f"CPU's Logical Cores: {count}")
p = psutil.Process()
cpu_lst = p.cpu_affinity()
print(F"CURRENT TASK ASSIGN TO CORE: {cpu_lst}")
assign_cpu_list = [0]
#p.cpu_affinity(assign_cpu_list)
print(F"ASSIGN TASK TO CORE: {assign_cpu_list}")

from utils import *

##########PATH########################
abspath = os.path.dirname(__file__)
sys.path.append(abspath)
base_dir = abspath # main
perl_temp_source_dir = os.path.join(base_dir, 'json_temp_file') # main and utils

if __name__ == '__main__':
    # ====== Start Main =====================================
    debug_mode = 0  # 0-enable perl script for parsing /1-not enable perl scrupt to run, should be used with "Parsing"
    regen_confyaml_parameter = True  # if True, must check sample_dir,yaml_config
    global_logic_level = 3
    global_scanff_replace = 0
    # ===grouping papra
    kfd_sample_reorder = True
    k_parameter = (True, 3) # (kgroup_enable,k_ara)  when kgroup_enable=False, k_ara is 'not care'
    test_set_enable = False

    fix_type_new_para = [
        # ('SINGLE', 'SINGLE', 'LIST', 'NUM'),
        ('sdir', 'Number of Trojans:', 'Type:', 'TRAIN_NUM','TEST_NUM'),  # parameter items
        ("c2670", 1, ["combinational"], 3, 5),
        ("c3540", 1, ["combinational"], 3, 5),
        ("c5315", 1, ["combinational"], 3, 5),
        ("c6288", 1, ["combinational"], 3, 5),
        ("s1423", 1, ["combinational"], 3, 5),
        ("s13207", 1, ["combinational"], 3, 5),
        ("s15850", 1, ["combinational"], 3, 5),
        ("s35932", 1, ["combinational"], 3, 5),

    ]

    process_query = [
        # 'Parsing',
        # 'Embedding',
        'NNTraining'
        ]  # 'Parsing', 'Embedding', 'NNTraining'

    sample_dir = ['thub_c2670', 'thub_c3540', 'thub_c5315', 'thub_c6288', 'thub_s1423', 'thub_s13207', 'thub_s15850',
                  'thub_s35932']
    yaml_config = {
        "DESIGN_FOLDER": [0, None],  # [0]: 0-not specified [1], 1- fixed [1](1,"XXXX")
        "SUB_TYPE": [0, None],  # [0]: 0-not specified [1], 1- fixed [1](1,"XXXX")
        "SOURCE_FILE": [0, None],  # 0-same as value in sub_type, (1,"_trojan_inserted"),
        "LIB_FOLDER": [1, "cell_lib2"],  # [0]: 0-not specified [1], 1- fixed [1](1,"XXXX")
        "LIB_FILE_NAME": [1, "lec25dscc25.v"],  # [0]: 0-not specified [1], 1- fixed [1](1,"XXXX")
        "LOGIC_LEVEL": [1, global_logic_level],  # [0]: 0-not specified [1], 1- fixed [1](1,"XXXX")
        "ASSIGN_REPLACE": [1, 1],  # [0]: 0-not specified [1], 1- fixed [1](1,"XXXX")
        "SCANFF_REPLACE": [1, global_scanff_replace],  # [0]: 0-not specified [1], 1- fixed [1](1,"XXXX")
        "STRLOGIC_REPLACE": [1, 0],
        # [0]: 0-not specified [1], 1- fixed [1](1,"XXXX") [1]: 0-bool value logic,1- string logic (Logic1,Logic0)
        "TRUSTHUB_LIB": [1, 1],  # [0]: 0-not specified [1], 1- fixed [1](1,"XXXX") [1]" 0-saed90nm.v, 1-lec25dscc25.v
        "EMBED_SOURCE": [1, 1],
        # [0]: 0-not specified [1], 1- fixed [1](1,"XXXX") indicate if source need to be embeded ,generally all needed
        "DETECT_SOURCE": [1, 1],
        # 0-not specified, 1- fixed (1,"XXXX") indicate if source need to be used in detection (training, testing)
        "TRAIN_SOURCE": [0, None]
        # 0-not specified, 1- fixed (1,"XXXX") indicate is detect source used in training or test

    }

    # ======check parameters ================
    kgroup_enable = k_parameter[0]
    k_para = k_parameter[1]  # 10
    # if kgroup_enable:
    if k_para < 1:
        raise Exception(
            "Error: k_para (k_parameter[1]) should >1 (1 is general mode, not k-fold) when kgroup_enable is 'True'!")
    #  ====== create "sample_info.csv" list and "sample_yaml.yaml" config file in "sample_info" folder ======
    #  ====== and return sample_info dict ======
    if regen_confyaml_parameter:
        sample_data_collect(sample_dir, yaml_config)
    # ====== read in basic config.yaml (contains basic design netlists files)
    source_config_path = os.path.join(base_dir, 'config.yaml')
    with open(source_config_path, 'r') as f:
        source_config = yaml.load(f, Loader=yaml.FullLoader)
    # ====== write parameters to initial 8 source_config dict
    for file_config in source_config['DATASET_SOURCE']:
        file_config["LOGIC_LEVEL"] = global_logic_level
        file_config["SCANFF_REPLACE"]= global_scanff_replace

    source_config['SAVE_PARA']['DEBUG_MODE'] = debug_mode
    source_config['SAVE_PARA']['LOGIC_LEVEL'] = global_logic_level
    source_config['SAVE_PARA']['YAML_CONFIG'] = yaml_config
    source_config['SAVE_PARA']['REGEN_YAML'] = regen_confyaml_parameter
    source_config['SAVE_PARA']['K_FOLD_ENA'] = kgroup_enable
    source_config['SAVE_PARA']['K_FOLD_NUM'] = k_para
    source_config['SAVE_PARA']['SAMPLE_CONFIG'] = fix_type_new_para
    source_config['SAVE_PARA']['TEST_SET_ENA'] = test_set_enable
    source_config['SAVE_PARA']['PROCS_QUEUE'] = process_query
    source_config['SAVE_PARA']['SAMPLE_DIRS'] = sample_dir
    # ====== Prepare source config, prepare final_data_idx
    if source_config['EXTRA_DATASET_SOURCE']['ENABLE'] == True:
        extra_dataset_path = os.path.join(perl_temp_source_dir,
                                          source_config['EXTRA_DATASET_SOURCE']['FOLDER'],
                                          source_config['EXTRA_DATASET_SOURCE']['CONFIG_FILE'])
        sampleinfo_csv_path = os.path.join(perl_temp_source_dir,
                                           source_config['EXTRA_DATASET_SOURCE']['FOLDER'],
                                           source_config['EXTRA_DATASET_SOURCE']['INFCSV_FILE'])

        with open(extra_dataset_path, 'r') as f:
            extra_dataset_list = yaml.load(f, Loader=yaml.FullLoader)
            print("Extra Dataset List Loaded...")

        list_data_dframe = data_grouping_new(sampleinfo_csv_path, fix_type_new_para)
        list_data_pre, list_data_idxs = data_flatten_new(list_data_dframe)
        final_data_idxs, final_test_idxs = kfold_data_grouping_new(list_data_idxs,kgroup_enable, k_para, kfd_sample_reorder,
                                                                    fix_type_new_para, test_set_enable)


        #####################################
        ################start################
        part_dataset_list = final_data_idxtrans(extra_dataset_list,
                                                perl_temp_source_dir,source_config, 'final_data_idxs.txt')
        dump_file = os.path.join(perl_temp_source_dir, source_config['SAVE_PARA']['FOLDER'], 'part_dataset_list.yaml')
        dump2yaml(part_dataset_list, dump_file)

        part_testset_list = final_data_idxtrans(extra_dataset_list,
                                                perl_temp_source_dir,source_config, 'final_test_idxs.txt')
        dump_file = os.path.join(perl_temp_source_dir, source_config['SAVE_PARA']['FOLDER'], 'part_testset_list.yaml')
        dump2yaml(part_testset_list, dump_file)

        # STAGE: parsing procedure, extract feature_traces based on logic_level in config file
        if 'Parsing' in process_query:

            for file_config in source_config['DATASET_SOURCE']:
                data_parsing(file_config, debug_mode=debug_mode)
            for file_config in part_dataset_list + part_testset_list:
                data_parsing(file_config, debug_mode=debug_mode)
        else:
            # or assign_strlogic_replace_rename() to match the parsed output file
            for file_config in source_config['DATASET_SOURCE']:
                source_file = assign_strlogic_replace_rename(file_config)
                file_config['SOURCE_FILE'] = source_file
            for file_config in part_dataset_list + part_testset_list:
                source_file = assign_strlogic_replace_rename(file_config)
                file_config['SOURCE_FILE'] = source_file
            # scanffreplace_rename
            for file_config in source_config['DATASET_SOURCE']:
                source_file = scanffreplace_rename(file_config)
                file_config['SOURCE_FILE'] = source_file
            for file_config in part_dataset_list + part_testset_list:
                source_file = scanffreplace_rename(file_config)
                file_config['SOURCE_FILE'] = source_file

        # STAGE: Netlist2Vector procedure wordembedding training skip-gram model
        if 'Embedding' in process_query:

            temp_config = copy.deepcopy(source_config)
            temp_config["DATASET_SOURCE"] += part_dataset_list + part_testset_list

            list_dir_ll = list()
            for file_config in temp_config["DATASET_SOURCE"]:
                if file_config['EMBED_SOURCE'] == 1:
                    logic_level = file_config['LOGIC_LEVEL']
                    temp_sub_path = os.path.join(file_config['DESIGN_FOLDER'],
                                                 file_config['SUB_TYPE'],
                                                 file_config['SOURCE_FILE'])
                    file_path = os.path.join(perl_temp_source_dir, temp_sub_path)
                    list_dir_ll.append((file_path, logic_level))

            sentence_count = dataset_shuffle(input_file_list=list_dir_ll,
                                             config=temp_config["WORD2VEC_PARA"],
                                             test_mode=False,
                                             max_cache_num=10)

            save_yaml_config(source_config, 'embedding_config.yaml')
            w2v_net = Word2VecTrainer_Netlist(base_path=perl_temp_source_dir,
                                              source_config=temp_config)
            
            w2v_net.train()

        # STAGE: NNTraining##########################
        # NN-based HT detection procedure,  NN-based classification model, traning and testing.
        if 'NNTraining' in process_query:
            if kgroup_enable is True:  # STEP 1: K-FOLD for parameters adjust
                for idx in range(2,k_para):  # range(k_para) range(3, 5):
                    source_config_copy = copy.deepcopy(source_config)
                    # part_dataset_list = final_grouping(extra_dataset_list, final_data_idxs, idx, k_para=10)
                    part_grouped_list = final_data_grouping(part_dataset_list, idx, k_para)  # group and random the idx
                    source_config_copy['DATASET_SOURCE'] += part_grouped_list
                    print("Data Grouped.")

                    # NN-based HT detection procedure,  NN-based classification model, traning and testing.
                    list_train_dir_ll = list()
                    list_test_dir_ll = list()
                    for file_config in source_config_copy["DATASET_SOURCE"]:
                        if file_config['DETECT_SOURCE'] == 1:
                            if file_config['TRAIN_SOURCE'] == 0:
                                logic_level = file_config['LOGIC_LEVEL']
                                temp_sub_path = os.path.join(file_config['DESIGN_FOLDER'],
                                                             file_config['SUB_TYPE'],
                                                             file_config['SOURCE_FILE'])
                                file_path = os.path.join(perl_temp_source_dir, temp_sub_path)
                                list_test_dir_ll.append((file_path, logic_level))

                            if file_config['TRAIN_SOURCE'] == 1:
                                logic_level = file_config['LOGIC_LEVEL']
                                temp_sub_path = os.path.join(file_config['DESIGN_FOLDER'],
                                                             file_config['SUB_TYPE'],
                                                             file_config['SOURCE_FILE'])
                                file_path = os.path.join(perl_temp_source_dir, temp_sub_path)
                                list_train_dir_ll.append((file_path, logic_level))

                    print("Shuffling and Caching TEST SET...")
                    sentence_test_count = dataset_shuffle(input_file_list=list_test_dir_ll,
                                                          config=source_config_copy["DETECT_PARA"]["TEST_PARA"],
                                                          test_mode=True,
                                                          max_cache_num=10)

                    print("Shuffling and Caching TRAIN SET...")
                    sentence_train_count = dataset_shuffle(input_file_list=list_train_dir_ll,
                                                           config=source_config_copy["DETECT_PARA"]["TRAIN_PARA"],
                                                           test_mode=False,
                                                           max_cache_num=10)
                    
                    print("Balancing TRAIN SET...")
                    dataset_balance(config=source_config_copy["DETECT_PARA"]["TRAIN_PARA"],
                                    ref_class=0,
                                    ubal_classes=[1])

                    save_yaml_config(source_config_copy, 'classifier_config_{:d}'.format(idx) + '.yaml')
                    
                    path = './json_temp_file/word2vec_emb/CNN_model_pretrained.pth'
                    HTnn_net = Classifier_Netlist(group_id=str(idx),
                                                  base_path=perl_temp_source_dir,
                                                  source_config=source_config_copy,
                                                  pretrained=path)   

                    for i, sample_batched in enumerate(HTnn_net.tra_dataloader):
                        print(sample_batched[1], sample_batched[2])

                    # import pickle
                    # with open('save/source_config.pkl', 'wb') as handle:
                    #     pickle.dump(source_config_copy, handle, protocol=pickle.HIGHEST_PROTOCOL)  
                    
                    HTnn_net.train()

