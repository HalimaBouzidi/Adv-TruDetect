######sci-kit######################
import time


######pytorch######################
from netlist_scraper import NetlistDataScraper

# from dnn_model.LSTM_netlist_softmax_save_results import Classifier_Netlist
# from dnn_model.CNN_netlist_softma_save_resluts import Classifier_Netlist

from logger import setup_logger

######python######################
import numpy as np
import sys
import os, shutil
import linecache
import re
import json
import subprocess
import csv
import pandas
import yaml
import copy
from functools import reduce

abspath = os.path.dirname(__file__)
sys.path.append(abspath)
base_dir = abspath # main
netlist_source_dir = os.path.join(base_dir, 'verilog') # utils
perl_temp_source_dir = os.path.join(base_dir, 'json_temp_file') # main and utils

# seif defined print to both file and console
def print_both(file, *args):
    to_print = ' '.join([str(arg) for arg in args])
    print(to_print)
    file.write(to_print)


def get_process_output(cmd, logfile, shell=True, env=None):
    log_file_path = os.path.join(perl_temp_source_dir, 'debug_log', logfile)
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    if env is None:
        env = os.environ
    # with

    process = subprocess.Popen(cmd, shell=shell, env=env,
                               stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE
                               )
    # process.wait()
    try:
        data, err = process.communicate(timeout=900)  # 15 minutes
    except subprocess.TimeoutExpired:
        process.kill()
        data, err = process.communicate()

    with open(log_file_path, 'w') as f:
        # for line in data.decode('utf-8'):
        f.write(data.decode('utf-8'))
        # sys.stdout.write("sub>>> %s" % data.decode('utf-8'))

    if process.returncode == 0:
        return data.decode('utf-8')
    else:
        print("sub>(get_process_output):Error:", err)
    return ""


def perl_run(sub_scrit_name, sub_args, sub_log_name):
    cmd = "perl ./" + sub_scrit_name + " " + sub_args
    log_info = get_process_output(cmd, sub_log_name)
    #for line in log_info.splitlines():
        # print(log_info)
    #    print('sub>(perl_run):%s>>: %s' % (sub_log_name, line))

def json_file_load(base_path, design_path, file_name):
    file = os.path.join(base_path, design_path, file_name)
    with open(file, 'r') as f:
        json_var = json.load(f)
    return json_var


def file_cache_in(base_path, design_path, file_name):
    file = os.path.join(base_path, design_path, file_name)
    cache_data = linecache.getlines(file)
    return cache_data


def label_info_reading(thlib_define, design_folder, sub_type, source_file, perl_temp_source_dir):
    #print("sub>(label_info_reading): start....")
    design_path = os.path.join(netlist_source_dir, design_folder, sub_type)
    perl_temp_path = os.path.join(perl_temp_source_dir, design_folder, sub_type, source_file)
    file_out = os.path.join(perl_temp_path, 'htlabel_cell.txt')

    if thlib_define == 0:
        # print("CSIT Hardware Trojan Libs.")
        if 'trojan_inserted' in source_file:
            trojan_file_list = is_filename_contain_word(design_path, 'trojaninfo')
            if len(trojan_file_list) < 1:
                raise Exception("Trojaninfo file not find!")
            else:
                trojancellinfo_list = list()
                for file in trojan_file_list:
                    rows = list()
                    with open(file, 'r') as f_input:
                        begin = False
                        for line in f_input:
                            if not begin:
                                if line.startswith("Netlist:"):
                                    begin = True
                            # elif line.startswith("end of file"):
                            #     break
                            else:
                                rows.append(line)
                    trojancellinfo_list.append(rows.copy())

                with open(file_out, 'w') as f_ouput:
                    for trojan in trojancellinfo_list:
                        for line in trojan:
                            line = line.strip()
                            line = line.split()
                            # print(line)
                            if (len(line) > 1):
                                ref_name = line[0]
                                cell_name = line[1]
                                f_ouput.write(cell_name + '\n')
                        f_ouput.write('=file_split=' + '\n')
        else:
            with open(file_out, 'w') as f_ouput:
                f_ouput.write(' ' + '\n')
        #print("sub>(label_info_reading):Label Info Loaded.")
    else:
        name = source_file.split("_")
        if (name[1][0] == "T"):
            # print("TrustHub Hardware Trojan Libs.")
            trojan_file_list = is_filename_contain_word(design_path, 'log.txt')
            if (len(trojan_file_list) < 1):
                raise Exception("Trojaninfo file not find!")
            else:
                trojancellinfo_list = list()
                for file in trojan_file_list:
                    rows = list()
                    with open(file, 'r') as f_input:
                        begin = False
                        for line in f_input:
                            if not begin:
                                if line.startswith("TROJAN BODY:"):
                                    begin = True
                            # elif line.startswith("end of file"):
                            #     break
                            else:
                                rows.append(line)
                    trojancellinfo_list.append(rows.copy())

                with open(file_out, 'w') as f_ouput:
                    for trojan in trojancellinfo_list:
                        for line in trojan:
                            if (line.startswith('---')):
                                f_ouput.write('---' + '\n')
                            else:
                                line = line.strip()
                                line = line.split()
                                # print(line)
                                if (len(line) > 1):
                                    ref_name = line[0]
                                    cell_name = line[1]
                                    f_ouput.write(cell_name + '\n')
                        f_ouput.write('=file_split=' + '\n')
        else:
            with open(file_out, 'w') as f_ouput:
                f_ouput.write(' ' + '\n')
        #print("sub>(label_info_reading):Label Info Loaded.")


def is_filename_contain_word(path, query_word):
    filename_list = list()
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path):
            if query_word in file:
                filename_list.append(file_path)
    #print("sub>(is_filename_contain_word):Finish searching.")
    return filename_list


def assignreplace(trusthub_lib, design_folder, sub_type, source_file):
    input_file = os.path.join(netlist_source_dir, design_folder, sub_type, source_file)
    filelist = open(input_file + '.v', 'r')

    #print("sub>(assignreplace): noassign_strlogic.v process...")
    # change assign XXX=1'b0 to nb1s2 AssRep%d( .DIN(Logic0), .Q(XXX) );
    # change .XXX(1'b0) to .XXX(Logic0)
    # file_output = open(input_file + '_reassign.v', 'w')
    file_output = open(input_file + '_noassign_strlogic.v', 'w')
    number = 0
    for line in filelist:
        temp_str = re.sub(r'\s+', ' ', line)
        temp_str = temp_str.strip()
        # print(temp_str)
        temp_str = temp_str.split(' ')
        # print(temp_str)
        if (temp_str[0] == 'assign'):

            #print("sub>(assignreplace): old ", end=" ")
            #print(*temp_str, sep=' ')

            source = temp_str[3][0:-1]
            if (source == "1'b0"):   source = "Logic0"
            if (source == "1'b1"):   source = "Logic1"
            drain = temp_str[1]
            if (trusthub_lib == 1):
                temp_str = ("nb1s2 AssRep%d( .DIN(%s), .Q(%s) );\n" % (number, source, drain))
            else:
                temp_str = ("AOBUFX2 AssRep%d( .INP(%s), .Z(%s) );\n" % (number, source, drain))

            #print("sub>(assignreplace): new %s" % temp_str)

            file_output.write(temp_str)
            number = number + 1

        else:
            temp_str = re.sub(r"1'b0", 'Logic0', line)
            temp_str = re.sub(r"1'b1", 'Logic1', temp_str)
            if (temp_str != line):
                print("sub>(assignreplace): old %s" % line)
                print("sub>(assignreplace): new %s" % temp_str)
            file_output.write(temp_str)
    file_output.close()

    # change assign XXX=1'b0 to nb1s2 AssRep%d( .DIN(1'b0), .Q(XXX) );
    # notchange 1'b0 to Logic0
    #print("sub>(assignreplace): noassign_vallogic.v process...")
    # file_output = open(input_file + '_noassign.v','w')
    file_output = open(input_file + '_noassign_vallogic.v', 'w')
    # 1'b1 and 1'b0 will not be replaced to Logic1 and Logic0, maybe useful under some software.
    number = 0
    filelist.seek(0)
    for line in filelist:

        temp_str = re.sub(r'\s+', ' ', line)
        temp_str = temp_str.strip()
        # print(temp_str)
        temp_str = temp_str.split(' ')
        # print(temp_str)
        if (temp_str[0] == 'assign'):

            #print("sub>(assignreplace): old ", end=" ")
            #print(*temp_str, sep=' ')

            source = temp_str[3][0:-1]
            drain = temp_str[1]
            if (trusthub_lib == 1):
                temp_str = ("nb1s2 AssRep%d( .DIN(%s), .Q(%s) );\n" % (number, source, drain))
            else:
                temp_str = ("AOBUFX2 AssRep%d( .INP(%s), .Z(%s) );\n" % (number, source, drain))

            #print("sub>(assignreplace): new %s" % temp_str)
            file_output.write(temp_str)
            number = number + 1

        else:
            temp_str = line
            # if (temp_str != line):
            #     print("sub assignreplace: old %s" % line)
            #     print("sub assignreplace: new %s" % temp_str)
            file_output.write(temp_str)
    file_output.close()

    filelist.close()
    #print('sub>(assignreplace):Assign Replacement Completed!\n')


def scanffreplace(trusthub_lib, design_folder, sub_type, source_file):
    input_file = os.path.join(netlist_source_dir, design_folder, sub_type, source_file)
    file_in = open(input_file + '.v', 'r')
    content = file_in.readlines()
    file_in.close()
    file_output = open(input_file + '_noscanff.v', 'w')

    #print("sub>(scanffreplace): _noscanff.v process...")

    # change assign XXX=1'b0 to nb1s2 AssRep%d( .DIN(Logic0), .Q(XXX) );
    # change .XXX(1'b0) to .XXX(Logic0)
    # file_output = open(input_file + '_reassign.v', 'w')

    number = 0
    line_mod = 0
    temp_content=list()
    for line in content:
        number = number + 1
        temp_content.append(line)

        temp_str = re.sub(r'\s+', ' ', line)
        temp_str = temp_str.strip()
        temp_str_list = temp_str.split(' ')
        # print(temp_str)
        if temp_str_list[0].startswith("sdff"):
            # pattern = re.compile(r'^sdffs\d+$')
            # match_obj = pattern.search(line)

            if re.match(r'^sdffs\d+$', temp_str_list[0]) is None:
                raise Exception("scanffreplace: un-prepard scandff detected!")

            line_mod = 1


        if line_mod == 1 and ";" in temp_str:
            # if(len(temp_content)==1):
            #     print("test: 1 line !")
            if not temp_str.endswith(");"):
                raise Exception("sub>(scanffreplace): line not end with ');'!")
            # map(str.strip, temp_content)
            # [x.strip() for x in temp_content]
            temp_complete = "".join(temp_content)
            temp_content.clear()
            temp_complete = re.sub(r'\s+', ' ', temp_complete)
            key2_list = ["DIN", "CLK", "Q", "QN"]

            ### CELL SCALE ###
            scale_size = None
            pattern = re.compile(r"^.+s(\d+)\s+")
            match_obj = pattern.search(temp_complete)
            if match_obj is not None:
                scale_size = match_obj.group(1)
            else:
                raise Exception("sub>(scanffreplace): can not get the size value of cell!")

            ### COMPONENT NAME ###
            comp_name = temp_complete.strip().split(' ')[1]

            ### PIN NET NAME ####
            key1_list = ["DIN", "SDIN", "SSEL", "CLK", "Q", "QN"]
            value_dict = dict()
            for key in key1_list:
                pattern = re.compile(r'\.\s*{}\s*\(\s*(\w+)\s*\)'.format(key))
                match_obj = pattern.search(temp_complete)
                if match_obj is not None:
                    value_dict[key] = match_obj.group(1)

            # template_dffcell = "dffsX CNAME ( .DIN(XX), .CLK(XX), .Q(XX), QN(XX));"
            pin_string = ''
            for idx, key in enumerate(key2_list):
                if key in value_dict:
                    if idx == 0:
                        pin_string += " .{}({})".format(key, value_dict[key])
                    else:
                        pin_string += ", .{}({})".format(key,value_dict[key])


            replaced_cell = "dffs{X} {CNAME} ({PIN_NET} );\n".format(
                X=scale_size,
                CNAME=comp_name,
                PIN_NET=pin_string)

            temp_content.append(replaced_cell)
            line_mod = 0

        if not line_mod:
            for line in temp_content:
                file_output.write(line)
            temp_content.clear()

    file_output.close()


    #print('sub>(scanffreplace):scanff Replacement Completed!\n')



def sample_data_collect(sample_dir, yaml_config):
    # sample_dir = ['thub_c2670', 'thub_c3540', 'thub_c5315', 'thub_c6288', 'thub_s1423', 'thub_s13207', 'thub_s15850',
    #               'thub_s35932']

    # list header information
    name_path = ["sdir", "spath"]
    key_dict = {"Number of Trojans:": 2, "Type:": 3, "Effect:": 4, "Activation Condition:": 5}
    name_headers = name_path + sorted(key_dict, key=key_dict.get)

    samples_info_list = list()
    for sdir in sample_dir:
        dpath = os.path.join(netlist_source_dir, sdir)
        for spath in sorted(os.listdir(dpath)):
            file_path = os.path.join(dpath, spath)
            if not os.path.isfile(file_path):
                filename_list = is_filename_contain_word(file_path, 'log.txt')
                if len(filename_list) < 1:
                    raise Exception("file not find!")
                else:
                    for file in filename_list:
                        rows = list()
                        with open(file, 'r') as f_input:
                            begin = False
                            for line in f_input:
                                if not begin:
                                    if line.startswith("TROJAN STATS:"):
                                        begin = True
                                elif line.startswith("***"):
                                    break
                                else:
                                    rows.append(line)

                        sample_info = [""] * (len(key_dict) + len(name_path))
                        curr_column = 0
                        for line in rows:
                            # print(line)
                            for key, value in key_dict.items():
                                if line.startswith(key):
                                    curr_column = value

                            sample_info[curr_column] += line.replace('\n', r' ')

                        sample_info[0] = sdir
                        sample_info[1] = spath

                        # Number of Trojans modify
                        temp = sample_info[2].split(":")
                        sample_info[2] = temp[1].strip()
                        # Type  modify
                        temp = sample_info[3].replace("Type:", "")
                        temp = temp.strip()
                        sample_info[3] = re.sub(r'\s\s+', ' ', temp)
                        # Effect  modify
                        temp = sample_info[4].split("...")
                        temp = temp[1].strip()
                        sample_info[4] = re.sub(r'\s\s+', ' ', temp)
                        # Activation Condition  modify
                        temp = sample_info[5].split("...")
                        temp = temp[1].strip()
                        sample_info[5] = re.sub(r'\s\s+', ' ', temp)

                        samples_info_list.append(sample_info)

    # save samples_info_list as csv list file
    df_samples_info_list = pandas.DataFrame(data=samples_info_list, columns=name_headers)
    file = os.path.join(perl_temp_source_dir, 'sample_info', 'samples_info.csv')
    os.makedirs(os.path.dirname(file), exist_ok=True)
    df_samples_info_list.to_csv(file)

    # generate sample yaml
    # return sample_info as dict
    sample_yaml_list = list()
    # samples_info_dict = dict()
    for line in samples_info_list:
        yaml_dict = dict()

        yaml_dict['DESIGN_FOLDER'] = line[0] if yaml_config['DESIGN_FOLDER'][0] == 0 \
            else yaml_config['DESIGN_FOLDER'][1]
        yaml_dict['SUB_TYPE'] = line[1] if yaml_config['SUB_TYPE'][0] == 0 \
            else yaml_config['SUB_TYPE'][1]  # comb_func, NULL
        yaml_dict['SOURCE_FILE'] = line[1] if yaml_config['SOURCE_FILE'][0] == 0 \
            else yaml_config['SOURCE_FILE'][1]
        yaml_dict['LIB_FOLDER'] = "undefined" if yaml_config['LIB_FOLDER'][0] == 0 \
            else yaml_config['LIB_FOLDER'][1]
        yaml_dict['LIB_FILE_NAME'] = "undefined" if yaml_config['LIB_FILE_NAME'][0] == 0 \
            else yaml_config['LIB_FILE_NAME'][1]
        yaml_dict['LOGIC_LEVEL'] = "undefined" if yaml_config['LOGIC_LEVEL'][0] == 0 \
            else yaml_config['LOGIC_LEVEL'][1]
        yaml_dict['ASSIGN_REPLACE'] = "undefined" if yaml_config['ASSIGN_REPLACE'][0] == 0 \
            else yaml_config['ASSIGN_REPLACE'][1]
        yaml_dict['SCANFF_REPLACE'] = "undefined" if yaml_config['SCANFF_REPLACE'][0] == 0 \
            else yaml_config['SCANFF_REPLACE'][1]
        yaml_dict['STRLOGIC_REPLACE'] = "undefined" if yaml_config['STRLOGIC_REPLACE'][0] == 0 \
            else yaml_config['STRLOGIC_REPLACE'][1]
        yaml_dict['TRUSTHUB_LIB'] = "undefined" if yaml_config['TRUSTHUB_LIB'][0] == 0 \
            else yaml_config['TRUSTHUB_LIB'][1]
        yaml_dict['EMBED_SOURCE'] = "undefined" if yaml_config['EMBED_SOURCE'][0] == 0 \
            else yaml_config['EMBED_SOURCE'][1]
        yaml_dict['DETECT_SOURCE'] = "undefined" if yaml_config['DETECT_SOURCE'][0] == 0 \
            else yaml_config['DETECT_SOURCE'][1]
        yaml_dict['TRAIN_SOURCE'] = "undefined" if yaml_config['TRAIN_SOURCE'][0] == 0 \
            else yaml_config['TRAIN_SOURCE'][1]

        sample_yaml_list.append(yaml_dict)

    file = os.path.join(perl_temp_source_dir, 'sample_info', 'samples_yaml.yaml')
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, 'w') as f:
        yaml.dump(sample_yaml_list, f, sort_keys=False)

    return df_samples_info_list


def data_grouping_new(info_csv, type_paras):
    # 1. random pick a sub-group datasets
    # 2. k-fold cross validation(divide into k group)
    # 3. k-1 for training, 1 for test

    # key1 combinational,
    # key2 counter (num of states) key2sub1=states
    # key3 FSM     (num of triggers) key3sub1=triggers
    # key4 (key1&key2)comb&counter
    # key5 (key1&key3)comb&FSM
    #print("Data Grouping V2")
    # param_types = type_paras[0]
    param_items = type_paras[0]
    param_requests = type_paras[1:]
    keysub_dict = {'FSM': 'triggers',
                   'counter': 'states'}


    df_sample_info = pandas.read_csv(info_csv, header=0, index_col=0)
    matched_design_df_list = list()
    for req in param_requests:
        #print(req)
        matched_design_table = pandas.Series([True] * len(df_sample_info))
        for idx in range(len(req)):
            # print(param_items[idx],":",req[idx])

            if param_items[idx] == 'sdir':
                temp_table = df_sample_info['sdir'].str.contains(req[idx])
                matched_design_table = matched_design_table & temp_table

            elif param_items[idx] == 'Number of Trojans:':
                temp_table = df_sample_info['Number of Trojans:'] == req[idx]
                matched_design_table = matched_design_table & temp_table

            elif param_items[idx] == 'Type:':
                for ele in req[idx]:
                    temp_table = df_sample_info['Type:'].str.contains(ele)
                    matched_design_table = matched_design_table & temp_table

            elif param_items[idx] in ['TRAIN_NUM','TEST_NUM']:
                # print("Do Not Need Operation:", param_items[idx])
                pass

            else:
                print("Undefined Operation!:", param_items[idx])


            # matched_design_df = df_sample_info[matched_design_table]

        matched_design_df = df_sample_info[matched_design_table]
        matched_design_df_list.append(matched_design_df)

    return matched_design_df_list


def data_flatten_new(list_data_df):
    #print('Data Flattening...')

    list_data_pre = list()
    list_data_idx = list()

    for data_df in list_data_df:
        data_pre = data_df.values.tolist()
        data_idx = data_df.index.tolist()
        list_data_pre.append(data_pre)
        list_data_idx.append(data_idx)

    return list_data_pre, list_data_idx


def kfold_data_grouping_new(list_data_idxs, k_enable, kpara, kfd_reorder, fix_type_new_para, test_set_en):
    #print('K={}-fold Grouping..'.format(kpara))
    param_items = fix_type_new_para[0]
    param_requests = fix_type_new_para[1:]
    #print('Samples Existing Checking..')
    for data_idxs, para_req in zip(list_data_idxs, param_requests):
        if len(data_idxs) == 0:
            raise Exception("Error: The Defined type in 'fix_type_new_para'not found in sample_info.csv!", para_req)

    k_dataset_size = 0
    list_train_size=list()
    list_test_size=list()
    for req in param_requests:
        #print(req)
        for idx in range(len(req)):
            # print(param_items[idx],":",req[idx])
            if param_items[idx] == 'TRAIN_NUM':
                #print("TRAIN_NUM:", req[idx])
                k_dataset_size += req[idx]
                list_train_size.append(req[idx])
            elif param_items[idx] == 'TEST_NUM':
                #print("TEST_NUM:", req[idx])
                list_test_size.append(req[idx])

    if k_enable and (k_dataset_size % kpara != 0):
        raise Exception("Error: can not get a int size for each KFD group!")
    else:
        k_size_each_group = k_dataset_size // kpara
        #print("Total number of sample: ",k_dataset_size)

    k_dataset = list()
    testset = list()

    for idx in range(len(param_requests)):
        data_idxs = list_data_idxs[idx]
        train_size = list_train_size[idx]
        test_size = list_test_size[idx]
        shuffle_list = data_idxs
        np.random.shuffle(shuffle_list)

        if train_size > len(shuffle_list):
            raise Exception("Error: can not get enough train data:",param_requests[idx])
        else:
            s_dataset_idxs = shuffle_list[:train_size]
            k_dataset += s_dataset_idxs

        if test_set_en:
            if test_size > (len(shuffle_list)-train_size):
                raise Exception("Error: can not get enough test data:", param_requests[idx])
            else:
                s_testset_idxs = shuffle_list[train_size: train_size + test_size]
                testset += s_testset_idxs

    if k_enable and kfd_reorder:
        temp_list = list()
        list_train_idx = [sum(list_train_size[:idx]) for idx in range(len(list_train_size))]
        for idx, num in zip(list_train_idx, list_train_size):
            temp_list.append(k_dataset[idx:idx + num])
        k_dataset.clear()
        for idx_k in range(kpara):
            k_dataset += [temp_list[idx_temp][idx_k] for idx_temp in range(len(temp_list))]

    else:
        np.random.shuffle(k_dataset)


    data_idxs_file = os.path.join(perl_temp_source_dir, 'word2vec_emb', 'final_data_idxs.txt')
    with open(data_idxs_file, 'w') as f_w:
        for item in k_dataset:
            f_w.write("%s\n" % item)

    # if test_set_en:
    data_idxs_file = os.path.join(perl_temp_source_dir, 'word2vec_emb', 'final_test_idxs.txt')
    with open(data_idxs_file, 'w') as f_w:
        for item in testset:
            f_w.write("%s\n" % item)

    return k_dataset, testset

def final_data_idxtrans(dataset_list, temp_dir, config, final_data_idxs_file):
    part_dataset_list = list()
    dump_file = os.path.join(temp_dir, config['SAVE_PARA']['FOLDER'], final_data_idxs_file)
    # final_data_idxs=list()
    with open(dump_file, mode='r') as f_rd:
        # final_data_idxs = f_rd.readlines()
        final_data_idxs = [int(line.strip()) for line in f_rd]

    for idx in final_data_idxs:
        temp_config = dataset_list[idx]
        part_dataset_list.append(temp_config)

    return part_dataset_list


def final_data_grouping(dataset_list, cur_idx, k_para):
    part_dataset_list = copy.deepcopy(dataset_list)
    if k_para <= 0:  # for test_set
        #print("Generate test-set source config...")
        for temp_config in part_dataset_list:
            temp_config['TRAIN_SOURCE'] = 0
    else:
        #print("Generate train-set source config...")
        if k_para == 1:
            for temp_config in part_dataset_list:
                temp_config['TRAIN_SOURCE'] = 1
        else:
            num_ingroup = len(dataset_list) // k_para
            for temp_config in part_dataset_list[cur_idx * num_ingroup:(cur_idx + 1) * num_ingroup]:
                temp_config['TRAIN_SOURCE'] = 0

            for temp_config in part_dataset_list[0:cur_idx * num_ingroup] + part_dataset_list[
                                                                            (cur_idx + 1) * num_ingroup:]:
                temp_config['TRAIN_SOURCE'] = 1

    return part_dataset_list


def data_parsing(file_config, debug_mode=1):
    # for file_config in extra_dataset_list:
    design_folder = file_config['DESIGN_FOLDER']
    sub_type = file_config['SUB_TYPE']  # comb_func, NULL
    source_file = file_config['SOURCE_FILE']
    lib_folder = file_config['LIB_FOLDER']
    lib_file_name = file_config['LIB_FILE_NAME']
    logic_level = file_config['LOGIC_LEVEL']
    assign_replace = file_config['ASSIGN_REPLACE']
    scanff_replace = file_config['SCANFF_REPLACE']
    strlogic_replace = file_config['STRLOGIC_REPLACE']
    trusthub_lib = file_config['TRUSTHUB_LIB']
    embed_source = file_config['EMBED_SOURCE']
    detect_source = file_config['DETECT_SOURCE']
    train_source = file_config['TRAIN_SOURCE']

    if trusthub_lib == 0:
        print("Undefined Libs.")
    else:
        print("TrustHub Hardware Trojan Auto_Gen_FRM Libs.")

    print("####################")
    print("Netlist Replace.")
    print("####################")

    # assign_replace
    # strlogic_replace
    # source_file = assign_strlogic_replace_rename(file_config)
    if assign_replace == 1:
        if 'noassign' in source_file:
            raise Exception("Error: Replace a already no-assign design!")
        else:
            # print("Assign Replacement sub start... ")
            assignreplace(trusthub_lib, design_folder, sub_type, source_file)

            if strlogic_replace == 1:
                #print("Source_file %s renamed to %s!" % (source_file, source_file + '_noassign_strlogic'))
                source_file = source_file + '_noassign_strlogic'
            else:
                #print("Source_file %s renamed to %s!" % (source_file, source_file + '_noassign_vallogic'))
                source_file = source_file + '_noassign_vallogic'

    #scandff replace ->normal dff
    if scanff_replace ==1:
        if 'noscanff' in source_file:
            raise Exception("Error: Replace a already noscanff design!")
        else:
            # print("Assign Replacement sub start... ")
            scanffreplace(trusthub_lib, design_folder, sub_type, source_file)
            #print("Source_file %s renamed to %s!" % (source_file, source_file + '_noscanff'))
            source_file = source_file + '_noscanff'


    file_config['SOURCE_FILE'] = source_file

    # my $base_dir =$Bin;
    # my $sub_dir = 'json_temp_file';
    # my $sub_path = $base_dir.'/'.$sub_dir;
    # my $sub_design_folder_path = $sub_path.'/'.$design_folder_dir;
    # my $sub_subtype_folder_path = $sub_design_folder_path.'/'.$type_folder_dir;
    # my $sub_source_file_folder_path = $sub_subtype_folder_path.'/'.$source_file_name;
    # my @ created = make_path($sub_source_file_folder_path);
    create_path = os.path.join(perl_temp_source_dir,design_folder,sub_type,source_file)
    os.makedirs(create_path, exist_ok=True)

    print("####################")
    print("Netlist Transition Probability & Netlist Hash data.")
    print("####################")

    scrit_name = "TransProb_script.pl"

    args = "--design_folder_path={} ".format(design_folder) + \
           "--subtype_folder_path={} ".format(sub_type) + \
           "--source_name={} ".format(source_file) + \
           "--lib_folder_path={} ".format(lib_folder) + \
           "--lib_file_name={}".format(lib_file_name)
    log_name = "_".join([design_folder, sub_type, 'log1.txt'])

    if debug_mode == 0:
        perl_run(scrit_name, args, log_name)

    print("####################")
    print("Feature Stage:")
    print("logic distance")
    print("####################")

    scrit_name = "CellsBlock_script.pl"

    args = "--design_folder_path={} ".format(design_folder) + \
           "--subtype_folder_path={} ".format(sub_type) + \
           "--source_name={} ".format(source_file) + \
           "--ref_Gate_HASH_ORI={} ".format('ref_Gate_HASH_ORI.json') + \
           "--ref_Netlink_HASH={} ".format('ref_Netlink_HASH.json') + \
           "--ref_Design_input_Port={} ".format('ref_Design_input_Port.json') + \
           "--ref_Design_output_Port={} ".format('ref_Design_output_Port.json') + \
           "--logic_level={:d}".format(logic_level)

    # log_name = "log2.txt"
    log_name = "_".join([design_folder, sub_type, 'log2.txt'])
    if debug_mode == 0:
        perl_run(scrit_name, args, log_name)

    # temp_sub_path = os.path.join(design_folder, sub_type, source_file)
    # ref_Gate_HASH_ORI      = json_file_load(perl_temp_source_dir, sub_path, 'ref_Gate_HASH_ORI.json')
    # ref_Netlink_HASH       = json_file_load(perl_temp_source_dir, sub_path, 'ref_Netlink_HASH.json')
    # ref_Design_input_Port  = json_file_load(perl_temp_source_dir, sub_path, 'ref_Design_input_Port.json')
    # ref_Design_output_Port = json_file_load(perl_temp_source_dir, sub_path, 'ref_Design_output_Port.json')
    # ref_hash_blocks        = json_file_load(perl_temp_source_dir, sub_path, 'ref_hash_blocks.json')

    print("####################")
    print("Word2vec Stage:")
    print("Read in")
    print("####################")
    # netlist_data = NetlistDataScraper(perl_temp_source_dir, temp_sub_path)
    # data = DataReader_Netlist(perl_temp_source_dir,temp_sub_path,100)
    label_info_reading(trusthub_lib, design_folder, sub_type, source_file, perl_temp_source_dir)
    temp_sub_path = os.path.join(design_folder, sub_type, source_file)
    if debug_mode == 0:
        netlist_data = NetlistDataScraper(perl_temp_source_dir, temp_sub_path, source_file, logic_level)

    # comp_cell
    # L3            L2         L1        CELL     R1        R2       R3
    # LPGP3         LPGP2      LPGP1     LPGP0    RPGP0     RPGP1    RPGP2
    # [iP0,iP1]     [iP0,iP1]  ...        ...     ...         ...     ...
    # [oP0,oP1]     [oP0,oP1]  ...        ...     ...         ...     ...
    print("Data Parsed.")


def assign_strlogic_replace_rename(f_config):
    # for file_config in extra_dataset_list:

    # design_folder = f_config['DESIGN_FOLDER']
    # sub_type = f_config['SUB_TYPE']  # comb_func, NULL
    source_file = f_config['SOURCE_FILE']
    assign_replace = f_config['ASSIGN_REPLACE']
    strlogic_replace = f_config['STRLOGIC_REPLACE']
    # trusthub_lib = f_config['TRUSTHUB_LIB']

    # assign_replace
    # strlogic_replace
    if assign_replace == 1:
        if 'noassign' in source_file:
            raise Exception("Error: Replace a already no-assign design!")
        else:
            # print("Assign Replacement sub start... ")
            # assignreplace(trusthub_lib, design_folder, sub_type, source_file)

            if strlogic_replace == 1:
                print("Source_file %s renamed to %s!" % (source_file, source_file + '_noassign_strlogic'))
                source_file = source_file + '_noassign_strlogic'
            else:
                print("Source_file %s renamed to %s!" % (source_file, source_file + '_noassign_vallogic'))
                source_file = source_file + '_noassign_vallogic'
    return source_file


def scanffreplace_rename(f_config):

    # for file_config in extra_dataset_list:
    source_file = f_config['SOURCE_FILE']
    scanff_replace = f_config['SCANFF_REPLACE']
    # trusthub_lib = f_config['TRUSTHUB_LIB']

    #scandff replace ->normal dff
    if scanff_replace ==1:
        if 'noscanff' in source_file:
            raise Exception("Error: Replace a already noscanff design!")
        else:
            # print("Assign Replacement sub start... ")
            # scanffreplace(trusthub_lib, design_folder, sub_type, source_file)
            #print("Source_file %s renamed to %s!" % (source_file, source_file + '_noscanff'))
            source_file = source_file + '_noscanff'

    return source_file

def dataset_balance(config, ref_class=1, ubal_classes=None, chunk_pieces=10):
    # temp_dict = {1: 23727, 0: 12962118}
    # config["SET_LABELS_CNT"] = temp_dict
    start = time.time()
    #print("Balancing other classes refer to class {}...".format(ref_class))
    dataset_in_file = os.path.join(perl_temp_source_dir,
                                   config["FOLDER"],
                                   config["SET_FILE"])
    file_name = config["SET_FILE"].split('.')
    out_file_name = file_name[0] + '_balanced.' + file_name[1]

    dataset_out_file = os.path.join(perl_temp_source_dir,
                                    config["FOLDER"],
                                    out_file_name)

    # class_sentence_count=dict()
    # neg_sentence_count = 0
    # pos_sentence_count = 0
    label_num_dict = config["SET_LABELS_CNT"]

    for ubal_classe in ubal_classes:
        if label_num_dict[ubal_classe] > label_num_dict[ref_class]:
            #print("Down scale class {}...".format(ubal_classe))
            thr = label_num_dict[ref_class] / label_num_dict[ubal_classe]
            with open(dataset_out_file, mode='w') as f_wr:
                with open(dataset_in_file, mode='r') as f_rd:
                    for ori_line in f_rd:
                        line = ori_line.strip()
                        line = line.split(',')
                        ll = int(line[0])
                        pgp_length = 2 * (ll - 1) + 1
                        label = 0 if line[1 + pgp_length] == '0' else 1

                        # label_dict[label] = label_dict.get(label, 0) + value
                        if label == ubal_classe:
                            if np.random.random() < thr:
                                f_wr.write(ori_line)
                                # neg_sentence_count += 1
                            else:
                                label_num_dict[label] = label_num_dict[label] - 1
                        else:
                            f_wr.write(ori_line)
                            # pos_sentence_count += 1
        else:  # number of ref_class > ubal_classe
            #print("Up scale class {}...".format(ubal_classe))
            thr = label_num_dict[ref_class] // label_num_dict[ubal_classe]
            with open(dataset_out_file, mode='w') as f_wr:
                with open(dataset_in_file, mode='r') as f_rd:
                    for ori_line in f_rd:
                        line = ori_line.strip()
                        line = line.split(',')
                        ll = int(line[0])
                        pgp_length = 2 * (ll - 1) + 1
                        label = 0 if line[1 + pgp_length] == '0' else 1
                        if label == ubal_classe:
                            for _ in range(thr):
                                f_wr.write(ori_line)
                                # pos_sentence_count += 1
                            label_num_dict[label] = label_num_dict[label] + (thr - 1)
                        else:
                            f_wr.write(ori_line)
                            # neg_sentence_count += 1


    #print("Re-shuffling...")
    out_file_name = file_name[0] + '_balanced_reshuffle.' + file_name[1]
    dataset_shuffled_out_file = os.path.join(perl_temp_source_dir,
                                             config["FOLDER"],
                                             out_file_name)
    try:
        os.remove(dataset_shuffled_out_file)
    except OSError:
        pass

    all_traces_balanced_num = sum(label_num_dict.values())
    chunk_size = all_traces_balanced_num // chunk_pieces
    if chunk_size <= 0:
        raise Exception("Chunk size is 0! all_traces_balanced_num should > chunk_pieces")
    local_trace_count = 0
    with open(dataset_out_file, mode='r') as f_rd:
        for chunk_idx in range(chunk_pieces - 1):
            #print('Caching...({}/{})'.format(chunk_idx + 1, chunk_pieces - 1))
            # read in lines to RAM
            cached_lines = list()
            line_num = 0
            while line_num < chunk_size:
                cached_lines.append(f_rd.readline())
                line_num += 1
                local_trace_count += 1
            # shuffle lines in RAM, accumulate sentence_count
            #print("Shuffling...")
            np.random.shuffle(cached_lines)
            # write out shuffled lines in RAM
            with open(dataset_shuffled_out_file, mode='a') as f_wr:
                #print('Writing...')
                for line in cached_lines:
                    f_wr.write(line)
        else:
            #print("Caching...Last")
            # read in lines to RAM
            cached_lines = list()
            while True:
                line = f_rd.readline()
                if not line:
                    break
                cached_lines.append(line)
                local_trace_count += 1
            # shuffle lines in RAM, accumulate sentence_count
            #print("Shuffling...")
            np.random.shuffle(cached_lines)
            # write out shuffled lines in RAM
            with open(dataset_shuffled_out_file, mode='a') as f_wr:
                #print('Writing...')
                for line in cached_lines:
                    f_wr.write(line)

    if all_traces_balanced_num == local_trace_count:
        print("All traces recorded!...")

    config["SET_FILE"] = out_file_name
    config["SET_LENG"] = all_traces_balanced_num
    config["SET_LABELS_CNT"] = label_num_dict
    #print("Have Cached {} lines...".format(local_trace_count))
    #print(label_num_dict)
    end = time.time()
    #print('Task runs {:.2f} seconds.'.format((end - start), ))
    return 1

def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copy2(s, d)


def dataset_shuffle(input_file_list, config, test_mode=True, max_cache_num=15):
    dataset_out_file = os.path.join(perl_temp_source_dir,
                                    config["FOLDER"],
                                    config["SET_FILE"])
    try:
        os.remove(dataset_out_file)
    except OSError:
        pass

    children_list_dir_ll = [input_file_list[i:i + max_cache_num] for i in
                            range(0, len(input_file_list), max_cache_num)]

    sentence_count = 0
    sub_labels_dict = dict()
    label_dict = dict()
    ll_dict = dict()
    sub_index = 0

    for sublist in children_list_dir_ll:
        cached_lines = list()
        # cache sublist files as lines in RAM
        #print('Caching...({}/{})'.format(sub_index + 1, len(children_list_dir_ll)))
        start = time.time()
        for (dir_path, ll) in sublist:

            # ll
            ll_dict[ll] = ll_dict.get(ll, 0) + 1

            # labels
            file_name = os.path.join(dir_path, 'labels_dict.yaml')
            with open(file_name, 'r') as f_read:
                sub_labels_dict = yaml.load(f_read, Loader=yaml.FullLoader)
            for ori_label, value in sub_labels_dict.items():
                label = 0 if ori_label == '0' else 1  # let all other ht label types to be 1
                label_dict[label] = label_dict.get(label, 0) + value

            # traces
            file_name = os.path.join(dir_path, 'df_keywords.csv')

            with open(file_name, mode='r') as f_read:
                # cached_lines += f_read.readlines()
                for line in f_read:
                    # [ll][x, x, ....x, x][label][comp]
                    cached_lines.append(str(ll) + ',' + line)

                    line = line.strip()
                    line = line.split(',')
                    pgp_length = 2 * (ll - 1) + 1
                    # 2*(ll-1)+1 is pgp length, 2 is the lable and comp.
                    # [x,x,....x,x][label][comp]
                    #  pgp_length,   1,     1
                    if len(line) != (pgp_length + 2):
                        raise Exception("Error: line length should be equal to pgp length!", len(line), pgp_length,
                                        file_name)


        # shuffle lines in RAM, accumulate sentence_count

        if not test_mode:
            #print("Shuffling...")
            np.random.shuffle(cached_lines)

        sentence_count += len(cached_lines)
        # write out shuffled lines in RAM
        with open(dataset_out_file, mode='a') as f_write:
            #print('Writing...')
            for line in cached_lines:
                f_write.write(line)
        sub_index += 1
        end = time.time()
        #print('Task runs {:.2f} seconds.'.format((end - start), ))

    config["SET_LENG"] = sentence_count
    config["SET_LABELS_CNT"] = label_dict
    config["SET_LOGICL_CNT"] = ll_dict
    #print("Have Cached {} lines...".format(sentence_count))
    return sentence_count

def save_yaml_config(config, save_name):
    file = os.path.join(perl_temp_source_dir, config['SAVE_PARA']['FOLDER'], save_name)
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, 'w') as f_wr:
        yaml.dump(config, f_wr, sort_keys=False)


def dump2yaml(data, data_save_path):
    os.makedirs(os.path.dirname(data_save_path), exist_ok=True)
    with open(data_save_path, 'w') as f_wr:
        yaml.dump(data, f_wr, sort_keys=False)

