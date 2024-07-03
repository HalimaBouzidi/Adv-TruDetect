import json
import os
import csv
import pandas
import re
import yaml
import polars as pl
import gc

class JsonLoader:
    def __init__(self,
                 base_path,
                 design_path,
                 file_name):
        self.data = dict()
        self.len = 0
        self.file_name = file_name

        self.json_file_load(base_path, design_path, file_name)

    def json_file_load(self, base_path, design_path, file_name):
        file = os.path.join(base_path, design_path, file_name)
        with open(file, 'r') as f:
            self.data = json.load(f)
        self.len = len(self.data)

    def data_len(self):
        print(self.file_name + " len:" + str(self.len))


class NetlistDataScraper:

    def __init__(self,
                 base_path,
                 design_path,
                 source_file_name,
                 logic_level):
        self.ref_Gate_HASH_ORI = dict()
        self.ref_Netlink_HASH = dict()
        self.ref_Design_input_Port = dict()
        self.ref_Design_output_Port = dict()
        self.ref_Blocks_HASH = dict()

        self.compref_blocks_slim_dict = dict()
        self.compref_path_dict = dict()
        self.compref_pathPGP_dict = dict()
        self.compref_PGP_data = dict()
        self.compref_PGP_data_count = 0
        self.compref_PGP_raw = list()

        self.compref_label_cell_dict = dict()
        self.compref_label_num_dict = dict()

        self.word2id = dict()
        self.id2word = dict()
        self.sentences_count = 0
        self.token_count = 0
        self.word_frequency = dict()

        self.ref_logic_level = logic_level
        self.ref_source_file_name = source_file_name
        self.ref_Gate_HASH_ORI = JsonLoader(base_path, design_path, 'ref_Gate_HASH_ORI.json')
        self.ref_Netlink_HASH = JsonLoader(base_path, design_path, 'ref_Netlink_HASH.json')
        self.ref_Design_input_Port = JsonLoader(base_path, design_path, 'ref_Design_input_Port.json')
        self.ref_Design_output_Port = JsonLoader(base_path, design_path, 'ref_Design_output_Port.json')
        self.ref_Blocks_HASH = JsonLoader(base_path, design_path, 'ref_hash_blocks.json')

        self.labelfile_read_in(base_path, design_path)

        self.hash_block_simplify()
        self.netlist_path_rewrite()
        self.pgp_path_rewrite()
        self.pgp_database()
        self.file_name = "PGP_keywords.txt"
        self.file_raw_name = "PGP_keywords_raw.txt"
        self.file_output_pgp(base_path, design_path)
        self.CSV_output_pgp(base_path, design_path)




    # self.compref_label_cell_dict
    def labelfile_read_in(self, base_path, design_path):
        file_name = 'htlabel_cell.txt'
        list_path = design_path.split("/")
        design_name = list_path[-2]
        file = os.path.join(base_path, design_path, file_name)
        cnt_block = 0
        cnt_file = 0
        with open(file, 'r') as f_input:
            for line in f_input:
                if ('---' in line):
                    cnt_block += 1
                elif ('=file_split=' in line):
                    cnt_file += 1
                else:
                    line = line.strip()
                    line = line.split()
                    if (len(line) > 0):
                        cell_name = line[0]
                        self.compref_label_cell_dict[cell_name] = design_name + '_T' + str(cnt_file) + '_' + str(
                            cnt_block)


    def hash_block_simplify(self):
        for key in sorted(self.ref_Blocks_HASH.data.keys()):
            temp_block = {}
            for side in ['L', 'R']:
                side_list = self.ref_Blocks_HASH.data[key][side]
                # print(side_list)
                temp_side_list = []
                for list in side_list:
                    if len(list) == 8:
                        refname_list = list[1:3] + list[5:]
                        name_list = list[:1] + list[3:5]
                        net_prob = [-2, -2]
                        temp_side_list.append((refname_list, name_list, net_prob))
                temp_block[side] = temp_side_list

            temp_block['self'] = {}
            temp_block['self']['list_input_pin'] = self.ref_Gate_HASH_ORI.data[key]['list_input_pin']
            temp_block['self']['list_output_pin'] = self.ref_Gate_HASH_ORI.data[key]['list_output_pin']
            temp_block['self']['cell_reference_name'] = self.ref_Gate_HASH_ORI.data[key]['cell_reference_name']

            # probibility code
            temp_block['self']['list_input_prob'] = []
            for ele in self.ref_Gate_HASH_ORI.data[key]['list_input_net']:
                net_prob = [-2, -2]
                temp_block['self']['list_input_prob'].append(net_prob)

            temp_block['self']['list_output_prob'] = []
            for ele in self.ref_Gate_HASH_ORI.data[key]['list_output_net']:
                net_prob = [-2, -2]
                temp_block['self']['list_output_prob'].append(net_prob)

            # probibility code

            # for pinin in self.ref_Gate_HASH_ORI.data[key]['list_input_pin']:
            #     for pinout in self.ref_Gate_HASH_ORI.data[key]['list_output_pin']:
            #         temp_list = [self.ref_Gate_HASH_ORI.data[key]['cell_reference_name'], pinout,
            #                      self.ref_Gate_HASH_ORI.data[key]['cell_reference_name'], pinin, 0]
            #         temp_self_list.append(temp_list)

            self.compref_blocks_slim_dict[key] = temp_block

    # 1.create a stack to store the different level parent node
    # 2. if next level >=current last level, than output the stack as a comp path
    # self.compref_path_dict = dict()
    def netlist_path_rewrite(self):
        # cell_path_dict = {}
        for key in self.compref_blocks_slim_dict:
            temp_block = {}
            for side in ['L', 'R']:
                comp_stack_top = -1
                comp_stack = []
                comp_path_list = []
                for comb in self.compref_blocks_slim_dict[key][side]:
                    line = comb[0]
                    name_list = comb[1]  # defined unsed
                    net_prob = comb[2]  # defined unsed
                    if (len(line) == 5):
                        # comp_key2=line[0]+'_'+line[1]
                        # comp_key1=line[2]+'_'+line[3]
                        # comp_level=line[4]
                        if len(comp_stack) == 0:
                            comp_stack.append(comb)
                            comp_stack_top = 0
                        else:
                            if (comp_stack[comp_stack_top][0][4] <= line[4]):
                                # print("Path End")
                                # print(comp_stack)
                                comp_path_list.append(comp_stack.copy())

                                while (len(comp_stack) > 0):
                                    if (comp_stack[comp_stack_top][0][4] <= line[4]):
                                        ele = comp_stack.pop()
                                        comp_stack_top -= 1
                                    else:
                                        break
                            comp_stack.append(comb)
                            comp_stack_top += 1
                else:
                    # print("Last Path")
                    # print(comp_stack)
                    comp_path_list.append(comp_stack.copy())
                    comp_stack.clear()
                temp_block[side] = comp_path_list
            temp_block['self'] = self.compref_blocks_slim_dict[key]['self']
            self.compref_path_dict[key] = temp_block


    def pgp_path_rewrite(self):
        for key in self.compref_path_dict:
            temp_block = {}
            # temp_block['self'] = []
            temp_block['L'] = []
            temp_block['R'] = []
            temp_block['self'] = self.compref_path_dict[key]['self']

            # if (len(self.compref_path_dict[key]['self']['list_output_pin']) > 1):
            #     print(key)
            for path_list in self.compref_path_dict[key]['L']:
                for pin_index, pin in enumerate(self.compref_path_dict[key]['self'][
                                                    'list_output_pin']):  # make left side list contain all the output pins on the cells
                    comp_path_list = []
                    length_path = len(path_list)
                    if (length_path == 0):
                        raise Exception("Error: path_list in compref_path_dict L <1!")
                    for index in range(self.ref_logic_level):
                        if (index < length_path):
                            comb = path_list[index]
                            line = comb[0]
                            name_list = comb[1]
                            if (len(line) != 5):
                                raise Exception("Error: line in path_list[x][0] should be 5!")
                            if (index < 1):
                                temp_line = [line[3], line[2], pin]
                                temp_name = [name_list[2]]
                            else:
                                pre_comp = path_list[index - 1]
                                pre_line = pre_comp[0]
                                pre_name_list = pre_comp[1]

                                temp_line = [line[3], line[2], pre_line[1]]  # path_list[XXXX-1][0] pre_line
                                temp_name = [pre_name_list[0], pre_name_list[1]]  # path_list[XXXX][1] name
                        elif (index == length_path):
                            pre_comp = path_list[index - 1]
                            pre_line = pre_comp[0]
                            # if line=(input,input,input,xxx,xxx,xxx)
                            if (pre_line[0] == 'input'):
                                temp_line = ['input', 'input', 'input']
                                temp_name = ['input', 'input']

                            else:
                                temp_line = ['None_I', 'None_I', 'None_I']
                                temp_name = ['None_I', 'None_I']

                        else:
                            temp_line = ['None_I', 'None_I', 'None_I']
                            temp_name = ['None_I', 'None_I']

                        temp_prob = [-2, -2]
                        comp_path_list.append((temp_line, temp_name, temp_prob))
                    temp_block['L'].append(comp_path_list)

            for path_list in self.compref_path_dict[key]['R']:
                comp_path_list = []
                length_path = len(path_list)
                if (length_path == 0):
                    raise Exception("Error: path_list in compref_path_dict R <1!")
                for index in range(self.ref_logic_level):
                    if (index < length_path):
                        comb = path_list[index]
                        line = comb[0]
                        name_list = comb[1]
                        if (len(line) != 5):
                            raise Exception("Error: line in path_list[x][0] should be 5!")

                        if (index < 1):
                            temp_line = [None, line[2], line[3]]
                            temp_name = [name_list[2]]
                        else:
                            pre_comp = path_list[index - 1]
                            pre_line = pre_comp[0]
                            pre_name_list = pre_comp[1]

                            temp_line = [pre_line[1], line[2], line[3]]
                            temp_name = [pre_name_list[0], pre_name_list[1]]

                    elif (index == length_path):
                        pre_comp = path_list[index - 1]
                        pre_line = pre_comp[0]
                        # if line=(input,input,input,xxx,xxx,xxx)
                        if (pre_line[0] == 'output'):
                            temp_line = ['output', 'output', 'output']
                            temp_name = ['output', 'output']
                        else:
                            temp_line = ['None_O', 'None_O', 'None_O']
                            temp_name = ['None_O', 'None_O']
                    else:
                        temp_line = ['None_O', 'None_O', 'None_O']
                        temp_name = ['None_O', 'None_O']
                    temp_prob = [-2, -2]
                    comp_path_list.append((temp_line, temp_name, temp_prob))
                temp_block['R'].append(comp_path_list)

            self.compref_pathPGP_dict[key] = temp_block

    # self.compref_PGP_data
    # self.compref_PGP_raw
    # allocate the Right side list to each left side list
    def pgp_database(self):
        for ele in self.compref_pathPGP_dict:
            temp_block = []
            temp_list = []
            label = '0'

            # print('sub>(pgp_database): '+ele)
            if (ele in self.compref_label_cell_dict):
                label = self.compref_label_cell_dict[ele]
            for path_list in self.compref_pathPGP_dict[ele]['L']:
                headpgp_comb = path_list[0]
                headpgp_ele = headpgp_comb[0]
                # headpgp_netprob = headpgp_comb[2]
                # print(path_list)
                keywords = headpgp_ele[1] + headpgp_ele[2]
                # print(keywords)
                for right_path_list in self.compref_pathPGP_dict[ele]['R']:
                    rheadpgp_comb = right_path_list[0]
                    rheadpgp_ele = rheadpgp_comb[0]
                    # rheadpgp_netprob = rheadpgp_comb[2]
                    if (rheadpgp_ele[1] + rheadpgp_ele[2] == keywords):
                        # print(right_path_list)
                        # temp_list[0]=path_list
                        # temp_list[1]=right_path_list
                        self.compref_PGP_data_count += 1
                        temp_block.append([path_list, right_path_list[1:], label])
                        self.compref_PGP_raw.append([path_list, right_path_list[1:], label])
            self.compref_PGP_data[ele] = temp_block
        print('sub>(pgp_database): Database Builded!')
        # print('sub>(pgp_database): ',end=" ")
        # print(*self.compref_pathPGP_dict.keys(), sep=' ')

    def file_output_pgp(self, base_path, design_path):
        file = os.path.join(base_path, design_path, self.file_name)
        file_raw = os.path.join(base_path, design_path, self.file_raw_name)

        f_output = open(file, 'w')
        f_raw_output = open(file_raw, 'w')

        for ele in self.compref_PGP_data:
            for path_list in self.compref_PGP_data[ele]:
                ############################################################################
                # pgp data line
                keywords = ''
                namewords = ''
                for comb in path_list[0]:
                    pgp = comb[0]
                    name = comb[1]
                    # keywords = pgp[0] + '_' + pgp[1] + '_' + pgp[2]
                    if (len(pgp) != 3):
                        raise Exception("Error: each pgp should have 3 elements!")
                    pgp[1] = re.sub(r"s\d+$", "", pgp[1])
                    keywords += '_'.join(pgp) + ','
                    namewords += ':'.join(name) + ','

                keywords += ';'
                namewords += ';'

                for comb in path_list[1]:
                    pgp = comb[0]
                    name = comb[1]
                    # keywords = pgp[0] + '_' + pgp[1] + '_' + pgp[2]
                    if (len(pgp) != 3):
                        raise Exception("Error: each pgp should have 3 elements!")
                    pgp[1] = re.sub(r"s\d+$", "", pgp[1])
                    keywords += '_'.join(pgp) + ','
                    namewords += ':'.join(name) + ','

                f_output.write('@cell: ' + ele + '\n')
                f_output.write('  ' + keywords + '###' + str(path_list[2]) + '\n')
                f_output.write('  ' + namewords + '\n')

                f_raw_output.write(keywords + '###' + str(path_list[2]) + '\n')

        f_output.close()
        f_raw_output.close()

    # self.compref_label_num_dict
    def CSV_output_pgp(self, base_path, design_path):
        # file = os.path.join(base_path, design_path, 'keywords_data.csv')
        # f_output = open(file, 'w')
        # f_output.close()
        if (self.ref_logic_level == 0):
            raise Exception("Error: logic level=0 extract nothing!")

        key_headers = ['PCP_0'] + \
                      ['PCP_L' + str(num) for num in range(1, self.ref_logic_level)] + \
                      ['PCP_R' + str(num) for num in range(1, self.ref_logic_level)] + \
                      ['Label'] + \
                      ['Comp']

        name_headers = ['Cell_0'] + \
                       ['CellNet_L' + str(num) for num in range(1, self.ref_logic_level)] + \
                       ['CellNet_R' + str(num) for num in range(1, self.ref_logic_level)] + \
                       ['Label']

        # self.compref_PGP_data
        keywords_rows = []
        namewords_rows = []
        for ele in self.compref_PGP_data:
            for path_list in self.compref_PGP_data[ele]:
                keywords_row = []
                namewords_row = []
                for comb in path_list[0]:
                    pgp = comb[0]
                    name = comb[1]
                    keywords_row.append('_'.join(pgp))
                    namewords_row.append(':'.join(name))
                for comb in path_list[1]:
                    pgp = comb[0]
                    name = comb[1]
                    keywords_row.append('_'.join(pgp))
                    namewords_row.append(':'.join(name))

                self.compref_label_num_dict[path_list[2]] = self.compref_label_num_dict.get(path_list[2], 0) + 1
                keywords_row.append(str(path_list[2]))  # label
                local_name = namewords_row[0] + '@' + self.ref_source_file_name
                keywords_row.append(local_name)  # name of the component belongs

                namewords_row.append(str(path_list[2]))  # label

                keywords_rows.append(keywords_row)
                namewords_rows.append(namewords_row)


        df_keywords = pandas.DataFrame(data=keywords_rows, columns=key_headers)
        del keywords_rows
        gc.collect()
        df_namewords = pandas.DataFrame(data=namewords_rows, columns=name_headers)
        del namewords_rows
        gc.collect()


        file = os.path.join(base_path, design_path, 'labels_dict.yaml')
        with open(file, 'w') as f:
            yaml.dump(self.compref_label_num_dict, f, sort_keys=False)

        file = os.path.join(base_path, design_path, 'df_keywords.csv')
        df_keywords.to_csv(file, header=False, index=False)
        del df_keywords
        gc.collect()
        file = os.path.join(base_path, design_path, 'df_namewords.csv')
        df_namewords.to_csv(file)
        del df_namewords
        gc.collect()


        # df_keywords_pl = pl.from_pandas(df_keywords)
        # del df_keywords
        # gc.collect()
        # df_namewords_pl = pl.from_pandas(df_namewords)
        # del df_namewords
        # gc.collect()
        #
        # index_series=pl.Series('None', list(range(0,len(df_namewords_pl))))
        # df_namewords_pl.insert_column(0, index_series)
        #
        # file = os.path.join(base_path, design_path, 'df_keywords.csv')
        # df_keywords_pl.write_csv(file,include_header=False)
        #
        # file = os.path.join(base_path, design_path, 'df_namewords.csv')
        # df_namewords_pl.write_csv(file)
