import torch
import datetime
import pandas as pd
import numpy as np
from torch.autograd import Variable
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torch.nn.functional as F
from torch.nn import init
import os
import time
import sys
import yaml
import csv
import linecache
from tqdm import tqdm
from logger import setup_logger


#####RANDOM SEED CONTROL###################
SEED = 0
np.random.seed(SEED)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)



class Cnn_classify(nn.Module):
    def __init__(self, in_dim, n_class):  # 9(7)*100*1
        super(Cnn_classify, self).__init__()
        # conv layer
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 6, kernel_size=(2, 3), stride=(1, 1), padding=(1, 1)),  # 10,(8) * 100 * 6
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(1, 2)),  # 10,(8) * 50 * 6
            nn.Conv2d(6, 16, kernel_size=(3, 3), stride=(1, 1), padding=(0, 1)),  # 8,(6)*50*16
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2)))  # 4,(3)*25*16
        # full connect
        self.fc = nn.Sequential(
            nn.Linear(800, 400),  # 1600,800: 4*25*16, 1200,600: (3*25*16)
            nn.Linear(400, 100),
            nn.Linear(100, n_class))
        # softmax
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)  # 384=4*6*16, (288=3*6*16)   *batch_size
        out_score = self.fc(out)
        out_softmax = self.log_softmax(out_score)
        return out_softmax
    

###########################################################
class DataReader:

    def __init__(self, base_path, source_config, config_type, num_workers):
        self.word2vec_file_name = str()
        # self.data_file_name = os.path.join(base_path, "thub_s13207/s13207_T002/s13207_T002_noassign_vallogic",'df_keywords.csv')  # for test
        self.embedding_dim = 0
        self.dict_size = 0
        self.word2id_dict = dict()
        self.word2vec_dict = dict()
        self.type = type

        self.file_path = ''
        self.sentences_count = 0

        self.chunk_file_paths = list()
        self.chunk_sentences_counts = list()

        self.read_embedding(base_path, source_config)
        self.read_data(base_path, source_config['DETECT_PARA'], config_type, num_workers)

    def read_embedding(self, base_path, source_config):
        embed_para = source_config['WORD2VEC_PARA']
        self.word2vec_file_name = os.path.join(base_path, embed_para['FOLDER'], embed_para['OUT_FILE'])
        with open(self.word2vec_file_name, mode='r') as f_input:
            line = f_input.readline()
            line = line.strip()
            self.dict_size, self.embedding_dim = line.split()
            for line in f_input:
                line = line.strip()
                line = line.split()
                word = line[0]
                vector = line[1:]
                if word not in self.word2id_dict:
                    self.word2id_dict[word] = len(self.word2id_dict)
                if word not in self.word2vec_dict:
                    # self.word2vec_dict[word] = list(np.float_(vector))
                    self.word2vec_dict[word] = list(np.float32(vector))
                else:
                    raise Exception("Error: redundant word in PGP_out.vec!")

    def read_data(self, base_path, sub_config, config_type, num_workers):
        config = sub_config[config_type]

        self.sentences_count = config["SET_LENG"]
        self.chunk_file_paths = list()
        folder_path = os.path.join(base_path, config["FOLDER"])
        self.file_path = os.path.join(folder_path, config["SET_FILE"])
        source_file_name = config["SET_FILE"].split('.')
        local_sentences_count = 0

        if num_workers > 1:
            chunk_size = self.sentences_count // num_workers
            with open(os.path.join(folder_path, config["SET_FILE"]), mode='r') as f_rd:
                for chunk_idx in range(num_workers - 1):
                    print("Chunking...{}/{}".format(chunk_idx, num_workers - 1))
                    chunk_file_name = source_file_name[0] + '_' + str(chunk_idx) + '.' + source_file_name[1]
                    self.chunk_file_paths.append(os.path.join(folder_path, chunk_file_name))
                    self.chunk_sentences_counts.append(chunk_size)
                    with open(os.path.join(folder_path, chunk_file_name), mode='w') as f_wr:
                        line_idx = 0
                        while line_idx < chunk_size:
                            line = f_rd.readline()
                            # f_wr.write(line)
                            content = line.split(',')
                            ll = int(content[0])
                            leftc = content[1:1 + ll]
                            leftc.reverse()
                            sorted_content = [content[0]] + leftc + content[1 + ll:]
                            f_wr.write(','.join(sorted_content))

                            local_sentences_count += 1
                            line_idx += 1
                            if local_sentences_count % 1000000 == 0:
                                print("Read " + str(int(local_sentences_count / 1000000)) + "M Feature Traces.")
                else:
                    print("Chunking...Last")
                    chunk_file_name = source_file_name[0] + '_' + str(chunk_idx + 1) + '.' + source_file_name[1]
                    self.chunk_file_paths.append(os.path.join(folder_path, chunk_file_name))
                    self.chunk_sentences_counts.append(self.sentences_count - chunk_size * (num_workers - 1))
                    with open(os.path.join(folder_path, chunk_file_name), mode='w') as f_wr:
                        while True:
                            line = f_rd.readline()
                            if not line:
                                break
                            else:
                                f_wr.write(line)
                                local_sentences_count += 1
                                if local_sentences_count % 1000000 == 0:
                                    print("Read " + str(int(local_sentences_count / 1000000)) + "M Feature Traces.")
        else:
            with open(self.file_path, mode='r') as f_rd:
                for _ in f_rd:
                    local_sentences_count += 1


############################################################################
class DataReader_Chunk:
    def __init__(self, data, worker_id=None):
        self.word2vec_dict = data.word2vec_dict

        if worker_id is not None:
            self.file_path = data.chunk_file_paths[worker_id]
            self.sentences_count = data.chunk_sentences_counts[worker_id]
        else:
            self.file_path = data.file_path
            self.sentences_count = data.sentences_count

    def __len__(self):
        return self.sentences_count


####################################################################
class NetlistDataset(IterableDataset):
    def __init__(self, data_list, batch_size):
        self.data_list = data_list
        self.batch_size = batch_size
        print("Iterable Dataset Loaded...")

    def __len__(self):
        sum_length = 0
        for x in range(len(self.data_list)):
            sum_length += self.data_list[x].sentences_count // self.batch_size + (
                    self.data_list[x].sentences_count % self.batch_size > 0)
        return sum_length
        # return len(self.file_data)

    def parse_file(self, data):
        with open(data.file_path, mode='r') as file_obj:
            for line in file_obj:
                line = line.strip()
                line = line.split(',')
                ll = int(line[0])
                pgp_length = 2 * (ll - 1) + 1
                # 2*(ll-1)+1 is pgp length, 3 is the ll and lable and comp
                if len(line) != (pgp_length + 3):
                    raise Exception("Error: line length should be equal to pgp length!", line, pgp_length)
                else:
                    words = line[1:1 + pgp_length]
                    label = 0 if line[1 + pgp_length] == '0' else 1
                    vectors = []
                    for word in words:
                        if word in data.word2vec_dict:
                            word_vec = data.word2vec_dict[word]
                            word_vec = torch.tensor(word_vec)
                            vectors.append(word_vec)
                        else:
                            raise Exception("Error: can not find %s in word2vec_Dict!", word)
                    comp = line[2 + pgp_length]

                    vectors = torch.stack(vectors)
                    label = torch.tensor(label)
                    yield vectors, label, comp

    def get_stream(self, data):
        # return cycle(self.parse_file(data))
        return self.parse_file(data)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return self.get_stream(self.data_list[0])
        else:
            worker_id = worker_info.id
            return self.get_stream(self.data_list[worker_id])



##########################################################################
class Classifier_Netlist:
    def __init__(self,
                 group_id,
                 base_path,
                 source_config,  # design_path, input_file,log_level,　
                 # output_model_file,
                 # output_result_file="score_pred_label_comp.csv",
                 # hyper_parameter
                 embedding_dim=100,
                 hidden_dim=128,
                 batch_size=32,  # 132
                 num_tra_workers=4,
                 num_val_workers=2,
                 num_epoches=20,
                 initial_lr=0.001,
                 pretrained=None):  # 1e-3
        
        self.use_cuda = torch.cuda.is_available()
        # self.use_cuda=False
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.num_tra_workers = num_tra_workers
        self.num_val_workers = num_val_workers

        tra_lls = source_config["DETECT_PARA"]["TRAIN_PARA"]["SET_LOGICL_CNT"].keys()
        val_lls = source_config["DETECT_PARA"]["TEST_PARA"]["SET_LOGICL_CNT"].keys()

        lls = list(set(list(tra_lls) + list(val_lls)))
        fix_ll = None
        fix_pgp_length = None
        if len(lls) == 1:
            fix_ll = lls[0]
            fix_pgp_length = 2 * (fix_ll - 1) + 1
        else:
            raise Exception("Various Logic_levels exist in train/test...")



        self.tra_data = DataReader(base_path=base_path, source_config=source_config, config_type='TRAIN_PARA',
                                   num_workers=num_tra_workers)
        self.tra_data_chunk_list = list()

        if self.num_tra_workers > 1:
            for idx in range(self.num_tra_workers):
                self.tra_data_chunk_list.append(DataReader_Chunk(self.tra_data, worker_id=idx))
        else:
            self.tra_data_chunk_list.append(DataReader_Chunk(self.tra_data))
        
        self.tra_dataset = NetlistDataset(self.tra_data_chunk_list, batch_size)
        self.tra_dataloader = DataLoader(self.tra_dataset, batch_size=batch_size, shuffle=False,
                                         num_workers=num_tra_workers)

        print('===TEST===')

        self.val_data = DataReader(base_path=base_path, source_config=source_config, config_type='TEST_PARA',
                                   num_workers=num_val_workers)
        self.val_data_chunk_list = list()
        if self.num_val_workers > 1:
            for idx in range(self.num_val_workers):
                self.val_data_chunk_list.append(DataReader_Chunk(self.val_data, worker_id=idx))
        else:
            self.val_data_chunk_list.append(DataReader_Chunk(self.val_data))
        self.val_dataset = NetlistDataset(self.val_data_chunk_list, batch_size)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False,
                                         num_workers=num_val_workers)

        self.folder_path = os.path.join(base_path, source_config['DETECT_PARA']["PROCS_PARA"]["FOLDER"])
        self.output_model_file = "KFD[{}]_".format(group_id) + source_config['DETECT_PARA']["PROCS_PARA"]["OUT_MODEL"]
        self.output_result_file = "KFD[{}]_".format(group_id) + source_config['DETECT_PARA']["PROCS_PARA"]["OUT_ANA_FILE"]
        self.log_task_name = "KFD[{}]_".format(group_id) + source_config['DETECT_PARA']["PROCS_PARA"]["OUT_LOG"]

        self.emb_dimension = embedding_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.initial_lr = initial_lr
        self.num_epochs = num_epoches
        self.model = Cnn_classify(in_dim=1,
                                  n_class=2)  # C:1 H:fix_pgp_length W: self.self.emb_dimension

        if self.use_cuda:
            # if self.device=='cuda':
            self.model.cuda()
            # self.weights = self.weights.cuda()

        if pretrained:
            checkpoint = torch.load(pretrained, map_location='cuda')
            self.model.load_state_dict(checkpoint)
            print('Model loaded')

    def train(self):  # train_data, valid_data, num_epochs, optimizer, loss_function
        logger = setup_logger(self.log_task_name, self.folder_path, 0, filename=self.log_task_name + '_log' + '.txt')

        # criterion = nn.CrossEntropyLoss(weight=self.weights)
        # criterion = nn.CrossEntropyLoss()  # # when use softmax

        # criterion = nn.NLLLoss(weight=self.weights)
        criterion = nn.NLLLoss()  # when use log_softmax

        optimizer = optim.Adam(self.model.parameters(), lr=self.initial_lr)
        prev_time = datetime.datetime.now()
        
        for epoch in range(0, self.num_epochs):


            # =====initialize parameters
            logger.info('epoch {}'.format(epoch + 1))
            logger.info('*' * 10)
            running_loss = 0.0
            running_acc = 0.0
            tra_confusion_matrix = torch.zeros(2, 2)  # nb_classes, nb_classes
            self.model.train()
            for i, sample_batched in enumerate(tqdm(self.tra_dataloader)):
                # batched_data = sample_batched[0].to(self.device)
                batched_data = torch.unsqueeze(sample_batched[0], 1).to(self.device)  # CNN input [N,C,H,W]
                batched_label = sample_batched[1].to(self.device)
                batched_comp = sample_batched[2]
                out = self.model(batched_data)
                loss = criterion(out, batched_label)
                running_loss += loss.item() * batched_label.size(0)

                _, pred = torch.max(out, 1)
                for p, t in zip(pred.view(-1), batched_label.view(-1)):
                    tra_confusion_matrix[p.long(), t.long()] += 1
                num_correct = (pred == batched_label).sum()
                running_acc += num_correct.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                # if i > 0 and i % 3000 == 0:
                #     logger.info('Train [{}] epoch, Loss: {:.6f}, Acc: {:.6f}'.format(epoch + 1,
                #                                                                running_loss / i,
                #                                                                running_acc / (self.batch_size * i)))
                #     logger.info(tra_confusion_matrix)
                #     # NPV,precision(PPV)
                #     precision = tra_confusion_matrix.diag() / tra_confusion_matrix.sum(1)  # ----> row add
                #     logger.info(precision)
                #     # TNR,recall(TPR)
                #     recall = tra_confusion_matrix.diag() / tra_confusion_matrix.sum(0)  # column add
                #     logger.info(recall)


            logger.info('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(epoch + 1,
                                                                      running_loss / (len(self.tra_dataset)),
                                                                      running_acc / (self.batch_size * len(
                                                                          self.tra_dataset))))
    
            self.evaluate_no_save()

            # logger.info(tra_confusion_matrix)
            # # To get the per-class accuracy: precision
            # precision = tra_confusion_matrix.diag() / tra_confusion_matrix.sum(1)
            # # print(precision)
            # logger.info(precision)
            # recall = tra_confusion_matrix.diag() / tra_confusion_matrix.sum(0)
            # # print(recall)
            # logger.info(recall)

            # if epoch == self.num_epochs -1:

            #     # print('====TEST====')# ===============================================================
            #     logger.info('====TEST====')  # ===============================================================
            #     # =====create output data file
            #     temp_name = self.output_result_file.split('.')
            #     result_file = temp_name[0] + '_test_epoch[{:d}].'.format(epoch) + temp_name[1]
            #     result_path_file = os.path.join(self.folder_path, result_file)
            #     # print("Creating {}...".format(result_file))
            #     logger.info("Creating {}...".format(result_file))
            #     try:
            #         os.remove(result_path_file)
            #     except OSError:
            #         pass
            #     df = pd.DataFrame(columns=['out_score', 'pred', 'label', 'compname'])
            #     df.to_csv(result_path_file, mode='w', header=True, index=False)
            #     output_data = dict()
            #     self.model.eval()
            #     eval_loss = 0.0
            #     eval_acc = 0.0
            #     eval_confusion_matrix = torch.zeros(2, 2)  # nb_classes, nb_classes
            #     for i, sample_batched in enumerate(tqdm(self.val_dataloader)):
            #         # batched_data = sample_batched[0].to(self.device)
            #         batched_data = torch.unsqueeze(sample_batched[0], 1).to(self.device)  # CNN input [N,C,H,W]
            #         batched_label = sample_batched[1].to(self.device)
            #         batched_comp = sample_batched[2]
            #         out = self.model(batched_data)
            #         loss = criterion(out, batched_label)
            #         eval_loss += loss.item() * batched_label.size(0)
            #         _, pred = torch.max(out, 1)
            #         for p, t in zip(pred.view(-1), batched_label.view(-1)):
            #             eval_confusion_matrix[p.long(), t.long()] += 1
            #         num_correct = (pred == batched_label).sum()
            #         eval_acc += num_correct.item()

            #         # =====save output data
            #         if 'out_score' not in output_data:
            #             output_data['out_score'] = out_score.detach().cpu()
            #         else:
            #             output_data['out_score'] = torch.cat((output_data['out_score'], out_score.detach().cpu()), dim=0)

            #         if 'pred' not in output_data:
            #             output_data['pred'] = pred.detach().cpu()
            #         else:
            #             output_data['pred'] = torch.cat((output_data['pred'], pred.detach().cpu()), dim=0)

            #         if 'label' not in output_data:
            #             output_data['label'] = batched_label.detach().cpu()
            #         else:
            #             output_data['label'] = torch.cat((output_data['label'], batched_label.detach().cpu()), dim=0)

            #         if 'compname' not in output_data:
            #             output_data['compname'] = list(batched_comp)
            #         else:
            #             output_data['compname'] += list(batched_comp)

            #         if i > 0 and i % 3000 == 0:
            #             logger.info("ANA_DATA saving...")
            #             output_data['out_score'] = list(output_data['out_score'].numpy())
            #             output_data['pred'] = output_data['pred'].numpy()
            #             output_data['label'] = output_data['label'].numpy()
            #             df = pd.DataFrame(output_data)
            #             output_data = dict()
            #             df.to_csv(result_path_file, mode='a', header=False, index=False)

            #         output_data['out_score'] = list(output_data['out_score'].numpy())
            #         output_data['pred'] = output_data['pred'].numpy()
            #         output_data['label'] = output_data['label'].numpy()
            #         df = pd.DataFrame(output_data)
            #         output_data = dict()
            #         df.to_csv(result_path_file, mode='a', header=False, index=False)

            #         logger.info('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(self.val_dataset)),
            #                                                     eval_acc * 1.0 / (self.batch_size * len(self.val_dataset))))
            #         logger.info(eval_confusion_matrix)

            #         # NPV, precision(PPV)
            #         precision = eval_confusion_matrix.diag() / eval_confusion_matrix.sum(1)
            #         logger.info(precision)
            #         # TNR, recall(TPR)
            #         recall = eval_confusion_matrix.diag() / eval_confusion_matrix.sum(0)
            #         logger.info(recall)

            # SAVE THE MODEL
            temp_name = self.output_model_file.split('.')
            result_file = temp_name[0] + '_epoch[{:d}].'.format(epoch) + temp_name[1]
            torch.save(self.model.state_dict(), os.path.join(self.folder_path, result_file))

    def evaluate_no_save(self):
        logger = setup_logger(self.log_task_name, self.folder_path, 0, filename=self.log_task_name + '_log' + '.txt')
        criterion = nn.NLLLoss()  # when use log_softmax

        # print('====TEST====')# ===============================================================
        logger.info('====TEST====')  # ===============================================================
        # =====create output data file

        self.model.eval()
        eval_loss = 0.0
        eval_acc = 0.0
        eval_confusion_matrix = torch.zeros(2, 2)  # nb_classes, nb_classes
        for i, sample_batched in enumerate(tqdm(self.val_dataloader)):
            # batched_data = sample_batched[0].to(self.device)
            batched_data = torch.unsqueeze(sample_batched[0], 1).to(self.device)  # CNN input [N,C,H,W]
            batched_label = sample_batched[1].to(self.device)
            batched_comp = sample_batched[2]
            out = self.model(batched_data)
            loss = criterion(out, batched_label)
            eval_loss += loss.item() * batched_label.size(0)
            _, pred = torch.max(out, 1)
            for p, t in zip(pred.view(-1), batched_label.view(-1)):
                eval_confusion_matrix[p.long(), t.long()] += 1
            num_correct = (pred == batched_label).sum()
            eval_acc += num_correct.item()

        logger.info('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(self.val_dataset)),
                                                        eval_acc * 1.0 / (self.batch_size * len(self.val_dataset))))
        logger.info(eval_confusion_matrix)

        # NPV, precision(PPV)
        precision = eval_confusion_matrix.diag() / eval_confusion_matrix.sum(1)
        logger.info(precision)
        # TNR, recall(TPR)
        recall = eval_confusion_matrix.diag() / eval_confusion_matrix.sum(0)
        logger.info(recall)
