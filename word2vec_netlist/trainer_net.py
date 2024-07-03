import torch
# import torch.optim
# from torch.optim import SparseAdam

import json
import os

from torch.utils.data import DataLoader
from tqdm import tqdm

from word2vec_netlist.data_reader_net import DataReader_Netlist, DataReader_Chunk, Word2vecIterDataset  # project package
from word2vec_netlist.model_net import SkipGramModel



from logger import setup_logger

"""
    Shichao Yu
"""


# self defined print to both file and console
def print_both(file, *args):
    to_print = ' '.join([str(arg) for arg in args])
    print(to_print)
    file.write(to_print)

class Word2VecTrainer_Netlist:
    def __init__(self,
                 base_path,
                 source_config,  # design_path, input_file,log_level,　
                 # hyper_parameter
                 emb_dimension=100,
                 batch_size=36,  # 132
                 num_workers=8,
                 # window_size=5,
                 iterations=4,  # 3 6
                 initial_lr=0.001,
                 min_count=1):  # 12

        self.num_workers = num_workers
        self.data = DataReader_Netlist(base_path=base_path,
                                       config=source_config["WORD2VEC_PARA"],
                                       num_workers=num_workers,
                                       min_count=min_count)  # design_path,input_file, log_level
        self.data_chunk_list = list()
        if self.num_workers > 1:
            for idx in range(self.num_workers):
                self.data_chunk_list.append(DataReader_Chunk(self.data, worker_id=idx))
        else:
            self.data_chunk_list.append(DataReader_Chunk(self.data))

        dataset = Word2vecIterDataset(self.data_chunk_list, batch_size)  # window_size

        self.dataloader = DataLoader(dataset, batch_size=batch_size,
                                     shuffle=False, num_workers=num_workers,
                                     collate_fn=dataset.collate)

        self.base_folder = os.path.join(base_path, source_config["WORD2VEC_PARA"]["FOLDER"])
        self.output_file_name = source_config["WORD2VEC_PARA"]["OUT_FILE"]
        self.output_model_name = source_config["WORD2VEC_PARA"]["OUT_MODEL"]
        self.log_task_name = source_config["WORD2VEC_PARA"]["OUT_LOG"]

        self.emb_size = len(self.data.word2id)
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.epochs = iterations
        self.initial_lr = initial_lr
        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.skip_gram_model.cuda()

    def get_lr(self, opt):
        for param_group in opt.param_groups:
            return param_group['lr']

    def train(self):
        # init
        logger = setup_logger(self.log_task_name, self.base_folder, 0, filename='log_' + self.log_task_name + '.txt')

        optimizer = torch.optim.SGD(self.skip_gram_model.parameters(), lr=self.initial_lr)  # optimizer

        lr_list = []

        # training
        for epoch in range(self.epochs):  # re-training times
            logger.info("Epoch: " + str(epoch))

            running_loss = 0.0
            epoch_loss = 0.0
            length_traindata = 0
            # training
            for i, sample_batched in enumerate(tqdm(self.dataloader)):  # times: 1563 , all batch number

                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)

                    optimizer.zero_grad()  # initialize the gradient
                    loss = self.skip_gram_model.forward(pos_u, pos_v, neg_v)  # forward transfer

                    loss.backward()  # gradient calculation
                    optimizer.step()  # update the weight parameters in network

                    running_loss += loss.item()
                    epoch_loss += pos_u.shape[0] * loss.item()
                    length_traindata += pos_u.shape[0]

                    if i > 0 and i % 10000 == 0:
                        logger.info("Current Loss: " + str(loss.item()))
                        logger.info("Loss over 10000 samples: " + str(running_loss / 10000))
                        running_loss = 0.0

            # print epoch loss
            logger.info("Loss over epoch " + str(epoch) + " : " + str(epoch_loss / length_traindata))
            torch.save(self.skip_gram_model.state_dict(), os.path.join(self.base_folder,self.output_model_name))
        self.skip_gram_model.save_embedding(self.data.id2word, os.path.join(self.base_folder,self.output_file_name))
