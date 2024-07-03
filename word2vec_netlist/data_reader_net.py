import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
import os


"""    --for netlist embedding data read in 
"""


# np.random.seed(12345)


class DataReader_Netlist:
    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, base_path, config, num_workers, min_count):  # design_path, inputFileName, log_level
        self.negatives = []
        self.discards = []
        self.negpos = 0

        self.file_path = ''
        self.sentences_count = 0

        self.chunk_file_paths = list()
        self.chunk_sentences_counts = list()

        self.word2id = dict()
        self.id2word = dict()
        self.token_count = 0
        self.word_frequency = dict()

        # self.read_dir(base_path, config_dict)
        self.read_words(min_count, base_path, config, num_workers)
        self.initTableNegatives()
        # self.initTableDiscards()

    def read_words(self, min_count, base_path, config, num_workers):
        # config = source_config["WORD2VEC_PARA"]
        self.sentences_count = config["SET_LENG"]
        self.chunk_file_paths = list()
        folder_path = os.path.join(base_path, config["FOLDER"])
        self.file_path = os.path.join(folder_path, config["SET_FILE"])
        source_file_name = config["SET_FILE"].split('.')
        local_sentences_count = 0
        word_frequency = dict()

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
                            f_wr.write(line)

                            line = line.strip()
                            line = line.split(',')
                            ll = int(line[0])
                            pgp_length = 2 * (ll - 1) + 1
                            # 2*(ll-1)+1 is pgp length, 3 is the ll , lable and comp.
                            if len(line) == (pgp_length + 3):
                                line = line[1:1 + pgp_length]
                                for word in line:
                                    if len(word) > 0:
                                        self.token_count += 1
                                        word_frequency[word] = word_frequency.get(word, 0) + 1
                                        if self.token_count % 1000000 == 0:
                                            print("Read " + str(int(self.token_count / 1000000)) + "M PGP words.")
                            else:
                                raise Exception("Line elements number do not match logic_level!")

                            local_sentences_count += 1
                            line_idx += 1
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
                                line = line.strip()
                                line = line.split(',')
                                ll = int(line[0])
                                pgp_length = 2 * (ll - 1) + 1
                                # 2*(ll-1)+1 is pgp length, 3 is the ll , lable and comp.
                                if len(line) == (pgp_length + 3):
                                    line = line[1:1 + pgp_length]
                                    for word in line:
                                        if len(word) > 0:
                                            self.token_count += 1
                                            word_frequency[word] = word_frequency.get(word, 0) + 1
                                            if self.token_count % 1000000 == 0:
                                                print("Read " + str(int(self.token_count / 1000000)) + "M PGP words.")
                                else:
                                    raise Exception("Line elements number do not match logic_level!")

                                local_sentences_count += 1

        else:
            with open(self.file_path, mode='r') as f_rd:
                for line in f_rd:
                    line = line.strip()
                    line = line.split(',')
                    ll = int(line[0])
                    pgp_length = 2 * (ll - 1) + 1
                    # 2*(ll-1)+1 is pgp length, 3 is the ll , lable and comp.
                    if len(line) == (pgp_length + 3):
                        line = line[1:1 + pgp_length]
                        for word in line:
                            if len(word) > 0:
                                self.token_count += 1
                                word_frequency[word] = word_frequency.get(word, 0) + 1
                                if self.token_count % 1000000 == 0:
                                    print("Read " + str(int(self.token_count / 1000000)) + "M PGP words.")
                    else:
                        raise Exception("Line elements number do not match logic_level!")
                    local_sentences_count += 1

        wid = 0
        for w, c in word_frequency.items():  # w:word_str ,c:word_count
            if c < min_count:
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1

        print("Total embeddings: " + str(len(self.word2id)))



    def initTableNegatives(self):
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.5  # possiblity on weight
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * DataReader_Netlist.NEGATIVE_TABLE_SIZE)
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)


# -----------------------------------------------------------------------------------------------------------------
class DataReader_Chunk:
    def __init__(self, data, worker_id=None):
        self.negatives = data.negatives
        self.discards = []
        self.negpos = 0
        self.word2id = data.word2id
        self.id2word = data.id2word
        self.token_count = data.token_count
        self.word_frequency = data.word_frequency

        if worker_id is not None:
            self.file_path = data.chunk_file_paths[worker_id]
            self.sentences_count = data.chunk_sentences_counts[worker_id]
        else:
            self.file_path = data.file_path
            self.sentences_count = data.sentences_count

    def __len__(self):
        return self.sentences_count

    def getNegatives(self, target, size):
        count = 0
        response = list()
        for index in range(size):
            while True:
                ele = self.negatives[(self.negpos + count) % len(self.negatives)]
                count += 1
                if (ele != target):
                    break
            response.append(ele)

        self.negpos = (self.negpos + count) % len(self.negatives)
        return response


# -----------------------------------------------------------------------------------------------------------------

class Word2vecIterDataset(IterableDataset):
    def __init__(self, data_list, batch_size):
        self.data_list = data_list
        self.batch_size = batch_size
        print("Embedding Dataset loaded")



    def __len__(self):
        sum_length = 0
        for x in range(len(self.data_list)):
            sum_length += self.data_list[x].sentences_count // self.batch_size + (
                          self.data_list[x].sentences_count % self.batch_size > 1)
        return sum_length

    def parse_file(self, data):
        with open(data.file_path, mode='r') as file_obj:
            for line in file_obj:
                line = line.strip()
                line = line.split(',')
                ll = int(line[0])
                pgp_length = 2 * (ll - 1) + 1
                # 2*(ll-1)+1 is pgp length, 3 is the ll , lable and comp.
                if len(line) != (pgp_length + 3):
                    raise Exception("Error: line length should be equal to pgp length!", line, pgp_length)
                else:
                    words = line[1:1 + pgp_length]
                    center_word = words[0]
                    center_wordid = None
                    word_ids = []
                    if center_word in data.word2id:
                        center_wordid = data.word2id[center_word]
                        # get contextid list
                        for w in words[1:]:
                            if w in data.word2id:
                                wordid = data.word2id[w]
                                word_ids.append(wordid)

                    # get the (center, context, negatives) pairs list
                    temp_list = []
                    for v in word_ids:
                        temp_array = data.getNegatives(v, 5)  # get negative samples for each (U,V)
                        temp_list.append((center_wordid, v, temp_array))

                    yield temp_list

    def get_stream(self, data):
        return self.parse_file(data)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return self.get_stream(self.data_list[0])
        else:
            worker_id = worker_info.id
            return self.get_stream(self.data_list[worker_id])

    @staticmethod
    def collate(batches):
        all_u = [u for batch in batches for u, _, _ in batch if len(batch) > 0]
        all_v = [v for batch in batches for _, v, _ in batch if len(batch) > 0]
        all_neg_v = [neg_v for batch in batches for _, _, neg_v in batch if len(batch) > 0]

        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v)
