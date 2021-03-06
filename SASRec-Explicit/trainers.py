import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.optim import Adam

from utils import ndcg_k, recall_at_k


class Trainer:
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        submission_dataloader,
        args,
    ):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.submission_dataloader = submission_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(
            self.model.parameters(),
            lr=self.args.lr,
            betas=betas,
            weight_decay=self.args.weight_decay,
        )

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        self.criterion = nn.BCELoss()

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch):
        return self.iteration(epoch, self.eval_dataloader, mode="valid")

    def test(self, epoch):
        return self.iteration(epoch, self.test_dataloader, mode="test")

    def submission(self, epoch):
        return self.iteration(epoch, self.submission_dataloader, mode="submission")

    def iteration(self, epoch, dataloader, mode="train"):
        raise NotImplementedError

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "RECALL@5": "{:.4f}".format(recall[0]),
            "NDCG@5": "{:.4f}".format(ndcg[0]),
            "RECALL@10": "{:.4f}".format(recall[1]),
            "NDCG@10": "{:.4f}".format(ndcg[1]),
        }
        print(post_fix)

        return [recall[0], ndcg[0], recall[1], ndcg[1]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size)  # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (
            (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()
        )  # [batch*seq_len]
        loss = torch.sum(
            -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget
            - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss

    def loss_fn(self, seq_out, pos_ids, target_ratings):
        # seq_out, pos_ids, target_ratings => [batch X seq_len]
        
        pos_emb = self.model.item_embeddings(pos_ids) # [batch X seq_len X hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2)) # [batch*seq_len X hidden_size]
        
        # seq_out => [batch X seq_len X hidden_size]
        # seq_emb => [batch*seq_len X hidden_size]
        seq_emb = seq_out.view(-1, self.args.hidden_size)  # [batch*seq_len X hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]

        istarget = (pos_ids > 0).view(-1).float()  # [batch*seq_len]

        loss = torch.sum(
            torch.log(torch.sigmoid(target_ratings) + 1e-24) * istarget
            -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss


    def rmse(self, seq_out, pos_ids, target_ratings):
        # seq_out, pos_ids, target_ratings => [batch X seq_len]
        
        pos_emb = self.model.item_embeddings(pos_ids) # [batch X seq_len X hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2)) # [batch*seq_len X hidden_size]
        
        # seq_out => [batch X seq_len X hidden_size]
        # seq_emb => [batch*seq_len X hidden_size]
        seq_emb = seq_out.view(-1, self.args.hidden_size)  # [batch*seq_len X hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]

        istarget = (pos_ids > 0).view(-1).float()

        loss_mse = torch.sum(
            (pos_logits - target_ratings.view(-1))**2 *istarget
        ) / torch.sum(istarget)

        loss_rmse = torch.sqrt(loss_mse)

        # mse = torch.nn.MSELoss()
        # loss = torch.sqrt(mse(pos_logits, target_ratings.view(-1)) + 1e-6)

        return loss_rmse

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred


class ExplicitTrainer(Trainer):
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        submission_dataloader,
        args,
    ):
        super(ExplicitTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            submission_dataloader,
            args,
        )

    def iteration(self, epoch, dataloader, mode="train"):

        # Setting the tqdm progress bar

        rec_data_iter = tqdm.tqdm(
            enumerate(dataloader),
            desc="Recommendation EP_%s:%d" % (mode, epoch),
            total=len(dataloader),
            bar_format="{l_bar}{r_bar}",
        )
        if mode == "train":
            self.model.train()
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0

            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)
                _, input_ids, input_ratings, target_pos, target_neg, target_ratings, _, _ = batch

                sequence_output = self.model.forward(input_ids, input_ratings)
                # Binary cross_entropy
                # loss = self.cross_entropy(sequence_output, target_pos, target_neg)
                
                # RMSE
                loss = self.rmse(sequence_output, target_pos, target_ratings)

                # BCE logits
                # loss = self.loss_fn(sequence_output, target_pos, target_ratings)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": "{:.4f}".format(rec_avg_loss / len(rec_data_iter)),
                "rec_cur_loss": "{:.4f}".format(rec_cur_loss),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

        else:
            self.model.eval()

            pred_list = []
            answer_list = []
            for i, batch in rec_data_iter:

                batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids, input_ratings, _, target_neg, target_ratings, answers, ratings_answer = batch
                recommend_output = self.model.forward(input_ids,  input_ratings) ############## TODO

                recommend_output = recommend_output[:, -1, :]
                # print(">>>>> recommend_output.size:", recommend_output.size())
                # print(">>>>> answers.size:", answers.size())
                
                # recommend_output : ???????????? ????????? ????????? ????????? ???????????? ???????????? ?????? ???????????? ??????????????? ???????????? ????????? ?????? # [batch hidden_size]
                # answers : ?????? ????????? ????????? ??? ?????? ?????????   # [batch * 1]
                # ratings_answer : ?????? ???????????? ?????? ????????? ??????????????? ???????????? ?????? # [batch hidden_size]

                answers_emb = self.model.item_embeddings(answers).squeeze(1) # [batch * hidden_size]
                rating_pred = torch.sum(answers_emb * recommend_output, dim = 1)

                pred_list.extend(rating_pred.cpu().data.numpy().copy().tolist())
                answer_list.extend(ratings_answer.squeeze(1).cpu().data.numpy().copy().tolist())

                # metric_fn = torch.nn.MSELoss()
                # score = metric_fn(rating_pred, ratings_answer.squeeze(1))

                # rating_pred = self.predict_full(recommend_output)

                # rating_pred = rating_pred.cpu().data.numpy().copy()
                # batch_user_index = user_ids.cpu().numpy()
                # rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0

                # ind = np.argpartition(rating_pred, -10)[:, -10:]

                # arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]

                # arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]

                # batch_pred_list = ind[
                #     np.arange(len(rating_pred))[:, None], arr_ind_argsort
                # ]

                # if i == 0:
                #     pred_list = batch_pred_list
                #     answer_list = answers.cpu().data.numpy()
                # else:
                #     pred_list = np.append(pred_list, batch_pred_list, axis=0)
                #     answer_list = np.append(
                #         answer_list, answers.cpu().data.numpy(), axis=0
                #     )

            def rmse(y_pred_arr, y_true_arr):
                return np.sqrt(((y_pred_arr - y_true_arr) ** 2).mean())

            pred_arr = np.array(pred_list)
            answer_arr = np.array(answer_list)

            score = rmse(pred_arr, answer_arr)
            
            if mode == "submission":
                return pred_list
            else:
                # return self.get_full_sort_score(epoch, answer_list, pred_list)
                post_fix = {"Epoch": epoch, "rmse":score}
                print(post_fix)
                return [score], str(post_fix)
