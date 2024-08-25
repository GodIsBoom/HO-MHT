import sys
from os.path import realpath, dirname
import torch
import os
import numpy as np
from torch_geometric.data import Data,Dataset
from torch_geometric.loader import DataLoader

from .GNBlock import _Model

torch.set_grad_enabled(False)

class Adap_TopK_Graph(torch.nn.Module):
    def __init__(self,step):
        super(Adap_TopK_Graph, self).__init__()
        self.step = step
    def knn_idx(self,distance_matrix,k):
        _, index = distance_matrix.sort(dim=1)

        return index[:, :k]
    def build_graph(self, distance_matrix,target_matrix):
        row_size,col_size=distance_matrix.shape
        k = min(row_size, 10 + self.step * int(row_size / 10))

        idx_knn=self.knn_idx(distance_matrix,k).reshape(-1,1)
        idx_row=torch.arange(0,row_size).view(-1,1).repeat(1,k).reshape(-1,1)
        idx_row=idx_row.type(torch.int64)

        edge_index=torch.cat((idx_row,idx_knn+row_size),dim=1).type(torch.long)
        edge_attr=distance_matrix[idx_row,idx_knn]
        ground_truth=target_matrix[idx_row,idx_knn]

        edge_attr=torch.cat((edge_attr,edge_attr),dim=1).view(-1,1)
        edge_index=torch.cat((edge_index,edge_index[:,1].unsqueeze(1),edge_index[:,0].unsqueeze(1)),dim=1)
        edge_index=edge_index.view(-1,2).permute(1,0)

        return edge_index,edge_attr,idx_row,idx_knn,k

    def forward(self,distance_matrix,target):
        gt_cost = torch.sum(distance_matrix * target)
        edge_index,edge_attr,idx_row,idx_knn,k=self.build_graph(distance_matrix,target)

        if torch.cuda.is_available():
            gt_cost = gt_cost.cuda()

        if torch.cuda.is_available():
            data = Data(x=torch.zeros((sum(distance_matrix.shape), 8)).cuda(), edge_index=edge_index.cuda(),
                        edge_attr=edge_attr.cuda(), y=target.view(-1, 1).cuda(),
                        kwargs=[distance_matrix.shape[0], k, idx_row, idx_knn, gt_cost.cuda(),edge_attr.shape[0]],
                        cost_vec = distance_matrix.view(-1,1).cuda())
        else:
            data=Data(x=torch.zeros((sum(distance_matrix.shape),8)),edge_index=edge_index,
                  edge_attr=edge_attr,y=target.view(-1,1),
                      kwargs=[distance_matrix.shape[0],k,idx_row,idx_knn, gt_cost,edge_attr.shape[0]],
                      cost_vec = distance_matrix.view(-1,1))

        return data

def sinkhorn_v1_np(mat):

    for _ in range(5):
        row_sum = np.expand_dims(np.sum(mat, axis=1), axis=1)
        mat = mat/ row_sum

        col_sum = np.expand_dims(np.sum(mat, axis = 0), axis = 0)
        mat = mat/col_sum

    return mat

class MatrixRealData(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, data_in, n_step = 2):
        super(MatrixRealData, self).__init__()#Initialization
        self.data = data_in
        self.transor = Adap_TopK_Graph(step=n_step)
        self.size = 1

    def __len__(self):
        'Denotes the total number of samples'
        return self.size

    def __getitem__(self, index):
        'Generates one sample of data'

        # Load data and get label
        matrix = self.data
        # print("see ori: ", matrix)
        matrix = torch.from_numpy(matrix.astype(np.float32))
        target = torch.zeros_like(matrix)
        # print("see ori: ", target)

        matrix_target = self.transor(matrix, target)

        return matrix,target,matrix_target
    def len(self) -> int:
        r"""Returns the number of graphs stored in the dataset."""
        return 1

    def get(self, idx: int):
        r"""Gets the data object at index :obj:`idx`."""
        return None

class GLAN4MHT():
    def __init__(self) -> None:
        self.model_file = "./glan/glan4mht.pth"
        self.model = _Model(layer_num = 5, edge_dim = 16, node_dim = 8)
        self.model.load_state_dict(torch.load(self.model_file))
        self.model.cuda()
        self.model.eval()

    def infer(self, input):
        tst_dataset = MatrixRealData(input, n_step = 2)
        tst_dataloader = DataLoader(tst_dataset, batch_size=1, shuffle=False)
        # 将 DataLoader 转换为迭代器
        loader_iter = iter(tst_dataloader)

        # 获取第一个批次
        cur_data = next(loader_iter)
        # print(cur_data)
        matrix, target,Dt_target = cur_data

        shapes_info=Dt_target.kwargs[0]

        shape, k, idx_row, idx_knn,_ , num_edges = shapes_info
        target=Dt_target['y'].view(-1,shape,shape)
        # print(Dt_target)
        pred = self.model(Dt_target)
        # print("see ori: ", pred.shape)
        pred=pred.view(-1,1)
        tag_scores = torch.zeros((1, shape, shape)).cuda()
        # print("debug", pred.shape, idx_row.shape, idx_knn.shape)
        tag_scores[:, idx_row, idx_knn] = pred

        matrix, target, tag_scores = matrix.squeeze(0), target.squeeze(0), tag_scores.squeeze(0)
        cost, target, pred_matrix = matrix.data.cpu().numpy(), target.data.cpu().numpy(), tag_scores.data.cpu().numpy()
        pred_matrix = sinkhorn_v1_np(pred_matrix + 1e-9)
        h, l = pred_matrix.shape
        prediction = np.zeros_like(pred_matrix)
        for hh in range(h):
            row, col = np.unravel_index(np.argmax(pred_matrix), pred_matrix.shape)
            prediction[row, col] = 1
            pred_matrix[row, :] = 0
            pred_matrix[:, col] = 0
            if np.sum(pred_matrix) == 0:
                break
        rows, cols = np.where(prediction == 1)
        return rows, cols


if __name__ == '__main__':
    def unit_test():
        glan = GLAN4MHT()
        mat = np.array([[0, 1, 3],
                        [1, 2, 1],
                        [3, 0, 1]])
        rows, cols = glan.infer(mat)
        print(rows, cols)

    unit_test()