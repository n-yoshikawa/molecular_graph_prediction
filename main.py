import torch
import torch.nn as nn
from torch.nn import Softplus
from torch.utils.data import Dataset, DataLoader
import dgl
import dgl.function as fn
from dgl.batched_graph import sum_nodes
from dgl.data.chem.utils import mol_to_complete_graph
import numpy as np
from collections import defaultdict
from rdkit import Chem

import matplotlib.pyplot as plt

class XYZDataset(Dataset):
    """Dataset of Molecules in XYZ Format."""

    def __init__(self, xyz_file):
        """
        Args:
            xyz_file (string): Path to the xyz file
        """
        self.molecule_list = []
        self.atom_counts = []
        self.atom2enum = {
            'H': 1,  # start at 1 just to be sure in cas 0 is a default embedding in DGL
            'C': 2,
            'N': 3,
            'O': 4,
            'F': 5,
            'P': 6,
            'S': 7,
            'Cl': 8,
            'Br': 9,
            'I': 10,
        }

        # XYZ format: https://openbabel.org/wiki/XYZ_(format)
        with open(xyz_file, "r") as f:
            atom_information = []
            line_number = 0
            num_atoms = 0
            for line in f:
                if line_number == 1:  # comment line
                    pass
                elif len(line.split()) == 1:  # number of molecules
                    if atom_information:  # start of next mol
                        assert len(atom_information) == num_atoms
                        self.molecule_list.append(atom_information)
                        atom_information = []
                        line_number = 0
                    num_atoms = int(line)
                else:
                    atomic_symbol, x, y, z = line.split()
                    atom_information.append((atomic_symbol, float(x), float(y), float(z)))
                line_number += 1

    def __len__(self):
        return len(self.molecule_list)

    def __getitem__(self, idx):
        mol = self.molecule_list[idx]
        atom_types = torch.tensor([self.atom2enum[atomic_symbol] for atomic_symbol, x, y, z in mol])
        dim = len(mol)
        distance_matrix = np.zeros((dim * dim, 1), dtype=np.float32)
        g = dgl.DGLGraph()
        g.add_nodes(dim)
        g.ndata['node_type'] = torch.LongTensor(atom_types)
        for i, (_, x1, y1, z1) in enumerate(mol):
            for j, (_, x2, y2, z2) in enumerate(mol):
                distance_matrix[i * dim + j, 0] = np.sqrt((x1-x2)**2.0+(y1-y2)**2.0+(z1-z2)**2.0)
                g.add_edges(i, j)
        distance_matrix = torch.tensor(distance_matrix)
        g.edata['distance'] = distance_matrix.view(-1, 1)
        return g

def get_bond_type(bond):
    if bond is None:
        return 0
    else:
        return 1
    #elif bond == Chem.rdchem.BondType.SINGLE:
    #    return 1
    #elif bond == Chem.rdchem.BondType.DOUBLE:
    #    return 2
    #elif bond == Chem.rdchem.BondType.TRIPLE:
    #    return 3
    #elif bond == Chem.rdchem.BondType.AROMATIC:
    #    return 4
    #else:
    #    return 5

def alchemy_edges(mol, self_loop=False):
    """Featurization for all bonds in a molecule.
    The bond indices will be preserved.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object
    self_loop : bool
        Whether to add self loops. Default to be False.

    Returns
    -------
    bond_feats_dict : dict
        Dictionary for bond features
    """
    bond_feats_dict = defaultdict(list)

    mol_conformers = mol.GetConformers()
    assert len(mol_conformers) == 1
    geom = mol_conformers[0].GetPositions()

    num_atoms = mol.GetNumAtoms()
    for u in range(num_atoms):
        for v in range(num_atoms):
            if u == v and not self_loop:
                continue

            e_uv = mol.GetBondBetweenAtoms(u, v)
            if e_uv is None:
                bond_type = None
            else:
                bond_type = e_uv.GetBondType()
            bond_feats_dict['e_feat'].append(get_bond_type(bond_type)
            )
            bond_feats_dict['distance'].append(
                np.linalg.norm(geom[u] - geom[v]))

    bond_feats_dict['e_feat'] = torch.LongTensor(
        np.array(bond_feats_dict['e_feat']))
    bond_feats_dict['distance'] = torch.tensor(
        np.array(bond_feats_dict['distance']).astype(np.float32)).reshape(-1 , 1)

    return bond_feats_dict

def alchemy_nodes(mol):
    """Featurization for all atoms in a molecule. The atom indices
    will be preserved.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object

    Returns
    -------
    atom_feats_dict : dict
        Dictionary for atom features
    """
    atom_feats_dict = defaultdict(list)

    num_atoms = mol.GetNumAtoms()
    for u in range(num_atoms):
        atom = mol.GetAtomWithIdx(u)
        atom_type = atom.GetAtomicNum()
        atom_feats_dict['node_type'].append(atom_type)

    atom_feats_dict['node_type'] = torch.tensor(np.array(
        atom_feats_dict['node_type']).astype(np.int64))

    return atom_feats_dict


class MoleculeDataset(Dataset):
    def __init__(self, sdf_name):
        self.graphs, self.smiles = [], []
        supp = Chem.SDMolSupplier(sdf_name)
        cnt = 0
        for mol in supp:
            cnt += 1
            if cnt > 100:
                break
            if cnt % 10 == 0:
                print('Processing molecule {:d}'.format(cnt))
            graph = mol_to_complete_graph(mol, atom_featurizer=alchemy_nodes,
                                               bond_featurizer=alchemy_edges)
            smiles = Chem.MolToSmiles(mol)
            self.smiles.append(smiles)
            self.graphs.append(graph)
        print(len(self.graphs), "loaded!")

    def __getitem__(self, item):
        """Get datapoint with index

        Parameters
        ----------
        item : int
            Datapoint index

        Returns
        -------
        str
            SMILES for the ith datapoint
        DGLGraph
            DGLGraph for the ith datapoint
        Tensor of dtype float32
            Labels of the datapoint for all tasks
        """
        return self.smiles[item], self.graphs[item]


    def __len__(self):
        """Length of the dataset

        Returns
        -------
        int
            Length of Dataset
        """
        return len(self.graphs)


class AtomEmbedding(nn.Module):
    """
    Convert the atom(node) list to atom embeddings.
    The atoms with the same element share the same initial embedding.
    Parameters
    ----------
    dim : int
        Size of embeddings, default to be 128.
    type_num : int
        The largest atomic number of atoms in the dataset, default to be 100.
    pre_train : None or pre-trained embeddings
        Pre-trained embeddings, default to be None.
    """
    def __init__(self, dim=128, type_num=100, pre_train=None):
        super(AtomEmbedding, self).__init__()

        self._dim = dim
        self._type_num = type_num
        self.embedding = nn.Embedding(type_num, dim, padding_idx=0)

    def forward(self, atom_types):
        """
        Parameters
        ----------
        atom_types : int64 tensor of shape (B1)
            Types for atoms in the graph(s), B1 for the number of atoms.
        Returns
        -------
        float32 tensor of shape (B1, self._dim)
            Atom embeddings.
        """
        return self.embedding(atom_types)

class ShiftSoftplus(Softplus):
    """
    Shiftsoft plus activation function:
        1/beta * (log(1 + exp**(beta * x)) - log(shift))
    """

    def __init__(self, beta=1, shift=2, threshold=20):
        super().__init__(beta, threshold)
        self.shift = shift
        self.softplus = Softplus(beta, threshold)

    def forward(self, input):
        return self.softplus(input) - np.log(float(self.shift))


class RBFLayer(nn.Module):
    """
    Radial basis functions Layer.
    e(d) = exp(- gamma * ||d - mu_k||^2)
    With the default parameters below, we are using a default settings:
    * gamma = 10
    * 0 <= mu_k <= 30 for k=1~300
    Parameters
    ----------
    low : int
        Smallest value to take for mu_k, default to be 0.
    high : int
        Largest value to take for mu_k, default to be 30.
    gap : float
        Difference between two consecutive values for mu_k, default to be 0.1.
        decay_rate = 0.9 * float(epoch)/max_epoch
        print(decay_rate)
        decay_rate = 0.9 * float(epoch)/max_epoch
        print(decay_rate)
    dim : int
        Output size for each center, default to be 1.
    """
    def __init__(self, low=0, high=30, gap=0.1, dim=1):
        super(RBFLayer, self).__init__()

        self._low = low
        self._high = high
        self._dim = dim

        self._n_centers = int(np.ceil((high - low) / gap))
        centers = np.linspace(low, high, self._n_centers)
        self.centers = torch.tensor(centers, dtype=torch.float, requires_grad=False)
        self.centers = nn.Parameter(self.centers, requires_grad=False)
        self._fan_out = self._dim * self._n_centers
        self._gap = centers[1] - centers[0]

    def forward(self, edge_distances):
        """
        Parameters
        ----------
        edge_distances : float32 tensor of shape (B, 1)
            Edge distances, B for the number of edges.
        Returns
        -------
        float32 tensor of shape (B, self._fan_out)
            Computed RBF results
        """
        radial = edge_distances - self.centers
        coef = -1 / self._gap
        return torch.exp(coef * (radial ** 2))


class CFConv(nn.Module):
    """
    The continuous-filter convolution layer in SchNet.
    Parameters
    ----------
    rbf_dim : int
        Dimension of the RBF layer output
    dim : int
        Dimension of output, default to be 64
    act : activation function or None.
        Activation function, default to be shifted softplus
    """

    def __init__(self, rbf_dim, dim=64, act="sp"):
        """
        Args:
            rbf_dim: the dimsion of the RBF layer
            dim: the dimension of linear layers
            act: activation function (default shifted softplus)
        """
        super(CFConv, self).__init__()

        self._rbf_dim = rbf_dim
        self._dim = dim

        if act is None:
            activation = nn.Softplus(beta=0.5, threshold=14)
        else:
            activation = act

        self.project = nn.Sequential(
            nn.Linear(self._rbf_dim, self._dim),
            activation,
            nn.Linear(self._dim, self._dim)
        )

    def forward(self, g, node_weight, rbf_out):
        """
        Parameters
        ----------
        g : DGLGraph
            The graph for performing convolution
        node_weight : float32 tensor of shape (B1, D1)
            The weight of nodes in message passing, B1 for number of nodes and
            D1 for node weight size.
        rbf_out : float32 tensor of shape (B2, D2)
            The output of RBFLayer, B2 for number of edges and D2 for rbf out size.
        """
        g = g.local_var()
        e = self.project(rbf_out)
        g.ndata['node_weight'] = node_weight
        g.edata['e'] = e
        g.update_all(fn.u_mul_e('node_weight', 'e', 'm'), fn.sum('m', 'h'))
        return g.ndata.pop('h')


class Interaction(nn.Module):
    """
    The interaction layer in the SchNet model.
    Parameters
    ----------
    rbf_dim : int
        Dimension of the RBF layer output
    dim : int
        Dimension of intermediate representations
    """

    def __init__(self, rbf_dim, dim):
        super(Interaction, self).__init__()

        self._dim = dim
        self.node_layer1 = nn.Linear(dim, dim, bias=False)
        self.cfconv = CFConv(rbf_dim, dim, Softplus(beta=0.5, threshold=14))
        self.node_layer2 = nn.Sequential(
            nn.Linear(dim, dim),
            Softplus(beta=0.5, threshold=14),
            nn.Linear(dim, dim)
        )

    def forward(self, g, n_feat, rbf_out):
        n_weight = self.node_layer1(n_feat)
        new_n_feat = self.cfconv(g, n_weight, rbf_out)
        new_n_feat = self.node_layer2(new_n_feat)
        return n_feat + new_n_feat

class SumPooling(nn.Module):
    r"""Apply sum pooling over the nodes in the graph.
    .. math::
        r^{(i)} = \sum_{k=1}^{N_i} x^{(i)}_k
    """
    def __init__(self):
        super(SumPooling, self).__init__()

    def forward(self, graph, feat):
        r"""Compute sum pooling.
        Parameters
        ----------
        graph : DGLGraph or BatchedDGLGraph
            The graph.
        feat : torch.Tensor
            The input feature with shape :math:`(N, *)` where
            :math:`N` is the number of nodes in the graph.
        Returns
        -------
        torch.Tensor
            The output feature with shape :math:`(*)` (if
            input graph is a BatchedDGLGraph, the result shape
            would be :math:`(B, *)`.
        """
        with graph.local_scope():
            graph.ndata['h'] = feat
            readout = sum_nodes(graph, 'h')
            return readout

class SchNet(nn.Module):
    """
    Schnet for feature extraction and regression to predict j-coupling constant
    """
    
    def __init__(self,
                 dim=64,
                 cutoff=5.0,
                 output_dim=1,
                 width=1,
                 n_conv=3,
                 norm=False,
                 atom_ref=None,
                 pre_train=None):
        super(SchNet, self).__init__()

        self._dim = dim
        self.cutoff = cutoff
        self.width = width
        self.n_conv = n_conv
        self.atom_ref = atom_ref
        self.norm = norm
        
        self.embedding_layer = AtomEmbedding(dim)
        self.rbf_layer = RBFLayer(0, cutoff, width)
        self.conv_layers = nn.ModuleList(
            [Interaction(self.rbf_layer._fan_out, dim) for _ in range(n_conv)])
        self.atom_update = nn.Sequential(
            nn.Linear(dim, 64),
            ShiftSoftplus(),
            nn.Linear(64, output_dim)
        )
        self.readout = nn.Sequential(
            nn.Linear(output_dim*2+self.rbf_layer._fan_out, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=0)
        )
    def forward(self, g, atom_types, edge_distances):
        """Predict molecule labels

        Parameters
        ----------
        atom_types : int64 tensor of shape (B1)
            Types for atoms in the graph(s), B1 for the number of atoms.
        edge_distances : float32 tensor of shape (B2, 1)
            Edge distances, B2 for the number of edges.
        Returns
        -------
        prediction : float32 tensor of shape (B, output_dim)
            Model prediction for the batch of graphs, B for the number
            of graphs, output_dim for the prediction size.
        """

        h = self.embedding_layer(atom_types)
        rbf_out = self.rbf_layer(edge_distances)
        for idx in range(self.n_conv):
            h = self.conv_layers[idx](g, h, rbf_out)
        h = self.atom_update(h)
        edge_pred = []
        cnt = 0
        for i, feat_1 in enumerate(h):
            for j, feat_2 in enumerate(h):
                if i==j:
                    continue
                edge_feat = torch.cat([feat_1, feat_2, self.rbf_layer(edge_distances[cnt])])
                edge_pred.append(self.readout(edge_feat))
                cnt += 1
        return torch.stack(edge_pred)

full_dataset = MoleculeDataset('platinum_dataset_2017_01.sdf')

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

model = SchNet()
loss_fn = nn.CrossEntropyLoss(reduction='mean')
loss_fn2 = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)


train_loss_list = []
test_loss_list = []
max_epoch = 20
for epoch in range(max_epoch):
    loss_list = []
    decay_rate = 0.5
    for data in train_dataset:
        smiles, graph = data
        atom_types = graph.ndata['node_type']
        edge_distances = graph.edata['distance']
        edge_types = graph.edata['e_feat']
        edge_pred = model(graph, atom_types, edge_distances)
        loss = (1.0 - decay_rate) * loss_fn(edge_pred, edge_types)
        loss += decay_rate * loss_fn2(edge_pred, edge_types)
        loss_list.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = torch.stack(loss_list).mean().item()
    train_loss_list.append(train_loss)
    print("train_loss at epoch {}:".format(epoch), train_loss)

    loss_list = []
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    accuracy_list = []
    recall_list = []
    with torch.no_grad():
        for data in test_dataset:
            smiles, graph = data
            atom_types = graph.ndata['node_type']
            edge_distances = graph.edata['distance']
            edge_types = graph.edata['e_feat']
            edge_pred = model(graph, atom_types, edge_distances)
            edge_types_pred = torch.argmax(edge_pred, dim=1)
            edge_mask = (edge_types != 0)
            print("ground truth:", edge_types)
            print("prediction:", edge_types_pred)
            print("{} / {}".format((edge_types == edge_types_pred).sum(), edge_types.shape[0]))
            print("{} / {}".format(((edge_types == edge_types_pred)*edge_mask).sum(), edge_mask.sum()))
            TP = ((edge_types == 1) * (edge_types_pred == 1)).sum()
            FP = ((edge_types == 0) * (edge_types_pred == 1)).sum()
            FN = ((edge_types == 1) * (edge_types_pred == 0)).sum()
            TN = ((edge_types == 0) * (edge_types_pred == 0)).sum()
            loss = (1.0 - decay_rate) * loss_fn(edge_pred, edge_types)
            loss += decay_rate * loss_fn2(edge_pred, edge_types)
            loss_list.append(loss)
            print("TP: {}, FP: {}, TN: {}, FN: {}".format(TP, FP, TN, FN))
            print("Accuracy: {}/{}, Recall: {}/{}".format(
                (TP+TN),(TP+FP+FN+TN),
                TP , (TP + FN)))
            accuracy_list.append(100.0*(TP+TN)/(TP+FP+FN+TN))
            recall_list.append(100.0*TP/(TP+FN))
    test_loss = torch.stack(loss_list).mean().item()
    test_loss_list.append(test_loss)
    print("result of epoch:", epoch)
    print("mean accuracy: {}%, mean recall: {}%".format(np.mean(accuracy_list), np.mean(recall_list)))

plt.plot(range(max_epoch), train_loss_list, label='train')
plt.plot(range(max_epoch), test_loss_list, label='test')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

