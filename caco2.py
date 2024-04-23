from typing import Union, cast
from tdc.single_pred import ADME
from tdc.chem_utils import MolConvert
from tdc import Evaluator

from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem import FragmentCatalog
from rdkit import RDConfig
from rdkit.Chem.Draw import rdMolDraw2D

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import to_hetero
import torch.nn.functional as F

import networkx as nx
import matplotlib.pyplot as plt

from attentive_fp import AttentiveFP
from molattention import MolAttention


def get_features_from_data(
    data: pd.DataFrame, smiles_col: str = "Drug", pbar: bool = True
) -> tuple[list, dict]:
    """Get rdkit features from a pandas DataFrame containing a column of SMILES data

    Args:
        data (pd.DataFrame): Input DataFrame
        smiles_col (str, optional): Column name with SMILES data. Defaults to "Data".
        pbar (bool, optional): Show progress bar via tqdm. Defaults to True.

    Returns:
        list: list of rdkit Features
    """
    mols = data[smiles_col].values

    mols_rd = [Chem.MolFromSmiles(x) for x in mols]

    fdef_name = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

    types = factory.GetFeatureDefs()

    mols_iter = mols_rd
    feats = []
    if pbar:
        mols_iter = tqdm(mols_rd)
    for mol in mols_iter:
        feats.append(factory.GetFeaturesForMol(mol))
    return list(feats), types


def get_functional_groups_from_data(
    data: pd.DataFrame, smiles_col: str = "Drug", pbar: bool = True
) -> tuple[list, list[str]]:
    """Get rdkit functional groups from a pandas DataFrame

    Args:
        data (pd.DataFrame): Input DataFrame
        smiles_col (str, optional): Column name with SMILES data. Defaults to "Data".
        pbar (bool, optional): Show progress bar. Defaults to True.

    Returns:
        list: list of rdkit FragCatalogs
        list[str]: list of rdkit Func Groups Names
    """
    mols = data[smiles_col].values

    mols_rd = [Chem.MolFromSmiles(x) for x in mols]

    fgroups = os.path.join(RDConfig.RDDataDir, "FunctionalGroups.txt")

    fparams = FragmentCatalog.FragCatParams(1, 10, fgroups)

    funcs = []
    for i in range(0, fparams.GetNumFuncGroups()):
        funcs.append(fparams.GetFuncGroup(i))

    fcgen = FragmentCatalog.FragCatGenerator()

    mols_iter = mols_rd
    fcats = []
    if pbar:
        mols_iter = tqdm(mols_rd)
    for mol in mols_iter:
        fcat = FragmentCatalog.FragCatalog(fparams)
        n = fcgen.AddFragsFromMol(mol, fcat)
        fcats.append((n, fcat))
    return fcats, funcs


def convert_smiles_pyg(data: pd.DataFrame, smiles_col: str = "Drug") -> list:
    """Converts a pandas DataFrame (with a SMILES column) to a list of PyG graphs

    Args:
        data (pd.DataFrame): Input Dataframe
        smiles_col (str, optional): Column name with SMILES data. Defaults to "Drug".

    Returns:
        list: list of PyG graphs
    """

    converter = MolConvert(src="SMILES", dst="PyG")

    return converter(data[smiles_col].values)


def add_label_pyg(
    graphs: Union[list[Data], list[HeteroData]], labels: list[float]
) -> Union[list[Data], list[HeteroData]]:
    """Adds labels to graphs

    Args:
        graphs (list[Data]): Input graphs
        labels (list[float]): Whole-graph label

    Returns:
        list[Data]: Graphs with labels
    """
    new_graphs = []
    for graph, label in zip(graphs, labels):
        graph.y = label
        new_graphs.append(graph)
    return new_graphs


def draw_pyg(graph: Data, name: str) -> None:
    """Draw a PyG graph, with ID labels
    For comparison with the rdkit visualization.

    Args:
        graph (torch_geometric.data.Data): Input graph
        name (str): Output filename
    """
    g = torch_geometric.utils.to_networkx(graph, to_undirected=True)
    nx.draw(g, with_labels=True)
    plt.savefig(name)


def draw_rdkit(smiles: str, name: str) -> None:
    """Draw an rdkit smiles molecule, with labels
    For comparison with the Pyg graph.

    Args:
        smiles (str): Input smiles string
        name (str): Output filename
    """
    d = rdMolDraw2D.MolDraw2DCairo(250, 250)
    d.drawOptions().addAtomIndices = True
    d.DrawMolecule(Chem.MolFromSmiles(smiles))
    d.FinishDrawing()
    d.WriteDrawingText(name)


def hash_types(types: list[str]) -> dict[str, int]:
    """Hash a list of features/functional groups to numeric codes

    Args:
        types (list[str]): List to hash

    Returns:
        dict[str, int]: Hash from types to codes
    """
    category = pd.Series(types).astype("category")

    return dict(zip(types, category.cat.codes))


def add_features_to_pyg(
    graph: Data, feature_hash: dict[str, int], features: list
) -> HeteroData:
    """Add molecule features from rdkit to a PyG graph

    Args:
        graph (torch_geometric.data.Data): Input graph
        feature_hash (dict[str, int]): Table to convert from feature type to code
        features (list): Features to add

    Returns:
        Data: Graph with features added as new nodes
    """
    # Generate the feature nodes, edges
    feat_nodes = torch.zeros(len(features), len(feature_hash))
    edges = []
    for feat_idx, feature in enumerate(features):
        feat_hashed = feature_hash[feature.GetFamily() + "." + feature.GetType()]
        feat_nodes[feat_idx, feat_hashed] = 1

        for node in feature.GetAtomIds():
            edges.append((node, feat_hashed))

    new_edges = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Make the heterogenous graph
    data = HeteroData()

    # Add the nodes
    data["molecule"].x = graph.x
    data["feature"].x = feat_nodes

    # Add the edges
    data["molecule", "conn", "molecule"].edge_index = graph.edge_index
    data["molecule", "part", "feature"].edge_index = new_edges

    return data


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_homog(train: pd.DataFrame, valid: pd.DataFrame, model_name: str = "attentivefp") -> torch.nn.Module:
    """Train the attentive fp model

    Args:
        train (pd.DataFrame): train dataset
        valid (pd.DataFrame): valid dataset

    Returns:
        torch.nn.Module: model
    """
    if model_name == "attentivefp":
        model = AttentiveFP(
            in_channels=39,
            hidden_channels=64,
            out_channels=1,
            num_layers=1,
            num_timesteps=10,
            dropout=0.0,
        )
    else:
        model = MolAttention(
            hidden_channels=64,
            out_channels=1,
            edge_dim=1,
            num_layers=1,
            num_heads=1,
            dropout=0.0
        )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()

    num_epochs = 30
    for epoch in range(0, num_epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.mse_loss(out, batch.y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} \t Loss: {loss}")

    return model


def eval_model(model: torch.nn.Module, test_graphs: list):
    """Evaluate model on test dataset

    Args:
        model (torch.nn.Module): Trained model
        test_graphs (list): Processed test graphs
    """
    model.eval()
    with torch.no_grad():
        preds = []
        for batch in test_loader:
            if INCLUDE_FEATURES:
                batch_dict = {x: batch[x].batch for x in ["molecule", "feature"]}
                out = model(batch.x_dict, batch.edge_index_dict, batch_dict)
            else:
                pred = model(batch.x, batch.edge_index, batch.batch)
            preds += pred.tolist()
        score = mean_absolute_error(test["Y"], preds)
        evaluator = Evaluator(name = 'MSE')
        # y_true: [0.8, 0.7, ...]; y_pred: [0.75, 0.73, ...]
        score = evaluator(y_true, y_pred)

        print(score)

    print(count_parameters(model))



if __name__ == "__main__":
    INCLUDE_FEATURES = True
    ATTENTIVE_FP = False

    data = ADME(name="Caco2_Wang")
    split = data.get_split()
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train, val, test = (
        cast(pd.DataFrame, split["train"]),
        cast(pd.DataFrame, split["valid"]),
        cast(pd.DataFrame, split["test"]),
    )

    types = None
    graphs = []
    tx = T.ToUndirected()
    for df in [train, val, test]:
        df = convert_smiles_pyg(df)

        if INCLUDE_FEATURES:
            if types is None:
                feats, types = get_features_from_data(df)
                types = hash_types(list(types.keys()))
            feats, _ = get_features_from_data(df)

            for idx, (graph, feat_list) in enumerate(zip(df, feats)):
                g = add_features_to_pyg(graph, types, feat_list)
                df[idx] = tx(g)

    
    graphs[0] = add_label_pyg(graphs[0], train["Y"].tolist())

    train_loader = DataLoader(graphs[0], batch_size=32)
    test_loader = DataLoader(graphs[2], batch_size=32)

    if ATTENTIVE_FP:
        model = train_homog(graphs[0], graphs[1], model_name="attentivefp")
    else:    
        if INCLUDE_FEATURES:
            pass
        else:
            pass
    

    eval_model(model, test_loader)

