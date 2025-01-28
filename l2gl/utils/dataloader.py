"""
Datasets loader for l2g-light, main file
"""

import datetime
from pathlib import Path
import polars as pl
import torch
import networkx as nx
import raphtory as rp
import torch_geometric.data

DATA_PATH = Path(__file__).parent.parent.parent / "data"
DATASETS = [x.stem for x in DATA_PATH.glob("*") if x.is_dir()]
EDGE_COLUMNS = {"source", "dest"}  # required columns

EdgeList = list[tuple[str, str]]


def is_graph_dataset(p: Path) -> bool:
    "Returns True if dataset represented by `p` is a valid dataset"
    return (p / (p.stem + "_edges.parquet")).exists()


class DataLoader:  # pylint: disable=too-many-instance-attributes
    """Take a dataframe representing a (temporal) graph and provide
    methods for loading the data in different formats.
    """

    def __init__(self, dset: str | Path, timestamp_fmt: str = "%Y-%m-%d"):
        if is_graph_dataset(Path(dset)):
            self.path = Path(dset)
        elif is_graph_dataset(DATA_PATH / dset):
            self.path = DATA_PATH / dset
        else:
            raise ValueError(f"Dataset either invalid or not found: {dset}")

        self.timestamp_fmt = timestamp_fmt
        self.paths = {"edges": self.path / (self.path.stem + "_edges.parquet")}
        if (nodes_path := self.path / (self.path.stem + "_nodes.parquet")).exists():
            self.paths["nodes"] = nodes_path

        self._load_files()

    def timestamp_from_string(self, ts: str) -> datetime.datetime:
        "Returns timestamp from string using `timestamp_fmt`"
        return datetime.datetime.strptime(ts, self.timestamp_fmt)

    def _load_files(self):
        "Loads dataset into memory"

        self.edges = pl.read_parquet(self.paths["edges"])
        assert EDGE_COLUMNS <= set(
            self.edges.columns
        ), f"Required edge columns not found: {EDGE_COLUMNS}"
        self.temporal = "timestamp" in self.edges.columns
        if not self.temporal:
            self.edges = self.edges.with_columns(pl.lit(0).alias("timestamp"))
        else:  # convert timestamp to datetime format
            if self.edges["timestamp"].dtype == pl.Utf8:
                self.edges = self.edges.with_columns(
                    pl.col("timestamp").str.to_datetime(self.timestamp_fmt)
                )

        self.datelist = self.edges.select("timestamp").to_series().unique()

        # Process nodes
        if self.paths.get("nodes"):
            self.nodes = pl.read_parquet(self.paths["nodes"])
            assert (
                "nodes" in self.nodes.columns
            ), "Required node columns not found: 'nodes'"
            if self.temporal:
                if "timestamp" not in self.nodes.columns:
                    raise ValueError(
                        "Nodes dataset missing 'timestamp' column, required"
                        " when edges dataset has 'timestamp'"
                    )
                if self.nodes["timestamp"].dtype == pl.Utf8:
                    self.nodes = self.nodes.with_columns(
                        pl.col("timestamp").str.to_datetime(self.timestamp_fmt)
                    )
        else:
            # build nodes from edges dataset
            self.nodes = (
                pl.concat(
                    [
                        self.edges.select(
                            pl.col("timestamp"), pl.col("source").alias("nodes")
                        ),
                        self.edges.select(
                            pl.col("timestamp"), pl.col("dest").alias("nodes")
                        ),
                    ]
                )
                .unique()
                .sort(by=["timestamp", "nodes"])
            )

        self.edge_features = [
            x
            for x in self.edges.columns
            if x not in ["timestamp", "label"] + sorted(EDGE_COLUMNS)
        ]
        self.node_features = [
            x for x in self.nodes.columns if x not in ["timestamp", "label", "nodes"]
        ]

    def get_dates(self) -> list[str]:
        "Returns list of dates"
        return self.datelist.to_list()

    def get_edges(self) -> pl.DataFrame:
        "Returns edges as a polars DataFrame"
        return self.edges

    def get_nodes(self, ts: str | None = None) -> pl.DataFrame:
        """Returns node data as a polars DataFrame

        Args:
            ts (str, optional): if specified, only return nodes with this timestamp

        Returns:
            polars.DataFrame
        """
        if ts is None:
            return self.nodes
        if isinstance(ts, str):
            ts = self.timestamp_from_string(ts)
        return self.nodes.filter(pl.col("timestamp") == ts)

    def get_node_list(self, ts: str | None = None) -> list[str]:
        """Returns node list

        Args:
            ts (str, optional): if specified, only return nodes with this timestamp

        Returns:
            list of str
        """
        nodes = self.nodes

        if ts is not None:
            if isinstance(ts, str):
                ts = self.timestamp_from_string(ts)
            nodes = nodes.filter(pl.col("timestamp") == ts)
        return nodes.select("nodes").unique(maintain_order=True).to_series().to_list()

    def get_node_features(self) -> list[str]:
        "Returns node features as a list of strings"
        return self.node_features

    def get_edge_features(self) -> list[str]:
        "Returns edge features as a list of strings"
        return self.edge_features

    def get_graph(self) -> rp.Graph:  # pylint: disable=no-member
        "Returns a raphtory.Graph representation"
        g = rp.Graph()  # pylint: disable=no-member

        g.load_edges_from_pandas(
            df=self.edges.to_pandas(),
            time="timestamp",
            src="source",
            dst="dest",
            properties=self.edge_features,
        )
        g.load_nodes_from_pandas(
            df=self.nodes.to_pandas(),
            time="timestamp",
            id="nodes",
            properties=self.node_features,
        )

        return g

    def get_edge_list(
        self, temp: bool = True
    ) -> EdgeList | dict[datetime.datetime, EdgeList]:
        """Returns edge list

        Args:
            temp (bool, optional, default=True): If true, then returns a dictionary of
                timestamps to edge lists (list of string tuples), if false, returns
                edge list for the entire graph
        """
        if self.temporal and temp:
            edge_list = {}
            for d in tqdm(self.datelist):
                edges = (
                    self.edges.filter(pl.col("timestamp") == d)
                    .select("source", "dest")
                    .to_numpy()
                )
                edge_list[d] = [tuple(x) for x in edges]
        else:
            edges = self.edges.select("source", "dest").unique().to_numpy()
            edge_list = [tuple(x) for x in edges]
        return edge_list

    def get_networkx(
        self, temp: bool = True
    ) -> nx.Graph | dict[datetime.datetime, nx.Graph]:
        """Returns networkx.DiGraph representation

        Args:
            temp (bool, optional, default=True): If true, then returns a dictionary of
                timestamps to networkx digraphs, if false, returns a networkx digraph
        """

        if self.temporal and temp:
            nx_graphs: dict[datetime.datetime, nx.Graph] = {}
            for d in tqdm(self.datelist):
                edges = (
                    self.edges.filter(pl.col("timestamp") == d)
                    .select("source", "dest")
                    .to_numpy()
                )
                edge_list = [tuple(x) for x in edges]
                nx_graphs[d] = nx.from_edgelist(edge_list, create_using=nx.DiGraph)
            return nx_graphs
        edges = self.edges.select("source", "dest").unique().to_numpy()
        edge_list = [tuple(x) for x in edges]
        return nx.from_edgelist(edge_list, create_using=nx.DiGraph)

    def get_edge_index(
        self, temp: bool = True
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Returns edge index as torch tensors

        Args:
            temp (bool, optional, default=True): If true, then returns a dictionary of
                timestamps to torch tensors (list of string tuples), if false, returns
                a torch tensor.
        """
        if self.temporal and temp:
            edge_index = {}
            for d in tqdm(self.datelist):
                edges = (
                    self.edges.filter(pl.col("timestamp") == d)
                    .select("source", "dest")
                    .to_numpy()
                )
                edge_list = [tuple(x) for x in edges]
                edge_index[d] = (
                    torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                )
        else:
            edges = self.edges.select("source", "dest").unique().to_numpy()
            edge_list = [tuple(x) for x in edges]
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        return edge_index

    def get_tgeometric(
        self, temp: bool = True
    ) -> torch_geometric.data.Data | dict[datetime.datetime, torch_geometric.data.Data]:
        """Returns torch_geometric representation

        Args:
            temp (bool, optional, default=True): If true, then returns a dictionary of
                timestamps to torch_geometric representations, if false, returns
                a torch_geometric representation.
        """
        nodes = self.nodes.select("nodes").unique().to_numpy()
        features = self.nodes.select(self.node_features).to_numpy()
        if self.temporal and temp:
            tg_graphs = {}
            for d in tqdm(self.datelist):
                edges = (
                    self.edges.filter(pl.col("timestamp") == d)
                    .select("source", "dest")
                    .to_numpy()
                )
                edge_list = [tuple(x) for x in edges]
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                tg = torch_geometric.data.Data(edge_index=edge_index)
                tg.nodes = torch.from_numpy(nodes).int()
                tg.x = torch.from_numpy(features).float()
                tg_graphs[d] = tg
        else:
            edges = self.edges.select("source", "dest").unique().to_numpy()
            edge_list = [tuple(x) for x in edges]
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            tg_graphs = torch_geometric.data.Data(edge_index=edge_index)
            tg_graphs.nodes = torch.Tensor(nodes).int()
            tg_graphs.x = torch.from_numpy(features).float()
        return tg_graphs


# TODO: integrate summary() into DataLoader
