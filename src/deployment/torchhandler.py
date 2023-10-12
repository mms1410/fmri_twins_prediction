"""Handler that should be used to convert input and outupt for torchserve."""
from ts.torch_handler.base_handler import BaseHandler

from src.data.twins_graphs_dataset import TwinsConnectomeDataset


class TwinsConnectomeHandler(BaseHandler):
    """Class to handle inputs and outputs for torchserve.

    Args:
        BaseHandler (_type_): _description_
    """

    def preprocess(self, pair_number):
        """
        Preprocess the input data before inference.

        Args:
            pair_number: number of the twin pair

        Returns:
            _type_: preprocessed data
        """
        twins_dataset = TwinsConnectomeDataset()
        graph1, graph2 = twins_dataset[pair_number][0]
        graph1 = [graph1]
        graph2 = [graph2]
        input_graphs = (graph1, graph2)
        preprocessed_data = input_graphs
        return preprocessed_data

    def postprocess(self, twin_prob):
        """Postprocess the model's output data.

        Args:
            twin_prob (_type_):probability whether the pair is a twin

        Returns:
            _type_: float of the twin probability
        """
        return float(twin_prob["output"])


_service = TwinsConnectomeHandler()
