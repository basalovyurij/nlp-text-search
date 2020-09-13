from abc import abstractmethod
from deeppavlov.models.ranking.siamese_model import SiameseModel
from keras import losses
from keras.models import Model
from keras.optimizers import Adam
from logging import getLogger
import numpy as np
from typing import List

from .keras_model import KerasModel


log = getLogger(__name__)


class KerasSiameseModel(SiameseModel, KerasModel):
    """The class implementing base functionality for siamese neural networks in keras.
    Args:
        learning_rate: Learning rate.
        use_matrix: Whether to use a trainable matrix with token (word) embeddings.
        emb_matrix: An embeddings matrix to initialize an embeddings layer of a model.
            Only used if ``use_matrix`` is set to ``True``.
        max_sequence_length: A maximum length of text sequences in tokens.
            Longer sequences will be truncated and shorter ones will be padded.
        dynamic_batch:  Whether to use dynamic batching. If ``True``, the maximum length of a sequence for a batch
            will be equal to the maximum of all sequences lengths from this batch,
            but not higher than ``max_sequence_length``.
        attention: Whether any attention mechanism is used in the siamese network.
        *args: Other parameters.
        **kwargs: Other parameters.
    """

    def __init__(self,
                 learning_rate: float = 1e-3,
                 use_matrix: bool = True,
                 emb_matrix: np.ndarray = None,
                 max_sequence_length: int = None,
                 dynamic_batch: bool = False,
                 attention: bool = False,
                 *args,
                 **kwargs) -> None:

        super(KerasSiameseModel, self).__init__(*args, **kwargs)

        self.learning_rate = learning_rate
        self.attention = attention
        self.use_matrix = use_matrix
        self.emb_matrix = emb_matrix
        if dynamic_batch:
            self.max_sequence_length = None
        else:
            self.max_sequence_length = max_sequence_length
        self.model = self.create_model()
        self.compile()
        if self.load_path.exists():
            self.load()
        else:
            self.load_initial_emb_matrix()

        if not self.attention:
            self.context_model = self.create_context_model()
            self.response_model = self.create_response_model()

    def compile(self) -> None:
        optimizer = Adam(lr=self.learning_rate)
        loss = losses.binary_crossentropy
        self.model.compile(loss=loss, optimizer=optimizer)

    def load(self) -> None:
        log.info("[initializing `{}` from saved]".format(self.__class__.__name__))
        self.model.load_weights(str(self.load_path))

    def save(self) -> None:
        log.info("[saving `{}`]".format(self.__class__.__name__))
        self.model.save_weights(str(self.save_path))

    def load_initial_emb_matrix(self) -> None:
        log.info("[initializing new `{}`]".format(self.__class__.__name__))
        if self.use_matrix:
            self.model.get_layer(name="embedding").set_weights([self.emb_matrix])

    @abstractmethod
    def create_model(self) -> Model:
        pass

    def create_context_model(self) -> Model:
        m = Model(self.model.inputs[:-1],
                  self.model.get_layer("sentence_embedding").get_output_at(0))
        return m

    def create_response_model(self) -> Model:
        m = Model(self.model.inputs[-1],
                  self.model.get_layer("sentence_embedding").get_output_at(1))
        return m

    def _train_on_batch(self, batch: List[np.ndarray], y: List[int]) -> float:
        loss = self.model.train_on_batch(batch, np.asarray(y))
        return loss

    def _predict_on_batch(self, batch: List[np.ndarray]) -> np.ndarray:
        y_pred = self.model.predict_on_batch(batch)
        return y_pred

    def _predict_context_on_batch(self, batch: List[np.ndarray]) -> np.ndarray:
        return self.context_model.predict_on_batch(batch)

    def _predict_response_on_batch(self, batch: List[np.ndarray]) -> np.ndarray:
        return self.response_model.predict_on_batch(batch)
