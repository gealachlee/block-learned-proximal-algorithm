from .block_fista import FISTA
from .block_lista import LISTA
from .block_ista import ISTA
from .block_lista_cp import LISTA_CP
from .block_alista import ALISTA
from .block_tilista import TiLISTA
from .mix_lista import MIX_LISTA
from .mix_fista import MIX_FISTA
from common.model import UnrollAlgorithm
from .block_lista_cp2 import LISTA_CP2


class ModelFactory:
    def get_model_list(self):
        return [
            'ISTA',
            'FISTA',
            'LISTA',
            'LISTA_CP',
            'TiLISTA',
            'ALISTA',
            'MIX_LISTA',
            'MIX_FISTA',
            'LISTA_CP2'
        ]

    @staticmethod
    def create_ALISTA(A, layers, tau, prox_cl, **opts):
        return ALISTA(A, layers, tau, prox_cl, **opts)
    @staticmethod
    def create_MIX_FISTA(A, layers, tau, prox_cl1,prox_cl2, **opts):
        return MIX_FISTA(A, layers, tau, prox_cl1, **opts)
    @staticmethod
    def create_MIX_LISTA(A, layers, tau, prox_cl1,prox_cl2, **opts):
        return MIX_LISTA(A, layers, tau, prox_cl1,prox_cl2, **opts)

    @staticmethod
    def create_TiLISTA(A, layers, tau, prox_cl, **opts):
        return TiLISTA(A, layers, tau, prox_cl, **opts)

    @staticmethod
    def create_LISTA_CP2(A, layers, tau, prox_cl, **opts):
        return LISTA_CP2(A, layers, tau, prox_cl, **opts)
    @staticmethod
    def create_ISTA(A, layers, tau, prox_cl, **opts):
        return ISTA(A, layers, tau, prox_cl, **opts)

    @staticmethod
    def create_LISTA_CP(A, layers, tau, prox_cl, **opts):
        return LISTA_CP(A, layers, tau, prox_cl, **opts)

    @staticmethod
    def create_FISTA(A, layers, tau, prox_cl, **opts):
        return FISTA(A, layers, tau, prox_cl, **opts)

    @staticmethod
    def create_LISTA(A, layers, tau, prox_cl, **opts):
        return LISTA(A, layers, tau, prox_cl, **opts)

    def create(self, model_name: str, *args, **kwargs) -> UnrollAlgorithm:
        assert model_name in self.get_model_list()
        model = getattr(self, f'create_{model_name}')(*args, **kwargs)
        return model
