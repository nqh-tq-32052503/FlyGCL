from .codaprompt import CodaPrompt
from .dualprompt import DualPrompt
from .flyprompt import FlyPrompt
from .l2p import L2P
from .mvp import MVP
from .moeranpac import MoERanPAC
from .ranpac import RanPAC
from .hide_norga_prefix_vit import HiDePrefixModel, NoRGaPrefixModel
from .sdlora import SDLoRAModel
from .sprompt import SPrompt

MODELS = {
    "codaprompt": CodaPrompt,
    "dualprompt": DualPrompt,
    "flyprompt": FlyPrompt,
    "l2p": L2P,
    "mvp": MVP,
    "moeranpac": MoERanPAC,
    "ranpac": RanPAC,
    "hide": HiDePrefixModel,
    "norga": NoRGaPrefixModel,
    "sdlora": SDLoRAModel,
    "sprompt": SPrompt,
}