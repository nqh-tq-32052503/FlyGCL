from .codaprompt import CodaPrompt
from .dualprompt import DualPrompt
from .flyprompt import FlyPrompt
from .l2p import L2P
from .mvp import MVP
from .moeranpac import MoERanPAC
from .ranpac import RanPAC
from .slca import SLCA
from .hide_norga_trainer import HiDeGCLTrainer, NoRGaGCLTrainer
from .sdlora import SDLoRAGCL

METHODS = {
    "codaprompt": CodaPrompt,
    "dualprompt": DualPrompt,
    "flyprompt": FlyPrompt,
    "l2p": L2P,
    "mvp": MVP,
    "moeranpac": MoERanPAC,
    "ranpac": RanPAC,
    "slca": SLCA,
    "hide": HiDeGCLTrainer,
    "norga": NoRGaGCLTrainer,
    "sdlora": SDLoRAGCL,
}