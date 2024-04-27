from .py.msw_msa_attention import ApplyMSWMSAAttention
from .py.raunet import ApplyRAUNet

NODE_CLASS_MAPPINGS = {
    "ApplyMSWMSAAttention": ApplyMSWMSAAttention,
    "ApplyRAUNet": ApplyRAUNet,
}
