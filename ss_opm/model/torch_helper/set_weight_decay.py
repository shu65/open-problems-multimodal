from torch import nn

from ss_opm.model.encoder_decoder.cite_encoder_decoder_module import CiteEncoderDecoderModule
from ss_opm.model.encoder_decoder.multi_encoder_decoder_module import MultiEncoderDecoderModule


def set_weight_decay(module, weight_decay):
    params_decay = []
    params_no_decay = []
    ignoring_params = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            params_decay.append(m.weight)
            if m.bias is not None:
                params_no_decay.append(m.bias)
        elif isinstance(m, nn.Conv1d):
            params_decay.append(m.weight)
            if m.bias is not None:
                params_no_decay.append(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
            if m.bias is not None:
                params_no_decay.append(m.weight)
            if m.bias is not None:
                params_no_decay.append(m.bias)
        elif isinstance(m, (CiteEncoderDecoderModule, MultiEncoderDecoderModule)):
            ignoring_params.append(m.inputs_decomposer_components)
            ignoring_params.append(m.targets_decomposer_components)
            if hasattr(m, "targets_global_median"):
                ignoring_params.append(m.targets_global_median)
            ignoring_params.append(m.y_loc)
            ignoring_params.append(m.y_scale)
            params_no_decay.append(m.gender_embedding)
    assert len(list(module.parameters())) == len(params_decay) + len(params_no_decay) + len(ignoring_params)
    params = [dict(params=params_decay, weight_decay=weight_decay), dict(params=params_no_decay, weight_decay=0.0)]

    return params
