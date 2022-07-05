def build_concatenation(config: dict, with_multi_scale_wrapper=True):
    from torch.nn import MultiheadAttention
    dim =  config['transformer']['backbone']['dim']
    # template_shape = config['transformer']['backbone']['template']['shape'] = [7, 7]

    return MultiheadAttention(dim,1)
    # head_config = config['head']
    # head_type = head_config['type']