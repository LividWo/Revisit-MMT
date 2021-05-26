import argparse
import torch


if __name__ == '__main__':
    # visual_awareness()
    parser = argparse.ArgumentParser(description='')
    # fmt: off
    parser.add_argument('--input', required=True)
    # fmt: on
    args = parser.parse_args()

    model_state_dict = torch.load(args.input, map_location="cpu")
    model_state_dict = model_state_dict['model']
    norm  = 0.
    filter_list = [
        'encoder.embed_positions._float_tensor', 'encoder.visual_features.weight',
        'encoder.dense.weight', 'encoder.gate_dense.weight', 'decoder.embed_tokens.weight',
        'decoder.embed_positions._float_tensor'
    ]
    ## encoder decoder vocab are the same, so only add once
    for k, v in model_state_dict.items():
        if k not in filter_list:
            v_norm = torch.norm(v, dim=-1).sum()
            # print(k, v_norm.item())
            norm += v_norm.item()
    print(norm)
    