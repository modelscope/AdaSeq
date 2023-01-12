import torch


def bio2span(sequences, device, id_to_label):
    """label sequence to span format"""

    bsz = sequences.size(0)

    all_spans = [[] for _ in range(bsz)]

    for b in range(len(all_spans)):
        all_spans[b] = convert2span(sequences[b].tolist(), id_to_label)

    max_num_spans = max(len(i) for i in all_spans)

    if max_num_spans == 0:
        max_num_spans = 1

    pred_mentions = [i + (max_num_spans - len(i)) * [[-1, -1]] for i in all_spans]

    pred_mentions = torch.tensor(pred_mentions).to(device)
    pred_mask = pred_mentions[..., 0] != -1

    return pred_mentions, pred_mask


def convert2span(label_list, id_to_label):
    """convert label sequence to span"""
    span_list = []
    i = 0
    label_list = [id_to_label[item] for item in label_list]
    label_list = fix_tag_sequence_error(label_list)
    while i < len(label_list):
        if label_list[i][0] == 'B' or label_list[i][0] == 'I':
            start_idx = i
            i += 1
            while i < len(label_list) and not (label_list[i][0] == 'O' or label_list[i][0] == 'E'):
                i += 1
            if i < len(label_list):
                end_idx = i if label_list[i][0] == 'E' else i - 1
                # Looks like a good trick
                span_list.append([start_idx, end_idx])
                i += 1
            else:
                span_list.append([start_idx, i - 1])

        elif label_list[i][0] == 'S' or label_list[i][0] == 'E':
            span_list.append([i, i])
            i += 1
        else:
            i += 1

    return span_list


def fix_tag_sequence_error(tag_seq):
    """fix label sequence errors"""
    fixed_tag_seq = []
    last_type = 'O'
    for t in tag_seq:
        if t.startswith('E-'):
            t = t.replace('E-', 'I-')
        if t.startswith('S-'):
            t = t.replace('S-', 'B-')
        if t == 'O':
            fixed_tag_seq.append(t)
        elif t.startswith('B-'):
            fixed_tag_seq.append('B-' + t[2:])
            last_type = t[2:]
        elif t.startswith('I-'):
            cur_type = t[2:]
            if cur_type == last_type:
                fixed_tag_seq.append('I-' + cur_type)
            else:
                fixed_tag_seq.append('B-' + cur_type)
                last_type = cur_type
    return fixed_tag_seq
