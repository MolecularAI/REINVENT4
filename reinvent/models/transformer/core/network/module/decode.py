import torch
from torch.autograd import Variable

from reinvent.models.transformer.core.network.module.subsequent_mask import subsequent_mask


def decode(model, src, src_mask, max_len, device, decode_type):
    """[summary]

    Args:
        max_len (int): max length for produced sequence (coded sequence)
        type (string): decode type, either greedy or multinomial

    Returns:
        ys (tensor): (batch_size, up to max length) the optimized molecules coded sequence
                      need decode and untokenization to produce smiles
    """

    ys = torch.ones(1)
    ys = ys.repeat(src.shape[0], 1).view(src.shape[0], 1).type_as(src.data)
    # ys shape [batch_size, 1]
    encoder_outputs = model.encode(src, src_mask)
    break_condition = torch.zeros(src.shape[0], dtype=torch.bool)
    # break_condition = np.zeros((src.shape[0]), dtype=bool)

    nlls = torch.zeros(src.shape[0]).to(device)
    loss = torch.nn.NLLLoss(reduction="none", ignore_index=0)  # note 0 is padding
    for i in range(max_len - 1):
        with torch.no_grad():
            out = model.decode(
                encoder_outputs,
                src_mask,
                Variable(ys),
                Variable(subsequent_mask(ys.size(1)).type_as(src.data)),
            )
            # (batch, seq, voc) need to exclude the probability of the start token "1"
            log_prob = model.generator(out[:, -1])
            prob = torch.exp(log_prob)

            assert len(log_prob.shape) == 2

            if decode_type == "greedy":
                _, next_word = torch.max(prob, dim=1)
                ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)  # [batch_size, i]

                logprob = torch.index_select(log_prob, 1, next_word)
                nlls += loss(log_prob, next_word)
            elif decode_type == "multinomial":
                next_word = torch.multinomial(prob, 1)
                ys = torch.cat([ys, next_word], dim=1)  # [batch_size, i]
                next_word = torch.reshape(next_word, (next_word.shape[0],))

                logprob = torch.index_select(log_prob, 1, next_word)
                nlls += loss(log_prob, next_word)

            # next_word = np.array(next_word.to('cpu').tolist())
            break_condition = break_condition | (next_word.to("cpu") == 2)
            if all(break_condition):  # end token
                break

    return ys, nlls


def likelihood_calculation(model, src, src_mask, ys, grad=False):
    loss = torch.nn.NLLLoss(reduction="none", ignore_index=0)

    if grad == True:
        model.train()  # this is used for agent
        encoder_outputs = model.encode(src, src_mask)
        out = model.decode(
            encoder_outputs,
            src_mask,
            Variable(ys),
            Variable(subsequent_mask(ys.size(1)).type_as(src.data)),
        )
        log_prob = model.generator(out).transpose(1, 2)  # (batch, seq, voc)
    else:
        model.eval()  # this is used for prior which dont need to learn
        with torch.no_grad():
            encoder_outputs = model.encode(src, src_mask)
            out = model.decode(
                encoder_outputs,
                src_mask,
                Variable(ys),
                Variable(subsequent_mask(ys.size(1)).type_as(src.data)),
            )
            log_prob = model.generator(out).transpose(1, 2)  # (batch, seq, voc)
    nll = loss(log_prob[:, :, :-1], ys[:, 1:]).sum(dim=1)
    return nll
    # :-1 because it also calculates the next token likelihood of input ys
    # 1: because the starting token is always the same "1"
