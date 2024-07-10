import torch
import torch.utils.data as tud


class Node:
    def __init__(self, model, x, vocabulary, device, data_device="cpu", batch_size=64):
        """
        Initialize a Node used for autoregression
        predictions, such as greedy search, multinomial
        sampling, or beam search

        Parameters:
            model (Any): any autoregressive model
            x (tuple(torch.tensor,)): a torch tensor representing
                              additional data to pass to
                              the regression model
            vocabulary (Vocabulary): a vocabulary object
            device (torch.device, str): device where to place
                                        the model
            data_device (torch.device, str): device where to place
                                             the data. WARNING! Use
                                             gpu here sparingly.
            batch_size(int): internal batch size used for the beam search
                             (Default: 64)

        """
        assert isinstance(device, torch.device) or isinstance(device, str)
        assert isinstance(data_device, torch.device) or isinstance(data_device, str)

        if isinstance(device, str):
            device = torch.device(device)

        if isinstance(data_device, str):
            data_device = torch.device(data_device)

        self.model = model
        self.device = device
        self.data_device = data_device
        src, src_mask = x
        self.batch_size = batch_size  # min(batch_size, len(src))

        with torch.no_grad():
            self.model = self.model.eval()

            if next(self.model.parameters()).device != self.device:
                self.model = self.model.to(self.device)
            if src.device != self.device:
                src = src.to(self.device)
            if src_mask.device != self.device:
                src_mask = src_mask.to(self.device)

            self.x = self.model.encode(src, src_mask).detach()
            self.x_mask = src_mask.detach()

            if self.x.device != self.data_device:
                self.x = self.x.to(self.data_device)
            if self.x_mask.device != self.data_device:
                self.x_mask = self.x_mask.to(self.data_device)

        self.vocabulary = vocabulary

        self.y = torch.ones((len(self.x), 1), dtype=torch.long) * self.vocabulary.bos_token
        self.y = self.y.detach()

        if self.y.device != self.data_device:
            self.y = self.y.to(self.data_device)

        self.ll_mask = torch.tensor([False])
        self.pos = 0

    def set_beam_width(self, beam_width):
        self.beam_width = beam_width

    def _get_topk(self, loglikelihood):
        v = loglikelihood.shape[-1]
        loglikelihood, next_chars = loglikelihood.topk(k=min(v, self.beam_width), axis=-1)
        if v < self.beam_width:
            d = self.beam_width - len(self.vocabulary)
            pl = -1e20 * torch.ones(
                (len(loglikelihood), d),
                dtype=loglikelihood.dtype,
                device=loglikelihood.device,
            )
            pc = torch.zeros(
                (len(next_chars), d),
                dtype=next_chars.dtype,
                device=loglikelihood.device,
            )
            loglikelihood = torch.cat((loglikelihood, pl), dim=-1)
            next_chars = torch.cat((next_chars, pc), dim=-1)
        return loglikelihood, next_chars

    def _init_action(self, loglikelihood):
        # Perform the first step
        loglikelihood, next_chars = self._get_topk(loglikelihood)

        self.loglikelihood = loglikelihood.view(-1, 1)
        next_chars = next_chars.view(-1, 1)

        self.y = self.y.view(len(self.y), 1, -1).repeat(1, self.beam_width, 1).view(-1, 1)
        self.x = (
            self.x[:, None].repeat(1, self.beam_width, 1, 1).view((-1,) + tuple(self.x.shape[1:]))
        )
        self.x_mask = (
            self.x_mask[:, None]
            .repeat(1, self.beam_width, 1, 1)
            .view((-1,) + tuple(self.x_mask.shape[1:]))
        )
        self.y = torch.cat((self.y, next_chars), dim=-1)

        # VERY IMPORTANT! we need a mask for
        # the log likelihood when reaching the eos
        # self.ll_mask = torch.zeros(len(self.loglikelihood), dtype=torch.bool)
        self.ll_mask = torch.any(self.y == self.vocabulary.eos_token, dim=-1)

    def get_actions(self):
        batch_size = self.batch_size
        y_mask = self.subsequent_mask(self.y.shape[-1]).to(self.device)
        next_loglikelihood = []

        local_dataset = tud.TensorDataset(self.x, self.x_mask, self.y)
        local_loader = tud.DataLoader(local_dataset, batch_size=batch_size)

        # make sure that the local_loader
        # will be iterated over only once
        iterator = iter(local_loader)

        with torch.no_grad():
            for x, x_mask, y in local_loader:
                if x.device != self.device:
                    x = x.to(self.device)
                if x_mask.device != self.device:
                    x_mask = x_mask.to(self.device)
                if y.device != self.device:
                    y = y.to(self.device)

                out = self.model.decode(x, x_mask, y, y_mask)
                ll = self.model.generator(out)[:, -1]
                next_loglikelihood.append(ll)
        next_loglikelihood = torch.cat(next_loglikelihood, axis=0)
        next_loglikelihood = next_loglikelihood.detach()
        if next_loglikelihood != self.data_device:
            next_loglikelihood = next_loglikelihood.to(self.data_device)
        return next_loglikelihood

    def action(self, next_loglikelhihood):
        if self.pos == 0:
            self._init_action(next_loglikelhihood)
        else:
            vocabulary_size = len(self.vocabulary)
            # set loglikehihood to the maxium (0)
            # when observed an eos_token
            next_loglikelhihood[self.ll_mask, :] = (
                torch.minimum(self.loglikelihood.min(), next_loglikelhihood.min()) - 1.0
            )
            next_loglikelhihood[self.ll_mask, self.vocabulary.eos_token] = 0.0
            # done

            ll = (self.loglikelihood + next_loglikelhihood).view(
                -1, self.beam_width, vocabulary_size
            )
            ll, idx = self._get_topk(ll.flatten(start_dim=1))

            # tricky indexing
            next_chars = torch.remainder(idx, vocabulary_size).flatten().unsqueeze(-1)
            best_candidates = (idx / vocabulary_size).long()
            if best_candidates.device != self.device:
                best_candidates = best_candidates.to(self.device)
            # done

            y = self.y.view(-1, self.beam_width, self.y.shape[-1])
            i = torch.arange(len(y)).unsqueeze(-1).repeat(1, self.beam_width).flatten()
            j = best_candidates.flatten()
            self.y = y[i, j].view(-1, self.y.shape[-1])

            self.y = torch.cat((self.y, next_chars), dim=-1)
            self.loglikelihood = ll.view(-1, 1)

            # update ll mask
            self.ll_mask = torch.any(self.y == self.vocabulary.eos_token, dim=-1)
        self.pos = self.pos + 1

    @staticmethod
    def subsequent_mask(size):
        "Mask out subsequent positions."
        attn_shape = (1, size, size)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
        A = subsequent_mask == 0
        return A.type(torch.long)


class Criterion:
    def __call__(self, node):
        raise NotImplementedError("Not implemented")


class MaxLength(Criterion):
    def __init__(self, max_length):
        super(MaxLength, self).__init__()
        self.max_length = max_length

    def __call__(self, node):
        return node.pos >= self.max_length


class EOS(Criterion):
    def __init__(self):
        super(EOS, self).__init__()

    def __call__(self, node):
        return torch.all(node.ll_mask).item()


class LogicalAnd(Criterion):
    def __init__(self, criteria):
        super(LogicalAnd, self).__init__()
        self.criteria = criteria

    def __call__(self, node):
        return all([c(node) for c in self.criteria])


class LogicalOr(Criterion):
    def __init__(self, criteria):
        super(LogicalOr, self).__init__()
        self.criteria = criteria

    def __call__(self, node):
        return any([c(node) for c in self.criteria])


def beamsearch(node, beamsize, stop_criterion):
    node.set_beam_width(beamsize)

    while not stop_criterion(node):
        a = node.get_actions()
        node.action(a)

    a = node.get_actions()

    end_tokens = node.vocabulary.eos_token * torch.logical_not(node.ll_mask).type(node.y.dtype)

    node.y = torch.cat((node.y, end_tokens.view(-1, 1)), dim=-1)

    ll_tail = a[torch.arange(len(a)), end_tokens] * torch.logical_not(node.ll_mask).type(a.dtype)
    node.loglikelihood = node.loglikelihood + ll_tail.view(-1, 1)
    return node
