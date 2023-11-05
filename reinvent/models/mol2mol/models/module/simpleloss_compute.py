class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, loss_function, opt):
        self.generator = generator
        self.loss_function = loss_function
        self.opt = opt

    def __call__(self, x, y, norm):

        x = self.generator(x)

        loss = (
            self.loss_function(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        )

        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()

        return loss.data * norm  # loss.data  making the tensor discounted from backprob
