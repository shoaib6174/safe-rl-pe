from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir: str = "./runs"):
        self.writer = SummaryWriter(log_dir)
        self.step = 0

    def log_scalar(self, tag: str, value: float, step: int = None):
        if step is None:
            step = self.step
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag: str, values: dict, step: int = None):
        if step is None:
            step = self.step
        self.writer.add_scalars(tag, values, step)

    def increment_step(self):
        self.step += 1

    def log_text(self, tag: str, text: str, step: int = None):
        if step is None:
            step = self.step
        self.writer.add_text(tag, text, step)

    def close(self):
        self.writer.close()
