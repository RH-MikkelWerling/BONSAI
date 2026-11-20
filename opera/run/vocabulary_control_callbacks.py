"""Opt-in scalar epoch counters for the exploratory vocabulary control."""
from lightning.pytorch.callbacks import Callback


class EpochTrainingCounters(Callback):
    def on_train_epoch_start(self,trainer,pl_module):
        self.tokens=int(pl_module.training_tokens_seen.item())
        self.examples=int(pl_module.training_examples_seen.item())
        self.steps=int(trainer.global_step)

    def on_train_epoch_end(self,trainer,pl_module):
        metrics={'epoch':int(trainer.current_epoch),'completed_epoch':int(trainer.current_epoch)+1,
            'audit/tokens_this_epoch':int(pl_module.training_tokens_seen.item())-self.tokens,
            'audit/examples_this_epoch':int(pl_module.training_examples_seen.item())-self.examples,
            'audit/optimizer_steps_this_epoch':int(trainer.global_step)-self.steps,
            'audit/tokens_cumulative':int(pl_module.training_tokens_seen.item())}
        for logger in trainer.loggers:logger.log_metrics(metrics,step=int(trainer.global_step))
