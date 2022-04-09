import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as k
from tensorflow.keras.callbacks import LearningRateScheduler
from eval.voc import Evaluate


class History(tf.keras.callbacks.Callback):
    def __init__(self, lr_scheduler=None, wd_scheduler=None):
        super(History, self).__init__()
        self.lr_scheduler = lr_scheduler
        self.wd_scheduler = wd_scheduler

    def on_epoch_begin(self, epoch, logs=None):
        if self.lr_scheduler:
            k.set_value(self.model.optimizer.lr, self.lr_scheduler(epoch))

        lr = float(k.get_value(self.model.optimizer.lr))

        if self.wd_scheduler:
            try:
                k.set_value(self.model.optimizer.weight_decay, self.wd_scheduler(epoch))
                wd = float(k.get_value(self.model.optimizer.weight_decay))
                print(f"[INFO] From History Callback: EP:{epoch} LR: {lr}, WD: {wd}")
            except:
                print(f"[INFO] From History Callback: EP:{epoch} LR: {lr}")


def create_callbacks(
        config,
        pred_mod,
        test_gen,
):
    cbs = []
    info_ = ''

    if config.EVALUATION:
        info_ = info_ + "Evaluation, "
        cbs.append(
            Evaluate(test_gen, pred_mod)
        )

    if config.TENSORBOARD:
        info_ = info_ + "Tensorboard, "

        cbs.append(
            keras.callbacks.TensorBoard(
                log_dir='./logs',
            )
        )

    if config.USING_HISTORY:
        info_ = info_ + "History, "

        _cosine_decay_dict = {
            "total_epochs": config.EPOCHs,
            "warm_up": config.USING_WARMUP,
            "warm_up_epochs": config.WP_EPOCHs,
            "warm_up_ratio": config.WP_RATIO,
            "alpha": config.ALPHA,
        }

        _steps_decay_dict = {
            "total_epochs": config.EPOCHs,
            "warm_up": config.USING_WARMUP,
            "warm_up_epochs": config.WP_EPOCHs,
            "warm_up_ratio": config.WP_RATIO
        }

        if config.LR_Scheduler == 1:
            cbs.append(
                History(
                    lr_scheduler=cosine_decay_scheduler(
                        param="LR",
                        init_val=config.BASE_LR,
                        **_cosine_decay_dict
                    ),
                    wd_scheduler=cosine_decay_scheduler(
                        param="WD",
                        init_val=config.DECAY,
                        **_cosine_decay_dict
                    )
                )
            )

        elif config.LR_Scheduler == 2:
            cbs.append(
                History(
                    lr_scheduler=steps_decay_scheduler(
                        init_val=config.BASE_LR,
                        **_steps_decay_dict
                    ),
                    wd_scheduler=steps_decay_scheduler(
                        init_val=config.DECAY,
                        **_steps_decay_dict
                    )
                )
            )

    if config.EARLY_STOPPING:
        # Early stopping.
        cbs.append(
            keras.callbacks.EarlyStopping(
                monitor='loss',
                mode='min',
                patience=3,
                restore_best_weights=True,
                verbose=1,
            )
        )
        print(info_ + 'EarlyStopping')
    return cbs


def cosine_decay_scheduler(
        param: str,
        total_epochs=100,
        init_val=1e-4,
        warm_up=0,
        warm_up_epochs=0,
        warm_up_ratio=0.1,
        alpha=0.,
        using_keras=0,
):
    t_eps = (total_epochs - warm_up_epochs) - 1 if warm_up else total_epochs
    min_val = warm_up_ratio * init_val

    def _warm_up(current_epoch):
        decay = min_val + (init_val - min_val) * (current_epoch / warm_up_epochs)
        print(f'-- [INFO] Warmup ({current_epoch + 1}/{warm_up_epochs}) {param} : {decay}')
        return decay

    def _cosine_decay(current_epoch):
        c_ep = (current_epoch - warm_up_epochs) if warm_up else current_epoch
        cosine = 0.5 * (1 + np.cos(np.pi * c_ep / t_eps))
        cosine = (1 - alpha) * cosine + alpha
        decay = init_val * cosine
        print(f'-- [INFO] Cosine-Decay {param} : {decay}')
        return decay

    def _output(epoch):
        return _warm_up(epoch) if warm_up and epoch < warm_up_epochs else _cosine_decay(epoch)

    if using_keras:
        def _scheduler(epoch, lr):
            return _output(epoch)

        return LearningRateScheduler(_scheduler)

    else:
        def _scheduler(epoch):
            return _output(epoch)

        return _scheduler


def steps_decay_scheduler(
        total_epochs=100,
        init_val=1e-4,
        warm_up=0,
        warm_up_epochs=0,
        warm_up_ratio=0.1,
        using_keras=0,
):
    min_val = warm_up_ratio * init_val

    def _warm_up(current_epoch):
        decay = min_val + (init_val - min_val) * (current_epoch / warm_up_epochs)
        print(f'-- [INFO] Warmup ({current_epoch + 1}/{warm_up_epochs}) : {decay}')
        return decay

    def _steps_decay(current_epoch):
        decay = init_val

        if (current_epoch / total_epochs) >= .9:
            decay = init_val * (.5 ** 3)

        elif (current_epoch / total_epochs) >= .75:
            decay = init_val * (.5 ** 2)

        elif (current_epoch / total_epochs) >= .5:
            decay = init_val * (.5 ** 1)

        print(f'-- [INFO] Steps-Decay : {decay}')
        return decay

    def _output(epoch):
        return _warm_up(epoch) if warm_up and epoch < warm_up_epochs else _steps_decay(epoch)

    if using_keras:
        def _scheduler(epoch, lr):
            return _output(epoch)

        return LearningRateScheduler(_scheduler)

    else:
        def _scheduler(epoch):
            return _output(epoch)

        return _scheduler
