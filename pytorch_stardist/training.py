import os
import sys
import time
import datetime
import warnings

import torch

from pathlib import Path

from .models.stardist_base import StarDistBase


def train(model: StarDistBase, train_dataloader, val_dataloader):
    """
    perform training with metrics logging and model checkpointing

    parameters
    ----------
    model: StarDist2D or StarDist3D
        StarDist model
    train_dataloader: torch.utils.data.DataLoader
        training data loader
    val_dataloader: torch.utils.data.DataLoader
        validation dataloader

    """
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    opt = model.opt
    logger = model.logger
    epoch_start = opt.epoch_count
    prev_time = time.time()
    max_steps_per_epoch = min(len(train_dataloader), opt.max_steps_per_epoch)

    for epoch in range(epoch_start, opt.n_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        time_start = time.time()
        
        for i, batch in enumerate(train_dataloader):          
            model.optimize_parameters(batch, epoch=epoch)

            time_spent = datetime.timedelta(seconds=time.time() - time_start)  
            
            if global_rank == 0:
                sys.stdout.write(
                    "\r[Epoch %d/%d] [Steps %d/%d] [Loss: %.4f Loss_dist: %.4f Loss_prob: %.4f Loss_prob_class: %.4f] Duration %s" %
                    (
                        epoch + 1,
                        opt.n_epochs,
                        i + 1,
                        max_steps_per_epoch,
                        logger.get_mean_value("loss", epoch) or 0.0,
                        logger.get_mean_value("loss_dist", epoch) or 0.0,
                        logger.get_mean_value("loss_prob", epoch) or 0.0,
                        logger.get_mean_value("loss_prob_class", epoch) or 0.0,
                        time_spent
                    )
                )
            if i+1 >= max_steps_per_epoch:
                break
                
        if global_rank == 0: print()     
        torch.distributed.barrier()  

        ### Evaluation
        if hasattr(opt, "evaluate") and opt.evaluate and val_dataloader is None:
            warnings.warn('"evaluate=True" but val_loader is None. Can \'t perform evaluation!')

        if hasattr(opt, "evaluate") and opt.evaluate:
            n_batches_val = len(val_dataloader)
            time_start = time.time()

            for i, batch in enumerate(val_dataloader):
                model.evaluate(batch)

                time_spent = datetime.timedelta(seconds=time.time() - time_start)
                
                if global_rank == 0:
                    sys.stdout.write(
                        "\r[Epoch %d/%d] [Batch %d/%d] [Val_loss: %.4f Val_loss_dist: %.4f Val_loss_prob: %.4f Val_loss_prob_class: %.4f]  Duration %s" %
                        (    
                            epoch + 1,
                            opt.n_epochs,
                            i + 1,
                            n_batches_val,
                            logger.get_mean_value("Val_loss", epoch) or 0.0,
                            logger.get_mean_value("Val_loss_dist", epoch) or 0.0,
                            logger.get_mean_value("Val_loss_prob", epoch) or 0.0,
                            logger.get_mean_value("Val_loss_prob_class", epoch) or 0.0,
                            time_spent
                        )
                    )
        ### End evaluation
        if global_rank == 0: print()     
        torch.distributed.barrier()  

        if hasattr(opt, "evaluate") and opt.evaluate:
            metric = logger.get_mean_value("Val_loss", epoch)
        else:
            metric = logger.get_mean_value("loss", epoch)

        model.update_lr(metric=metric)
              
        if not hasattr(model.opt, "best_metric") or model.opt.best_metric >= metric:
            model.opt.best_metric = metric
            opt.best_metric = metric
            if epoch >= opt.start_saving_best_after_epoch and global_rank == 0:
                model.save_state(name="best")

        if global_rank == 0:
            print("---")
            logger.plot(path=Path(opt.log_dir) / f"{opt.name}/figures")

        opt.epoch_count += 1
        
        if (epoch + 1) % opt.save_epoch_freq == 0 and global_rank == 0:
            print("*** Saving ...")

            model.save_state()

            print("*** Saving done.")
            print()
    
        torch.distributed.barrier()
