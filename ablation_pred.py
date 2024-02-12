
import fire
import os
import pytorch_lightning as pl
# from clearml import Task
from pytorch_lightning import loggers
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from utils import get_config, get_logger, make_result_dir
import shutil
from pipeline import Net # !!!!

def main(config_file: str, name: str) -> None:
    """Run training script

    Parameters
    ----------
    config_file : str
        Path to config file.
    """

    # task = Task.init(project_name="ablation_prediction", task_name=name)
    # task.set_resource_monitor_iteration_timeout(180)
    #  set random seed
    # np.random.seed(42)

    #  setup logger
    logger = get_logger()

    #  read config file
    config = get_config(config_file)

    #  tracl config with ClearML
    # _ = task.connect_configuration(config)

    #  print config info into log
    log_message =\
        f"Run training for config: {config}"
    logger.info(log_message)

    result_dir_name = name
    result_dir = make_result_dir(result_dir_name)
    
    #  save config file into result directory
    shutil.copyfile(config_file, os.path.join(result_dir, result_dir_name+'.yaml'))

    
    pipeline = Net(result_dir, **config)

    checkpoint_callback = [pl.callbacks.ModelCheckpoint(
                dirpath=result_dir,
                filename='{epoch:03d}-{val_loss:.4f}',
                save_top_k=-1,
                save_weights_only=True,
                verbose=True,
                monitor='val_loss',
                mode='min')]    
    
    logger=loggers.TensorBoardLogger(result_dir)
    
    trainer = pl.Trainer(
        accelerator="gpu", # 'cpu'
        devices=config['hardware']['gpus'],
        logger=logger,
        max_epochs=pipeline.max_epochs,
        check_val_every_n_epoch=pipeline.check_val,
        callbacks=checkpoint_callback,
        gradient_clip_val=0.5, 
        gradient_clip_algorithm="value",
    )   
    trainer.fit(pipeline)

if __name__ == '__main__':
    fire.Fire(main)

