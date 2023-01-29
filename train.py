import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from src.args import handle_arguments
from src.data.datamodule import DataModule
from src.models.get import get_model

import wandb
from pytorch_lightning.loggers import WandbLogger

def main():
    args = handle_arguments()

    isExist = os.path.exists(args.save_dir)
    if not isExist:
        os.makedirs(args.save_dir)
    
    # Setup logging and weight saves.
    logger = TensorBoardLogger(os.path.join(args.save_dir, 'tb-logs'), name=args.name, version=args.version)
    args.logger = logger
    args.weights_save_path = os.path.join(args.save_dir, 'tb-logs', args.name, f'version_{args.version}')
    
    dm = DataModule(**args.__dict__)
    dm.setup()
    num_classes = dm.train_dataset.num_labels

    model = get_model(args.model_type)(num_classes=num_classes, **args.__dict__)

    wandb_logger = WandbLogger(project='ViT')
    
    trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger)
    
    if args.resume:
        trainer.fit(model, datamodule=dm, ckpt_path=args.resume)
    else:
        trainer.fit(model, datamodule=dm)

    wandb.finish()

if __name__ == '__main__':
    main()
    