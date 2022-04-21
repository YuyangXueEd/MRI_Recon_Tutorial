import argparse
from pathlib import Path

KNEE_ACQ = ['CORPD_FBK', 'CORPDFS_FBK']
BRAIN_ACQ = ['AXFLAIR', 'AXT1POST', 'AXT1PRE', 'AXT1', 'AXT2']


class Args(argparse.ArgumentParser):
    """
    Defines default arguments
    """
    def __init__(self, **overrides):
        """
        :param **overrides -> (dict, optional): keyword arguments
                                                       used to override default argument values
        """

        super(Args, self).__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # global
        self.add_argument(
            '--seed',
            default=42,
            type=int,
            help='Seed for random number generators'
        )

        self.add_argument(
            '--device',
            default='cuda',
            type=str,
            help='which device to train on. Set to "cuda" to use the GPU.'
        )

        self.add_argument(
            '--num-workers',
            default=8,
            type=int,
            help='threads'
        )

        self.add_argument(
            '--batch-size',
            default=32,
            type=int,
            help='Mini-batch size'
        )

        # Dataset

        self.add_argument(
            '--challenge',
            choices=['singlecoil', 'multicoil'],
            required=True,
            help='Which challenge'
        )

        self.add_argument(
            '--acquisition',
            choises = KNEE_ACQ + BRAIN_ACQ,
            default=None,
            help = 'If set, only volumes of the specified acquisition type'
                   'are used for evaluation. By default, all volumes are included.'
        )

        self.add_argument(
            '--datapath',
            type=Path,
            required=True,
            help='Path to the dataset'
        )

        self.add_argument(
            '--csvpath',
            type=Path,
            required=True,
            help='Path to the csv files'
        )

        # data

        self.add_argument(
            '--nsmaps',
            type=int,
            default=2,
            help='Number of soft sensitivity maps to include'
        )

        self.add_argument(
            '--norm',
            choice=['lfimg_max', 'lfimg_med'],
            default='lfimg_max',
            help='Data normalisation. Options: lfimg_{max|med}'
        )

        # Mask

        self.add_argument(
            '--accelerations',
            nargs='+',
            default=[4, 8],
            type=int,
            help='Ratio of kspace columns to be sampled. '
                 'If multiple values are provided, then one of those is '
                 'chosen uniformly at random for each volume.'
        )

        self.add_argument(
            '--certer-fractions',
            nargs='+',
            default=[0.08, 0.04],
            type=float,
            help='Fraction of low-frequency kspace columns to be sampled.'
                 'Should have the same length as accelerations.'
        )

        self.set_defaults(**overrides)



class TrainArgs(Args):
    """
    Defines global default arguments for training.
    """

    def __init__(self, **overrides):
        """
        Args: **overrides (dict, optional): keyword arguments used to
               override default argument values
        """

        super(TrainArgs, self).__init__(**overrides)

        self.add_argument(
            '--report-interval',
            type=int,
            default=100,
            help='Period of loss reporting'
        )

        self.add_argument(
            '--data-parallel',
            action='store_true',
            default=True,
            help='If set, use multiple GPUs using data parallelism'
        )

        self.add_argument(
            '--outdir',
            type=Path,
            default='checkpoints',
            help='Path where model and results should be saved'
        )

        self.add_argument(
            '--nepochs',
            type=int,
            default=50,
            help='Number of trianing epochs'
        )

        self.add_argument(
            '--optimizer',
            type=str,
            default='rmsprop',
            choices=['rmsprop', 'adam'],
            help='Options: {adam | rmsprop}'
        )

        self.add_argument(
            '--lr',
            type=float,
            default=1e-4,
            help='Learning rate'
        )

        self.add_argument(
            '--lr-gamma',
            type=float,
            default=0.5,
            help='Learning rate scheduler'
        )

        self.add_argument(
            '--lr-step-size',
            type=int,
            default=15,
            help='Period of learning rate decay'
        )

        self.add_argument(
            '--weight-decay',
            type=float,
            default=0.,
            help='Strength of weight decay regularisation'
        )

        self.add_argument(
            '--fe-patch-size',
            type=int,
            default=96,
            help='Patch size along frequency encoding direction.'
        )

        self.add_argument(
            '--use-fg-mask',
            action='store-true',
            help='Using a foreground mask for loss evaluation.'
        )

        self.set_defaults(**overrides)

class TestArgs(Args):
    """
    Defines global default arguments for testing / validation.
    """
    def __init__(self, **overrides):
        """
        Args: **overrides (dict, optional): Keyword arguments used to
            override default argument values
        """
        super().__init__(**overrides)

        self.add_argument(
            '--data-split',
            choices=['val', 'test_v2', 'challenge'],
            required=True,
            help='Which data partition to run on: "val", "test_v2", "challenge"',
        )
        self.add_argument(
            '--recon-dir',
            type=Path,
            required=True,
            help='Path to save the reconstructions to',
        )
        self.add_argument(
            '--mask-bg', action='store_true',
            help='Whether to apply a mask to bg and replace the values'
            'by a mean value estimated from the undersample RSS',
        )
        self.add_argument(
            '--save-true-acceleration', action='store_true',
            help='Flag to additionally store true acceleration factor.',
        )


