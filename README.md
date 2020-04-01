# Code for training models in *nussl*

This repository contains the code for training many of the models
in *nussl*. It contains a few extra utilities, such as a simple way
to schedule jobs with GPUs and recipes for creating datasets.

## Setting up environment

```
conda env create -f conda.yml
conda activate nussl-models
```

Set the paths appropriately in `env.sh`. Please ensure all paths in `env.sh` are absolute paths. Then `source env.sh`.

If you update the `conda.yml` file to say, add new requirements, update the conda environment after activating it:

```
conda env update -f conda.yml --prune
```

Note that the `--prune` option will get rid of any packages that were removed in `conda.yml`.

## Running experiments

All of the experiments in this repository use 
[gin-config](https://github.com/google/gin-config), which is a flexible
way to specify hyperparameters and configuration in a hierarchical and
reusable fashion. There are 5 main functions whose arguments are all
configured with gin:

1. `train`: This function will train a model. This needs datasets 
   (see `wham/wham8k.gin` for an example), and model and training 
   parameters (see `wham/exp/dpcl.gin` for an example).
2. `cache`: This function takes the datasets and caches them to the 
   desired location.
3. `evaluate`: This function evaluates a separation algorithm on the 
   test dataset.
4. `instantiate`: This function instantiates an experiment by compiling a
   gin config file on the fly, possibly sweeping across some hyperparameters,
   and writing it all to an output directory.
5. `analyze`: This analyzes the experiment after evaluation. Spits out a report
   card with the metrics.

Finally `all` will run `train`, then `evaluate`, then `analyze`.

### Usage

The `main.py` script takes 3 arguments and 1 positional argument. It takes an
environment config file (`--environment_config`, `-env`) which sets all the 
machine-specific variables like path to data directories and so on, a 
data config file (`--data_config`, `-dat`) which describes all of the datasets, 
and an experiment config file (`--experiment_config`, `-exp`) which describes
the model settings, the optimizer, the algorithm settings, and whatever else
is needed. Generally, you'll want to follow a process like this:

```
python main.py --dat [path/to/data.gin] --env [path/to/env.gin] --exp [path/to/exp.gin] cache
```

This will cache all of the datasets so that things train faster. Now, go and set the key
that says `cache_populated = False` to `cache_populated = True` in your `data.gin` file.
Then:

```
python main.py --dat [path/to/data.gin] --env [path/to/env.gin] --exp [path/to/exp.gin] instantiate
```

This will instantiate your experiments into a single config file and place that config
into a folder. The location of the folder depends on the location of the path to the
experiment config. If the config file, for example, was at `wham/exp/dpcl.gin`, then
the output will be at `wham/exp/out/dpcl/run0:[descriptive_string]/config.gin`. If you
had a sweep in your experiment config file, then you'll see multiple of these. For
example, if you had lines in your experiment config like this:

```
sweep.parameters = {
    'add_clip_gradient_handler.clip_value': [1e-4, 1e-3, 1e-2]
}
```

Then, the output configs will be at:

```
wham/exp/out/dpcl/run0:clip_value:0.0001/config.gin 
wham/exp/out/dpcl/run1:clip_value:0.001/config.gin 
wham/exp/out/dpcl/run2:clip_value:0.01/config.gin 
```

Now, you're ready to run experiments. To run, say the first experiment you would do:

```
python main.py --exp wham/exp/out/dpcl/run0:clip_value:0.0001/config.gin train
python main.py --exp wham/exp/out/dpcl/run0:clip_value:0.0001/config.gin evaluate
python main.py --exp wham/exp/out/dpcl/run0:clip_value:0.0001/config.gin analyze
```

Or to do all of them in one fell swoop:

```
python main.py --exp wham/exp/out/dpcl/run0:clip_value:0.0001/config.gin all
```

At the end, you'll see some output like this:

```
❯ python main.py --exp wham/exp/out/dpcl/run0:clip_value:0.0001/config.gin all
/Users/prem/miniconda3/envs/nussl-models/lib/python3.7/site-packages/torch/nn/modules/rnn.py:50: UserWarning: dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.3 and num_layers=1
  "num_layers={}".format(dropout, num_layers))
2020-04-01:02:36:57,921 [train.py:39] SeparationModel(
  (layers): ModuleDict(
    (embedding): Embedding(
      (linear): Linear(in_features=100, out_features=2580, bias=True)
    )
    (log_spectrogram): AmplitudeToDB()
    (normalization): BatchNorm(
      (batch_norm): BatchNorm1d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (recurrent_stack): RecurrentStack(
      (rnn): LSTM(129, 50, batch_first=True, dropout=0.3, bidirectional=True)
    )
  )
)
Number of parameters: 332982
2020-04-01:02:36:57,922 [train.py:45] Saving to /Users/prem/Dropbox/research/nussl-models/wham/exp/out/dpcl/run0:clip_value:0.0001
2020-04-01:02:36:57,924 [engine.py:837] Engine run starting with max_epochs=1.
2020-04-01:02:36:59,235 [engine.py:939] Epoch[1] Complete. Time taken: 00:00:01
2020-04-01:02:36:59,236 [engine.py:837] Engine run starting with max_epochs=1.
2020-04-01:02:36:59,875 [engine.py:939] Epoch[1] Complete. Time taken: 00:00:00
2020-04-01:02:36:59,876 [engine.py:947] Engine run complete. Time taken 00:00:00
2020-04-01:02:36:59,907 [trainer.py:340]

EPOCH SUMMARY
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Epoch number: 0001 / 0001
- Training loss:   0.553462
- Validation loss: 0.552009
- Epoch took: 00:00:01
- Time since start: 00:00:01
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Saving to /Users/prem/Dropbox/research/nussl-models/wham/exp/out/dpcl/run0:clip_value:0.0001/checkpoints/best.model.pth.
Output @ /Users/prem/Dropbox/research/nussl-models/wham/exp/out/dpcl/run0:clip_value:0.0001

2020-04-01:02:36:59,911 [engine.py:947] Engine run complete. Time taken 00:00:01
/Users/prem/miniconda3/envs/nussl-models/lib/python3.7/site-packages/nussl/separation/base/separation_base.py:71: UserWarning: input_audio_signal has no data!
  warnings.warn('input_audio_signal has no data!')
/Users/prem/miniconda3/envs/nussl-models/lib/python3.7/site-packages/nussl/core/audio_signal.py:445: UserWarning: Initializing STFT with data that is non-complex. This might lead to weird results!
  warnings.warn('Initializing STFT with data that is non-complex. '
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:16<00:00,  3.37s/it]
┌────────────────────┬───────────────────┬────────────────────┐
│                    │ OVERALL (N = 20)  │                    │
╞════════════════════╪═══════════════════╪════════════════════╡
│        SAR         │        SDR        │        SIR         │
├────────────────────┼───────────────────┼────────────────────┤
│ 13.487467288970947 │ -5.10146589204669 │ -5.028965532779694 │
└────────────────────┴───────────────────┴────────────────────┘
```
