import os
import subprocess
import argparse

from ewiser import preprocess
from dataset import WSDData, train_test_split


# Takes a dir with preprocessed corpora and relevant dictionaries, a model output dir, and a bunch of ewiser params
def train(out_dir: str, modelname: str, **params):

    # Epochs and learning rates have to be handled separately
    shell_params = {"epochs1": 50,
                    "epochs2": 50}

    # Setup params for training script
    DEFAULT_PARAMS = {
        "arch": "linear_seq",
        "task": "sequence_tagging",
        "criterion": "weighted_cross_entropy",
        "tokens-per-sample": 100,
        "max-tokens": 1000,
        "optimizer": "adam",
        "min-lr": 1e-7,
        "lr-scheduler": "fixed",
        "decoder-embed-dim": 512,
        "update-freq": 4,
        "dropout": 0.2,
        "clip-norm": 1.0,
        "context-embeddings": True,
        "context-embeddings-type": "bert",
        "context-embeddings-bert-model": "bert-base-multilingual-cased",
        "context-embeddings-cache": True,
        "only-use-targets": True,
        "log-format": "tqdm",
        "decoder-layers": 2,
        "decoder-norm": True,
        "decoder-last-activation": True,
        "decoder-activation": "swish",
        "no-epoch-checkpoints": True
    }

    for key, value in DEFAULT_PARAMS.items():
        if key not in params:
            params[key] = value

    # Process parameters
    arg_list = []
    for key, value in params.items():
        if not type(value) == bool:
            arg_list.append("--{} {}".format(key, value))
        else:
            if value:
                arg_list.append("--{}".format(key))

    # Build shell file
    outlines = ["#!/bin/bash",
                "MODEL='{}'".format(modelname),
                "CORPUS_DIR='{}'".format(out_dir),
                "EMBEDDINGS='{}'".format(
                    os.path.join(preprocess.EMB_PATH, 'sensembert+lmms.svd512.synset-centroid.vec')),
                "EDGES='{}'".format(preprocess.EDGE_PATH),
                "EPOCHS_1={}".format(shell_params["epochs1"]),
                "EPOCHS_2={}\n".format(shell_params["epochs2"]),
                # Savedir
                "SAVEDIR='{}'".format(os.path.join(out_dir, modelname)),
                "mkdir -p ${SAVEDIR}\n",
                "args1=(\\\n{}\n)\n".format(" \\\n".join(arg_list)),
                "args2=( --decoder-output-pretrained $EMBEDDINGS --decoder-use-structured-logits"
                " --decoder-structured-logits-edgelists ${EDGES}/hypernyms.tsv )\n",
                # Stage 1 training
                "CUDA_VISIBLE_DEVICES=0 python3 bin/train.py $CORPUS_DIR \"${args1[@]}\" \"${args2[@]}\" --lr 1e-4"
                " --save-dir $SAVEDIR --max-epoch $EPOCHS_1 --decoder-output-fixed"
                " --decoder-structured-logits-trainable\n",
                # Setup stage 2
                "mkdir -p $SAVEDIR/stage2",
                "cp $SAVEDIR/checkpoint_best.pt $SAVEDIR/stage2/init.pt",
                "args3=( --restore-file $SAVEDIR/stage2/init.pt --decoder-structured-logits-trainable"
                " --only-load-weights --reset-optimizer --reset-dataloader --reset-meters)\n",
                # Stage 2 training
                "CUDA_VISIBLE_DEVICES=0 python3 bin/train.py $CORPUS_DIR \"${args1[@]}\" \"${args2[@]}\""
                " \"${args3[@]}\" --lr 1e-5 --save-dir $SAVEDIR/stage2 --max-epoch $EPOCHS_2"]

    with open("train_bash.sh", "w", encoding="utf8") as f:
        for line in outlines:
            f.write(line + "\n")

    subprocess.call(["bash", "train_bash.sh"])


# CLI will take datasets as files, tmpdir, modeldir and any params, then load datasets and call train
# Datasets can be either as param "datasets", which will be train/eval split according to ratios,
# or else as train/eval sets
# TODO: Ewiser params somehow
# TODO: Warning that setting --include-wn to true will produce some Warnings about missing entries in lemma pos \
#  dictionary
# TODO: Option to directly train with already created training directories. Should ask the user if we want to train \
#  with directory if it already exists
def cli():
    parser = argparse.ArgumentParser(description="Training script for ewiser")

    # Training sets
    parser.add_argument("--train", required=False, type=str, nargs='*', help=
                        "JSON Datafiles that should be used for training")

    # Evaluation sets
    parser.add_argument("--eval", required=False, type=str, nargs='*', help=
                        "JSON Datafiles that will be used for validation during training")

    parser.add_argument("--test", required=False, type=str, nargs='*', help=
                        "JSON Datafiles that you intend to use for testing. No actual testing is done, but "
                        "dictionary entries are updated")

    # Plain datasets, that we will split ourselves
    parser.add_argument("--data", required=False, type=str, nargs='*', help=
                        "JSON Datafiles that will be split into train/eval/test. Note that the test set will not be "
                        "used during training. This argument and train/eval are mutually exclusive!")
    # If we split we have to know ratios
    parser.add_argument("--ratio-eval", required=False, type=float, help=
                        "ratio of the datasets that will be used as evaluation data")
    parser.add_argument("--ratio-test", required=False, type=float, help=
                        "ratio of the datasets that will be used as test data")

    # Training directory (will be created if not exists)
    parser.add_argument("--train-dir", required=True, type=str, help=
                        "Directory where relevant data and preprocessed corpora will be stored")

    # Directory in the training directory for models
    parser.add_argument("--model-dir", required=True, type=str, help=
                        "Subdirectory in the training directory where checkpoints will be saved")

    # If we should include the wordnet glosses and examples in the training data
    parser.add_argument("--include-wn", required=False, action="store_true", help=
                        "Wordnet glosses and examples are included in training data if set.")

    # Path to dictionaries for already created xml format files
    parser.add_argument("--dict-dir", required=False, type=str, help=
                        "Directory containing additional dictionary files.\n"
                        "Dictionaries are normally built, saved and set during preprocessing."
                        " If you setup two training runs, the set dictionaries will match the last run, not the first."
                        " It is important that the dictionaries used during training match those used during "
                        "preprocessing to avoid errors. This parameter can be used to point to the directory where the "
                        "specific dictionaries were saved.")

    args = parser.parse_args()

    # Either we have args.data or args.train and args.eval
    assert (args.data is None and args.train is not None and args.eval is not None) or \
           (args.data is not None and args.train is None and args.eval is None and args.test is None), \
           "data and train/eval are mutually excluve arguments."

    # Load data
    trainsets = []
    evalsets = []
    testsets = []

    if args.data is not None:
        assert args.ratio_test is not None and args.ratio_eval is not None, \
               "When passing unsplit data you have to specify splitting ratios with ratio-test and ratio-eval"
        for datapath in args.data:
            dataset = WSDData.load(datapath)
            trainset, evalset, testset = train_test_split(dataset, args.ratio_eval, args.ratio_test)
            trainsets.append(trainset)
            evalsets.append(evalset)
            testsets.append(testset)

    elif args.train is not None and args.eval is not None:
        for datapath in args.train:
            dataset = WSDData.load(datapath)
            trainsets.append(dataset)
        for datapath in args.eval:
            dataset = WSDData.load(datapath)
            evalsets.append(dataset)
        if args.test is not None:
            for datapath in args.test:
                dataset = WSDData.load(datapath)
                testsets.append(dataset)

    print("Preprocessing...")
    if args.include_wn:
        print("INFO: Including wordnet glosses and examples will produce warnings about missing lemma pos entries, these are "
              "expected for the wordnet data only. ")
    # Do preprocessing
    preprocess.preproc(trainsets,
                       evalsets,
                       args.train_dir,
                       data_for_dicts_only=testsets,
                       dict_dir=args.dict_dir,
                       include_wn=args.include_wn)

    # Write out testsets as raganato for convenience
    if testsets:
        for testset in testsets:
            preprocess.make_raganato(testset, args.train_dir)
    print("Training...")
    # Train model
    train(args.train_dir, args.model_dir)


if __name__ == "__main__":
    cli()
