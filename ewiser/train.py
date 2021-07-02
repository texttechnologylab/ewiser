import os
import subprocess

import preprocess


# Takes a list of train/eval datasets, a temp dir for preprocessed corpora etc..., and a bunch of ewiser params
def train(trainsets, evalsets, out_dir, modelname, **params):
    # Do preprocessing
    preprocess.preproc(trainsets, evalsets, out_dir)

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
    outlines = ["!/bin/bash",
                "MODEL='{}'".format(modelname),
                "CORPUS_DIR='{}'".format(out_dir),
                "EMBEDDINGS='{}'".format(
                    os.path.join(preprocess.EMB_PATH, '/sensembert+lmms.svd512.synset-centroid.vec')),
                "EDGES='{}'".format(preprocess.EDGE_PATH),
                "EPOCHS_1={}".format(params["epochs1"]),
                "EPOCHS_2={}".format(params["epochs2"]),
                "\n",
                "SAVEDIR='{}'".format(os.path.join(out_dir, modelname)),
                "mkdir -p ${SAVEDIR}",
                "\n",
                "args1=(\\\n{}\n)".format("\\\n".join(arg_list)),
                "\n",
                "args2= --decoder-output-pretrained $EMBEDDINGS --decoder-use-structured-logits --decoder-structured-logits-edgelists ${EDGES}/hypernyms.tsv",
                "\n",
                "CUDA_VISIBLE_DEVICES=0 python3 bin/train.py $CORPUS_DIR '${args1[@]}' '${args2[@]}' --lr 1e-4 --save-dir $SAVEDIR --max-epoch $EPOCHS_1 \
                    --decoder-output-fixed --decoder-structured-logits-trainable",
                "\n",
                "mkdir -p $SAVEDIR/stage2",
                "cp $SAVEDIR/checkpoint_best.pt $SAVEDIR/stage2/init.pt",
                "args3=(--restore-file $SAVEDIR/stage2/init.pt --decoder-structured-logits-trainable --only-load-weights --reset-optimizer --reset-dataloader --reset-meters)",
                "\n",
                "CUDA_VISIBLE_DEVICES=0 python3 bin/train.py $CORPUS_DIR '${args1[@]}' #${args2[@]}' '${args3[@]}' --lr 1e-5 --save-dir $SAVEDIR/stage2 --max-epoch $EPOCHS_2"]

    # Should somehow make this part of shell script as well

    # Stage 1 training

    # Setup for stage2

    # Stage 2 training

    with open("train_bash.sh", "w", encoding="utf8") as f:
        for line in outlines:
            f.write(line + "\n")

    subprocess.call("./train_bash.sh")


# TODO: CLI 
# CLI will take datasets as files, tmpdir, modeldir and any params, then load datasets and call train
