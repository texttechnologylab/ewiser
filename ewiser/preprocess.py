import os
import shutil
from lxml import etree as et
from typing import List

from fairseq.data import Dictionary

import dataset as ds

from ewiser.fairseq_ext.data.wsd_dataset import WSDDatasetBuilder

VALID_POS = ["NOUN", "VERB", "ADJ", "ADJ"]
VALID_LABELS = ["wnoffsets", "bnids"]
# These are not freely adjustable as they are referenced in EWISER code itself
dir_path = os.path.dirname(os.path.realpath(__file__))
DICT_PATH = os.path.abspath(os.path.join(dir_path, "../res/dictionaries"))
EMB_PATH = os.path.abspath(os.path.join(dir_path, "../res/embeddings"))
EDGE_PATH = os.path.abspath(os.path.join(dir_path, "../res/edges"))

"""
Functions for preprocessing corpora into EWISER-appropriate formats and creating relevant data files.
Note that most functions assume that any labels appearing the testset will also appear in the training set, 
dictionaries may not be consistent otherwise
"""


def set_dicts(datasets: List[ds.WSDData], built_ins=False):
    # TODO: built ins
    # TODO: option to add dictionaries from some other directory
    # Write out the dictionaries
    # Note: This might not work properly for english corpora?
    print("Creating dictionary entries...")
    # Relevant languages:
    langs = set()
    # Remove current dictionaries
    for dataset in datasets:
        langs.add(dataset.lang)
    main_dict_path = os.path.join(DICT_PATH, "dict.txt")
    if os.path.exists(main_dict_path):
        os.remove(os.path.join(DICT_PATH, "dict.txt"))
    for lang in langs:
        pos_path = os.path.join(DICT_PATH, "lemma_pos." + lang + ".txt")
        if os.path.exists(pos_path):
            os.remove(pos_path)
        offsets_path = os.path.join(DICT_PATH, "lemma_pos2offsets." + lang + ".txt")
        if os.path.exists(offsets_path):
            os.remove(offsets_path)

    # Load wordnet to babelnet mapping
    wn2bn = {}
    with open(os.path.join(DICT_PATH, "bnids_map.txt"), "r", encoding="utf8") as f:
        for line in f:
            line = line.strip().split("\t")
            bn = line[0]
            wn = line[2]
            wn2bn[wn] = bn

    # Create new current ones using datasets and optionally built ins.
    forms = {}
    lemma_pos_lang = {}
    lemma_pos2offsets_lang = {}
    paths = []
    for lang in langs:
        lemma_pos_lang[lang] = set()
        lemma_pos2offsets_lang[lang] = {}
    for dataset in datasets:
        print("Processing dataset {}".format(dataset.name))
        lang = dataset.lang
        assert dataset.labeltype in VALID_LABELS, "Labels must be one of {} for EWISER".format(VALID_LABELS)
        for entry in dataset.entries:
            # lemma pos
            lemma = entry.lemma
            upos = entry.upos
            assert len(entry.tokens) > 0, "Entries must have list of tokens"
            assert upos in VALID_POS, "EWISER cannot process pos other than NOUN, VERB, ADJ or ADV"
            pos = None
            if upos == "NOUN":
                pos = "n"
            elif upos == "VERB":
                pos = "v"
            elif upos == "ADJ":
                pos = "a"
            elif upos == "ADV":
                pos = "r"
            # If pos is still None here, something went horribly wrong
            lemma_pos_key = lemma + "#" + pos
            lemma_pos_lang[lang].add(lemma_pos_key)

            # possible babelnet ids
            label = entry.label
            # pos2offsets have to be babelnet ids.
            assert label.startswith("wn:") or label.startswith("bn:"), \
                "Ewiser labels must be wordnet offsets or babelnet ids in the format 'wn:<offset>' or 'bn:<id>'"
            bnlabel = None
            if label.startswith("wn:"):
                # Translate from wordnet to bn using the provided mapping
                try:
                    bnlabel = wn2bn[label]
                except IndexError as e:
                    raise IndexError("Couldn't find a babelnet id for wordnet label {}".format(label)) from e
            elif label.startswith("bn:"):
                bnlabel = label
            if lemma_pos_key in lemma_pos2offsets_lang[lang]:
                lemma_pos2offsets_lang[lang][lemma_pos_key].add(bnlabel)
            else:
                lemma_pos2offsets_lang[lang][lemma_pos_key] = {bnlabel}

            # form dict
            for token in entry.tokens:
                form = token.form
                if form in forms:
                    forms[form] += 1
                else:
                    forms[form] = 1

    for lang in langs:
        with open(os.path.join(DICT_PATH, "lemma_pos." + lang + ".txt"), "w", encoding="utf8") as f:
            paths.append(os.path.realpath(f.name))
            for lemma_pos in lemma_pos_lang[lang]:
                f.write(lemma_pos + " 1\n")

        with open(os.path.join(DICT_PATH, "lemma_pos2offsets." + lang + ".txt"), "w", encoding="utf8") as f:
            paths.append(os.path.realpath(f.name))
            for lemma_pos in lemma_pos2offsets_lang[lang]:
                f.write(lemma_pos + "\t" + "\t".join(lemma_pos2offsets_lang[lang][lemma_pos]) + "\n")

    # Sort forms by frequency.
    out = [item[0] + " " + str(item[1]) for item in sorted(forms.items(), key=lambda item: item[1], reverse=True)]
    with open(os.path.join(DICT_PATH, "dict.txt"), "w", encoding="utf8") as f:
        paths.append(os.path.realpath(f.name))
        for line in out:
            f.write(line + "\n")

    # Return the paths to all dictionaries we created, so we can back them up
    return paths
    # TODO: Built_ins should load and include the dicts that ewiser came with


def make_raganato(dataset: ds.WSDData, directory):
    # Writes out the dataset in the raganto xml format. This is used by EWISER for eval and its own preprocessing
    # XML filename is be dataset.name + ".data.xml"
    root = et.Element("corpus")

    gold_keys = []

    doc_counter = 0
    for entry in dataset.entries:
        doc_counter += 1
        doc_id = "d{:07d}".format(doc_counter)
        text = et.SubElement(root, "text")
        text.set("id", doc_id)
        source_id = entry.source_id
        if source_id is None:
            source_id = dataset.name + str(doc_counter)
        text.set("source", source_id)

        sent_id = "s001"
        sentence = et.SubElement(text, "sentence")
        sentence.set("id", doc_id + "." + sent_id)

        for token in entry.tokens:
            instance_counter = 0
            if token.is_pivot:
                instance_counter += 1
                instance_id = "h{:03d}".format(instance_counter)
                instance = et.SubElement(sentence, "instance")
                instance.set("id", doc_id + "." + sent_id + "." + instance_id)
                instance.set("lemma", token.lemma)
                instance.set("pos", token.upos)
                instance.text = token.form
                gold_keys.append(doc_id + "." + sent_id + "." + instance_id + "\t" + entry.label)
            else:
                word = et.SubElement(sentence, "wf")
                word.set("lemma", token.lemma)
                word.set("pos", token.upos)
                word.text = token.form

    # Write out files
    tree = et.ElementTree(root)
    outpath = os.path.join(directory, dataset.name + ".data.xml")
    if os.path.exists(outpath):
        raise RuntimeError("Cannot create new datafile {} since it already exists!".format(outpath))
    tree.write(outpath,
               encoding="utf-8",
               pretty_print=True,
               xml_declaration=True)

    with open(os.path.join(directory, dataset.name + ".gold.key.txt"), "w", encoding="utf8") as f:
        for key in gold_keys:
            f.write(key + "\n")

    return outpath


def preproc(trainsets: List[ds.WSDData],
            evalsets: List[ds.WSDData],
            directory: str,
            data_for_dicts_only: List[ds.WSDData] = [],
            max_length=100,
            on_error="skip",
            quiet=False):
    # Setup
    try:
        os.makedirs(directory)
        os.makedirs(os.path.join(directory, "data"))
    except OSError:
        print("Could not create data directories")

    # Make dictionaries
    created_dicts = set_dicts(trainsets + evalsets + data_for_dicts_only)
    # Copy form dictionary that we need for preprocessing and training
    # Copy other dictionaries as well for backup
    for dictpath in created_dicts:
        shutil.copy(dictpath, directory)

    # Setup EWISER preproc
    dictionary = Dictionary.load(os.path.join(directory, "dict.txt"))

    # Make raganatos and create preprocessed files from those
    print("Creating/processing training data...")
    for i, dataset in enumerate(trainsets):
        assert dataset.labeltype in VALID_LABELS, "Labels must be one of {}".format(VALID_LABELS)
        dirname = "train{}".format(i if i > 0 else "")
        os.mkdir(os.path.join(directory, "data", dirname))
        make_raganato(dataset, os.path.join(directory, "data", dirname))

        # Ewiser preproc
        output = WSDDatasetBuilder(
            os.path.join(directory, dirname),
            dictionary=dictionary,
            use_synsets=True,
            keep_string_data=True,
            lang=dataset.lang)

        # adjust keys:
        input_keys = None
        if dataset.labeltype == "wnoffsets":
            input_keys = "offsets"
        elif dataset.labeltype == "bnids":
            input_keys = "bnids"

        output.add_raganato(
            xml_path=os.path.join(directory, "data", dirname, dataset.name + ".data.xml"),
            max_length=max_length,
            input_keys=input_keys,
            on_error=on_error,
            quiet=quiet,
            read_by="text",
        )

        output.finalize()

    print("Creating/processing eval data...")
    for i, dataset in enumerate(evalsets):
        assert dataset.labeltype in VALID_LABELS, "Labels must be one of {}".format(VALID_LABELS)
        dirname = "valid{}".format(i if i > 0 else "")
        os.mkdir(os.path.join(directory, "data", dirname))
        make_raganato(dataset, os.path.join(directory, "data", dirname))

        # Ewiser preproc
        output = WSDDatasetBuilder(
            os.path.join(directory, dirname),
            dictionary=dictionary,
            use_synsets=True,
            keep_string_data=True,
            lang=dataset.lang)

        # adjust keys:
        input_keys = None
        if dataset.labeltype == "wnoffsets":
            input_keys = "offsets"
        elif dataset.labeltype == "bnids":
            input_keys = "bnids"

        output.add_raganato(
            xml_path=os.path.join(directory, "data", dirname, dataset.name + ".data.xml"),
            max_length=max_length,
            input_keys=input_keys,
            on_error=on_error,
            quiet=quiet,
            read_by="text",
        )

        output.finalize()
