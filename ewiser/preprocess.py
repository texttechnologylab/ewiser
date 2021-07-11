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
CORPORA_PATH = os.path.abspath(os.path.join(dir_path, "../res/corpora"))

"""
Functions for preprocessing corpora into EWISER-appropriate formats and creating relevant data files.
Note that most functions assume that any labels appearing the testset will also appear in the training set, 
dictionaries may not be consistent otherwise
"""


def load_form_dict(forms, dict_path):
    with open(dict_path, "rt", encoding="utf8") as f:
        for line in f:
            line = line.strip().split(" ")
            form, freq = line
            if form in forms:
                forms[form] = forms[form] + int(freq)
            else:
                forms[form] = int(freq)


def load_lemma_pos(lp_set, dict_path):
    with open(dict_path, "rt", encoding="utf8") as f:
        for line in f:
            line = line.strip().split(" ")
            lemmapos = line[0]
            lp_set.add(lemmapos)


def set_dicts(datasets: List[ds.WSDData], dict_dir: str = None, include_wn: bool = False):
    # dict_dir can be used to load dictionaries that were created previously and add them
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
    print("Loading babelnet mapping")
    wn2bn = {}
    with open(os.path.join(DICT_PATH, "bnids_map.txt"), "r", encoding="utf8") as f:
        for line in f:
            line = line.strip().split("\t")
            bn = line[0]
            wn = line[2]
            wn2bn[wn] = bn

    # Create new dictionaries ones using datasets, directory and optionally the wordnet ones
    # Setup dictionaries
    forms = {}
    lemma_pos_lang = {}
    lemma_pos2offsets_lang = {}
    for lang in langs:
        lemma_pos_lang[lang] = set()
        lemma_pos2offsets_lang[lang] = {}

    if include_wn:
        langs.add("en")
        load_form_dict(forms, os.path.join(DICT_PATH, "builtin_dict.txt"))
        if "en" not in lemma_pos_lang:
            lemma_pos_lang["en"] = set()
        load_lemma_pos(lemma_pos_lang["en"], os.path.join(DICT_PATH, "builtin_lemma_pos.en.txt"))

    # Load dictionaries from directory
    if dict_dir:
        load_form_dict(forms, os.path.join(dict_dir, "dict.txt"))

        # lemmapos dictionaries for each language
        lemma_pos_files = [filename for filename in os.listdir(dict_dir) if
                           os.path.isfile(os.path.join(dict_dir, filename)) and filename.startswith("lemma_pos.")]
        for lemma_pos_file in lemma_pos_files:
            lang = lemma_pos_file.split(".")[1]
            langs.add(lang)
            if lang not in lemma_pos_lang:
                lemma_pos_lang[lang] = set()

            # load lemmapos
            load_lemma_pos(lemma_pos_lang[lang], os.path.join(dict_dir, lemma_pos_file))

        # lemmapos2offsets dictionaries for each language
        lemma_pos_offset_files = [filename for filename in os.listdir(dict_dir) if
                                  os.path.isfile(os.path.join(dict_dir, filename)) and
                                  filename.startswith("lemma_pos2offsets.")]
        for lemma_pos_file in lemma_pos_offset_files:
            lang = lemma_pos_file.split(".")[1]
            langs.add(lang)
            if lang not in lemma_pos2offsets_lang:
                lemma_pos2offsets_lang[lang] = {}

            with open(os.path.join(dict_dir, lemma_pos_file), "rt", encoding="utf8") as f:
                for line in f:
                    line = line.strip().split("\t")
                    lemmapos = line[0]
                    offsets = line[1:]
                    if lemmapos in lemma_pos2offsets_lang[lang]:
                        lemma_pos2offsets_lang[lang][lemmapos].update(offsets)
                    else:
                        lemma_pos2offsets_lang[lang][lemmapos] = set(offsets)

    # Build dictionaries from datasets
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

    paths = []
    for lang in langs:
        with open(os.path.join(DICT_PATH, "lemma_pos." + lang + ".txt"), "w", encoding="utf8") as f:
            paths.append(os.path.realpath(f.name))
            for lemma_pos in lemma_pos_lang[lang]:
                f.write(lemma_pos + " 1\n")

        if lang in lemma_pos2offsets_lang:
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


def _preproc_dataset(dataset: ds.WSDData,
                     directory: str,
                     dictionary,
                     subdir_name: str,
                     **kwargs):

    assert dataset.labeltype in VALID_LABELS, "Labels must be one of {}".format(VALID_LABELS)
    os.mkdir(os.path.join(directory, "data", subdir_name))
    make_raganato(dataset, os.path.join(directory, "data", subdir_name))

    # adjust keys:
    input_keys = None
    if dataset.labeltype == "wnoffsets":
        input_keys = "offsets"
    elif dataset.labeltype == "bnids":
        input_keys = "bnids"

    # Ewiser preproc
    _raganato_preproc(os.path.join(directory, "data", subdir_name, dataset.name + ".data.xml"),
                      os.path.join(directory, subdir_name),
                      dictionary,
                      dataset.lang,
                      input_keys,
                      **kwargs)


def _raganato_preproc(input_dir: str,
                      output_dir: str,
                      dictionary,
                      language: str,
                      input_keys: str,
                      max_length=100,
                      on_error="skip",
                      quiet=False):

    output = WSDDatasetBuilder(
        output_dir,
        dictionary=dictionary,
        use_synsets=True,
        keep_string_data=True,
        lang=language)

    output.add_raganato(
        xml_path=input_dir,
        max_length=max_length,
        input_keys=input_keys,
        on_error=on_error,
        quiet=quiet,
        read_by="text",
    )

    output.finalize()


def preproc(trainsets: List[ds.WSDData],
            evalsets: List[ds.WSDData],
            directory: str,
            data_for_dicts_only: List[ds.WSDData] = [],
            dict_dir: str = None,
            max_length=100,
            on_error="skip",
            quiet=False,
            include_wn=False):
    # Setup
    try:
        os.makedirs(directory)
        os.makedirs(os.path.join(directory, "data"))
    except OSError:
        print("Could not create data directories")

    # Make dictionaries
    created_dicts = set_dicts(trainsets + evalsets + data_for_dicts_only, dict_dir=dict_dir, include_wn=include_wn)
    # Copy form dictionary that we need for preprocessing and training
    # Copy other dictionaries as well for backup
    for dictpath in created_dicts:
        shutil.copy(dictpath, directory)

    # Setup EWISER preproc
    dictionary = Dictionary.load(os.path.join(directory, "dict.txt"))

    # Make raganatos and create preprocessed files from those
    print("Creating/processing training data...")
    # Make and preproc our data
    train_counter = 0
    for dataset in trainsets:
        subdir_name = "train{}".format(train_counter if train_counter > 0 else "")
        _preproc_dataset(dataset,
                         directory,
                         dictionary,
                         subdir_name,
                         max_length=max_length,
                         on_error=on_error,
                         quiet=quiet)
        train_counter += 1

    # Preproc wordnet with our dict if we include it
    if include_wn:
        _raganato_preproc(os.path.join(CORPORA_PATH, "training", "orig", "examples.data.xml"),
                          os.path.join(directory, "train{}".format(train_counter)),
                          dictionary,
                          "en",
                          "sensekeys",
                          max_length=max_length,
                          on_error=on_error,
                          quiet=quiet)
        _raganato_preproc(os.path.join(CORPORA_PATH, "training", "orig", "glosses_main.data.xml"),
                          os.path.join(directory, "train{}".format(train_counter + 1)),
                          dictionary,
                          "en",
                          "sensekeys",
                          max_length=max_length,
                          on_error=on_error,
                          quiet=quiet
                          )

    print("Creating/processing eval data...")
    for i, dataset in enumerate(evalsets):
        subdir_name = "valid{}".format(i if i > 0 else "")
        _preproc_dataset(dataset,
                         directory,
                         dictionary,
                         subdir_name,
                         max_length=max_length,
                         on_error=on_error,
                         quiet=quiet)
