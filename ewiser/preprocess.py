import os
import shutil
from lxml import etree as et
from typing import List

from fairseq.data import Dictionary

from wsdUtils.dataset import WSDData

from ewiser.fairseq_ext.data.wsd_dataset import WSDDatasetBuilder

VALID_POS = ["NOUN", "VERB", "ADJ", "ADV"]
VALID_LABELS = ["wnoffsets", "bnids"]
# These are not freely adjustable as they are referenced in EWISER code itself
dir_path = os.path.dirname(os.path.realpath(__file__))
DICT_PATH = os.path.abspath(os.path.join(dir_path, "../res/dictionaries"))
EMB_PATH = os.path.abspath(os.path.join(dir_path, "../res/embeddings"))
EDGE_PATH = os.path.abspath(os.path.join(dir_path, "../res/edges"))
CORPORA_PATH = os.path.abspath(os.path.join(dir_path, "../res/corpora"))
LANGS = ["en", "de", "es", "fr", "it"]

"""
Functions for preprocessing corpora into EWISER-appropriate formats and creating relevant data files.
Note that most functions assume that any labels appearing the testset will also appear in the training set, 
dictionaries may not be consistent otherwise
"""


def load_label_freq_dict(l_dict, dict_path):
    with open(dict_path, "rt", encoding="utf8") as f:
        for line in f:
            line = line.strip().split(" ")
            form, freq = line
            if form in l_dict:
                l_dict[form] = l_dict[form] + int(freq)
            else:
                l_dict[form] = int(freq)


def load_lemma_pos(lp_set, dict_path):
    with open(dict_path, "rt", encoding="utf8") as f:
        for line in f:
            line = line.strip().split(" ")
            lemmapos = line[0]
            lp_set.add(lemmapos)


def load_lemma2offsets(lemma2offset_dict, dict_path):
    with open(dict_path, "rt", encoding="utf8") as f:
        for line in f:
            line = line.strip().split("\t")
            lemmapos = line[0]
            offsets = line[1:]
            if lemmapos in lemma2offset_dict:
                lemma2offset_dict[lemmapos].update(offsets)
            else:
                lemma2offset_dict[lemmapos] = set(offsets)


def load_bnidmap():
    wn2bn = {}
    with open(os.path.join(DICT_PATH, "bnids_map.txt"), "r", encoding="utf8") as f:
        for line in f:
            line = line.strip().split("\t")
            bn = line[0]
            wn1 = line[1]
            wn2 = line[2]
            wn2bn[wn1] = bn
            wn2bn[wn2] = bn
    return wn2bn


def set_dicts(datasets: List[WSDData], dict_dir: str = None, include_wn: bool = False, on_error: str = "skip"):
    """ skip excludes entries which dop not have corresponding babelnet entries"""
    # dict_dir can be used to load dictionaries that were created previously and add them
    # Write out the dictionaries
    # Note: This might not work properly for english corpora?
    print("Creating dictionary entries...")

    do_offsets = False  # Updating offsets.txt causes issues with the adjacency list, seems to be a bigger rework

    # Remove current dictionaries
    paths_to_remove = []
    paths_to_remove.append("dict.txt")  # form dict
    if do_offsets:
        paths_to_remove.append("offsets.txt")  # offsets
    for lang in LANGS:
        paths_to_remove.append("lemma_pos." + lang + ".txt")
        paths_to_remove.append("lemma_pos2offsets." + lang + ".txt")

    for path in paths_to_remove:
        full_path = os.path.join(DICT_PATH, path)
        if os.path.exists(full_path):
            os.remove(full_path)

    # The following dictionary files are only used in certain circumstances that we do not care about
    # "mfs.txt" - no idea when this is relevant
    # "sensekeys.txt" - same as offsets.txt but with sensekeys, only relevant sensekey labels, which we do not allow

    # Load wordnet to babelnet mapping
    print("Loading babelnet mapping")
    wn2bn = load_bnidmap()

    # Relevant languages:
    langs = set()
    for dataset in datasets:
        langs.add(dataset.lang)
    if include_wn:
        langs.add("en")

    # Create new dictionaries ones using datasets, directory and optionally the wordnet ones
    # Setup dictionaries
    forms = {}
    lemma_pos_lang = {}
    lemma_pos2offsets_lang = {}
    offsets = {}
    for lang in langs:
        lemma_pos_lang[lang] = set()
        lemma_pos2offsets_lang[lang] = {}

    if include_wn:
        load_label_freq_dict(forms, os.path.join(DICT_PATH, "builtin_dict.txt"))  # Load dict
        if do_offsets:
            load_label_freq_dict(offsets, os.path.join(DICT_PATH, "builtin_offsets.txt"))
        if "en" not in lemma_pos_lang:
            lemma_pos_lang["en"] = set()
        load_lemma_pos(lemma_pos_lang["en"], os.path.join(DICT_PATH, "builtin_lemma_pos.en.txt"))
        # lemmapos2offsets does not exist for english, presumably built from bn?

    # Load dictionaries from directory
    if dict_dir:
        load_label_freq_dict(forms, os.path.join(dict_dir, "dict.txt"))
        load_label_freq_dict(offsets, os.path.join(dict_dir, "offsets.txt"))

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

            load_lemma2offsets(lemma_pos2offsets_lang[lang], os.path.join(dict_dir, lemma_pos_file))

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
            assert upos in VALID_POS, "Invalid pos {}, EWISER cannot process pos other than " \
                                      "NOUN, VERB, ADJ or ADV".format(upos)
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
                except KeyError as e:
                    if on_error == "raise":
                        raise KeyError("Couldn't find a babelnet id for wordnet label {}".format(label)) from e
                    elif on_error == "skip":
                        continue
                    else:
                        raise NotImplementedError from e
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

            # offsets
            assert label.startswith("wn:"), "Labels must be wordnet offsets in the format 'wn:<offset>'"
            if label in offsets:
                offsets[label] += 1
            else:
                offsets[label] = 1

    paths = []
    for lang in langs:
        with open(os.path.join(DICT_PATH, "lemma_pos." + lang + ".txt"), "w", encoding="utf8", newline="") as f:
            paths.append(os.path.realpath(f.name))
            for lemma_pos in lemma_pos_lang[lang]:
                f.write(lemma_pos + " 1\n")

        with open(os.path.join(DICT_PATH, "lemma_pos2offsets." + lang + ".txt"), "w", encoding="utf8", newline="") as f:
            paths.append(os.path.realpath(f.name))
            for lemma_pos in lemma_pos2offsets_lang[lang]:
                f.write(lemma_pos + "\t" + "\t".join(lemma_pos2offsets_lang[lang][lemma_pos]) + "\n")

    # Sort forms by frequency.
    forms_out = [item[0] + " " + str(item[1]) for item in sorted(forms.items(), key=lambda item: item[1], reverse=True)]
    with open(os.path.join(DICT_PATH, "dict.txt"), "w", encoding="utf8", newline="") as f:
        paths.append(os.path.realpath(f.name))
        for line in forms_out:
            f.write(line + "\n")

    if do_offsets:
        offsets_out = [item[0] + " " + str(item[1]) for item in sorted(offsets.items(), key=lambda item: item[1], reverse=True)]
        with open(os.path.join(DICT_PATH, "offsets.txt"), "w", encoding="utf8", newline="") as f:
            paths.append(os.path.realpath(f.name))
            for line in offsets_out:
                f.write(line + "\n")

    # Return the paths to all dictionaries we created, so we can back them up
    return paths


def make_raganato(dataset: WSDData, directory, on_error: str = "skip"):
    # Writes out the dataset in the raganto xml format. This is used by EWISER for eval and its own preprocessing
    # XML filename is be dataset.name + ".data.xml"
    root = et.Element("corpus")

    valid_wnids = load_bnidmap()
    gold_keys = []

    # Map entries to sentences
    sentence_entry_map = {}
    for entry in dataset.entries:
        idx = entry.sentence_idx
        if idx in sentence_entry_map:
            sentence_entry_map[idx].append(entry)
        else:
            sentence_entry_map[idx] = [entry]

    doc_counter = 1
    doc_id = "d{:07d}".format(doc_counter)
    text = et.SubElement(root, "text")
    text.set("id", doc_id)
    source_id = dataset.name
    text.set("source", source_id)
    sent_counter = 0
    for sentence_idx in sentence_entry_map:
        # Grab tokens and source from first entry in list, are identical for all
        tokens = sentence_entry_map[sentence_idx][0].tokens

        sent_id = "s{:03d}".format(sent_counter)
        sentence_element = et.SubElement(text, "sentence")
        sentence_element.set("id", doc_id + "." + sent_id)

        instance_counter = 0
        for token in tokens:
            # check if this token is a pivot in any of the entries for this sentence
            label = None
            for entry in sentence_entry_map[sentence_idx]:
                # Filter out illegal wordnet labels
                if not entry.label.startswith("bn") and entry.label not in valid_wnids:
                    if on_error == "skip":
                        print("Skipping entry with illegal label {}".format(entry.label))
                        continue
                    else:
                        raise KeyError("Invalid label!")
                if token.begin == entry.pivot_start and token.end == entry.pivot_end:
                    label = entry.label
            if label is not None:
                instance_counter += 1
                instance_id = "h{:03d}".format(instance_counter)
                instance = et.SubElement(sentence_element, "instance")
                instance.set("id", doc_id + "." + sent_id + "." + instance_id)
                instance.set("lemma", token.lemma)
                instance.set("pos", token.upos)
                instance.text = token.form
                gold_keys.append(doc_id + "." + sent_id + "." + instance_id + "\t" + label)
            else:
                word = et.SubElement(sentence_element, "wf")
                word.set("lemma", token.lemma)
                try:
                    word.set("pos", token.upos)
                except TypeError as e:
                    print(token)
                    print(repr(token.pos), repr(token.upos))
                    raise e
                word.text = token.form
        sent_counter += 1

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


def _preproc_dataset(dataset: WSDData,
                     directory: str,
                     dictionary,
                     subdir_name: str,
                     on_error: str = "skip",
                     **kwargs):

    assert dataset.labeltype in VALID_LABELS, "Labels must be one of {}".format(VALID_LABELS)
    os.mkdir(os.path.join(directory, "data", subdir_name))
    make_raganato(dataset, os.path.join(directory, "data", subdir_name), on_error=on_error)

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


def preproc(trainsets: List[WSDData],
            evalsets: List[WSDData],
            directory: str,
            data_for_dicts_only: List[WSDData] = [],
            dict_dir: str = None,
            max_length=100,
            on_error="skip",
            quiet=False,
            include_wn=False):
    # Setup
    # TODO: Should absolutely also split built in corpora, don't know how to just yet. Maybe load as own corpus?
    #  This will lead to out-of-memory issues otherwise
    #  Could also fix the loading in ewiser code itself somehow
    try:
        os.makedirs(directory)
        os.makedirs(os.path.join(directory, "data"))
    except OSError:
        print("Could not create data directories")

    # Make dictionaries
    created_dicts = set_dicts(trainsets + evalsets + data_for_dicts_only,
                              dict_dir=dict_dir,
                              include_wn=include_wn,
                              on_error=on_error)
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
        outputs = [(os.path.join(CORPORA_PATH, "training", "orig", "examples.data.xml"),
                    os.path.join(directory, "train{}".format(train_counter))),
                   (os.path.join(CORPORA_PATH, "training", "orig", "glosses_main.data.xml"),
                    os.path.join(directory, "train{}".format(train_counter + 1)))]
                   #(os.path.join(CORPORA_PATH, "training", "orig", "glosses_main.untagged.data.xml"),
                   # os.path.join(directory, "train{}".format(train_counter + 2)))]
        for (data_path, outdir) in outputs:
            _raganato_preproc(data_path,
                              outdir,
                              dictionary,
                              "en",
                              "sensekeys",
                              max_length=max_length,
                              on_error=on_error,
                              quiet=quiet)

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
