import os
import shutil
import lxml
from lxml import etree as ET
from dataset import wsdToken, wsdEntry, wsdData

from fairseq.data import Dictionary

from ewiser.fairseq_ext.data.dictionaries import MFSManager, ResourceManager, DEFAULT_DICTIONARY
from ewiser.fairseq_ext.data.wsd_dataset import WSDDataset, WSDDatasetBuilder

class ewiser():
    
    VALID_POS = ["NOUN", "VERB", "ADJ", "ADJ"]
    VALID_LABELS = ["wnoffsets", "bnids"]
    DICT_PATH = "res/dictionaries" # This is not freely adjustable as it is referenced in EWISER code itself

    def set_dicts(self, datasets, built_ins = False):
        # TODO: built ins
        # Write out the dictionaries
        # Note: This might not work properly for english corpora?
        
        print("Creating dictionary entries...")
        # Relevant languages:
        langs = set()
        # Remove current dictionaries
        for dataset in datasets:
            langs.add(dataset.lang)
        os.remove(os.path.join(self.DICT_PATH, "dict.txt"))
        for lang in langs:
            os.remove(os.path.join(self.DICT_PATH, "lemma_pos." + lang + ".txt"))
            os.remove(os.path.join(self.DICT_PATH, "lemma_pos2offsets." + lang + ".txt"))
            
        # Load wordnet to babelnet mapping
        wn2bn = {}
        with open(os.path.join(self.DICT_PATH, "bnids_map.txt"), "r", encoding="utf8") as f:
            for line in f:
                line = line.strip().split("\t")
                bn = line[0]
                wn = line[2]
                wn2bn[wn] = bn
    
        # Create new current ones using datasets and optionally built ins.
        forms = {}
        lemma_pos_lang = {}
        lemma_pos2offsets_lang = {}
        for lang in langs:
            lemma_pos_lang[lang] = set()
            lemma_pos2offsets_lang[lang] = {}
        for dataset in datasets:
            lang = dataset.lang
            assert dataset.labetype in self.VALID_LABELS, "Labels must be one of {}".format(self.VALID_LABELS)
            for entry in dataset:
                # lemma pos
                lemma = entry.lemma
                upos = entry.upos
                assert "tokens" in entry, "Entries must have list of tokens"
                assert upos in self.VALID_POS, "EWISER cannot process pos other than NOUN, VERB, ADJ or ADV"
                if upos == "NOUN":
                    pos = "n"
                elif upos == "VERB":
                    pos = "v"
                elif upos == "ADJ":
                    pos = "a"
                elif upos == "ADV":
                    pos = "r"
                lemma_pos_key = lemma + "#" + pos
                lemma_pos_lang[lang].add(lemma_pos_key)
                
                # possible babelnet ids
                label = entry.label
                # pos2offsets have to be babelnet ids.
                assert label.startswith("wn:") or label.startswith("bn:"), "Ewiser labels must be wordnet offsets or babelnet ids in the format 'wn:<offset>' or 'bn:<id>'"
                if label.startswith("wn:"):
                    # Translate from wordnet to bn using the provided mapping
                    try:
                        bnlabel = wn2bn[wn]
                    except IndexError as e:
                        raise IndexError("Couldn't find a babelnet id for wordnet label {}".format(label)) from e
                elif label.startswith("bn:"):
                    bnlabel = label
                if lemma_pos_key in lemma_pos2offsets_lang[lang]:
                    lemma_pos2offsets_lang[lang][lemma_pos_key].add(bnlabel)
                else:
                    lemma_pos2offsets_lang[lang][lemma_pos_key] = set([bnlabel])
                    
                # form dict
                
                for token in entry.tokens:
                    form = token.form
                    if form in forms:
                        forms[form] += 1
                    else:
                        forms[form] = 1
                
        
        for lang in langs:
            with open(os.path.join(self.DICT_PATH, "lemma_pos." + lang + ".txt"), "w", encoding="utf8") as f:
                for lemma_pos in lemma_pos_lang[lang]:
                    f.write(lemma_pos + " 1\n")
            
            with open(os.path.join(self.DICT_PATH, "lemma_pos2offsets." + lang + ".txt"), "w", encoding = "utf8") as f:
                for lemma_pos in lemma_pos2offsets_lang[lang]:
                    f.write(lemma_pos + "\t" + "\t".join(lemma_pos2offsets_lang[lang][lemma_pos]))
        
        # Sort forms by frequency.
        out = [item[0] + " " + str(item[1]) for item in sorted(forms.items(), key=lambda item: item[1], reverse=True)]
        with open(os.path.join(self.DICT_PATH, "dict.txt"), "w", encoding="utf8") as f:
            for line in out:
                f.write(line + "\n")
        
        # TODO: Built_ins should load and include the dicts that ewiser came with
        
    
    def make_raganato(self, dataset, directory):
        # Writes out the dataset in the raganto xml format. This is used by EWISER for its own preprocessing
        # XML filename is be dataset.name + ".data.xml"
        # TODO: All of it
        root = ET.Element("corpus")
        
        gold_keys = []
        
        doc_counter = 0
        for entry in dataset.entries:
            doc_counter += 1
            doc_id = "d{:07d}".format(doc_counter)
            text = ET.SubElement(root, "text")
            text.set("id", doc_id)
            text.set("source", dataset.name)
            
            sent_id = "s001"
            sentence = ET.SubElement(text, "sentence")
            sentence.set("id", doc_id + "." + sent_id)
            
            for token in entry.tokens:
                instance_counter = 0
                if token.is_pivot:
                    instance_counter += 1
                    instance_id = "h{:3d}".format(instance_counter)
                    instance = ET.SubElement(sentence, "instance")
                    instance.set("id", doc_id + "." + sent_id + "." + instance_id)
                    instance.set("lemma", token.lemma)
                    instance.set("pos", token.upos)
                    instance.text = token.form
                    
                    gold_keys.append(doc_id + "." + sent_id + "." + instance_id + "\t" + entry.label)
                else:
                    word = ET.SubElement(sentence, "wf")
                    word.set("lemma", token.lemma)
                    word.set("pos", token.upos)
                    word.text = token.form
                    
        # Write out files
        tree = ET.ElementTree(root)
        tree.write(os.path.join(directory, dataset.name + ".data.xml"), encoding="utf8", pretty_print=True, xml_declaration=True)
        
        with open(os.path.join(directory, dataset.name + ".gold.key.txt"), "w", encoding="utf8") as f:
            for key in gold_keys:
                f.write(key + "\n")
                
    
    def preproc(self, trainsets, evalsets, directory, max_length = 100, on_error="skip", quiet = quiet):
        # Setup
        try:
            os.makedirs(directory)
            os.makedirs(os.path.join(directory, "data"))
        except OSError:
            print("Could not create data directories")
        
        # Make dictionaries
        self.set_dicts(datasets)
        # Copy form dictionary that we need for preprocessing and training
        shutil.copy(os.path.join(self.DICT_PATH, "dict.txt"), directory)
            
        # Setup EWISER preproc
        dictionary = Dictionary.load(os.path.join(directory, "dict.txt"))
        output_dictionary = ResourceManager.get_senses_dictionary(use_synsets=True)
        
        # Make raganatos and create preprocessed files from those
        print("Creating/processing training data...")
        for i, dataset in enumerate(trainsets):
            assert dataset.labetype in VALID_LABELS, "Labels must be one of {}".format(VALID_LABELS)
            dirname = "train{}".format(i if i > 0 else "")
            os.mkdir(os.path.join(directory, "data", dirname))
            self.make_raganato(datasets, os.path.join(directory, "data", dirname))
            
            output = WSDDatasetBuilder(
                os.path.join(directory, dirname),
                dictionary=dictionary,
                use_synsets=True,
                keep_string_data=True,
                lang=dataset.lang)
                
            # adjust keys:
            input_keys = None
            if dataset.labetype == "wnoffsets":
                input_keys = "offsets"
            elif dataset.labetype == "bnids":
                input_keys = "bnids"
                
            output.add_raganato(
                xml_path=os.path.join(directory, dirname, dataset.name + ".data.xml"),
                max_length=max_length,
                input_keys=input_keys,
                on_error=on_error,
                quiet=quiet,
                read_by="text",
            )
            
            output.finalize()
            
        print("Creating/processing eval data...")
        for i, dataset in enumerate(evalsets):
            assert dataset.labetype in VALID_LABELS, "Labels must be one of {}".format(VALID_LABELS)
            dirname = "valid{}".format(i if i > 0 else "")
            os.mkdir(os.path.join(directory, "data", dirname))
            self.make_raganato(datasets, os.path.join(directory, "data", dirname))

            output = WSDDatasetBuilder(
                os.path.join(directory, dirname),
                dictionary=dictionary,
                use_synsets=True,
                keep_string_data=True,
                lang=dataset.lang)
                
            # adjust keys:
            input_keys = None
            if dataset.labetype == "wnoffsets":
                input_keys = "offsets"
            elif dataset.labetype == "bnids":
                input_keys = "bnids"
                
            output.add_raganato(
                xml_path=os.path.join(directory, dirname, dataset.name + ".data.xml"),
                max_length=max_length,
                input_keys=input_keys,
                on_error=on_error,
                quiet=quiet,
                read_by="text",
            )
            
            output.finalize()

        
def train_test_split(dataset, ratio_eval = 0.2, ratio_test = 0.2):
    # Split dataset into train/eval/test datasets with stratification using the gold labels
    assert ratio_eval + ratio_test <= 1.0
    assert ratio_eval >= 0.0
    assert ratio_test >= 0.0
    
    entries_by_label = {}
    
    for entry in dataset.entries:
        label = entry.label
        if label in entries_by_label:
            entries_by_label[label].append(entry)
        else:
            entries_by_label[label] = [entry]
    
    trainset = wsdData(dataset.name + "_train", dataset.lang, dataset.labeltype, entries = [])
    evalset = wsdData(dataset.name + "_eval", dataset.lang, dataset.labeltype, entries = [])
    testset = wsdData(dataset.name + "_test", dataset.lang, dataset.labeltype, entries = [])
    
    for label, entries in entries_by_label.items():
        # Dump labels with single instance
        if len(entries) == 1:
            continue
        
        # Fix sizes for low count labels to ensure we have at least one in train/eval/test if at all possible
        eval_size = math.floor(len(entries)*ratio_eval)
        if eval_size == 0 and ratio_eval > 0.0 and len(entries) >= 3:
            eval_size = 1
            
        test_size = math.floor(len(entries)*ratio_test)
        if test_size == 0 and ratio_test > 0.0 and len(entries) >= 2:
            test_size = 1
            
        train_size = len(entries) - eval_size - test_size
        if train_size == 0 and ratio_eval + ratio_test < 1.0:
            if eval_size > 1:
                eval_size = eval_size -1
                train_size += 1
            elif test_size > 1:
                test_size = test_size - 1
                train_size += 1
            elif eval_size == 1 and len(entries) == 2:
                eval_size = 0
                train_size = 1
                
                
        evalset.entries.extend(entries[:eval_size])
        testset.entries.extend(entries[eval_size:eval_size+test_size])
        trainset.entries.extend(entries[eval_size+test_size:])
    
    return trainset, evalset, testset
    
    
def tokenize(dataset):
    # Run the Java UIMA thingy somehow
    pass