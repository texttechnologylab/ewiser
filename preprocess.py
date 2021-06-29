import os
from dataset import wsdToken, wsdEntry, wsdData, DICT_PATH

class ewiser():
    
    VALID_POS = ["NOUN", "VERB", "ADJ", "ADJ"]

    def set_dicts(self, datasets, built_ins):
        # TODO: built ins
        # TODO: Write out the dictionaries
        # Note: This might not work properly for english corpora?
        
        # Relevant languages:
        langs = set()
        # Remove current dictionaries
        for dataset in datasets:
            langs.add(dataset.lang)
        os.remove(os.path.join(DICT_PATH, "dict.txt"))
        for lang in langs:
            os.remove(os.path.join(DICT_PATH, "lemma_pos." + lang + ".txt"))
            os.remove(os.path.join(DICT_PATH, "lemma_pos2offsets." + lang + ".txt"))
            
        # Load wordnet to babelnet mapping
        wn2bn = {}
        with open(os.path.join(DICT_PATH, "bnids_map.txt"), "r", encoding="utf8") as f:
            for line in f:
                line = line.strip().split("\t")
                bn = line[0]
                wn = line[2]
                wn2bn[wn] = bn
    
        # Create new current ones using datasets and optionally built ins.
        lemma_pos_lang = {}
        lemma_pos2offsets_lang = {}
        for lang in langs:
            lemma_pos_lang[lang] = set()
            lemma_pos2offsets_lang[lang] = {}
        for dataset in datasets:
            lang = dataset.lang
            for entry in dataset:
                lemma = entry.lemma
                upos = entry.upos
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
                    
                    
        
        # Update lemma_pos and lemma_pos2offsets for each language
        # Built_ins will load and include the dicts that ewiser came with
        
    
    def preproc(self, datasets, directory):
        # Make raganato xmls from datasets
        # Update dictionaries
        # Call the builtin preprocessing function on xmls

        
def train_test_split(dataset):
    # Split dataset into train/eval/test datasets with stratification using the gold labels
    
    
def tokenize(dataset):
    # Run the Java UIMA thingy somehow
