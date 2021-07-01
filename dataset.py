import os
import json
import jsonpickle


"""
Defines the data format for the pipeline scripts train/eval for wsd.
Files are lists of json objects that each must contain the following fields for EWISER processsing:
    'label': String, Gold label for disambiguation. This must be either wordnet offsets or babelnet ids.
    'lemma': String, target lemma for disambiguation.
    'upos': String, target pos. Must be either 'NOUN', 'VERB', 'ADJ' or 'ADV'
    'tokens': A list of json objects which each object representing a token. Tokens must have the fields:
        'form': String, Wordform
        'lemma': String, lemma
        'pos': String, pos. Assumed to be TIGER STTS
        'upos': String, coarse upos. If this field is missing we will try to produce it from 'pos'.
        'begin': Integer, index of the first character of the token in the text.
        'end': Integer, index of the last character of the token in the text
        'is_pivot': Boolean, whether or not this is the target token for the label
        
Entries can contain additional fields without interfering, but they will not be used/considered for EWISER
"""

class wsdToken():
    def __init__(self, form, lemma, pos, begin, end, upos = None, is_pivot = False):
        self.form = form
        self.lemma = lemma
        self.pos = pos
        self.upos = upos
        self.begin = begin
        self.end = end
        self.is_pivot = is_pivot
        
        
class wsdEntry():
    def __init__(self, label, lemma, upos, tokens = [], sentence = None, source_Id = None):
        self.label = label
        self.lemma = lemma
        self.tokens = tokens
        self.sentence = sentence
        self.upos = upos
        self.source_Id = source_Id
        
        
class wsdData():
    def __init__(self, name, lang, labeltype, entries = [], json=None):
        assert labeltype in ["wnoffsets", "bnids", "gnet"]
        self.name = name
        self.entries = entries
        self.lang = lang
        self.labeltype = labeltype
        
    @classmethod
    def load(cls, json_path):
        with open(json_path, "r", encoding="utf8") as f:
            loaded=json.load(f)
            lang = loaded["lang"]
            name = loaded["name"]
            labeltype = loaded["labeltype"]
            entries = []
            for entry in loaded["entries"]:
                label = entry["label"]
                target_lemma = entry["lemma"]
                entry_pos = entry["upos"]
                
                if "sentence" in entry:
                    sentence = entry["sentence"]
                else:
                    sentence = None
                    
                l_tokens = []
                if "tokens" in entry:
                    tokens = entry["tokens"]
                    for token in tokens:
                        form = token["form"]
                        lemma = token["lemma"]
                        pos = token["pos"]
                        if "upos" in token:
                            upos = token["upos"]
                        else:
                            # TODO: Do the pos to upos conversion
                            pass
                        begin = token["begin"]
                        end = token["end"]
                        is_pivot = token["pivot"]
                        l_tokens.append(wsdToken(form, lemma, pos, begin, end, upos=upos, is_pivot=is_pivot))
                        
                entries.append(wsdEntry(label, target_lemma, entry_pos, tokens=l_tokens, sentence=sentence))
            return cls(name, lang, labeltype, entries)
                
    def save(self, outpath):
        out = jsonpickle.encode(self.entries, unpicklable=False, indent=2)
        with open(outpath, "w+", encoding="utf8") as f:
            f.write(out)
            
    def add(self, other):
        """ Merges the other dataset into this one. This can only be done if both have the same language"""
        assert self.lang == other.lang
        self.name = self.name + "+" + other.name
        self.entries.extend(other.entries)
            