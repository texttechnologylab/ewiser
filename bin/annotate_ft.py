import re
import os

SEP = '@#*'
MAP_DIR = "mappings"


def load_lemma_ids(filename):
    lemma_id_dict = {}  
    id_lemma_dict = {}  
    with open(filename, encoding = "utf8") as f:
        for line in f:
            line = line.strip().split("\t")
            lemma = line[0]
            IDs = line[1:]
            lemma_id_dict[lemma] = IDs
            for ID in IDs:
                id_lemma_dict[ID] = lemma
    return lemma_id_dict, id_lemma_dict


def read_ft_file(it):
    for line in it:
        line = line.strip()
        if line:
            line = line.split("\t")
            sentence = line[1]
            sentence = re.sub(r'\s+', ' ', sentence)
            yield sentence
   
   
def read_mapping(filename):
    mapping = {}
    with open(filename, "rt", encoding="utf8") as f:
        for line in f:
            line = line.strip().split("\t")
            key = line[0]
            labels = line[1].split(" ")
            mapping[key] = labels
    return mapping
        

def annotate(filename, nlp, pos = "v"):
    verbose = False
    spacy_pos = {"v": "VERB", "n": "NOUN"}

    gold = []
    # Load gold labels
    with open(filename, "rt", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line:
                line = line.split("\t")
                gold.append(line[0].split("__")[2])
                   
    # load Lemma-Label maps
    lemma_id_dict, id_lemma_dict = load_lemma_ids(os.path.join(MAP_DIR, "verbLemmaIds"))
    
    # Load WNet to GermaNet mapping
    gnet_map = read_mapping(os.path.join(MAP_DIR, "PWN3toGNET"))
    
    preds = []
    instances = 0
    instances_annotated = 0
    lemmas_total = set()
    lemmas_annotated = set()
    lemmas_annotated_gnet = set()
    # Get EWISER predictions and map to GermaNet
    with open(filename, "rt", encoding="utf8") as f:
        sentences = read_ft_file(f)
        counter = 0
        for par in nlp.pipe(sentences, batch_size = 5):
            labels = []
            if verbose:
                print("\n")
            for token in par:
                
                if verbose:
                    # nicely print to console
                    if token.text == '\n':
                        print()
                    else:
                        new_string = token.text + SEP + token.lemma_ + SEP + token.pos_ + SEP
                        if token._.offset:
                            new_string += token._.offset
                        print(new_string, end=' ')
                # Relevant token
                if token.pos_ == spacy_pos[pos]:
                    lemmas_total.add(token.lemma_)
                # Relevant token with an annotation
                if token.pos_ == spacy_pos[pos] and token._.offset:
                    lemmas_annotated.add(token.lemma_)
                    # Get wordnet offset and convert to GermaNet labels
                    wn_offset = token._.offset.split(":")[1][:-1]
                    wn_id = "ENG30-{0:08d}-{1}".format(int(wn_offset), pos)
                    if wn_id in gnet_map:
                        gnet_labels = gnet_map[wn_id]
                        labels.extend(gnet_labels)
                        lemmas_annotated_gnet.add(token.lemma_)
                        
            preds.append(labels)
            if verbose:
                print("\n", gold[counter], labels)
            counter += 1
            if counter % 1000 == 0:
                print("Processed {} lines".format(counter))
    
    tot = 0
    hit = 0
    tot_with_preds = 0
    hit_filtered = 0
    tot_filtered = 0
    for i in range(len(gold)):
        if not gold[i] in id_lemma_dict:
            continue
        lemma = id_lemma_dict[gold[i]]
        valids = lemma_id_dict[lemma]
        tot += 1
        if preds[i]:
            tot_with_preds += 1
        if gold[i] in preds[i]:
            hit += 1
        filtered_preds = [pred for pred in preds[i] if pred in valids]
        if filtered_preds:
            tot_filtered += 1
            if gold[i] in filtered_preds:
                hit_filtered += 1
    print("{} distinct verb lemmas occurred in the data set, of which {} and {} had wordnet and germanet annotations respectively".format(len(lemmas_total), len(lemmas_annotated), len(lemmas_annotated_gnet)))
    print("{} Instances with {} correct predictions".format(tot, hit))
    print("Instances with predictions: {}".format(tot_with_preds))
    print("Instances with relevant predictions: {}".format(tot_filtered))
    print("Micro out-of-the-box: {}".format(hit/tot))
    print("Micro filtered: {}".format(hit_filtered/tot_filtered))
        #print(gold[i], preds[i])
                

def read_paragraphs(it):
    doc = []
    for line in it:
        line = line.strip()
        line = re.sub(r'\s+', ' ', line)
        if not line and doc:
            yield "\n".join(doc)
            doc.clear()
        else:
            if line:
                doc.append(line)
    if doc:
        yield "\n".join(doc)

def annotate_and_print(it_par, nlp, batch_size = 5):
    for par in nlp.pipe(it_par, batch_size=batch_size):
        for token in par:
            if token.text == '\n':
                print()
            else:
                new_string = token.text + SEP + token.lemma_ + SEP + token.pos_ + SEP
                if token._.offset:
                    new_string += token._.offset
                print(new_string, end=' ')
        print()
        print()


if __name__ == '__main__':

    from argparse import ArgumentParser
    import fileinput

    from ewiser.spacy.disambiguate import Disambiguator
    from spacy import load

    parser = ArgumentParser(description='Script to produce EWISER scores for fasttext files')
    parser.add_argument(
        'input', type=str,
        help='Input lines. FastText file or stdin (if arg == "-").')
    parser.add_argument(
        '-c', '--checkpoint', type=str,
        help='Trained EWISER checkpoint.')
    parser.add_argument(
        '-d', '--device', default='cpu',
        help='Device to use. (cpu, cuda, cuda:0 etc.)')
    parser.add_argument(
        '-l', '--language', default='de')
    parser.add_argument(
        '-s', '--spacy', default='de_core_news_sm')
    parser.add_argument(
        '-p', '--pos', default="v")
        
    args = parser.parse_args()

    wsd = Disambiguator(args.checkpoint, lang=args.language, batch_size=5, save_wsd_details=False).eval()
    wsd = wsd.to(args.device)
    nlp = load(args.spacy, disable=['ner', 'parser'])
    wsd.enable(nlp, 'wsd')

    if args.input == '-':
        lines = fileinput.input(['-'])
        pars = read_paragraphs(lines)
        annotate_and_print(pars, nlp, batch_size = 1)
    else:
        labels = annotate(args.input, nlp, pos = args.pos)
            
