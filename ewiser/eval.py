import os.path

from bin.eval_wsd import predict
from eval import compute_scores, pretty_print_results
from preprocess import make_raganato, set_dicts
from dataset import WSDData
from typing import List


def eval_ewiser(checkpoint_path: str, output_dir: str, test_datasets: List[WSDData] = [], test_xmls: List[str] = [], dict_dir: str = None):
    """
    Wraps bin/eval_wsd.py, performing all the relevant dictionary updating as we go.
    We compute result scores for each test corpus separately and return them separately as well
    """
    assert len(test_datasets) > 0 or len(test_xmls) > 0
    # TODO: Optionally load dictionaries from training/some other directory (provided as argument)
    # Turn datasets into raganatos
    for dataset in test_datasets:
        xml_path = make_raganato(dataset, os.path.abspath(output_dir))
        test_xmls.append(xml_path)
    # TODO: Set dictionaries
    # TODO: Copy dictionaries into output directory
    # Turn xml paths into abspaths for the sake of it
    test_paths = [os.path.abspath(test_xml) for test_xml in test_xmls]
    ewiser_results = predict([checkpoint_path])
    scores = {}
    for test_path in test_paths:
        gold_dict, pred_dict = ewiser_results[test_path]
        golds = []
        preds = []
        for key, value in pred_dict.items():
            if key in gold_dict:
                if gold_dict[key]:
                    preds.append(value)
                    if len(gold_dict[key]) == 1:
                        golds.append(gold_dict[key][0])
                    else:
                        for gold in gold_dict[key]:
                            if gold == value:
                                golds.append(gold)
                                break
                else:
                    print("No valid gold labels for key {} in {}, skipping".format(key, test_path))
            else:
                print("Mismatch between gold and data keys, missing key {} in {}}, skipping".format(key, test_path))
        scores[test_path] = compute_scores(golds, preds)
    return scores

def cli():
    # Arguments are: Model checkpoint, Test directories OR XMLs, optionally directory with lemmapos dictionaries for filtering
    # Results will be printed to Console, optionally to files as latex tables
    # TODO: All of it
    pass


if __name__=="__main__":
    cli()
