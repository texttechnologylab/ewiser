import os
import argparse
import shutil

from bin.eval_wsd import predict
from eval import compute_scores, pretty_print_results
from preprocess import make_raganato, set_dicts
from dataset import WSDData
from typing import List


# TODO: Test
def eval_ewiser(checkpoint_path: str,
                output_dir: str,
                lang: str = None,
                test_datasets: List[WSDData] = [],
                test_xmls: List[str] = []):
    """
    Wraps bin/eval_wsd.py, performing all the relevant dictionary updating as we go.
    We compute result scores for each test corpus separately and return them separately as well
    """
    assert len(test_datasets) > 0 or len(test_xmls) > 0
    if len(test_xmls) > 0:
        assert lang is not None, "Language has to specified manually for xml corpora"

    # Turn xml paths into abspaths for the sake of it
    test_paths = [os.path.abspath(test_xml) for test_xml in test_xmls]
    # Run tests on already raganatod corpora
    ewiser_results = predict([checkpoint_path],
                             test_paths,
                             device="cuda",
                             dict_path=os.path.join(output_dir, "dict.txt"),  # dict_path is specifically the form dict
                             lang=lang,
                             predictions=output_dir)

    # Get results for json sets which can use different languages
    for dataset in test_datasets:
        xml_path = make_raganato(dataset, os.path.abspath(output_dir))
        test_paths.append(xml_path)
        json_result = predict([checkpoint_path],
                              xml_path,
                              device="cuda",
                              dict_path=os.path.join(output_dir, "dict.txt"),
                              lang=dataset.lang,
                              predictions=output_dir)
        ewiser_results.update(json_result)

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
    # Results will be printed to Console, optionally to files as latex tables
    pass
    parser = argparse.ArgumentParser(description="Training script for ewiser")

    parser.add_argument("--checkpoint", required=True, type=str, help=
                        "Ewiser checkpoint")

    parser.add_argument("--output", required=True, type=str, help=
                        "Directory for temporary data storage and file outputs")

    parser.add_argument("--jsons", required=False, type=str, help=
                        "Json datasets to be used for testing")

    parser.add_argument("--xmls", required=False, type=str, help=
                        "Raganato XML style datasets to be used for testing")

    parser.add_argument("--lang", required=True, type=str, help=
                        "Language of the xml corpora. Must be identical for all, undefined behaviour otherwise")

    parser.add_argument("--dicts", required=False, type=str, help=
                        "Dictionary entries will be set using the json datasets. This argument can be specified to add"
                        " additional dictionary entries. This is necessary if you have xml datasets whose keys are not"
                        " included in the json datasets")

    args = parser.parse_args()

    # Make sure we actually have data
    assert args.jsons is not None or args.xmls is not None

    datasets = []
    for json_path in args.jsons:
        dataset = WSDData.load(json_path)
        datasets.append(dataset)

    # Set dictionaries. If we have created test XMLs ahead of time we have to pass dict_dir
    created_dicts = set_dicts(datasets, dict_dir=args.dicts)
    # Copy form dictionary that we need for preprocessing and training
    # Copy other dictionaries as well for backup
    for dictpath in created_dicts:
        shutil.copy(dictpath, args.output)

    scores = eval_ewiser(args.checkpoint,
                         args.output,
                         lang=args.lang,
                         test_datasets=datasets,
                         test_xmls=args.xmls)

    for corpus in scores:
        print("Results for {}:".format(corpus))
        pretty_print_results(scores[corpus])


if __name__ == "__main__":
    cli()
