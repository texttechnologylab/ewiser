import os
import argparse
import shutil

from bin.eval_wsd import predict
from wsdUtils.eval import compute_scores, pretty_print_results
from ewiser.preprocess import make_raganato, set_dicts
from wsdUtils.dataset import WSDData
from typing import List


# TODO: Test
def eval_ewiser(checkpoint_path: str,
                output_dir: str,
                lang: str = None,
                test_datasets: List[WSDData] = None,
                test_xmls: List[str] = None):
    """
    Wraps bin/eval_wsd.py, performing all the relevant dictionary updating as we go.
    We compute result scores for each test corpus separately and return them separately as well
    """
    if test_datasets is None:
        test_datasets = []
    if test_xmls is None:
        test_xmls = []

    assert len(test_datasets) > 0 or len(test_xmls) > 0
    if len(test_xmls) > 0:
        assert lang is not None, "Language has to specified manually for xml corpora. " \
                                 "All xml corpora must have the same language"

    # Turn xml paths into abspaths for the sake of it
    test_paths = [os.path.abspath(test_xml) for test_xml in test_xmls]
    # Run tests on already raganatoed corpora
    ewiser_results = {}
    if test_paths:
        ewiser_results.update(predict([checkpoint_path],
                                      test_paths,
                                      device="cuda",
                                      dict_path=os.path.join(output_dir, "dict.txt"),  # dict_path is specifically the form dict
                                      lang=lang,
                                      predictions=output_dir))

    # Get results for json sets which can use different languages
    for dataset in test_datasets:
        xml_path = make_raganato(dataset, os.path.abspath(output_dir))
        test_paths.append(xml_path)
        json_result = predict([checkpoint_path],
                              [xml_path],
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
                    for gold in gold_dict[key]:
                        golds.append(gold)  # Currently only support single label per instance
                else:
                    print("No valid gold labels for key {} in {}, skipping".format(key, test_path))
            else:
                print("Mismatch between test and train keys, missing key {} in {}, skipping".format(key, test_path))
        scores[test_path] = compute_scores(golds, preds)
    return scores


def cli():
    # Results will be printed to Console, optionally to files as latex tables
    pass
    parser = argparse.ArgumentParser(description="Training script for ewiser")

    parser.add_argument("--checkpoint", required=True, type=str,
                        help="Ewiser checkpoint")

    parser.add_argument("--output", required=True, type=str,
                        help="Directory for temporary data storage and file outputs")

    parser.add_argument("--json", required=False, type=str, nargs='*',
                        help="Json datasets to be used for testing")

    parser.add_argument("--xmls", required=False, type=str,  nargs='*',
                        help="Raganato XML style datasets to be used for testing")

    parser.add_argument("--lang", required=False, type=str,
                        help="Language of the xml corpora. Must be identical for all, undefined behaviour otherwise")

    parser.add_argument("--dict-dir", required=False, type=str,
                        help="Dictionary entries will be set using the json datasets. This argument can be specified "
                             "to add additional dictionary entries. This is necessary if you have xml datasets whose "
                             "keys are not included in the json datasets")

    args = parser.parse_args()

    # Make sure we actually have data
    assert args.json is not None or args.xmls is not None
    if args.xmls:
        assert args.lang is not None, "Must provide language for xml corpora."

    datasets = []
    for json_path in args.json:
        dataset = WSDData.load(json_path)
        datasets.append(dataset)

    try:
        os.makedirs(args.output)
    except OSError:
        print("Could not create output directories")

    # Set dictionaries. If we have created test XMLs ahead of time we have to pass dict_dir
    created_dicts = set_dicts(datasets, dict_dir=args.dict_dir, include_wn=False)
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
