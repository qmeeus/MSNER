import editdistance
import json
import numpy as np
import pandas as pd
import sys

from pathlib import Path
from typing import List
from collections import defaultdict
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from typing import Dict, List, Union, Tuple


NamedEntity = Tuple[str, str, int]
Json = Dict[str, Any]

whisper_norm = BasicTextNormalizer()


def get_ner_scores(all_gt:List[NamedEntity], all_predictions[List[NamedEntity]]) -> Json:
    """
    Evalutes per-label and overall (micro and macro) metrics of precision, recall, and fscore

    Input:
        all_gt/all_predictions:
            List of list of tuples: (label, phrase, identifier)
            Each list of tuples correspond to a sentence:
                label: entity tag
                phrase: entity phrase
                tuple_identifier: identifier to differentiate repeating (label, phrase) pairs

    Returns:
        Dictionary of metrics

    Example:
        List of GT (label, phrase) pairs of a sentence: [(GPE, "eu"), (DATE, "today"), (GPE, "eu")]
        all_gt: [(GPE, "eu", 0), (DATE, "today", 0), (GPE, "eu", 1)]
    """
    metrics = {}
    stats = get_ner_stats(all_gt, all_predictions)
    num_correct, num_gt, num_pred = 0, 0, 0
    prec_lst, recall_lst, fscore_lst = [], [], []
    for tag_name, tag_stats in stats.items():
        precision, recall, fscore = get_metrics(
            np.sum(tag_stats["tp"]),
            np.sum(tag_stats["gt_cnt"]),
            np.sum(tag_stats["pred_cnt"]),
        )
        _ = metrics.setdefault(tag_name, {})
        metrics[tag_name]["precision"] = precision
        metrics[tag_name]["recall"] = recall
        metrics[tag_name]["fscore"] = fscore

        num_correct += np.sum(tag_stats["tp"])
        num_pred += np.sum(tag_stats["pred_cnt"])
        num_gt += np.sum(tag_stats["gt_cnt"])

        prec_lst.append(precision)
        recall_lst.append(recall)
        fscore_lst.append(fscore)

    precision, recall, fscore = get_metrics(num_correct, num_gt, num_pred)
    metrics["overall_micro"] = {}
    metrics["overall_micro"]["precision"] = precision
    metrics["overall_micro"]["recall"] = recall
    metrics["overall_micro"]["fscore"] = fscore

    metrics["overall_macro"] = {}
    metrics["overall_macro"]["precision"] = np.mean(prec_lst)
    metrics["overall_macro"]["recall"] = np.mean(recall_lst)
    metrics["overall_macro"]["fscore"] = np.mean(fscore_lst)

    return metrics


def get_ner_stats(all_gt:List[NamedEntity], all_predictions:List[NamedEntity]) -> Json:
    stats = {}
    cnt = 0
    for gt, pred in zip(all_gt, all_predictions):
        entities_true = defaultdict(set)
        entities_pred = defaultdict(set)
        for type_name, entity_info1, entity_info2 in gt:
            entities_true[type_name].add((entity_info1, entity_info2))
        for type_name, entity_info1, entity_info2 in pred:
            entities_pred[type_name].add((entity_info1, entity_info2))
        target_names = sorted(set(entities_true.keys()) | set(entities_pred.keys()))
        for tag_name in target_names:
            _ = stats.setdefault(tag_name, {})
            _ = stats[tag_name].setdefault("tp", [])
            _ = stats[tag_name].setdefault("gt_cnt", [])
            _ = stats[tag_name].setdefault("pred_cnt", [])
            entities_true_type = entities_true.get(tag_name, set())
            entities_pred_type = entities_pred.get(tag_name, set())
            stats[tag_name]["tp"].append(len(entities_true_type & entities_pred_type))
            stats[tag_name]["pred_cnt"].append(len(entities_pred_type))
            stats[tag_name]["gt_cnt"].append(len(entities_true_type))
    return stats


def safe_divide(numerator:List[Union[int,float]], denominator:List[Union[int,float]]) -> List[float]:
    numerator = np.array(numerator)
    denominator = np.array(denominator)
    mask = denominator == 0.0
    denominator = denominator.copy()
    denominator[mask] = 1  # avoid infs/nans
    return numerator / denominator


def ner_error_analysis(all_gt:List[NamedEntity], all_predictions:List[NamedEntity], gt_text) -> Json:
    """
    Print out predictions and GT
    all_gt: [GT] list of tuples of (label, phrase, identifier idx)
    all_predictions: [hypothesis] list of tuples of (label, phrase, identifier idx)
    gt_text: list of GT text sentences
    """
    analysis_examples_dct = {}
    analysis_examples_dct["all"] = []
    for idx, text in enumerate(gt_text):
        if isinstance(text, list):
            text = " ".join(text)
        gt = all_gt[idx]
        pred = all_predictions[idx]
        entities_true = defaultdict(set)
        entities_pred = defaultdict(set)
        for type_name, entity_info1, entity_info2 in gt:
            entities_true[type_name].add((entity_info1, entity_info2))
        for type_name, entity_info1, entity_info2 in pred:
            entities_pred[type_name].add((entity_info1, entity_info2))
        target_names = sorted(set(entities_true.keys()) | set(entities_pred.keys()))
        analysis_examples_dct["all"].append("\t".join([text, str(gt), str(pred)]))
        for tag_name in target_names:
            _ = analysis_examples_dct.setdefault(tag_name, [])
            entities_true_type = entities_true.get(tag_name, set())
            entities_pred_type = entities_pred.get(tag_name, set())
            num_gt = len(entities_true_type)
            num_correct = len(entities_true_type & entities_pred_type)

            new_gt = [(item1, item2) for item1, item2, _ in gt]
            new_pred = [(item1, item2) for item1, item2, _ in pred]
            analysis_examples_dct[tag_name].append(
                "\t".join([text, str(new_gt), str(new_pred)])
            )

    return analysis_examples_dct


def get_metrics(num_correct, num_gt, num_pred):
    precision = safe_divide([num_correct], [num_pred])
    recall = safe_divide([num_correct], [num_gt])
    fscore = safe_divide([2 * precision * recall], [(precision + recall)])
    return precision[0], recall[0], fscore[0][0]


def remap_entities(entities):

    def _map(*entities):
        new_entities = []
        for tag, entity in entities:
            i = 0
            while True:
                if (tag, entity, i) not in new_entities:
                    new_entities.append((tag.replace(" ", "_").lower(), whisper_norm(entity), i))
                    break
                i += 1
        return new_entities

    if type(entities) is str:
        entities = eval(entities)

    if entities is None:
        entities = []

    if type(entities) is not list:
        raise TypeError(f"Unexpected type for entities {type(entities)}")

    return _map(*entities)


def eval_ner(prediction_dir_or_file, target_file=None):

    prediction_dir_or_file = Path(prediction_dir_or_file)
    if prediction_dir_or_file.is_dir():
        model_dir = prediction_dir_or_file
        predictions = pd.concat([
            pd.read_csv(fn, sep="\t").assign(checkpoint=int(fn.parent.name.split("-")[1]), subset=fn.stem.split("_")[1])
            for fn in model_dir.rglob("predictions_*.tsv")
        ], axis=0)
    else:
        model_dir = prediction_dir_or_file.parent
        predictions = pd.read_csv(prediction_dir_or_file, sep="\t").assign(checkpoint=-1, subset="test")

    if target_file is not None:
        assert "id" in predictions.columns
        target = pd.read_csv(target_file, sep="\t").set_index("id")
        predictions = predictions.join(target, on="id", how="inner")

    if "entities" not in predictions.columns:
        raise ValueError("Missing target")

    all_gt = predictions["entities"].map(remap_entities).tolist()
    all_predictions = predictions["ner"].map(remap_entities).tolist()
    entity_scores = get_ner_scores(all_gt, all_predictions)

    all_gt_label, all_preds_label = (
        [[(typ, "DUMMY", i) for typ, _, i in entities] for entities in annotations]
        for annotations in (all_gt, all_predictions)
    )

    label_scores = get_ner_scores(all_gt_label, all_preds_label)

    print(json.dumps({"entity": entity_scores, "label": label_scores}, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--prediction_file", type=str, default=None)
    parser.add_argument("--target_file", type=str, default=None)
    args = parser.parse_args()
    if not (bool(args.model_dir) ^ bool(args.prediction_file)):
        raise ValueError("Either model_dir or prediction_file should be specified")
    eval_ner(args.model_dir or args.prediction_file, target_file=args.target_file)

