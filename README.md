# MSNER
Repository for the paper "MSNER: A Multilingual Speech Dataset for Named Entity Recognition"

## Prediction file format
The `evaluate.py` script expects files of a specific format: one tab-separated text file (TSV) for the references and one for the predictions.
Each file must have at least a column containing entities and optionally, a column containing the transcript.
The entities must be formatted as follows: `[('ENTITY_TYPE1', 'ENTITY_STRING1'), ('ENTITY_TYPE2', 'ENTITY_STRING2')]` or `[]` for no entities.
The column names in the TSV file can be passed to the script with `--text_column_name` and `--entity_column_name` (defaults to `text` and `entities`).
Refer to the code for additional information.

## Usage
```bash
$ python src/evaluate.py \
  --hyps predictions.tsv --refs targets.tsv \
  --entity_column_name entities --text_column_name sentence \
  --normalize | tee test_metrics.json
{
  "entity": {
    "geopolitical_area":{"precision":0.7887931034482759,"recall":0.7065637065637066,"fscore":0.7454175152749491},
    "date":{"precision":0.6270096463022508,"recall":0.6351791530944625,"fscore":0.6310679611650484},
    "organization":{"precision":0.5030211480362538,"recall":0.5362318840579711,"fscore":0.519095869056898},
    "group":{"precision":0.6041666666666666,"recall":0.3717948717948718,"fscore":0.46031746031746035},
    "person":{"precision":0.34558823529411764,"recall":0.3821138211382114,"fscore":0.3629343629343629},
    ...,
    "overall_micro":{"precision":0.8601532567049809,"recall":0.7666476949345475,"fscore":0.8107132109539574},
    "overall_macro":{"precision":null,"recall":0.5665167064112204,"fscore":null}
  },
  "wer":0.16122358504507167
}
```

## Citation
```
@inproceedings{MSNER,
author = {Meeus, Quentin and Moens, Marie-Francine and Van hamme, Hugo},
booktitle = {20th Joint ACL-ISO Workshop on Interoperable Semantic Annotation at LREC-COLING},
title = {{MSNER: A Multilingual Speech Dataset for Named Entity Recognition}},
year = {2024}
}
```
