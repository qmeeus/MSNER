# MSNER
Repository for the paper "MSNER: A Multilingual Speech Dataset for Named Entity Recognition"

## Prediction file format
The `evaluate.py` script expects files of a specific format: one tab-separated text file (TSV) for the references and one for the predictions.
Each file must have at least a column containing entities and optionally, a column containing the transcript.
The entities must be formatted as follows: `[('ENTITY_TYPE1', 'ENTITY_STRING1'), ('ENTITY_TYPE2', 'ENTITY_STRING2')]` or `[]` for no entities.
The column names in the TSV file can be passed to the script with `--text_column_name` and `--entity_column_name` (defaults to `text` and `entities`).
Refer to the code for additional information.

Example:
```
$ head predictions/nl/targets.tsv
audio_id        sentence        entities
20110705-0900-PLENARY-8-nl_20110705-16:29:24_6   En daar die eigen middelen het ook mogelijk maken om de bijdragen van de staten te verminderen, is het meteen ook een manier om bij te dragen tot hun begroting.       []
20170201-0900-PLENARY-9-nl_20170201-16:51:24_8   Je hebt een zeer goede en sterke Europese gedreven governance nodig en ook op dat punt zullen wij samen met de andere fracties versterkte voorstellen indienen.        [('group', 'Europese')]
20090504-0900-PLENARY-13-nl_20090504-21:08:18_3  Tegen die achtergrond vindt het voornemen dat nu in de Ministerraad is geuit, om niet alleen de zelfstandige bestuurder uit te sluiten van de werkingssfeer, maar ook om niets afdoende te doen tegen de schijnzelfstandigen, in de ogen van de PSE Fractie geen genade.       [('organization', 'Ministerraad'), ('organization', 'PSE Fractie')]
```

A simple method for generating the file in the right format from the provided json files:
```python
import json

def join(x):
    return "".join(x).strip()

with open("targets.tsv", "w") as outfile:
    print("\t".join(["audio_id", "text", "entities"]), file=outfile)
    with open("data/transcript-de-test-ann-ontonotes-v1.jsonl") as infile:
        for line in f.readlines():
            data = json.loads(line)
            text = join(data["input"])
            entities = str([(annot["type"], join(annot["entity"])) for annot in data["annotation"]])])
            print("\t".join([data["id"], text, entities, file=outfile)
```

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
