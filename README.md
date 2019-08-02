# Speculation detection of concepts in Dutch clinical text

This repository contains the source code for a Dutch speculation detector for clinical text developed in the scope of the  
[ACCUMULATE](https://github.com/clips/accumulate) project. The speculation detection is performed specifically for detected clinical concepts within a sentence, rather than on the token level.

## Requirements

* Python 3
* [Frog](https://languagemachines.github.io/frog/)

## Usage

### Data preprocessing

This module processes raw clinical text using Frog and integrates the preprocessed output with user-provided concept annotations on the raw text.
Gold standard negation annotations can be included for later evaluation.

```
from preprocessing import PreprocessCorpus
    
preprocessor = PreprocessCorpus()
preprocessed_instances = preprocessor(file_ids)
# file_ids = list of paths to .json files containing one dictionary each with the relevant input data
    
# example input dictionary:
# input_dictionary['text'] = raw clinical text to be processed by Frog
# input_dictionary['concept_spans'] = [{'begin': start_idx, 'end': end_index},
                                       {'begin': start_idx, 'end': end_index}]                           
# if gold standard annotations are present for negation:
# input_dictionary['speculation_status'] = [True, False]
```

### Tagging of speculation cues

```
from speculation_tagger import SpeculationTagger
    
# if gold standard is included, gold_included should be True, else False
tagger = SpeculationTagger(gold_included)
tagged_sentences = tagger(preprocessed_instances)
```

### Speculation detection of clinical concepts

```
from speculation_detector import SpeculationDetector, SpeculationDetectorEvaluation

# choose model from ['forward', 'backward', 'forward_punct', 'backward_punct', 'finetuned_baseline', 'finetuned_hybrid']
sentence_instances = tagged_sentences['sentence_instances']

# usage for data WITHOUT gold standard speculation annotations
detector = SpeculationDetector()
instances_detection_data = detector.detect(sentence_instances, model)             

# usage for data WITH gold standard speculation annotations
detector = SpeculationDetectorEvaluation()
results = detector(sentence_instances, model)
```

##### Forward model

Matches the first following concept after a detected speculation cue.

##### Backward model

Matches the first preceding concept before a detected speculation cue.

##### Forward punctuation model

Matches all following concepts before the first following punctuation.

##### Backward punctuation model

Matches all preceding concepts after the first preceding punctuation.

##### Fine-tuned baseline model

Applies for each cue separately the most effective of the four baseline models.

##### Fine-tuned ybrid model

Replaces the fine-tuned baseline model for every cue it can outperform with a rule selected from simple rules on the Frog dependency parse.
