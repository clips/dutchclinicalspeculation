from collections import defaultdict
import json
from speculation_triggers import speculation_triggers
from sys import exit
from finetuned_models import finetuned_baseline, finetuned_hybrid


class SpeculationDetector:

    def __init__(self):

        self.speculation_cues = speculation_triggers

        self.allowed_dependency_steps = 1
        self.fuse_conjuncted = True
        self.concept_only_headwords = False
        self.dependency_upstream = True

        self.single_reference_cue = ''
        self.evaluation_mode = False

    def load_instances(self, path_to_instances):

        with open(path_to_instances, 'r') as f:
            sentence_instances, gold = json.load(f)

        if self.evaluation_mode:
            if not gold:
                raise ValueError('If SpeculationDetector used in evaluation mode, loaded data should be gold data!')

        return sentence_instances

    def detect(self, sentence_instances, model):

        all_detection_data = []
        for sentence_instance in sentence_instances:
            detection_data = self.detect_instance(sentence_instance, model)
            all_detection_data += detection_data

        return all_detection_data

    def detect_instance(self, sentence_instance, model):

        if model in ['forward', 'backward', 'forward_punct', 'backward_punct', 'dependency', 'finetuned_baseline', 'finetuned_hybrid']:
            detection_data = self.detect_speculation(sentence_instance, model)
        else:
            raise ValueError('{} is no valid model, choose between baseline, dependency and ensemble'.format(model))

        return detection_data

    def detect_speculation(self, instance_data, model):

        # unpack data
        instance, instance_id = instance_data
        tags, metadata, sentence = instance
        tokens, spans, dependency_data = list(zip(*sentence))

        # extract valid modality-concept matches
        valid_modality_concept_matches = self.extract_valid_modality_concept_matches(tags)

        # extract concept idxs for reference
        concept_tagsets = self.extract_concept_tagsets(tags)
        if self.concept_only_headwords:
            dependency_tree = self.build_dependency_tree(dependency_data)
            concept_tagsets = {concept: self.extract_headwords(dependency_tree, dependency_data, idxs)
                               for concept, idxs in concept_tagsets.items()}

        # match concepts with modality
        if model == 'forward':
            matched_concepts = self.baseline_forward(valid_modality_concept_matches, concept_tagsets,
                                                                      metadata)
        elif model == 'backward':
            matched_concepts = self.baseline_backward(valid_modality_concept_matches, concept_tagsets,
                                                                      metadata)
        elif model == 'forward_punct':
            matched_concepts = self.baseline_forward_punctuation(valid_modality_concept_matches, concept_tagsets,
                                                                      metadata, tokens)
        elif model == 'backward_punct':
            matched_concepts = self.baseline_backward_punctuation(valid_modality_concept_matches, concept_tagsets,
                                                                      metadata, tokens)

        elif model == 'dependency':
            matched_concepts = self.dependency_detector(valid_modality_concept_matches, concept_tagsets, metadata,
                                                        dependency_data)

        elif model == 'finetuned_baseline':
            matched_concepts = self.finetuned_baseline(valid_modality_concept_matches, concept_tagsets, metadata,
                                                        tokens)
        elif model == 'finetuned_hybrid':
            matched_concepts = self.finetuned_hybrid(valid_modality_concept_matches, concept_tagsets, metadata,
                                                        dependency_data, tokens, dependency_data)

        else:
            raise ValueError('{} is not a valid model, choose between baseline and dependency')

        concepts = sorted(concept_tagsets.keys(), key=lambda x: int(x[1:]))
        detection_data = [matched_concepts, concepts, metadata, instance_id]

        # fuse predictions for conjuncted concepts
        if self.fuse_conjuncted:
            detection_data = self.detect_conjunct_speculation(instance_data, detection_data)

        return detection_data

    def finetuned_baseline(self, valid_modality_concept_matches, concept_tagsets, metadata, tokens):

        matched_concepts = defaultdict(lambda: defaultdict(list))

        finetuned_baseline_reference = {k: v[0] for k, v in finetuned_baseline.items()}
        
        for (modality_cue, cue_idxs), valid_concepts in valid_modality_concept_matches.items():
            lexical_modality_cue = metadata['speculation'][modality_cue]['cue']
            model = finetuned_baseline_reference[lexical_modality_cue]
            if model == 'forward':
                matched_concepts = self._baseline_forward_per_cue(valid_modality_concept_matches, matched_concepts, modality_cue, cue_idxs, concept_tagsets, metadata)
            elif model == 'forward_punct':
                matched_concepts = self._baseline_forward_punctuation_per_cue(valid_modality_concept_matches, matched_concepts, modality_cue, cue_idxs, concept_tagsets, metadata, tokens)
            elif model == 'backward':
                matched_concepts = self._baseline_backward_per_cue(valid_modality_concept_matches, matched_concepts, modality_cue, cue_idxs, concept_tagsets, metadata)
            elif model == 'backward_punct':
                matched_concepts = self._baseline_backward_punctuation_per_cue(valid_modality_concept_matches, matched_concepts, modality_cue, cue_idxs, concept_tagsets, metadata, tokens)

        return matched_concepts

    def finetuned_hybrid(self, valid_modality_concept_matches, concept_tagsets, metadata, dependency_data, tokens, depdata):
        
        matched_concepts = defaultdict(lambda: defaultdict(list))

        finetuned_hybrid_reference = {k: v[0] for k, v in finetuned_hybrid.items()}

        dependency_tree = self.build_dependency_tree(depdata)
        
        for (modality_cue, cue_idxs), valid_concepts in valid_modality_concept_matches.items():
            lexical_modality_cue = metadata['speculation'][modality_cue]['cue']
            model_info = finetuned_hybrid_reference[lexical_modality_cue]
            model = model_info[0]
            if len(model_info) == 2:
                self.fuse_conjuncted = model_info[1]
            elif len(model_info) == 5:
                self.fuse_conjuncted, self.allowed_dependency_steps, self.concept_only_headwords, self.dependency_upstream = model_info[1:]
            else:
                raise ValueError

            if model == 'forward':
                matched_concepts = self._baseline_forward_per_cue(valid_modality_concept_matches, matched_concepts, modality_cue, cue_idxs, concept_tagsets, metadata)
            elif model == 'forward_punct':
                matched_concepts = self._baseline_forward_punctuation_per_cue(valid_modality_concept_matches, matched_concepts, modality_cue, cue_idxs, concept_tagsets, metadata, tokens)
            elif model == 'backward':
                matched_concepts = self._baseline_backward_per_cue(valid_modality_concept_matches, matched_concepts, modality_cue, cue_idxs, concept_tagsets, metadata)
            elif model == 'backward_punct':
                matched_concepts = self._baseline_backward_punctuation_per_cue(valid_modality_concept_matches, matched_concepts, modality_cue, cue_idxs, concept_tagsets, metadata, tokens)
            elif model == 'dependency':
                matched_concepts = self._dependency_detector_per_cue(valid_modality_concept_matches, valid_concepts, matched_concepts, modality_cue, cue_idxs, concept_tagsets, metadata, dependency_tree)

        return matched_concepts

    def detect_conjunct_speculation(self, instance, detection_data):
        conjuncted_concepts = self._extract_conjunct_concepts(instance)
        conjuncted_detection_data = self._fuse_conjunct_concept_modalities(conjuncted_concepts, detection_data)

        return conjuncted_detection_data

    @staticmethod
    def _fuse_conjunct_concept_modalities(conjuncted_concepts, detection_data):
        # takes conjuncted concepts and fuses their extracted modality
        matched_concepts = detection_data[0]
        concepts = detection_data[1]

        for concept in concepts:
            if not matched_concepts[concept]:
                for conjuncted_concept in conjuncted_concepts[concept]:
                    conjuncted_concept_modality = matched_concepts[conjuncted_concept]
                    matched_concepts[concept] = conjuncted_concept_modality

        detection_data[0] = matched_concepts

        return detection_data

    def _extract_conjunct_concepts(self, instance):

        # check if concepts are conjuncted by checking a concept for any conjunction relation with other concepts in
        # the sentence, then tie their faiths together!

        instance, instance_id = instance
        tags, metadata, sentence = instance
        tokens, spans, dependency_data = list(zip(*sentence))

        # extract direct conjunction dependencies from dependency data
        conjunction_dependencies = defaultdict(set)
        for token_index, (dependency_relation, dependency_idx) in enumerate(dependency_data):
            if dependency_relation == 'cnj':
                # see what other cnj tokens depend on the same conjuncting head
                head_conjuncting_idx = int(dependency_idx) - 1
                conjunction_dependencies[head_conjuncting_idx].add(token_index)

        concept_tagsets = self.extract_concept_tagsets(tags)
        inverted_concept_tagsets = defaultdict(set)
        for concept, idxs in concept_tagsets.items():
            for idx in idxs:
                inverted_concept_tagsets[idx].add(concept)

        conjuncted_concept_sets = []
        for conjuncted_tokens_idxs in conjunction_dependencies.values():
            conjuncted_concept_set = set()
            for token_idx in conjuncted_tokens_idxs:
                token_idx_concepts = inverted_concept_tagsets[token_idx]
                if not token_idx_concepts:
                    continue
                if len(token_idx_concepts) > 1:
                    # take longest concept
                    token_idx_concept_lens = {concept: len(concept_tagsets[concept]) for concept in token_idx_concepts}
                    token_idx_concept = max(token_idx_concept_lens.items(), key=lambda x: x[1])[0]
                else:
                    token_idx_concept = next(iter(token_idx_concepts))
                conjuncted_concept_set.add(token_idx_concept)
            conjuncted_concept_sets.append(conjuncted_concept_set)

        conjuncted_concepts = defaultdict(set)
        for conjuncted_concept_set in conjuncted_concept_sets:
            for conjuncted_concept in conjuncted_concept_set:
                conjuncted_concepts[conjuncted_concept].update(conjuncted_concept_set)

        conjuncted_concepts = {k: {x for x in v if x != k} for k, v in conjuncted_concepts.items()}
        conjuncted_concepts = defaultdict(set, conjuncted_concepts)  # convert back to defaultdict

        return conjuncted_concepts

    def baseline_forward(self, valid_modality_concept_matches, concept_tagsets, metadata):
        matched_concepts = defaultdict(lambda: defaultdict(list))
        for (modality_cue, cue_idxs), valid_concepts in valid_modality_concept_matches.items():
            matched_concepts = self._baseline_forward_per_cue(valid_modality_concept_matches, matched_concepts, modality_cue, cue_idxs, concept_tagsets, metadata)

        return matched_concepts

    def _baseline_forward_per_cue(self, valid_modality_concept_matches, matched_concepts, modality_cue, cue_idxs, concept_tagsets, metadata):

        start_modality_index = cue_idxs[0]

        lexical_modality_cue = metadata['speculation'][modality_cue]['cue']
        if self.single_reference_cue and (lexical_modality_cue.lower() != self.single_reference_cue):
            return matched_concepts
        # assign to each modality cue the nearest concept

        # check for pre-modifier modality cues
        if lexical_modality_cue in self.speculation_cues:
            # extract first following concept
            concept_start_idxs = {concept: min(tagset) for
                                  concept, tagset in concept_tagsets.items()
                                  if min(tagset) > start_modality_index}
            if concept_start_idxs:
                first_following_concept = sorted(concept_start_idxs.items(), key=lambda x:x[1])[0][0]
                # match with first following concept if allowed
                if first_following_concept in valid_modality_concept_matches[(modality_cue, cue_idxs)]:
                    matched_concepts[first_following_concept]['forward'].append(lexical_modality_cue)

        return matched_concepts

    def baseline_forward_punctuation(self, valid_modality_concept_matches, concept_tagsets, metadata, tokens):
        matched_concepts = defaultdict(lambda: defaultdict(list))
        for (modality_cue, cue_idxs), valid_concepts in valid_modality_concept_matches.items():
            matched_concepts = self._baseline_forward_punctuation_per_cue(valid_modality_concept_matches, matched_concepts, modality_cue, cue_idxs, concept_tagsets, metadata, tokens)

        return matched_concepts

    def _baseline_forward_punctuation_per_cue(self, valid_modality_concept_matches, matched_concepts, modality_cue, cue_idxs, concept_tagsets, metadata, tokens):

        start_modality_index = cue_idxs[0]

        lexical_modality_cue = metadata['speculation'][modality_cue]['cue']
        if self.single_reference_cue and (lexical_modality_cue.lower() != self.single_reference_cue):
            return matched_concepts

        # tag each following concept before punctuation

        # check for pre-modifier modality cues
        if lexical_modality_cue in self.speculation_cues:

            # determine where first punctuation encountered

            rest_of_sentence = tokens[start_modality_index + 1:]
            punct_index = None
            for i, token in enumerate(rest_of_sentence):
                if token in '?!.,:;':
                    punct_index = i + 1 + start_modality_index
            
            if punct_index:
                # all following concepts which end before the punctuation
                for concept, tagset in concept_tagsets.items():
                    if min(tagset) > start_modality_index:
                        if max(tagset) < punct_index:
                            if concept in valid_modality_concept_matches[(modality_cue, cue_idxs)]:
                                matched_concepts[concept]['punct'].append(lexical_modality_cue)
            else:  # means all remaining following concepts negated
                for concept, tagset in concept_tagsets.items():
                    if min(tagset) > start_modality_index:
                        if concept in valid_modality_concept_matches[(modality_cue, cue_idxs)]:
                            matched_concepts[concept]['forward_punct'].append(lexical_modality_cue)

        return matched_concepts

    def baseline_backward(self, valid_modality_concept_matches, concept_tagsets, metadata):
        matched_concepts = defaultdict(lambda: defaultdict(list))
        for (modality_cue, cue_idxs), valid_concepts in valid_modality_concept_matches.items():
            matched_concepts = self._baseline_backward_per_cue(valid_modality_concept_matches, matched_concepts, modality_cue, cue_idxs, concept_tagsets, metadata)

        return matched_concepts

    def _baseline_backward_per_cue(self, valid_modality_concept_matches, matched_concepts, modality_cue, cue_idxs, concept_tagsets, metadata):
        try:
            end_modality_index = cue_idxs[1]
        except:
            end_modality_index = cue_idxs[0]

        lexical_modality_cue = metadata['speculation'][modality_cue]['cue']
        if self.single_reference_cue and (lexical_modality_cue.lower() != self.single_reference_cue):
            return matched_concepts
        
        # assign to each modality cue the nearest concept

        # check for post-modifier modality cue
        if lexical_modality_cue in self.speculation_cues:
            # extract first following concept
            concept_end_idxs = {concept: max(tagset) for
                                  concept, tagset in concept_tagsets.items()
                                  if max(tagset) < end_modality_index}
            if concept_end_idxs:
                first_preceding_concept = sorted(concept_end_idxs.items(), key=lambda x:x[1], reverse=True)[0][0]
                # match with first following concept if allowed
                if first_preceding_concept in valid_modality_concept_matches[(modality_cue, cue_idxs)]:
                    matched_concepts[first_preceding_concept]['backward'].append(lexical_modality_cue)

        return matched_concepts

    def baseline_backward_punctuation(self, valid_modality_concept_matches, concept_tagsets, metadata, tokens):
        matched_concepts = defaultdict(lambda: defaultdict(list))
        for (modality_cue, cue_idxs), valid_concepts in valid_modality_concept_matches.items():
            matched_concepts = self._baseline_backward_punctuation_per_cue(valid_modality_concept_matches, matched_concepts, modality_cue, cue_idxs, concept_tagsets, metadata, tokens)

        return matched_concepts

    def _baseline_backward_punctuation_per_cue(self, valid_modality_concept_matches, matched_concepts, modality_cue, cue_idxs, concept_tagsets, metadata, tokens):

        try:
            end_modality_index = cue_idxs[1]
        except:
            end_modality_index = cue_idxs[0]

        lexical_modality_cue = metadata['speculation'][modality_cue]['cue']
        if self.single_reference_cue and (lexical_modality_cue.lower() != self.single_reference_cue):
            return matched_concepts

        # check for pre-modifier modality cues
        if lexical_modality_cue in self.speculation_cues:

            # determine where first punctuation encountered

            rest_of_sentence = tokens[:end_modality_index]
            punct_index = None
            for i, token in enumerate(rest_of_sentence[::-1]):
                if token in '?!.,:;':
                    punct_index = end_modality_index - i - 1
            
            if punct_index:
                # all preceding concepts which start after the punctuation
                for concept, tagset in concept_tagsets.items():
                    if max(tagset) < end_modality_index:
                        if min(tagset) > punct_index:
                            if concept in valid_modality_concept_matches[(modality_cue, cue_idxs)]:
                                matched_concepts[concept]['punct'].append(lexical_modality_cue)
            else:  # means all remaining preceding concepts negated
                for concept, tagset in concept_tagsets.items():
                    if max(tagset) < end_modality_index:
                        if concept in valid_modality_concept_matches[(modality_cue, cue_idxs)]:
                            matched_concepts[concept]['backward_punct'].append(lexical_modality_cue)

        return matched_concepts

    def dependency_detector(self, valid_modality_concept_matches, concept_tagsets, metadata, depdata):
        matched_concepts = defaultdict(lambda: defaultdict(list))
        dependency_tree = self.build_dependency_tree(depdata)
        for (modality_cue, cue_idxs), valid_concepts in valid_modality_concept_matches.items():
            matched_concepts = self._dependency_detector_per_cue(valid_modality_concept_matches, valid_concepts, matched_concepts, modality_cue, cue_idxs, concept_tagsets, metadata, dependency_tree)

        return matched_concepts

    def _dependency_detector_per_cue(self, valid_modality_concept_matches, valid_concepts, matched_concepts, modality_cue, cue_idxs, concept_tagsets, metadata, dependency_tree):

        lexical_modality_cue = metadata['speculation'][modality_cue]['cue']
        if self.single_reference_cue and (lexical_modality_cue.lower() != self.single_reference_cue):
            return matched_concepts

        # assign to each modality cue all concepts within a dependency span
        # pre- or post-modifiers become irrelevant here, should be inferred by the dependency parse...

        # traverse tree, assign all concepts within n dependencies
        # accumulate paths of all token idxs of the cue
        for cue_idx in cue_idxs:
            dependent_terms, governor_terms = self.extract_dependency_path(dependency_tree, cue_idx)
            idxs_to_match = set()  # extract from dependent terms

            if self.dependency_upstream:
                for numsteps, dep_idxs in dependent_terms.items():
                    if numsteps <= self.allowed_dependency_steps:
                        idxs_to_match.update(dep_idxs)
            else:
                for numsteps, gov_idxs in governor_terms.items():
                    if numsteps <= self.allowed_dependency_steps:
                        idxs_to_match.update(gov_idxs)

            for concept, concept_idxs in concept_tagsets.items():
                if concept in valid_concepts:
                    if idxs_to_match.intersection(concept_idxs):
                        matched_concepts[concept]['dependency'].append(lexical_modality_cue)

        return matched_concepts

    def extract_dependency_path(self, dependency_tree, token_index):
        forward_dependency = dependency_tree['forward']
        backward_dependency = dependency_tree['backward']
        dependent_terms = defaultdict(list)
        governor_terms = defaultdict(list)
        steps = 1
        self.traverse_dependency_path(token_index, forward_dependency, dependent_terms, steps)
        self.traverse_dependency_path(token_index, backward_dependency, governor_terms, steps)

        return dependent_terms, governor_terms

    def traverse_dependency_path(self, reference_idx, dependency_tree, token_idxs, steps):
        dependencies = dependency_tree[reference_idx]
        token_idxs[steps] += dependencies
        steps += 1
        for dependency in dependencies:
            self.traverse_dependency_path(dependency, dependency_tree, token_idxs, steps)

    @staticmethod
    def extract_concept_tagsets(tags):
        concepts_idxs = defaultdict(set)
        for i, position_tags in enumerate(tags):
            for tag in position_tags:
                if tag.startswith('C'):
                    concepts_idxs[tag].add(i)

        return concepts_idxs

    @staticmethod
    def extract_valid_modality_concept_matches(tags):
        # disables modality cues for concepts which they are a part of!
        modality_tagsets = defaultdict(set)
        concept_tagsets = defaultdict(set)
        for i, position_tags in enumerate(tags):
            for tag in position_tags:
                if tag.startswith('speculation'):
                    modality_tagsets[tag].add(i)
                elif tag.startswith('C'):
                    concept_tagsets[tag].add(i)
                else:
                    pass

        valid_modality_concept_matches = defaultdict(list)
        for modality_cue, cue_tagindexset in modality_tagsets.items():
            for concept, concept_tagindexset in concept_tagsets.items():
                if not cue_tagindexset.intersection(concept_tagindexset):
                    modality_data = (modality_cue, tuple(sorted(cue_tagindexset)))
                    valid_modality_concept_matches[modality_data].append(concept)

        return valid_modality_concept_matches

    @staticmethod
    def print_dependency_tree(dependency_tree, tokens):
        for dep, heads in dependency_tree['forward'].items():
            for head in heads:
                print(tokens[dep], '--->', tokens[head])

    @staticmethod
    def build_dependency_tree(depdata):
        # build dependency dict
        forward_dependency = defaultdict(set)
        # tokens, depdata = list(zip(*sentence))
        for index, (dep, depindex) in enumerate(depdata):
            depindex = int(depindex)
            if depindex > 0:  # don't include ROOT for dependency
                real_depindex = depindex - 1
                forward_dependency[index].add(real_depindex)

        # create backward dependency
        backward_dependency = defaultdict(set)
        for dependent_index, governor_indexes in forward_dependency.items():
            for governor_index in governor_indexes:
                backward_dependency[governor_index].add(dependent_index)

        dependency_tree = {'forward': forward_dependency, 'backward': backward_dependency}

        return dependency_tree

    def extract_headwords(self, dependency_tree, depdata, idxs):
        headwords_idxs = set()

        # per ROOT: concept words closest to root!
        root_idxs = [i for i, x in enumerate(depdata) if x[0] == 'ROOT']

        for root_idx in root_idxs:
            _, governor_terms = self.extract_dependency_path(dependency_tree, root_idx)
            for numsteps, governed_idxs in sorted(governor_terms.items()):
                headwords_candidates = set(governed_idxs).intersection(idxs)
                if headwords_candidates:
                    headwords_idxs.update(headwords_candidates)
                    break

        # default to all words if no proper head words can be found for any reason
        if not headwords_idxs:
            headwords_idxs = idxs

        headwords_idxs = tuple(sorted(headwords_idxs))

        return headwords_idxs


class SpeculationDetectorEvaluation(SpeculationDetector):

    def __init__(self):

        super(SpeculationDetectorEvaluation, self).__init__()

        self.evaluation_mode = True
        self.confusion_matrix = {'true_pos': [],
                                 'true_neg': [],
                                 'false_pos': [],
                                 'false_neg': []}

    def __call__(self, sentence_instances, model, outfile='', verbose=True):

        self.reset_confusion_matrix()
        for sentence_instance in sentence_instances:
            self.detect_and_evaluate_instance(sentence_instance, model)

        results = self.evaluation()
        if verbose:
            print(results)

        if outfile:
            results_outfile = '{}_results.json'.format(outfile)
            confusion_matrix_outfile = '{}_confusion.json'.format(outfile)
            print('Saving...')
            with open(outfile, 'w') as f:
                json.dump(results_outfile, f)
            print('Done')
            self.save_confusion_matrix(confusion_matrix_outfile)

        return results

    def detect_and_evaluate_instance(self, sentence_instance, model):
        # assert self.assert_gold_data(sentence_instance)

        detection_data = self.detect_instance(sentence_instance, model)
        self.update_confusion_matrix(*detection_data)

    def reset_confusion_matrix(self):
        self.confusion_matrix = {'true_pos': [],
                                 'true_neg': [],
                                 'false_pos': [],
                                 'false_neg': []}

    def tune_step(self, cue, sentence_instances, model):
        cue_count = 0
        self.reset_confusion_matrix()
        for sentence_instance in sentence_instances:
            greenlight = False
            metadata = sentence_instance[0][1]
            if 'speculation' in metadata:
                for speculation_id, cue_data in metadata['speculation'].items():
                    if cue_data['cue'] == cue:
                        greenlight = True
                        cue_count += 1
                        break
            if not greenlight:
                continue
            detection_data = self.detect_instance(sentence_instance, model=model)
            self.update_cue_specific_confusion_matrix(*detection_data)

        scores = self.evaluation()
        prec, recall = scores['positive_precision'], scores['positive_recall']
        try:
            F1 = 2 * ((prec * recall) / (prec + recall))
        except:
            F1 = 0

        return F1, cue_count

    def tune_baseline(self, sentence_instances):
        # store using model names for easy reference! model = winner, can extract using max of F1
        tune_dict = defaultdict(lambda: defaultdict(dict))
        for cue in sorted(self.speculation_cues):

            self.single_reference_cue = cue
            
            # forward
            F1, cue_count = self.tune_step(cue, sentence_instances, model='forward')
            tune_dict[(cue, cue_count)]['forward'] = F1

            # backward
            F1, cue_count = self.tune_step(cue, sentence_instances, model='backward')
            tune_dict[(cue, cue_count)]['backward'] = F1

            # forward punctuation
            F1, cue_count = self.tune_step(cue, sentence_instances, model='forward_punct')
            tune_dict[(cue, cue_count)]['forward_punct'] = F1

            # backward punctuation
            F1, cue_count = self.tune_step(cue, sentence_instances, model='backward_punct')
            tune_dict[(cue, cue_count)]['backward_punct'] = F1

            print('Checked for cue {}'.format(cue))

        print('Finished!')

        return tune_dict

    def tune_hybrid(self, sentence_instances):
        # store using model names for easy reference! model = winner, can extract using max of F1
        # also store 
        tune_dict = defaultdict(dict)
        for cue in sorted(self.speculation_cues):

            self.single_reference_cue = cue
            
            for conjunct in [True, False]:
                self.fuse_conjuncted = conjunct
                # forward
                F1, cue_count = self.tune_step(cue, sentence_instances, model='forward')
                tune_dict[(cue, cue_count)][('forward', conjunct)] = F1

                # backward
                F1, cue_count = self.tune_step(cue, sentence_instances, model='backward')
                tune_dict[(cue, cue_count)][('backward', conjunct)] = F1

                # forward punctuation
                F1, cue_count = self.tune_step(cue, sentence_instances, model='forward_punct')
                tune_dict[(cue, cue_count)][('forward_punct', conjunct)] = F1

                # backward punctuation
                F1, cue_count = self.tune_step(cue, sentence_instances, model='backward_punct')
                tune_dict[(cue, cue_count)][('backward_punct', conjunct)] = F1
            
            # dependency
            for conjunct in [True, False]:
                self.fuse_conjuncted = conjunct
                for step_size in range(1, 4):
                    self.allowed_dependency_steps = step_size
                    for only_headwords in [True, False]:
                        self.concept_only_headwords = only_headwords
                        for dependency_upstream in [True, False]:
                            self.dependency_upstream = dependency_upstream
                            F1, cue_count = self.tune_step(cue, sentence_instances, model='dependency')
                            tune_dict[(cue, cue_count)][('dependency', conjunct, step_size, only_headwords, dependency_upstream)] = F1

            print('Checked for cue {}'.format(cue))

        print('Finished!')

        return tune_dict

    @staticmethod
    def assert_gold_data(instance):
        if 'true_modality' not in instance[0][1]['concepts']['C0']:
            raise ValueError('This is not proper gold data, please provide gold data')

    def reset_confusion_matrix(self):
        self.confusion_matrix = {'true_pos': [], 'true_neg': [], 'false_pos': [], 'false_neg': []}

    def save_confusion_matrix(self, outfile):
        print('Saving confusion matrix...')
        with open(outfile, 'w') as f:
            json.dump(self.confusion_matrix, f)
        print('Done')

    def update_confusion_matrix(self, matched_concepts, concepts, metadata, instance_id):
        for concept in concepts:
            if matched_concepts[concept]:
                modality_prediction = matched_concepts[concept]
                # check for true positive
                if metadata['concepts'][concept]['true_modality'] == True:
                    self.confusion_matrix['true_pos'].append((concept, instance_id, modality_prediction))
                # check for false positive
                else:
                    self.confusion_matrix['false_pos'].append((concept, instance_id, modality_prediction))
            else:
                # check for false negative
                if metadata['concepts'][concept]['true_modality'] == True:
                    self.confusion_matrix['false_neg'].append((concept, instance_id))
                # check for true negative
                else:
                    self.confusion_matrix['true_neg'].append((concept, instance_id))

    def update_cue_specific_confusion_matrix(self, matched_concepts, concepts, metadata, instance_id):
        for concept in concepts:
            if matched_concepts[concept]:
                modality_prediction = matched_concepts[concept]
                # check for true positive
                if metadata['concepts'][concept]['true_modality'] == True:
                    self.confusion_matrix['true_pos'].append((concept, instance_id, modality_prediction))
                # check for false positive
                else:
                    # check if concept is considered speculated by our specific cue...
                    # print(matched_concepts[concept].values())
                    if self.single_reference_cue in list(matched_concepts[concept].values())[0]:
                        # print('False positive for {}'.format(self.single_reference_cue))
                        self.confusion_matrix['false_pos'].append((concept, instance_id, modality_prediction))
            else:
                # check for false negative
                if metadata['concepts'][concept]['true_modality'] == True:
                    self.confusion_matrix['false_neg'].append((concept, instance_id))
                # check for true negative
                else:
                    self.confusion_matrix['true_neg'].append((concept, instance_id))

    def evaluation(self):
        # uses the confusion matrix to calculate various evaluation metrics

        true_pos = len(self.confusion_matrix['true_pos'])
        true_neg = len(self.confusion_matrix['true_neg'])
        false_pos = len(self.confusion_matrix['false_pos'])
        false_neg = len(self.confusion_matrix['false_neg'])

        try:
            positive_precision = true_pos / (true_pos + false_pos)
        except ZeroDivisionError:
            positive_precision = None
        try:
            positive_recall = true_pos / (true_pos + false_neg)
        except ZeroDivisionError:
            positive_recall = None
        try:
            negative_precision = true_neg / (true_neg + false_neg)
        except ZeroDivisionError:
            negative_precision = None
        try:
            negative_recall = true_neg / (true_neg + false_pos)
        except ZeroDivisionError:
            negative_recall = None

        accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)

        majority_baseline = max(true_pos + false_neg, true_neg + false_pos) / (true_pos + false_neg + true_neg + false_pos)

        results = {'accuracy': accuracy,
                   'majority_baseline': majority_baseline,
                   'positive_precision': positive_precision,
                   'positive_recall': positive_recall,
                   'negative_precision': negative_precision,
                   'negative_recall': negative_recall}

        return results
