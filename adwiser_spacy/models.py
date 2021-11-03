# import warnings
# warnings.filterwarnings("ignore")
import re
import spacy
from spacy.tokens import Token
# from .annotator import output_maker
from spacy.matcher import DependencyMatcher
from spacy.matcher import Matcher
from spellchecker import SpellChecker
# import treetaggerwrapper

nlp = spacy.load("en_core_web_lg")
# tagger = treetaggerwrapper.TreeTagger(TAGLANG='en', TAGDIR='tt/')
char_span = lambda token: (token.idx, token.idx + len(token.text))
Token.set_extension(name='span', getter=char_span)


#
# def tree_tag_(token):
#     doc_ = token.doc
#     tags_ = tagger.tag_text(str(doc_))
#     tags_ = tags_[token.i]
#     _, tag, _ = tags_.split('\t')
#     return tag
#
#
# tree_tag = lambda token: tree_tag_(token)
# Token.set_extension(name='tree_tag', getter=tree_tag)


def find_span(tokens):
    if len(tokens) == 1:
        token = tokens[0]
        return [token._.span[0], token._.span[1]]
    ss = min([s._.span[0] for s in tokens])
    s: Token
    sm = max([s._.span[1] for s in tokens])
    return [ss, sm]


def without_child(token, values):
    wrong_keys = {'dep', 'lemma', 'norm', 'tag', 'pos'}
    construct_pattern = '[i for i in token.children if '
    for value in values.keys():
        if value not in dir(token) or value in wrong_keys:
            raise AttributeError(f'Invalid attribute {value}')
        temp = 'i.' + value
        if isinstance(values[value], list):
            temp_2 = '{' + str(values[value])[1:-1] + '}'
            temp += ' in '
            temp += temp_2
        elif isinstance(values[value], str):
            temp += ' == ' + '\'' + str(values[value]) + '\''
        else:
            raise KeyError(f'Invalid key: {type(values[value])}')
        temp += ' and '
        construct_pattern += temp
    construct_pattern = construct_pattern[:-5] + ']'
    try:
        flag = exec(construct_pattern)
        if flag: return False
    except:
        raise KeyError(f'Invalid key')
    return True


def models(text, test_mode=False):
    def pp_time(sent):

        errors_pp = []
        all_errors = []
        dep_matcher = DependencyMatcher(vocab=nlp.vocab)
        present_perfect = [{'RIGHT_ID': 'verb', 'RIGHT_ATTRS': {'TAG': 'VBN'}},
                           {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'have',
                            'RIGHT_ATTRS': {'LEMMA': 'have', 'DEP': 'aux', 'TAG': {'IN': ['VBP', 'VBZ']}}}]
        dep_matcher.add("present_perfect", patterns=[present_perfect])
        results = dep_matcher(sent)
        if results:

            error_message = 'Present Perfect does not go along with indication of time in the past.'
            all_matches = []
            verbs = []
            dep_matcher_ = DependencyMatcher(vocab=nlp.vocab)

            # in/from/over/between + CD
            patt_one = [{'RIGHT_ID': 'prep', 'RIGHT_ATTRS':
                {'LEMMA': {'IN': ['from', 'in', 'over', 'between']}}},
                        {'LEFT_ID': 'prep', 'REL_OP': '>', 'RIGHT_ID': 'year',
                         'RIGHT_ATTRS': {'TAG': 'CD', 'DEP': 'pobj'}}]

            dep_matcher_.add('prep+cd', patterns=[patt_one])
            dep_matches = dep_matcher_(sent)

            # + у последнего нет детей с леммой recent/last
            if dep_matches and without_child(sent[dep_matches[0][1][1]], {'lemma_': ['last', 'recent']}):
                all_matches.append(dep_matches[0][1][0])
                if sent[dep_matches[0][1][0]].head.tag_ == 'VBN':
                    verbs.append(sent[dep_matches[0][1][0]].head)
            # in/from/over/between+year
            patt_two = [{'RIGHT_ID': 'prep', 'RIGHT_ATTRS': {'LEMMA': {'IN': ['at', 'over', 'in']}}},
                        {'LEFT_ID': 'prep', 'REL_OP': '>', 'RIGHT_ID': 'year',
                         'RIGHT_ATTRS': {'LEMMA': 'year', 'DEP': 'pobj'}}]

            dep_matcher_.add('prep+year', patterns=[patt_two])
            # at/in/over + last/recent + NN
            patt_three = [{'RIGHT_ID': 'prep', 'RIGHT_ATTRS': {'LEMMA': {'IN': ['at', 'over', 'in']}}},
                          {'LEFT_ID': 'prep', 'REL_OP': '>', 'RIGHT_ID': 'noun',
                           'RIGHT_ATTRS': {'TAG': 'NN', 'DEP': {'IN': ['pobj', 'npadvmod']}}},
                          {'LEFT_ID': 'noun', 'REL_OP': '>', 'RIGHT_ID': 'adj',
                           'RIGHT_ATTRS': {'LEMMA': {'IN': ['last', 'initial']}}}]

            dep_matcher_.add('prep+noun+last/initial', patterns=[patt_three])
            # at/in/over + ordinal + NN
            patt_four = [{'RIGHT_ID': 'prep', 'RIGHT_ATTRS': {'LEMMA': {'IN': ['at', 'over', 'in']}}},
                         {'LEFT_ID': 'prep', 'REL_OP': '>', 'RIGHT_ID': 'noun',
                          'RIGHT_ATTRS': {'TAG': 'NN', 'DEP': {'IN': ['pobj', 'npadvmod']}}},
                         {'LEFT_ID': 'noun', 'REL_OP': '>', 'RIGHT_ID': 'ord_adj',
                          'RIGHT_ATTRS': {'ENT_TYPE': 'ORDINAL'}}]

            dep_matcher_.add('prep+noun+ordinal', patterns=[patt_four])
            # ago (but not since ...a long ago )
            # I have done it since 2 years ago.
            find_ago = [ago for ago in sent if ago.lemma_ == 'ago']
            for ago_ in find_ago:
                if ago_.head.tag_ == 'VBN':
                    verbs.append(ago_.head)

            # last + noun of periods
            patt_five = [{'RIGHT_ID': 'noun', 'RIGHT_ATTRS': {'TAG': 'NN', 'DEP': {'IN': ['pobj', 'npadvmod']}, 'LEMMA':
                {'IN': ['year', 'term', 'week', 'semester', 'century'
                                                            'day', 'month', 'decade', 'spring', 'fall',
                        'autumn', 'winter', 'summer', 'night', 'evening',
                        'morning', 'season', 'stage', 'point', 'phase']}}},
                         {'LEFT_ID': 'noun', 'REL_OP': '>', 'RIGHT_ID': 'last', 'RIGHT_ATTRS': {'LEMMA': 'last'}}]

            dep_matcher_.add('last+nn', patterns=[patt_five])

            # since/from cd to cd
            patt_six = [{'RIGHT_ID': 'vbn', 'RIGHT_ATTRS': {'TAG': 'VBN'}},
                        {'LEFT_ID': 'vbn', 'REL_OP': '>', 'RIGHT_ID': 'prep',
                         'RIGHT_ATTRS': {'LEMMA': {'IN': ['from', 'since']}, 'DEP': 'prep'}},
                        {'LEFT_ID': 'prep', 'REL_OP': '>', 'RIGHT_ID': 'cd',
                         'RIGHT_ATTRS': {'TAG': 'CD', 'DEP': {'IN': ['pobj', 'npadvmod']}}},
                        {'LEFT_ID': 'vbn', 'REL_OP': '>', 'RIGHT_ID': 'to',
                         'RIGHT_ATTRS': {'LEMMA': 'to', 'DEP': 'prep'}},
                        {'LEFT_ID': 'to', 'REL_OP': '>', 'RIGHT_ID': 'cd_two', 'RIGHT_ATTRS': {'TAG': 'CD'}}]

            dep_matcher_.add('since_1998_to_2000', patterns=[patt_six])
            yesterday = [yestdy for yestdy in sent if yestdy.lemma_ == 'yesterday']
            for yestdy in yesterday:
                if yestdy.head.head.head.tag_ == 'VBN':
                    verbs.append(yestdy.head.head.head)
            # Verbs
            if dep_matcher_(sent):
                for match_dep in dep_matcher_(sent):
                    if sent[match_dep[1][0]].head.tag_ == 'VBN':
                        verbs.append(sent[match_dep[1][0]].head)
                    if nlp.vocab[match_dep[0]].text == 'since_1998_to_2000':
                        if sent[match_dep[1][1]].lemma_ == 'since':
                            tok = sent[match_dep[1][1]]
                            all_errors.append([find_span([tok]),
                                               'You may need \'from\' instead of \'since\'.'])

            # have + not от каждого глагола + ошибки
            for verb in verbs:
                if verb.tag_ == 'VBN':

                    have_not = [have for have in verb.children if have.lemma_ == 'have' and have.dep_ == 'aux'
                                and have.tag_ in {'VBZ', 'VBP'}]
                    if have_not:
                        have_not = have_not[0]
                        errors_pp.append(have_not)
                        not_ = [i for i in have_not.children if i.dep_ == 'neg' and i.norm_ == 'not']
                        if not_:
                            errors_pp.append(not_)
                        all_errors.append([find_span(errors_pp), error_message])
        return all_errors

    def inversion(sent):

        error_message = 'You may need inverted word order here.'

        def find_wrong_order(token_i):

            possible_inversions = {'barely', 'never', 'rarely', 'seldom',
                                   'scarcely', 'nowhere', 'neither', 'nor'}
            if token_i.lemma_ in possible_inversions and not re.search(r'^Never the', token.sent.text):
                return True
            elif token_i.lemma_ in {'hardly', 'not'} and token.sent[token.i + 1].pos_ != 'NOUN':
                return True
            elif token_i.lemma_ == 'little' and token_i.dep_ in {'npadvmod', 'dobj', 'advmod'}:
                if without_child(token_i, {'lemma_': 'by'}):
                    return True
            elif token_i.lemma_ == 'not':
                flag = [i for i in token_i.children if i.lemma_ in {'only', 'until'}]
                if flag:
                    return True

            return False

        for token in sent:  # for future modifications (for several clauses in a sent)
            flag = False
            if not (token.i - sent.start):
                prep_no_noun = [{'RIGHT_ID': 'prep',
                                 'RIGHT_ATTRS': {'LEMMA': {'IN': ['under', 'in', 'over', 'at', 'for', 'to']},
                                                 'TAG': 'IN'}},
                                {'LEFT_ID': 'prep', 'REL_OP': '>', 'RIGHT_ID': 'noun', 'RIGHT_ID': 'noun',
                                 'RIGHT_ATTRS': {'POS': 'NOUN'}},
                                {'LEFT_ID': 'noun', 'REL_OP': '>', 'RIGHT_ID': 'no', 'RIGHT_ATTRS': {'LEMMA': 'no'}}]
                no_sooner = [{'RIGHT_ID': 'no', 'RIGHT_ATTRS': {'LEMMA': 'no', 'DEP': 'neg'}},
                             {'LEFT_ID': 'no', 'REL_OP': '<', 'RIGHT_ID': 'noun',
                              'RIGHT_ATTRS': {'LEMMA': {'IN': ['soon', 'later']}}}]
                dep_matcher_ = DependencyMatcher(vocab=nlp.vocab)
                dep_matcher_.add('inversions', patterns=[prep_no_noun, no_sooner])
                for match_dep in dep_matcher_(sent):

                    if sent[match_dep[1][0]].i == token.i:
                        flag = True
                        del dep_matcher_
                        break
                if not flag:
                    flag = find_wrong_order(token)

                if flag:

                    verb = token.head if token.head.pos_ == 'AUX' else ''
                    if not verb:
                        aux = [v for v in token.head.children if v.pos_ == 'AUX']
                        verb_ = token.head if token.head.pos_ == 'VERB' else ''
                        aux_s = [v for v in token.head.head.children if v.pos_ == 'AUX']
                        verb2_ = token.head.head if token.head.head.pos_ == 'VERB' else ''
                        if aux:
                            verb = aux[0]
                        elif aux_s:
                            verb = aux_s[0]
                        elif verb_:
                            verb = verb_
                        elif verb2_:
                            verb = verb2_

                    if verb:
                        noun = [noun for noun in token.head.children if noun.dep_ in {'nsubj', 'nsubjpass'}]
                        noun = noun or [noun for noun in token.head.head.children if
                                        noun.dep_ in {'nsubj', 'nsubjpass'}]

                        if noun:

                            wrong_order = True if verb.i > noun[0].i else False
                            if wrong_order:
                                errors = []
                                errors.append([find_span([verb]), error_message])
                                return errors
            break
        return

    def only(sent):

        error_message = 'You may need inverted word order here.'

        only_1 = [{'RIGHT_ID': 'only', 'RIGHT_ATTRS': {'LEMMA': 'only', 'DEP': 'advmod'}},
                  {'LEFT_ID': 'only', 'REL_OP': '<', 'RIGHT_ID': 'advcl', 'RIGHT_ATTRS': {'TAG': 'WRB'}},
                  {'LEFT_ID': 'advcl', 'REL_OP': '$++', 'RIGHT_ID': 'verb', 'RIGHT_ATTRS': {'DEP': 'ccomp'}},
                  {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'noun',
                   'RIGHT_ATTRS': {'DEP': {'IN': ['nsubj', 'nsubjpass']}}}]

        only_1_2 = [{'RIGHT_ID': 'only', 'RIGHT_ATTRS': {'LEMMA': 'only', 'DEP': 'advmod'}},
                    {'LEFT_ID': 'only', 'REL_OP': '<<', 'RIGHT_ID': 'advcl', 'RIGHT_ATTRS': {'DEP': 'advcl'}},
                    {'LEFT_ID': 'advcl', 'REL_OP': '<', 'RIGHT_ID': 'verb', 'RIGHT_ATTRS': {'POS': 'VERB'}},
                    {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'noun',
                     'RIGHT_ATTRS': {'DEP': {'IN': ['nsubj', 'nsubjpass']}}}]

        dep_matcher_ = DependencyMatcher(vocab=nlp.vocab)
        dep_matcher_.add('only_1', patterns=[only_1, only_1_2])

        verb_start = None
        noun = None
        if dep_matcher_(sent):

            if not sent[dep_matcher_(sent)[0][1][0]].i - sent.start:
                verb_start = sent[dep_matcher_(sent)[0][1][-2]]
                noun = sent[dep_matcher_(sent)[0][1][-1]]
            for child in sent[dep_matcher_(sent)[0][1][0]].head.children:
                if sent[dep_matcher_(sent)[0][1][0]].head.dep_ == 'nummod' or \
                        not without_child(child, {'dep_': 'prep', 'lemma_': 'of'}):
                    verb_start = ''

                    break
        elif sent[0].lemma_ == 'only' and sent[0].dep_ == 'advmod':
            if sent[0].head.dep_ != 'nummod':
                verbs = {}
                matcher_only = Matcher(vocab=nlp.vocab)
                matcher_only.add('verb', patterns=[[{'POS': 'VERB'}]])
                if len(matcher_only(sent)) == 1:
                    verb_start = sent[matcher_only(sent)[0][1]]
                    noun = [n for n in verb_start.children if n.dep_ in ['nsubj', 'nsubjpass']]
                    if noun:
                        noun = noun[0]
            if noun and verb_start:
                aux_ = [aux for aux in verb_start.children if aux.pos_ == 'AUX']
                if aux_:
                    verb_start = aux_[0]
                if verb_start.i > noun.i:
                    err = [[find_span([verb_start]), error_message]]
                    return err
        return

    def extra_inversion(sent):

        error_message = 'You may have used the wrong word order'
        if '"' not in sent.text and '"' not in sent.text:
            dep_matcher_ = DependencyMatcher(vocab=nlp.vocab)
            ccomp_ = [{'RIGHT_ID': 'verb', 'RIGHT_ATTRS': {'DEP': 'ROOT'}},
                      {'LEFT_ID': 'verb', 'REL_OP': '>', 'RIGHT_ID': 'ccomp', 'RIGHT_ATTRS': {'DEP': 'ccomp'}}]
            dep_matcher_.add('ccomp_', patterns=[ccomp_])
            if dep_matcher_(sent):
                ccomp = sent[dep_matcher_(sent)[0][1][-1]]
                i = ccomp.i
                del dep_matcher_
                dep_matcher_ = DependencyMatcher(vocab=nlp.vocab, validate=True)

                wrb = [{'RIGHT_ID': 'ccomp', 'RIGHT_ATTRS': {'DEP': 'ccomp'}},
                       {'LEFT_ID': 'ccomp', 'REL_OP': '>', 'RIGHT_ID': 'wrb',
                        'RIGHT_ATTRS': {'DEP': 'advmod', 'TAG': 'WRB'}}]

                whether_if = [{'RIGHT_ID': 'ccomp', 'RIGHT_ATTRS': {'DEP': 'ccomp'}},
                              {'LEFT_ID': 'ccomp', 'REL_OP': '>', 'RIGHT_ID': 'sconj',
                               'RIGHT_ATTRS': {'DEP': 'mark', 'POS': 'SCONJ', 'LEMMA': {'IN': ['whether', 'if']}}}]
                dep_matcher_.add('wrb', patterns=[wrb, whether_if])
                for match in dep_matcher_(sent):
                    ccomp_verb = sent[match[1][0]]
                    noun = [i for i in ccomp_verb.children if i.dep_ in ['nsubj', 'nsubjpass']]
                    aux = [i for i in ccomp_verb.children if i.pos_ == 'AUX']
                    if noun and aux:
                        if aux[0].i < noun[0].i:
                            words = [sent[k] for k in range(aux[0].i, noun[0].i + 1)]
                            return ([find_span(words), error_message])
            return

    def spelling(sent):
        spell = SpellChecker()
        errors = []
        for token in sent:
            if len(token.text) <4:continue;
            if str(spell.correction(token.text)) != str(token.text):
                candidates = "/".join(spell.candidates(token.text))
                errors.append([find_span([token]),
                               f'You might have misspelled that word, possible '
                               f'correction: {candidates}.'])
        return errors

    def hardly(sent):

        hardly = [
            {'RIGHT_ID': 'hardly', 'RIGHT_ATTRS': {'DEP': 'advmod', 'LEMMA': {'IN': ['hardly', 'scarcely', 'barely']}}},
            {'LEFT_ID': 'hardly', 'REL_OP': '<', 'RIGHT_ID': 'verb', 'RIGHT_ATTRS': {'POS': 'VERB'}}]

        no_sooner = [{'RIGHT_ID': 'sooner', 'RIGHT_ATTRS': {'DEP': 'advmod', 'LEMMA': 'soon'}},
                     {'LEFT_ID': 'sooner', 'REL_OP': '<', 'RIGHT_ID': 'verb', 'RIGHT_ATTRS': {'POS': 'VERB'}},
                     {'LEFT_ID': 'sooner', 'REL_OP': '>', 'RIGHT_ID': 'no',
                      'RIGHT_ATTRS': {'DEP': 'neg', 'LEMMA': 'no'}}]

        dep_matcher_ = DependencyMatcher(vocab=nlp.vocab)
        dep_matcher_.add('hardly', patterns=[hardly])
        dep_matcher_.add('sooner', patterns=[no_sooner])
        errors = []
        if dep_matcher_(sent):
            if not sent[dep_matcher_(sent)[0][1][0]].i - sent.start:
                matched = nlp.vocab[dep_matcher_(sent)[0][0]]

                root_ = sent.root
                advcls = [x for x in sent if x.dep_ == 'advcl']
                if advcls:

                    advcl = advcls[0]
                    conj = ''
                    allowed = ''
                    if matched.text == 'hardly':  # TODO italics comment
                        conj = [wh for wh in advcl.children if wh.dep_ == 'advmod' and wh.lemma_ in {'when', 'before'}]
                        allowed = '\'when\' or \'before\''
                    else:

                        conj = [wh for wh in advcl.children if wh.dep_ == 'mark' and wh.lemma_ == 'than']
                        allowed = 'than'
                    if not conj:
                        hardly = sent[dep_matcher_(sent)[0][1][0]]
                        errors.append([find_span([hardly]),
                                       f'With such construction of the main clause, the next clause should be introduced with {allowed}.'])

                    if not (sent[dep_matcher_(sent)[0][1][0]].tag_ == 'VBN' and [aux for aux in
                                                                                 sent[dep_matcher_(sent)[0][1][
                                                                                     0]].children
                                                                                 if
                                                                                 aux.dep_ == 'aux' and aux.pos_ == 'VBD' and aux.lemma_ == 'have']):
                        errors.append([find_span([sent[dep_matcher_(sent)[0][1][0]]]),
                                       'Past Perfect should be used in the main clause.'])
                    for clause in advcls:
                        verb = clause

                        if not (verb.tag_ == 'VBD' and without_child(verb, {'pos_': 'AUX', 'dep_': 'aux'})):
                            errors.append([find_span([verb]), 'Past Simple should be used in this clause.'])
            return errors

    def conditionals(sent):

        if [x for x in sent if x.lemma_ in {'if'}]:
            root = sent.root if sent.root.pos_ == 'VERB' else ''
            if root:
                errors = []
                will = [x for x in root.children if x.dep_ == 'aux' and x.lemma_ == 'will']
                would = [x for x in root.children if
                         x.dep_ == 'aux' and x.lemma_ == 'would' and x.head.lemma_ != 'like']
                advcls = [x for x in sent if x.dep_ == 'advcl' and [y for y in x.children if
                                                                    y.lemma_ in {'if'} and y.dep_ == 'mark']]
                for advcl in advcls:

                    ss = 'http://realec-reference.site/viewArticle/CONDITIONAL%20SENTENCES'
                    will_if_clause = [x for x in advcl.children if x.dep_ == 'aux' and x.lemma_ in {'will', 'would'}]

                    if will_if_clause:
                        errors.append([find_span([will_if_clause[0]]),
                                       '\'Will\' or \'would\' after \'if\' are not used in conditionals.'])
                    verb = advcl
                    if will:

                        have = [have for have in verb.children if
                                (have.lemma_ == 'have' and have.dep_ == 'aux' and have.tag_ in {'VBP', 'VBZ'})]
                        pres_perfect = True if verb.tag_ == 'VBN' and have else False
                        pres_simple = True if verb.tag_ in {'VB', 'VBP', 'VBZ'} else False
                        if not (pres_perfect | pres_simple):
                            errors.append([find_span([verb]),
                                           'In if-clauses talking about future, Present Simple or Present Perfect are expected. (More examples: http://realec-reference.site/viewArticle/CONDITIONAL%20SENTENCES )'])

                    elif would:
                        past_perfect = True if verb.tag_ == 'VBN' \
                                               and [have for have in verb.children if have.dep_ == 'aux' \
                                                    and have.text == 'had'] else False
                        past_simple = True if verb.tag_ == 'VBD' \
                                              and without_child(verb, {'dep_': 'aux'}) else False

                        if not (past_perfect | past_simple):
                            errors.append([find_span([verb]),
                                           r'In if-clauses talking about unreal conditions, Past Simple or '
                                           r'Past Perfect are expected.(More examples: '
                                           r'http://realec-reference.site/viewArticle/CONDITIONAL%20SENTENCES) '])

                return errors
            # TODO if/provided/providing/unless/on condition/lest

    def that_comma(sent):

        error_that = DependencyMatcher(vocab=nlp.vocab)
        comma_that = [{'RIGHT_ID': 'comma', 'RIGHT_ATTRS': {"LEMMA": ",", "POS": "PUNCT"}},
                      {'LEFT_ID': 'comma', 'REL_OP': '.', 'RIGHT_ID': 'that', 'RIGHT_ATTRS': {'LEMMA': 'that'}}]
        error_that.add('comma_that', [comma_that])
        if error_that(sent):
            found_subj = [token.i for token in sent if 'subj' in str(token.dep_)]
            found_subj.sort()
            found_subj = found_subj[0]
            errors = []
            for match in error_that(sent):
                that = sent[match[1][-1]]
                if 'subj' in str(that.dep_):
                    if that.i != found_subj:  # deleting cases as Unfortunately, that
                        errors.append([find_span([sent[match[1][0]], that]),
                                       'Instead of the comma, semicolon has to be used in front of \'that\'.'])
            first_that = sent[error_that(sent)[0][1][-1]]
            if first_that.dep_ == 'mark':
                errors.append([find_span([sent[error_that(sent)[0][1][0]], first_that]),
                               'You may have used a redundant comma in front of \'that\'.'])
            return errors
        return

    def punct(sent):
        # TODO in addition, - однозначный случай / in addition to зависимые от addition
        pass

    def redundant_comma(sent):

        error_clause = DependencyMatcher(vocab=nlp.vocab)
        wh_clause = [{'RIGHT_ID': 'root', 'RIGHT_ATTRS': {'DEP': {'IN': ['ROOT', 'ccomp']}}},
                     {'LEFT_ID': 'root', 'REL_OP': '>', 'RIGHT_ID': 'ccomp', 'RIGHT_ATTRS': {'DEP': 'ccomp'}},
                     {'LEFT_ID': 'ccomp', 'REL_OP': '>', 'RIGHT_ID': 'wrb',
                      'RIGHT_ATTRS': {'TAG': {'IN': ['WP', 'WRB']}}},
                     {'LEFT_ID': 'wrb', 'REL_OP': ';', 'RIGHT_ID': 'punct', 'RIGHT_ATTRS': {'POS': 'PUNCT'}}]

        whether_clause = [{'RIGHT_ID': 'root', 'RIGHT_ATTRS': {'DEP': {'IN': ['ROOT', 'ccomp']}}},
                          {'LEFT_ID': 'root', 'REL_OP': '>', 'RIGHT_ID': 'ccomp', 'RIGHT_ATTRS': {'DEP': 'ccomp'}},
                          {'LEFT_ID': 'ccomp', 'REL_OP': '>', 'RIGHT_ID': 'whether',
                           'RIGHT_ATTRS': {'DEP': 'mark', 'LEMMA': {'IN': ['whether', 'if']}}},
                          {'LEFT_ID': 'whether', 'REL_OP': ';', 'RIGHT_ID': 'punct', 'RIGHT_ATTRS': {'POS': 'PUNCT'}}]

        error_clause.add('wh_clause', [wh_clause, whether_clause])
        errors = []
        first = True
        for match in error_clause(sent):
            if first:
                if 'subj' not in sent[match[1][-2]].dep_:
                    first = False

                    errors.append([find_span([sent[match[1][-1]], sent[match[1][-2]]]),
                                   'You may have used a redundant comma in front of this conjunction.'])
        return errors

    def past_cont(sent):
        p_cont = [{'RIGHT_ID': 'vbg', 'RIGHT_ATTRS': {'TAG': 'VBG'}},
                  {'LEFT_ID': 'vbg', 'REL_OP': '>', 'RIGHT_ID': 'was', 'RIGHT_ATTRS': {'DEP': 'aux', 'TAG': 'VBD'}}]
        error_clause = DependencyMatcher(vocab=nlp.vocab)
        error_clause.add('p_cont', [p_cont])
        for match in error_clause(sent):
            verb = sent[match[1][0]]
            errors = []
            if without_child(verb, {'dep_': 'advmod', 'tag_': 'RB'}):
                advcls = [x for x in verb.children if x.dep_ in {'relcl', 'advcl'}]
                advcls = advcls or [x for x in sent if x.dep_ in {'relcl', 'advcl'}]
                while_ = False
                for advcl in advcls:
                    if any([while_.dep_ == 'mark'
                            and while_.lemma_ in {'as', 'while'} for while_ \
                            in advcl.children]): while_ = True;break;
                # TODO always/never/forever/constantly/permanently/eternally  | for ever
                # Исключения: every ..
                if not while_ or errors:
                    errors.append([find_span([sent[match[1][1]], sent[match[1][0]]]),
                                   'The usage of Past Continuous might be erroneous.'])

            return errors

        return

    def consider_that(sent):
        error_message = 'You may have wrongly used the verb CONSIDER with THAT.'
        consider_that_errs = []
        consider_that = [{'LEMMA': 'consider'}, {'LEMMA': 'that'}]
        matcher = Matcher(vocab=nlp.vocab)
        matcher.add("consider_that", patterns=[consider_that])
        for matched in matcher(sent):
            consider_that_errs.append([find_span([sent[matched[1]], sent[matched[2]]]), error_message])
        return consider_that_errs

    def agreement_s_v(sent):
        pass

    def prep(sent):
        pass

    def adj(sent):
        pass

    def quantifiers(sent):
        erroneous = []
        error_message = 'That might be an erroneous use of quantifiers'
        uncount = ['much', 'less', 'least']
        uncount_of = ['deal', 'amount']
        count_pl = ['many', 'few', 'fewer', 'fewest', 'several']
        count_pl_of = ['number', 'numbers', 'couple', 'hundreds', \
                       'thousands', 'millions', 'billions']
        count_sg = ['each', 'every', 'another']
        # uncount_count_pl = ['some', 'all', 'no'] невозможно проверить без списка uncountable
        # uncount_count_pl_of = ['lot', 'lots', 'none'] невозможно проверить без списка uncountable
        quantifiers_ = uncount + count_pl + count_sg
        quantifiers_of = uncount_of + count_pl_of
        for i in sent:
            if i.text in quantifiers_:
                if (i.text in uncount or i.text in count_sg) and 'Plur' in i.head.morph.get("Number"):
                    erroneous.append([find_span([i, i.head]), error_message])
            elif i.text in count_pl and 'Sing' in i.head.morph.get("Number"):
                erroneous.append([find_span([i, i.head]), error_message])
            if i.text in quantifiers_of:
                for child in i.children:
                    if child.text == 'of':
                        for c in child.children:
                            if i.text in uncount_of and 'Plur' in c.morph.get("Number"):
                                erroneous.append([find_span([i, c]), error_message])
                            elif i.text in count_pl_of and 'Sing' in c.morph.get("Number"):
                                erroneous.append([find_span([i, c]), error_message])
            if i.text == 'both':
                all_c = [x for x in i.head.children]
                if i.head.morph.get("Number") == 'Plur':
                    pass
                elif len(all_c) > 2:
                    if all_c[1].text != 'and' and (all_c[2].pos_ != 'NOUN' or all_c[2].pos_ != 'PRON'):
                        erroneous.append([find_span([i]), error_message])
                else:
                    erroneous.append([find_span([i]), error_message])
        return erroneous

    def polarity(sentence):
        # Checks if any polarity items were used in the wrong context

        neg_error_message = 'This item can only be used in negative contexts.'
        pos_error_message = 'This item can only be used in positive contexts.'
        polarity_errors = []

        negation = r"[Nn]['`’o]t\b|[Nn]ever(?!theless| the less)|\b[Nn]o\b|[Nn]owhere|[Nn]othing|[Nn]one|[Nn]oone|[Nn]either|"
        negation += r'[Hh]ardly|[Ss]carcely|[Bb]arely|^[Ff]ew|(?<![Aa] )[Ff]ew|[Nn]or|[Ll]ack|[Ss]top|[Aa]ny'
        ifclause = r'\b[Ii]f\b|[Ww]hether'
        negative_lic = re.compile('|'.join((negation, ifclause)))
        neg_gr = r'at all|whatsoever|just yet|yet'
        neg_exp = r'lift[a-z]{0,3} a finger|(sleep[a-z]{0,3}|slept) a wink|bat[a-z]{0,4} an eye|((takes?|took|taking)|(last[a-z]{0,3})) long\b|(drink[a-z]{0,3}|drank|drunk) a drop|(mean|small) feat'
        neg_exp += r'|put( \w+?| ) ?finger on|(thinks?|thought) much '
        temporal_neg_exp = r'in (?:hours|days|weeks(?! [0-9])|months(?! [JFMASOD])|years(?! gone| past| [a-zA-Z]*? ?[0-9])|decades|yonks|eons|a million years|ages(?! [0-9])|donkey\'s years)'
        neg_pol = re.compile('|'.join([neg_gr, neg_exp, temporal_neg_exp]))
        pos_pol = re.compile(r'already|somewhat|too')

        words = ' '.join([token.text for token in sentence])
        # When there is a negative polarity item but no prior negation
        neg = re.search(neg_pol, words)
        if sentence.text[-1] != '?' and neg:
            neg_token = None
            licensed = False
            for token in sentence:
                # Presuming if the whole NP expression is in the scope of negation then its every element will be too
                # Hence only check the first word of the NPI to match a token
                if token.text == neg.group().split()[0]:
                    neg_token = token  # technically a faulty verification, esp. with 'at all' where 'at' can be repeated later and therefore mixed up, but will have to do for now
            if neg_token:
                if (neg_token.text == 'at' and neg_token.dep_ != 'advmod') or (
                        neg_token.text == 'yet' and neg_token.head.lemma_ != 'have' and neg_token.head.dep_ != 'aux'):
                    licensed = True  # Not licensed per se, but rather not an NPI at all
                lic = re.findall(negative_lic, words)
                for l in lic:
                    for token in sentence:
                        # An NPI is licensed when it's within the scope of (i.e. c-commanded by) a negation
                        if token.text == l and (token.head.is_ancestor(neg_token) or token.is_ancestor(neg_token)):
                            licensed = True
                if not licensed:
                    polarity_errors.append([find_span([neg_token]),
                                            neg_error_message])  # if more than one word in polarity item, only finds the first one

        # todo add superlative licensing for temporal PI

        # When there is a positive polarity item but it is negated
        pos = re.search(pos_pol, words)
        if pos:
            pos_token = None
            licensed = True
            for token in sentence:
                if token.text == pos.group().split()[0]:
                    pos_token = token
            if pos_token:
                anti_lic = re.findall(negation, words)
                for al in anti_lic:
                    for token in sentence:
                        if token.text == al and token.is_ancestor(pos_token):
                            licensed = False
                if pos_token.text == 'too' and pos_token.i != len(doc) - 1:
                    if pos_token.nbor().text in ['much', 'many'] or pos_token.nbor().pos_ == 'ADJ':
                        licensed = True  # Not licensed per se, but rather not a PPI at all
                if not licensed:
                    polarity_errors.append([find_span([pos_token]), pos_error_message])

        # Checking if negation's parent is an ancestor of a PPI yields too many false positives

        return polarity_errors

    def preprocess(sentences):
        sentence_dots: str = re.sub(r'\.', '. ', sentences)
        sentence_dots = re.sub(r'\.\s\.\s\.', '...', sentence_dots)
        sentence_dots = re.sub(r'\.\s\.', '..', sentence_dots)
        sentence_dots = re.sub(r'\!', '! ', sentence_dots)
        sentence_dots = re.sub(r'\?', '? ', sentence_dots)
        sentence_dots = re.sub(r'\s\s', ' ', sentence_dots)
        return sentence_dots

    def apply_models(sentence_process, test_mode):
        result = []
        if test_mode:
            for sent in sentence_process.sents:
                for function in test_mode:
                    exec(f'result.append({function}(sent))')


        else:
            observed_functions = {quantifiers, past_cont, redundant_comma, hardly, that_comma, pp_time,
                                  only, inversion, extra_inversion, spelling, conditionals,
                                  consider_that, polarity}

            apply_ = lambda f, given_: f(given_)
            for function in observed_functions:
                sentence_err = apply_(function, sentence_process)
                if sentence_err:
                    result.extend({
                        'start': start,
                        'end': end,
                        'type': function.__name__,
                        'msg': msg,
                    } for (start, end), msg in sentence_err)
        return result

    text_ = preprocess(text)
    doc = nlp(text_)
    if test_mode:
        return apply_models(doc, test_mode)
    all_errors = dict()

    sents = []
    for num, sentence in enumerate(doc.sents):
        errors = apply_models(sentence, test_mode)
        if errors:
            if isinstance(errors[-1], str):
                all_errors[num] = [errors]
            else:
                all_errors[num] = errors
        sents.append({
            'text': sentence.text,
            'annotations': errors
        })

    return sents


def generate_text(text, test_mode=False):
    return models(text, test_mode=test_mode)
    # text_, errors = models(text, test_mode=test_mode)
    # annotated_text, comments = output_maker(text_, errors)
    # return text_, errors, annotated_text, comments

#
#
# nlp = spacy.load("en_core_web_lg")
# text_ = 'I have been a student last summer.'
# doc_ = nlp(text_)
# for d in doc_.sents:
#     for t in d:
#         print(t, t.head, t.dep_, t.tag_, t.pos_, t._.tree_tag)
#
# print(generate_text('If I am there, there wouldn\'t be..'))
# 12 % nummod CD NUM CRD
# sixteen percent nummod CD NUM CRD
