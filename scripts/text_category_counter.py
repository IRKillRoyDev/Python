"""Count word categories across period text files.

Each .txt file is one period (the last number in its name). The tool cleans
the text, saves the top 30 words, suggests categories, and counts each
category in the full cleaned text.

Run with no arguments to open the window, or pass file names for the command
line. Results go in a new category_output folder next to the text files.
"""

import argparse
import os
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime

import nltk
import pandas as pd
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize


def download_nltk_data():
    """Download NLTK data if it is not already present."""
    packages = [
        'stopwords',
        'punkt',
        'punkt_tab',
        'wordnet',
        'omw-1.4',
        'averaged_perceptron_tagger_eng',
        'averaged_perceptron_tagger',
    ]
    for package in packages:
        nltk.download(package, quiet=True)


# How many top words to save per file.
num_words = 30

DEFAULT_STOPWORDS_FILE = 'extra_stopwords.txt'
OUTPUT_FOLDER_NAME = 'category_output'
TOP_WORDS_FOLDER = 'top_words'
CATEGORY_FOLDER = 'category_counts'

# Sentence clustering, then a WordNet pass. See suggest_categories.
MIN_SHARED_SENTENCES = 0.2
WORDNET_JOIN = 0.7
WORDNET_MOVE_GAP = 0.2
WORDNET_FLAG = 0.2

# Pairs the tokenizer emits for contractions (can't -> ca + n't). Dropped only
# as a pair, so a real acronym such as CA or AI is kept.
_CONTRACTION_PAIRS = frozenset({
    ('ca', "n't"),
    ('wo', "n't"),
    ('ai', "n't"),
    ('sha', "n't"),
    ('gon', 'na'),
    ('wan', 'na'),
    ('gim', 'me'),
    ('lem', 'me'),
    ('got', 'ta'),
})
_ABBREV_RE = re.compile(r'^[A-Za-z]{1,3}(?:\.[A-Za-z]{1,3})+\.?$')
_ABBREV_STOP = frozenset({'eg', 'ie'})


def find_stopwords_file(filename):
    """Return filename if that file exists, otherwise ''."""
    if filename and os.path.isfile(filename):
        return filename
    return ''


def read_extra_stopwords(filename):
    """Read extra stopwords, one word per line.

    Blank lines and # comments are skipped. Returns [] if there is no file.
    """
    if not filename:
        return []
    try:
        with open(filename, encoding='utf-8') as handle:
            lines = handle.read().splitlines()
    except OSError:
        return []
    words = []
    for line in lines:
        word = line.split('#')[0].strip().lower()
        if word != '':
            words.append(word)
    return words


def get_stop_words(stopwords_filename, more_words=None):
    """Return the NLTK English stopwords plus the file and any typed words."""
    stop = stopwords.words('english')
    stop.extend(read_extra_stopwords(stopwords_filename))
    if more_words:
        stop.extend(w.strip().lower() for w in more_words if w.strip() != '')
    return set(stop)


def get_period(filename):
    """Return the last number in the file name, or the name without .txt."""
    name, _ext = os.path.splitext(os.path.basename(filename))
    numbers = re.findall(r'\d+', name)
    if numbers:
        return int(numbers[-1])
    return name


def _is_word_chars(piece):
    """True when every character is a letter, digit, or combining mark."""
    return bool(piece) and all(
        ch.isalnum() or unicodedata.category(ch).startswith('M') for ch in piece
    )


def _is_acronym(token):
    """True when the first and last letters are capitals (POS, DoS, IoT)."""
    letters = [ch for ch in token if ch.isalpha()]
    return len(letters) >= 2 and letters[0].isupper() and letters[-1].isupper()


def _is_plural_acronym(token):
    """True for a capital stem plus one lowercase plural s (APIs, IoTs)."""
    letters = [ch for ch in token if ch.isalpha()]
    if len(letters) < 3 or letters[-1] != 's' or letters[-1].isupper():
        return False
    return letters[0].isupper() and letters[-2].isupper()


def _word_from_piece(piece):
    """Return (lowercase word, is_acronym) or None if the piece is punctuation."""
    if not piece:
        return None
    if _ABBREV_RE.fullmatch(piece):
        core = piece[:-1] if piece.endswith('.') else piece
        letters = ''.join(ch for ch in core if ch.isalpha())
        return core.replace('.', '').lower(), _is_acronym(letters)
    if not _is_word_chars(piece):
        return None
    if _is_plural_acronym(piece):
        return piece.lower()[:-1], True
    return piece.lower(), _is_acronym(piece)


def _expand_token(token):
    """Split one tokenizer token into words.

    Hyphens and slashes separate words. Dotted abbreviations such as U.S.A.
    and Ph.D. become a single word. e.g. and i.e. are dropped.
    """
    words = []
    for part in re.split(r'[-/]', token):
        item = _word_from_piece(part)
        if item is None:
            continue
        word, acronym = item
        if word in _ABBREV_STOP and not acronym:
            continue
        if word:
            words.append((word, acronym))
    return words


def _contraction_indexes(tokens):
    """Indexes of contraction leftovers such as ca/n't from can't."""
    skip = set()
    lowered = [token.lower() for token in tokens]
    for index in range(len(lowered) - 1):
        if (lowered[index], lowered[index + 1]) in _CONTRACTION_PAIRS:
            skip.add(index)
            skip.add(index + 1)
    return skip


def _wordnet_pos(tag):
    """Map a Penn tag to a WordNet part of speech."""
    if tag.startswith('J'):
        return wordnet.ADJ
    if tag.startswith('V'):
        return wordnet.VERB
    if tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN


def clean_and_count(text, stop_words, lemmatizer):
    """Clean one text and return (total_words, word_counts).

    total_words is the count after punctuation is removed and before stopwords
    are removed. word_counts is after part-of-speech lemmatization and
    stopword removal. Acronyms are kept as written and are not stopwords.
    """
    raw_tokens = word_tokenize(text)
    if not raw_tokens:
        return 0, Counter()
    skip = _contraction_indexes(raw_tokens)
    pieces = []
    for index, (token, tag) in enumerate(pos_tag(raw_tokens)):
        if index in skip:
            continue
        for word, acronym in _expand_token(token):
            pieces.append((word, acronym, tag))
    total_words = len(pieces)
    kept = []
    for word, acronym, tag in pieces:
        if acronym:
            kept.append(word)
            continue
        if word in stop_words:
            continue
        lemma = lemmatizer.lemmatize(word, _wordnet_pos(tag))
        # WordNet maps some non-words onto stopwords ("dos" -> "do").
        if lemma in stop_words:
            lemma = word
        if lemma not in stop_words:
            kept.append(lemma)
    return total_words, Counter(kept)


def make_output_folder(filenames):
    """Create category_output next to the first text file.

    If that name exists, use category_output_2, and so on, so older runs
    are left in place. Returns the new folder path.
    """
    parent = os.path.dirname(os.path.abspath(filenames[0]))
    if not os.path.isdir(parent):
        parent = os.getcwd()
    number = 1
    while True:
        folder_name = OUTPUT_FOLDER_NAME if number == 1 else OUTPUT_FOLDER_NAME + '_' + str(number)
        output_folder = os.path.join(parent, folder_name)
        try:
            os.mkdir(output_folder)
            break
        except FileExistsError:
            number += 1
    print('All output files go in this folder: ' + output_folder)
    return output_folder


def save_top_words(fullname, word_freq, output_folder):
    """Write one file's top words to top_words/<name>_top_words.csv."""
    most_common_words = word_freq.most_common(num_words)
    frame = pd.DataFrame(most_common_words, columns=['Word', 'Count'])
    name, _ext = os.path.splitext(os.path.basename(fullname))
    os.makedirs(os.path.join(output_folder, TOP_WORDS_FOLDER), exist_ok=True)
    output_filename = os.path.join(output_folder, TOP_WORDS_FOLDER, name + '_top_words.csv')
    print_name = os.path.join(TOP_WORDS_FOLDER, name + '_top_words.csv')
    frame.to_csv(output_filename, index=False)
    print('Your output file has been written.  Its name is ' + print_name)
    return frame


def read_and_clean_files(filenames, stopwords_filename, more_words=None):
    """Read each text file, clean it, and save its top words.

    Returns (results, run). results is keyed by period. A missing text file
    stops the run. A missing stopwords file does not: only the NLTK list is used.
    """
    if not filenames:
        raise ValueError('No text files were given. Upload or choose your .txt files first.')
    if (
        stopwords_filename
        and stopwords_filename != DEFAULT_STOPWORDS_FILE
        and not find_stopwords_file(stopwords_filename)
    ):
        print('The stopwords file "' + str(stopwords_filename) + '" is not there, so no stopwords file is used.')
    stopwords_filename = find_stopwords_file(stopwords_filename)
    run = {
        'time': datetime.now(),
        'output_folder': None,
        'results': {},
        'stopwords_file': stopwords_filename,
        'more_words': [w.strip().lower() for w in (more_words or []) if w.strip() != ''],
        'stop_words': None,
        'counts': [],
        'errors': [],
        'warnings': [],
        'suggested': None,
    }

    texts = {}
    for fullname in filenames:
        try:
            with open(fullname, encoding='utf-8') as handle:
                texts[fullname] = handle.read()
        except (OSError, UnicodeDecodeError):
            run['errors'].append(
                'Could not read the text file "' + str(fullname) + '". '
                'Check the spelling and that it is a plain text file in the working folder.'
            )
    run['output_folder'] = make_output_folder(filenames)
    if run['errors']:
        write_summary(run)
        raise FileNotFoundError(run['errors'][0])

    run['stop_words'] = get_stop_words(stopwords_filename, more_words)
    print(stopwords_line(run))

    lemmatizer = WordNetLemmatizer()
    results = run['results']
    for fullname in filenames:
        print('File read successfully: ' + os.path.basename(fullname))
        period = get_period(fullname)
        if not re.findall(r'\d+', os.path.basename(fullname)):
            run['warnings'].append(
                os.path.basename(fullname) + ' has no year or period number in its name, '
                'so the file name is used as the period.'
            )
        if period in results:
            print('Warning: two files give the period', period, '- the second one replaces the first.')
            run['warnings'].append(
                'Two files give the period ' + str(period) + ', so ' + results[period]['file']
                + ' was replaced by ' + os.path.basename(fullname) + '.'
            )
        total_words, word_freq = clean_and_count(texts[fullname], run['stop_words'], lemmatizer)
        frame = save_top_words(fullname, word_freq, run['output_folder'])
        results[period] = {
            'file': os.path.basename(fullname),
            'path': fullname,
            'text': texts[fullname],
            'total_words': total_words,
            'word_freq': word_freq,
            'top_words': list(zip(frame['Word'], frame['Count'])),
        }
    write_summary(run)
    return results, run


def top_words_union(results):
    """Return every word in any period's top 30, with its total count, A to Z."""
    words = sorted({w for period in results for w, _count in results[period]['top_words']})
    counts = [sum(results[period]['word_freq'][w] for period in results) for w in words]
    return pd.DataFrame({'Word': words, 'Count': counts})


def count_shared_sentences(results, words, stop_words):
    """Count how often words share sentences across all files.

    Returns in_sentences[word] and together[(word1, word2)] for word1 < word2.
    """
    lemmatizer = WordNetLemmatizer()
    wanted = set(words)
    in_sentences = {w: 0 for w in words}
    together = {}
    for period in results:
        text = results[period].get('text')
        if text is None:
            with open(results[period]['path'], encoding='utf-8') as handle:
                text = handle.read()
        for sentence in sent_tokenize(text):
            _total, word_freq = clean_and_count(sentence, stop_words, lemmatizer)
            present = sorted(w for w in word_freq if w in wanted)
            for i, first in enumerate(present):
                in_sentences[first] += 1
                for second in present[i + 1:]:
                    together[(first, second)] = together.get((first, second), 0) + 1
    return in_sentences, together


def group_by_shared_sentences(words, in_sentences, together):
    """Group words that often share sentences.

    Closeness is sentences-with-both / sqrt(count1 * count2). Groups merge
    while the closest pair's average closeness is at least MIN_SHARED_SENTENCES.
    Ties keep the earlier pair, so the same input always gives the same groups.
    Returns groups of 2 or more words.
    """
    def closeness(first, second):
        both = together.get((min(first, second), max(first, second)), 0)
        if both == 0:
            return 0.0
        return both / (in_sentences[first] * in_sentences[second]) ** 0.5

    groups = [[w] for w in sorted(words)]
    while True:
        best, best_pair = -1.0, None
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                pairs = [(w1, w2) for w1 in groups[a] for w2 in groups[b]]
                average = sum(closeness(w1, w2) for w1, w2 in pairs) / len(pairs)
                if average > best:
                    best, best_pair = average, (a, b)
        if best_pair is None or best < MIN_SHARED_SENTENCES:
            break
        a, b = best_pair
        groups[a] = sorted(groups[a] + groups[b])
        del groups[b]
    return [g for g in groups if len(g) >= 2]


def check_with_wordnet(groups, words):
    """Adjust groups using WordNet similarity. Returns (groups, notes).

    Fit is the average Wu-Palmer similarity to the other known words in a
    group, using each word's most common meaning. Scores are computed on the
    groups as they stood before this pass.
    An ungrouped word joins its best group at WORDNET_JOIN or above.
    A grouped word moves at that same level when its own group has no score,
    or when the new fit is at least WORDNET_MOVE_GAP better.
    A fit under WORDNET_FLAG is only noted.
    """
    meaning = {}
    for word in words:
        synsets = wordnet.synsets(word)
        meaning[word] = synsets[0] if synsets else None

    def fit(word, group):
        others = [item for item in group if item != word and meaning[item] is not None]
        if meaning[word] is None or not others:
            return None
        return sum(meaning[word].wup_similarity(meaning[item]) or 0 for item in others) / len(others)

    notes = []
    moves = {}
    home = {word: index for index, group in enumerate(groups) for word in group}
    for word in sorted(words):
        own = fit(word, groups[home[word]]) if word in home else None
        best, best_i = None, None
        for index, group in enumerate(groups):
            if index == home.get(word):
                continue
            score = fit(word, group)
            if score is not None and (best is None or score > best):
                best, best_i = score, index
        if word not in home:
            if best is not None and best >= WORDNET_JOIN:
                moves[word] = best_i
                notes.append(
                    word + ' joins the group with ' + ', '.join(groups[best_i][:3])
                    + ' (meaning fit ' + str(round(best, 2)) + ')'
                )
        elif best is not None and best >= WORDNET_JOIN and (own is None or best - own >= WORDNET_MOVE_GAP):
            moves[word] = best_i
            notes.append(
                word + ' moves to the group with ' + ', '.join(groups[best_i][:3])
                + ' (meaning fit ' + str(round(best, 2)) + ')'
            )
        elif own is not None and own < WORDNET_FLAG:
            notes.append(word + ' may not fit its group (meaning fit ' + str(round(own, 2)) + ')')

    new_groups = [[word for word in group if word not in moves] for group in groups]
    for word, index in moves.items():
        new_groups[index].append(word)
    return [sorted(group) for group in new_groups if len(group) >= 2], notes


def name_groups(groups, words_table):
    """Name groups and order them by total word count, largest first.

    The short name is the most-used word that is not already a name. The long
    name is the two most-used words. Returns (short -> words, short -> long).
    """
    total = dict(zip(words_table['Word'], words_table['Count']))
    ordered = sorted(
        (sorted(group, key=lambda w: (-total.get(w, 0), w)) for group in groups),
        key=lambda group: (-sum(total.get(w, 0) for w in group), group[0]),
    )
    named = {}
    long_names = {}
    for group in ordered:
        short = next((word for word in group if word not in named), None)
        if short is None:
            number = 2
            while group[0] + ' ' + str(number) in named:
                number += 1
            short = group[0] + ' ' + str(number)
        named[short] = group
        if len(group) >= 2:
            long_names[short] = group[0] + ' and ' + group[1]
        else:
            long_names[short] = short
    return named, long_names


def long_name_of(category, long_names):
    """Return the long name, or the short name when none was set."""
    if long_names and long_names.get(category, '').strip() != '':
        return long_names[category].strip()
    return category


def custom_name_of(category, custom_names):
    """Return the custom name, or '' when none was set."""
    if custom_names and custom_names.get(category, '').strip() != '':
        return custom_names[category].strip()
    return ''


def long_names_from_words(groups, words_table):
    """Rebuild each long name from that category's current words."""
    total = dict(zip(words_table['Word'], words_table['Count']))
    long_names = {}
    for category, words in groups.items():
        ordered = sorted(set(words), key=lambda w: (-total.get(w, 0), w))
        if len(ordered) >= 2:
            long_names[category] = ordered[0] + ' and ' + ordered[1]
        elif ordered:
            long_names[category] = ordered[0]
        else:
            long_names[category] = category
    return long_names


def suggest_categories(results, stop_words, run=None):
    """Suggest categories from shared sentences, then check them with WordNet.

    Returns (short name -> words, short name -> long name).
    """
    words_table = top_words_union(results)
    words = list(words_table['Word'])
    in_sentences, together = count_shared_sentences(results, words, stop_words)
    first = group_by_shared_sentences(words, in_sentences, together)
    print('Suggested groups from shared sentences (short name (long name): words):')
    first_named, first_long = name_groups(first, words_table)
    for name, group in first_named.items():
        print('  ' + name + ' (' + first_long[name] + '): ' + ', '.join(group))
    second, notes = check_with_wordnet(first, words)
    print('After the WordNet meaning check:')
    for note in notes:
        print('  - ' + note)
    suggested, long_names = name_groups(second, words_table)
    for name, group in suggested.items():
        print('  ' + name + ' (' + long_names[name] + '): ' + ', '.join(group))
    if run is not None:
        flagged = [note for note in notes if 'may not fit' in note]
        run['suggested'] = {
            'words': len(words),
            'first_groups': len(first),
            'changed': len(notes) - len(flagged),
            'flagged': len(flagged),
            'groups': len(suggested),
        }
        write_summary(run)
    return suggested, long_names


def read_categories(categories_filename):
    """Read a categories CSV (Category, Word, and optional name columns).

    Words are lowercased. A blank category or word is skipped. Older files
    with only Category and Word still load.
    """
    try:
        table = pd.read_csv(categories_filename, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    except OSError:
        raise FileNotFoundError(
            'Could not open the categories file "' + str(categories_filename) + '". '
            'Check the spelling and that the file is in the working folder.'
        ) from None
    if 'Category' not in table.columns or 'Word' not in table.columns:
        raise ValueError(
            'The categories file "' + str(categories_filename) + '" needs two columns named Category and Word.'
        )
    blank = [''] * len(table)
    long_column = list(table['Long Name']) if 'Long Name' in table.columns else blank
    custom_column = list(table['Custom Name']) if 'Custom Name' in table.columns else blank
    groups = {}
    long_names = {}
    custom_names = {}
    rows = zip(table['Category'], long_column, custom_column, table['Word'])
    for category, long_name, custom_name, word in rows:
        category = category.strip()
        word = word.strip().lower()
        if category == '' or word == '':
            continue
        groups.setdefault(category, [])
        if word not in groups[category]:
            groups[category].append(word)
        if category not in long_names and long_name.strip() != '':
            long_names[category] = long_name.strip()
        if category not in custom_names and custom_name.strip() != '':
            custom_names[category] = custom_name.strip()
    for category in groups:
        long_names[category] = long_name_of(category, long_names)
    return groups, long_names, custom_names


def save_categories(groups, long_names, custom_names, categories_filename):
    """Save categories as Category, Long Name, Custom Name, Word."""
    rows = [
        [category, long_name_of(category, long_names), custom_name_of(category, custom_names), word]
        for category, words in groups.items()
        for word in sorted(set(words))
    ]
    table = pd.DataFrame(rows, columns=['Category', 'Long Name', 'Custom Name', 'Word'])
    table.to_csv(categories_filename, index=False)
    return table


def count_categories(results, groups, long_names=None, custom_names=None):
    """Count each category in every period's full cleaned text.

    Columns: Period, Category, Long Name, Custom Name, Count, Total Words,
    Words Included. Count uses every cleaned word, not only the top 30.
    """
    rows = []
    for period in results:
        word_freq = results[period]['word_freq']
        for category, words in groups.items():
            if not words:
                continue
            word_list = sorted(set(words))
            count = sum(word_freq[w] for w in word_list)
            rows.append([
                period,
                category,
                long_name_of(category, long_names),
                custom_name_of(category, custom_names),
                count,
                results[period]['total_words'],
                ', '.join(word_list),
            ])
    return pd.DataFrame(rows, columns=[
        'Period', 'Category', 'Long Name', 'Custom Name', 'Count', 'Total Words', 'Words Included',
    ])


def output_name(categories_filename):
    """Results file name derived from the categories file name."""
    folder, name = os.path.split(categories_filename)
    name2, _ext = os.path.splitext(name)
    if 'categories' in name2:
        new_name = name2.replace('categories', 'category_counts', 1) + '.csv'
    else:
        new_name = name2 + '_category_counts.csv'
    return os.path.join(folder, new_name)


def free_names(folder, categories_name, results_name=None):
    """Pick category and results names that do not overwrite existing files."""
    name2, ext2 = os.path.splitext(os.path.basename(categories_name))
    ext2 = ext2 or '.csv'
    res2 = res_ext = None
    if results_name:
        res2, res_ext = os.path.splitext(os.path.basename(results_name))
        res_ext = res_ext or '.csv'
    number = 1
    while True:
        tail = '' if number == 1 else '_' + str(number)
        categories_path = os.path.join(folder, name2 + tail + ext2)
        if results_name:
            results_path = os.path.join(folder, res2 + tail + res_ext)
        else:
            results_path = output_name(categories_path)
        if categories_path == results_path:
            root, ext = os.path.splitext(results_path)
            results_path = root + '_counts' + ext
        if not os.path.exists(categories_path) and not os.path.exists(results_path):
            return categories_path, results_path
        number += 1


def count_and_save(results, groups, long_names, custom_names, categories_name, run, results_name=None):
    """Count categories and save both CSVs under category_counts/."""
    kept_groups = {}
    kept_long_names = {}
    kept_custom_names = {}
    for category, words in groups.items():
        if category.strip() != '' and words:
            key = category.strip()
            kept_groups[key] = words
            kept_long_names[key] = long_name_of(category, long_names)
            kept_custom_names[key] = custom_name_of(category, custom_names)
    if not kept_groups:
        raise ValueError('No category has any words yet. Make a category and pick its words first.')
    folder = os.path.join(run['output_folder'], CATEGORY_FOLDER)
    os.makedirs(folder, exist_ok=True)
    categories_filename, output_filename = free_names(folder, categories_name or 'categories.csv', results_name)
    save_categories(kept_groups, kept_long_names, kept_custom_names, categories_filename)
    print(
        'Your categories file has been written.  Its name is '
        + os.path.join(CATEGORY_FOLDER, os.path.basename(categories_filename))
    )
    frame = count_categories(results, kept_groups, kept_long_names, kept_custom_names)
    frame.to_csv(output_filename, index=False)
    print(
        'Your output file has been written.  Its name is '
        + os.path.join(CATEGORY_FOLDER, os.path.basename(output_filename))
    )
    print('Both are in this folder: ' + run['output_folder'])
    run['counts'].append({
        'groups': kept_groups,
        'long_names': kept_long_names,
        'custom_names': kept_custom_names,
        'table': frame,
        'categories_file': categories_filename,
        'results_file': output_filename,
    })
    write_summary(run)
    return frame, categories_filename, output_filename


def note_error(run, error):
    """Record an error in run_summary.txt once."""
    if run is None or not run.get('output_folder'):
        return
    message = str(error)
    if message in run['errors']:
        return
    run['errors'].append(message)
    write_summary(run)


def stopwords_line(run):
    """One line describing which stopword list was used."""
    if run['stopwords_file']:
        return (
            'Stopwords file: ' + os.path.abspath(run['stopwords_file'])
            + ' (' + str(len(read_extra_stopwords(run['stopwords_file']))) + ' words)'
        )
    if run['more_words']:
        return 'Stopwords file: none (NLTK list plus the words typed in)'
    return 'Stopwords file: none (NLTK list only)'


def write_summary(run):
    """Rewrite run_summary.txt from the current run record."""
    results = run['results']
    lines = [
        'Text Category Counter - run summary',
        'Date and time: ' + run['time'].strftime('%Y-%m-%d %I:%M %p'),
        'Output folder: ' + run['output_folder'],
        '',
        'Files read:',
    ]
    for period in results:
        lines.append(
            '  ' + results[period]['file'] + ' (period ' + str(period) + '): '
            + str(results[period]['total_words']) + ' words'
        )
    if not results:
        lines.append('  none')
    if run['stop_words'] is not None:
        lines.append(stopwords_line(run))
        if run['more_words']:
            lines.append('Stopwords typed in: ' + ', '.join(run['more_words']))
    if results:
        lines.append('Top 30 words saved in: ' + os.path.join(run['output_folder'], TOP_WORDS_FOLDER))
    suggested = run.get('suggested')
    if suggested:
        lines += [
            '',
            'How the suggested groups were made:',
            '  1. Words: every word in the top ' + str(num_words) + ' of at least one file ('
            + str(suggested['words']) + ' words).',
            '  2. Shared sentences: closeness = sentences with both / square root of',
            '     (sentences with the first x sentences with the second). Each word starts',
            '     alone; the closest groups merge until none have closeness '
            + str(MIN_SHARED_SENTENCES) + ' or more',
            '     (' + str(suggested['first_groups']) + ' groups of two or more words).',
            '  3. WordNet check (Wu-Palmer similarity of the most common meaning):',
            '     an ungrouped word joins at ' + str(WORDNET_JOIN) + ' or more. A grouped word moves',
            '     when the new fit is at least ' + str(WORDNET_JOIN) + ', and either its own group has no',
            '     score or the new fit is at least ' + str(WORDNET_MOVE_GAP) + ' better.',
            '     A fit under ' + str(WORDNET_FLAG) + ' is only flagged (joined or moved: '
            + str(suggested['changed']) + '; flagged: ' + str(suggested['flagged']) + ').',
            '  4. Short name = the most-used word; long name = the two most-used words.',
            '  Result: ' + str(suggested['groups']) + ' suggested groups. They are a starting point.',
            '  The categories that were counted are listed under each Count below.',
        ]
    elif results and run.get('used_saved_categories'):
        lines += ['', 'Suggested groups: not made in this run (a saved categories file was used).']
    elif results:
        lines += ['', 'Suggested groups: not made in this run.']
    for number, count in enumerate(run['counts'], start=1):
        lines += ['', 'Count ' + str(number), 'Categories used (short name (long name): words):']
        for category, words in count['groups'].items():
            line = '  ' + category + ' (' + count['long_names'][category] + ')'
            if count['custom_names'][category] != '':
                line += ', custom name "' + count['custom_names'][category] + '"'
            lines.append(line + ': ' + ', '.join(sorted(set(words))))
        lines.append('Counts:')
        table = count['table']
        for period, category, number_found in zip(table['Period'], table['Category'], table['Count']):
            lines.append('  ' + str(period) + ', ' + category + ': ' + str(number_found))
        lines.append('Categories saved in: ' + count['categories_file'])
        lines.append('Counts saved in: ' + count['results_file'])
    if run['counts']:
        lines += [
            '',
            'How the counts are worked out:',
            '  Cleaning: split into words, drop punctuation, split on hyphens,',
            '  collapse dotted abbreviations (U.S.A. -> usa), lemmatize with part of speech',
            '  (acronyms such as POS, DoS, and IT stay as written), then remove stopwords.',
            '  Categories are built from the words in the per-file top ' + str(num_words) + ' lists.',
            '  Count = uses of the category words in that period full cleaned text,',
            '  including words that are outside that file top ' + str(num_words) + '.',
            '  Total Words = words after punctuation is removed, before stopwords are removed.',
            '  To compare periods: Count / Total Words x 1,000 = uses per 1,000 words.',
        ]
    lines.append('')
    if run.get('warnings'):
        lines.append('Warnings:')
        lines += ['  ' + warning for warning in run['warnings']]
    if run['errors']:
        lines.append('Errors:')
        lines += ['  ' + error for error in run['errors']]
    else:
        lines.append('Errors: none')

    summary_file = os.path.join(run['output_folder'], 'run_summary.txt')
    # Plain ASCII so the summary opens anywhere; other characters become "?".
    with open(summary_file, 'w', encoding='ascii', errors='replace') as handle:
        handle.write('\n'.join(lines) + '\n')
    return summary_file


def words_from_answer(answer, words):
    """Turn a comma-separated answer into category words.

    A token that is itself in the word list is that word. Any other number
    from 1 to the list length picks the word at that position.
    """
    chosen = []
    for item in answer.split(','):
        item = item.strip().lower()
        if item == '':
            continue
        if item in words:
            picked = item
        elif item.isdigit() and 1 <= int(item) <= len(words):
            picked = words[int(item) - 1]
        else:
            picked = item
        if picked not in chosen:
            chosen.append(picked)
    return chosen


def type_categories(words_table, groups=None):
    """Build categories by typing. Enter an empty name when finished.

    "clear" removes every category. An existing name with no words removes
    that category. Returns short name -> words.
    """
    groups = dict(groups or {})
    words = list(words_table['Word'])
    print('Words that made the top 30 in at least one file (with total count):')
    for index, (word, count) in enumerate(zip(words_table['Word'], words_table['Count'])):
        print(str(index + 1) + '. ' + word + ' (' + str(count) + ')')
    for category, chosen in groups.items():
        print('Starting category:', category, '=', ', '.join(chosen))
    while True:
        category = input('Category name (Enter when done, "clear" to remove all) : ').strip()
        if category == '':
            break
        if category.lower() == 'clear':
            groups = {}
            print('All categories removed.')
            continue
        answer = input('Words for ' + category + ' (numbers or words, separated by commas; Enter to remove it) : ')
        chosen = words_from_answer(answer, words)
        if chosen:
            groups[category] = chosen
            print(category, '=', ', '.join(chosen))
        else:
            groups.pop(category, None)
            print(category, 'removed.')
    return groups


def run_window():
    """Open the Tkinter window and run the same steps as the command line."""
    import tkinter as tk
    from tkinter import filedialog
    from tkinter import messagebox

    state = {
        'text_files': [],
        'stopwords_file': '',
        'results': None,
        'run': None,
        'groups': {},
        'long_names': {},
        'custom_names': {},
        'current': None,
        'words': [],
    }

    root = tk.Tk()
    root.title('Text Category Counter')

    status = tk.StringVar(value='Start by choosing your .txt files.')
    files_label = tk.StringVar(value='No text files chosen')
    stop_label = tk.StringVar(value='')
    typed_words = tk.StringVar(value='')
    new_category = tk.StringVar(value='')
    custom_name_text = tk.StringVar(value='')
    save_name = tk.StringVar(value='categories.csv')

    if os.path.isfile(DEFAULT_STOPWORDS_FILE):
        state['stopwords_file'] = os.path.abspath(DEFAULT_STOPWORDS_FILE)
        stop_label.set(DEFAULT_STOPWORDS_FILE + ' (from the working folder)')
    else:
        stop_label.set('Stopwords file: none (NLTK list only)')

    def set_status(message):
        status.set(message)
        root.update_idletasks()

    def choose_text_files():
        paths = filedialog.askopenfilenames(title='Choose the text files', filetypes=[('Text files', '*.txt')])
        if paths:
            state['text_files'] = sorted(paths)
            files_label.set(str(len(paths)) + ' text files chosen')
            set_status('Text files chosen. Next: Find top words.')

    def choose_stopwords_file():
        path = filedialog.askopenfilename(
            title='Choose a stopwords file',
            filetypes=[('Text files', '*.txt'), ('All files', '*.*')],
        )
        if path:
            state['stopwords_file'] = path
            stop_label.set(os.path.basename(path))
            set_status('Stopwords file chosen.')

    def find_top_words():
        if not state['text_files']:
            messagebox.showinfo('Find top words', 'Choose your .txt files first.')
            return
        try:
            download_nltk_data()
            more_words = [w.strip() for w in typed_words.get().split(',') if w.strip() != '']
            set_status('Reading and cleaning ' + str(len(state['text_files'])) + ' files...')
            state['results'], state['run'] = read_and_clean_files(
                state['text_files'],
                state['stopwords_file'] or DEFAULT_STOPWORDS_FILE,
                more_words,
            )
            if not state['run']['stopwords_file']:
                stop_label.set(stopwords_line(state['run']))
            elif not state['stopwords_file']:
                stop_label.set(DEFAULT_STOPWORDS_FILE + ' (from the working folder)')
            if not state['groups']:
                set_status('Working out suggested groups...')
                state['groups'], state['long_names'] = suggest_categories(
                    state['results'], state['run']['stop_words'], state['run']
                )
                state['current'] = next(iter(state['groups']), None)
        except (OSError, ValueError) as error:
            messagebox.showerror('Find top words', str(error))
            set_status('Stopped: ' + str(error))
            return
        state['words'] = list(top_words_union(state['results'])['Word'])
        for words in state['groups'].values():
            for word in words:
                if word not in state['words']:
                    state['words'].append(word)
        refresh_word_list()
        refresh_category_list()
        set_status(
            'Top 30 words saved for ' + str(len(state['results'])) + ' files in '
            + os.path.join(state['run']['output_folder'], TOP_WORDS_FOLDER)
            + '. Check the suggested groups: rename, edit or remove them, or add your own.'
        )

    def word_label(word):
        total = sum(state['results'][period]['word_freq'][word] for period in state['results'])
        return word + ' (' + str(total) + ')'

    def refresh_word_list():
        word_list.delete(0, tk.END)
        for word in state['words']:
            word_list.insert(tk.END, word_label(word))
        show_category_words()

    def refresh_category_list():
        category_list.delete(0, tk.END)
        for index, category in enumerate(state['groups']):
            line = category + ' (' + long_name_of(category, state['long_names']) + ')'
            custom = custom_name_of(category, state['custom_names'])
            if custom != '':
                line += ' - ' + custom
            category_list.insert(tk.END, line)
            if category == state['current']:
                category_list.selection_set(index)
        show_category_words()

    def show_category_words():
        word_list.selection_clear(0, tk.END)
        for word in state['groups'].get(state['current'], []):
            if word in state['words']:
                word_list.selection_set(state['words'].index(word))

    def add_category():
        name = new_category.get().strip()
        if name == '':
            messagebox.showinfo('Add category', 'Type a name for the new category first.')
            return
        state['groups'].setdefault(name, [])
        state['current'] = name
        new_category.set('')
        custom_name_text.set(custom_name_of(name, state['custom_names']))
        refresh_category_list()
        set_status('Category "' + name + '" added. Click its words in the list.')

    def set_custom_name():
        if state['current'] not in state['groups']:
            messagebox.showinfo('Set custom name', 'Pick a category first.')
            return
        custom = custom_name_text.get().strip()
        if custom == '':
            state['custom_names'].pop(state['current'], None)
            set_status('Custom name removed from ' + state['current'] + '.')
        else:
            state['custom_names'][state['current']] = custom
            set_status(state['current'] + ' now has the custom name "' + custom + '".')
        refresh_category_list()

    def remove_category():
        if state['current'] in state['groups']:
            del state['groups'][state['current']]
            state['long_names'].pop(state['current'], None)
            state['custom_names'].pop(state['current'], None)
            state['current'] = None
            custom_name_text.set('')
            refresh_category_list()

    def on_category_picked(_event):
        picked = category_list.curselection()
        if picked:
            state['current'] = list(state['groups'])[picked[0]]
            custom_name_text.set(custom_name_of(state['current'], state['custom_names']))
            show_category_words()

    def on_words_picked(_event):
        if state['current'] is None or state['results'] is None:
            return
        state['groups'][state['current']] = [state['words'][i] for i in word_list.curselection()]
        set_status(state['current'] + ': ' + ', '.join(state['groups'][state['current']]))

    def load_categories():
        path = filedialog.askopenfilename(title='Choose a saved categories file', filetypes=[('CSV files', '*.csv')])
        if not path:
            return
        try:
            state['groups'], state['long_names'], state['custom_names'] = read_categories(path)
        except (OSError, ValueError) as error:
            messagebox.showerror('Load categories', str(error))
            return
        state['current'] = next(iter(state['groups']), None)
        if state['results'] is not None:
            for words in state['groups'].values():
                for word in words:
                    if word not in state['words']:
                        state['words'].append(word)
            refresh_word_list()
        refresh_category_list()
        set_status('Loaded ' + str(len(state['groups'])) + ' categories from ' + os.path.basename(path) + '.')

    def count_and_save_clicked():
        if state['results'] is None:
            messagebox.showinfo('Count', 'Click "Find top words" first.')
            return
        try:
            _frame, categories_filename, output_filename = count_and_save(
                state['results'],
                state['groups'],
                state['long_names'],
                state['custom_names'],
                save_name.get().strip(),
                state['run'],
            )
        except (OSError, ValueError) as error:
            note_error(state['run'], error)
            messagebox.showerror('Count', str(error))
            set_status('Stopped: ' + str(error))
            return
        set_status(
            'Done: saved ' + os.path.basename(categories_filename) + ' and '
            + os.path.basename(output_filename) + ' in ' + os.path.dirname(output_filename)
            + '. A summary is in run_summary.txt.'
        )

    pad = {'padx': 10, 'pady': 4, 'sticky': 'w'}
    tk.Button(root, text='Choose text files...', width=30, command=choose_text_files).grid(row=0, column=0, **pad)
    tk.Label(root, textvariable=files_label).grid(row=0, column=1, **pad)
    tk.Button(root, text='Choose stopwords file...', width=30, command=choose_stopwords_file).grid(
        row=1, column=0, **pad
    )
    tk.Label(root, textvariable=stop_label).grid(row=1, column=1, **pad)
    tk.Label(root, text='More stopwords (optional, separated by commas):').grid(row=2, column=0, **pad)
    tk.Entry(root, textvariable=typed_words, width=40).grid(row=2, column=1, **pad)
    tk.Button(root, text='Find top words', width=30, command=find_top_words).grid(row=3, column=0, **pad)

    left = tk.Frame(root)
    left.grid(row=4, column=0, padx=10, pady=4, sticky='nw')
    tk.Label(left, text='Categories').pack(anchor='w')
    category_list = tk.Listbox(left, height=12, width=40, exportselection=False)
    category_list.pack(anchor='w')
    category_list.bind('<<ListboxSelect>>', on_category_picked)
    tk.Label(left, text='Custom name:').pack(anchor='w')
    tk.Entry(left, textvariable=custom_name_text, width=30).pack(anchor='w', pady=2)
    tk.Button(left, text='Set custom name', width=26, command=set_custom_name).pack(anchor='w')
    tk.Label(left, text='New category name:').pack(anchor='w', pady=(8, 0))
    tk.Entry(left, textvariable=new_category, width=30).pack(anchor='w', pady=2)
    tk.Button(left, text='Add category', width=26, command=add_category).pack(anchor='w')
    tk.Button(left, text='Remove category', width=26, command=remove_category).pack(anchor='w')
    tk.Button(left, text='Load categories file...', width=26, command=load_categories).pack(anchor='w', pady=(8, 0))

    right = tk.Frame(root)
    right.grid(row=4, column=1, padx=10, pady=4, sticky='nw')
    tk.Label(right, text='Words for the chosen category (top 30 in any file, with total count):').pack(anchor='w')
    word_list = tk.Listbox(right, height=18, width=40, selectmode=tk.MULTIPLE, exportselection=False)
    scroll = tk.Scrollbar(right, command=word_list.yview)
    word_list.config(yscrollcommand=scroll.set)
    word_list.pack(side='left')
    scroll.pack(side='left', fill='y')
    word_list.bind('<<ListboxSelect>>', on_words_picked)

    count_row = tk.Frame(root)
    count_row.grid(row=5, column=0, columnspan=2, padx=10, pady=4, sticky='w')
    tk.Label(count_row, text='Save categories as:').pack(side='left')
    tk.Entry(count_row, textvariable=save_name, width=30).pack(side='left', padx=4)
    tk.Button(count_row, text='Count and save', width=20, command=count_and_save_clicked).pack(side='left', padx=4)
    tk.Label(root, textvariable=status, anchor='w', relief='sunken', width=100).grid(
        row=6, column=0, columnspan=2, padx=10, pady=8
    )

    def get_ready():
        old = status.get()
        set_status('Getting NLTK data ready...')
        download_nltk_data()
        set_status(old)

    root.after(100, get_ready)
    root.mainloop()


def type_custom_names(groups, custom_names=None):
    """Ask for custom names. Enter an empty category name when finished."""
    custom_names = dict(custom_names or {})
    print('Custom names (optional). Categories:', ', '.join(groups))
    while True:
        category = input('Category to give a custom name (Enter when done) : ').strip()
        if category == '':
            break
        if category not in groups:
            print('There is no category named', category)
            continue
        custom = input('Custom name for ' + category + ' (Enter for none) : ').strip()
        if custom != '':
            custom_names[category] = custom
        else:
            custom_names.pop(category, None)
    return custom_names


def run_command_line(argv):
    """Run from file names. Returns 0 on success and 1 when the run stops."""
    parser = argparse.ArgumentParser(description='Text Category Counter (run with no options to open the window).')
    parser.add_argument('files', nargs='+', help='the .txt files, one per period')
    parser.add_argument(
        '-c', '--categories',
        help='a saved categories CSV (Category, Long Name, Custom Name, Word; '
             'Long Name and Custom Name are optional) for a repeat run',
    )
    parser.add_argument('-o', '--output', help='results file name (default: named from the categories file)')
    parser.add_argument(
        '--save-categories', default='categories.csv',
        help='file name for typed categories (default categories.csv)',
    )
    parser.add_argument(
        '--stopwords', default=DEFAULT_STOPWORDS_FILE,
        help='stopwords file, one word per line (default: extra_stopwords.txt in the working folder; '
             'if the file is not there, none is used)',
    )
    args = parser.parse_args(argv)

    run = None
    try:
        download_nltk_data()
        groups, long_names, custom_names = None, None, None
        if args.categories:
            groups, long_names, custom_names = read_categories(args.categories)
        results, run = read_and_clean_files(args.files, args.stopwords)
        if groups is None:
            groups, long_names = suggest_categories(results, run['stop_words'], run)
            words_table = top_words_union(results)
            groups = type_categories(words_table, groups)
            long_names = long_names_from_words(groups, words_table)
            custom_names = type_custom_names(groups)
            categories_name = args.save_categories
        else:
            run['used_saved_categories'] = True
            categories_name = args.categories
        frame, _categories_filename, _output_filename = count_and_save(
            results, groups, long_names, custom_names, categories_name, run, args.output
        )
        print(frame)
    except (OSError, ValueError) as error:
        note_error(run, error)
        print('Stopped: ' + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    if len(sys.argv) > 1:
        sys.exit(run_command_line(sys.argv[1:]))
    else:
        run_window()
