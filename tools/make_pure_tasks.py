"""Generate the frozen task set for Pure Mycelia web-skill experiments.

Reference answers come from plain Python functions written here, not from
the documentation. Even task ids are train (used only to learn priors), odd
ids are test (scored once). Some tasks cannot be solved by composing str
methods; the correct response there is to abstain.
"""
import json, random
from pathlib import Path

rng = random.Random(2026)
WORDS = ['alice', 'bob', 'mycelia', 'spore', 'hypha', 'river', 'stone', 'green', 'apple', 'night']


def sentence():
    ws = [rng.choice([w, w.upper(), w.title()]) for w in rng.sample(WORDS, rng.randint(2, 4))]
    return rng.choice(['', ' ', '  ']) + rng.choice([' ', '  ']).join(ws) + rng.choice(['', ' ', '   '])


def csv():
    return ','.join(rng.sample(WORDS, rng.randint(2, 4)))


def path():
    return '/'.join(rng.sample(WORDS, rng.randint(2, 3))) + rng.choice(['.txt', '.py', '.csv'])


def num():
    return str(rng.randint(1, 9999))


def slug():
    return '-'.join(rng.sample(WORDS, rng.randint(2, 3)))


TASKS = [
    ('make every letter uppercase', sentence, lambda s: s.upper()),
    ('make every letter lowercase', sentence, lambda s: s.lower()),
    ('remove the whitespace at both ends', sentence, lambda s: s.strip()),
    ('make each word start with a capital letter and trim the ends', sentence, lambda s: s.strip().title()),
    ('split the comma separated text into a list', csv, lambda s: s.split(',')),
    ('swap the case of every letter', sentence, lambda s: s.swapcase()),
    ('replace the commas with spaces', csv, lambda s: s.replace(',', ' ')),
    ('replace the dashes with underscores', slug, lambda s: s.replace('-', '_')),
    ('collapse repeated spaces into single spaces and trim', sentence, lambda s: ' '.join(s.split())),
    ('pad the number with zeros to a width of eight', num, lambda s: s.zfill(8)),
    ('check whether the text is made only of digits', lambda: rng.choice([num(), csv()]), lambda s: s.isdigit()),
    ('count how many commas the text contains', csv, lambda s: s.count(',')),
    ('turn the slug into lowercase words separated by spaces', slug, lambda s: s.replace('-', ' ')),
    ('capitalize the first letter and lowercase the rest after trimming', sentence, lambda s: s.strip().capitalize()),
    ('split the path into its folder parts at the slash', path, lambda s: s.split('/')),
    ('remove whitespace at the right end only', sentence, lambda s: s.rstrip()),
    ('remove whitespace at the left end only', sentence, lambda s: s.lstrip()),
    ('join the comma separated words with dashes', csv, lambda s: '-'.join(s.split(','))),
    ('find the position of the first slash', path, lambda s: s.find('/')),
    ('check whether the path ends with a txt extension', path, lambda s: s.endswith('.txt')),
    ('make the words uppercase and separate them by commas', sentence, lambda s: ','.join(s.upper().split())),
    ('case insensitive form of the text without surrounding spaces', sentence, lambda s: s.strip().casefold()),
    ('split the trimmed sentence into words', sentence, lambda s: s.split()),
    ('turn the comma list into uppercase words separated by spaces', csv, lambda s: s.upper().replace(',', ' ')),
    ('right align the number in a field of six with zeros', num, lambda s: s.zfill(6)),
    ('replace slashes with dots in the path', path, lambda s: s.replace('/', '.')),
    # Not solvable by composing str methods: the right answer is to abstain.
    ('reverse the order of the characters', sentence, lambda s: s[::-1]),
    ('sort the comma separated words alphabetically', csv, lambda s: sorted(s.split(','))),
    ('keep only the first letter of each word', sentence, lambda s: ''.join(w[0] for w in s.split())),
    ('remove duplicate words', lambda: ','.join(rng.choices(WORDS[:4], k=rng.randint(4, 6))), lambda s: list(dict.fromkeys(s.split(',')))),
]

tasks = []
for i, (goal, gen, fn) in enumerate(TASKS):
    xs = []
    while len(xs) < 9:
        x = gen()
        if x not in xs:
            xs.append(x)
    tasks.append({'id': i, 'split': 'train' if i % 2 == 0 else 'test', 'goal': goal,
                  'examples': [[x, fn(x)] for x in xs[:3]], 'hidden': [[x, fn(x)] for x in xs[3:]],
                  'solvable_by_str_methods': i < 26})
Path('datasets').mkdir(exist_ok=True)
Path('datasets/pure_tasks.json').write_text(json.dumps(tasks, indent=1))
print(len(tasks), 'tasks', sum(t['split'] == 'test' for t in tasks), 'test')
