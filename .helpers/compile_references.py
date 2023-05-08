import argparse
import os
import re

LINK_REGEX = re.compile(r'\[([^[]*?)\]\((http.*?)\)', re.DOTALL)
WHITESPACE_REGEX = re.compile(r'\s+')
IGNORE_SUFFIXES = [
    '.png',
    '.pdf',
    '.pyc',
    '.jpeg',
    '.jpg',
    '.gif',
    '-checkpoint.ipynb',
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'result_filename',
        nargs='?',
        default='references.md',
        help='Where to save the results.',
    )
    parser.add_argument(
        'top_directory',
        nargs='?',
        default='.',
        help='Where to start searching from.',
    )
    return parser.parse_args()


def collect_references(top_directory, result_file):
    references = {}

    for (dirpath, dirnames, filenames) in os.walk(top_directory):
        if is_ignored_directory(dirpath):
            continue

        for filename in filenames:
            if is_ignored_file(filename, result_file):
                continue

            filepath = os.path.join(dirpath, filename)

            matches = find_matches(filepath)
            if not matches:
                continue

            references.setdefault(filepath, {})

            for (description, link) in matches:
                references[filepath][link] = truncate_whitespace(
                    description,
                )

    return references


def is_ignored_directory(dirpath):
    # Skip Git directory
    if dirpath.startswith(os.path.join('.', '.git')):
        return True
    return False


def is_ignored_file(filename, result_file):
    if filename == os.path.basename(result_file):
        return True
    for suffix in IGNORE_SUFFIXES:
        if filename.endswith(suffix):
            return True
    return False


def find_matches(filepath):
    with open(filepath, 'r') as f:
        try:
            contents = f.read()
        except UnicodeDecodeError:
            print(f'Warning: failed reading {filepath}.')
            return None

        return LINK_REGEX.findall(contents)


def truncate_whitespace(text):
    return re.sub(WHITESPACE_REGEX, ' ', text)


def write_references(result_filename, references):
    first_write = True

    with open(result_filename, 'w') as result_file:
        for (filepath, links) in references.items():
            if not first_write:
                result_file.write('\n')

            result_file.write(f'- [{filepath}]({filepath}):\n')
            for (link, description) in links.items():
                result_file.write(f'    - [{description}]({link})\n')

            first_write = False


def main():
    args = parse_args()
    references = collect_references(args.top_directory, args.result_filename)
    write_references(args.result_filename, references)


if __name__ == '__main__':
    main()
