import re
import string


def clean_text(text):
    """Cleans the input string ``text``.

    Cleans a Reuters-specific new headline following the procedure explained in
    the report of the thesis. In short:

        1. Removes the introductory text of many Reuter headlines, if present.
        2. Removes possible non-informative tags at the end of the headline.
        3. Transforms the text to lowercase.
        4. Removes points that may be present.
        5. Removes numbers that may be present.
        6. Removes commas that may be present.
        7. Replaces special characters with spaces.
        8. Removes non-single spaces.
        9. Removes possible spaces at the beginning and the end of the headline
           (inserted by previous steps).

    Args:
        text (str): Reuters headline.

    Returns:
        str: sanitized Reuters headline.
    """
    text_sanitized = re.sub(pattern=r'^[A-Z0-9-\s]*-', repl='', string=text)
    text_sanitized = re.sub(
        pattern=r'\s-[\s\w]*$', repl='', string=text_sanitized
    )
    text_sanitized = text_sanitized.lower()
    text_sanitized = re.sub(pattern=r'[\.]+', repl='', string=text_sanitized)
    text_sanitized = re.sub(pattern=r'\w*\d\w*', repl='',
                            string=text_sanitized)
    text_sanitized = re.sub(pattern=r'[\']', repl='', string=text_sanitized)
    text_sanitized = re.sub(
        pattern=r'[^a-z0-9\s]', repl=' ', string=text_sanitized
    )
    text_sanitized = re.sub(pattern=r'\s{2,}', repl=' ', string=text_sanitized)
    return re.sub(pattern=r'^\s|\s$', repl='', string=text_sanitized)


def tokenize_text(text, tokenizer):
    """Tokenizes a the string ``text`` using ``tokenizer``.

    Args:
        text (str): text to tokenize.
        tokenizer (callable): tokenizer to use.

    Returns:
        list: list containing the tokens of the input text.
    """
    return tokenizer(text.translate(str.maketrans('', '', string.punctuation)))


def clean_and_tokenize(text, tokenizer):
    """Cleans and tokenizes the string ``text`` using ``tokenizer``.

    Args:
        text (str): text to tokenize.
        tokenizer (callable): tokenizer to use.

    Returns:
        list: list containing the tokens of the input text.
    """
    return tokenize_text(clean_text(text), tokenizer)
