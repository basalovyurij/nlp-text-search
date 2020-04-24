import html
import re


def normalize_phrase(s):
    # remove special characters
    r = html.unescape(s).lower().replace('&', '')
    # remove end numbers
    r = re.sub(r'\s{5,}\d+(\.\d+)?', '', r)
    # remove whitespaces
    r = re.sub(r'\s+', ' ', r)
    return r
