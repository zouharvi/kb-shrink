import math


class FixedWidthSplitter:
    def __init__(self, width):
        self.width = width

    def split(self, line):
        line = line.rstrip("\n").split(" ")
        for i in range(0, math.ceil(len(line) / self.width)):
            yield " ".join(line[i * self.width:(i + 1) * self.width])

class OverlapSplitter:
    def __init__(self, width, ):
        self.width = width

    def split(self, line):
        line = line.rstrip("\n").split(" ")
        for i in range(0, math.ceil(len(line) / self.width)):
            yield " ".join(line[i * self.width:(i + 1) * self.width])

_SPLITTERS = {
    "fixed": FixedWidthSplitter,
    "overlap": OverlapSplitter,
}

def get_splitter(name, args):
    if name in _SPLITTERS:
        return _SPLITTERS[name](args.splitter_fixed_width)
    else:
        raise Exception(f"Unknown splitter {name}")


def split_paragraph_list(text_list, splitter):
    return [
        span for span_list
        in [
            splitter.split(text) for text in text_list
            if not text.startswith("BULLET::::") and not text.startswith("Section::::")
        ]
        for span in span_list
    ]
