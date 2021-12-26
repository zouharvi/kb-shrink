import math

class FixedWidthSplitter:
    def __init__(self, token_count):
        self.token_count = token_count
    
    def split(self, line):
        line = line.rstrip("\n").split(" ")
        for i in range(0, math.ceil(len(line) / self.token_count)):
            if len(line[i * self.token_count:(i + 1) * self.token_count]) == 0:
                raise Exception("Empty chunk")
            yield " ".join(line[i * self.token_count:(i + 1) * self.token_count])


_SPLITTERS = {
    "fixed": FixedWidthSplitter
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