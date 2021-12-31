import math


class FixedWidthSplitter:
    def __init__(self, width):
        self.width = width

    def split(self, line):
        line = line.rstrip("\n").split(" ")
        for i in range(0, math.ceil(len(line) / self.width)):
            pos_left = i * self.width
            pos_right = (i + 1) * self.width
            yield " ".join(line[pos_left:pos_right])


class FixedOverlapSplitter:
    def __init__(self, width, overlap):
        self.width = width
        self.overlap = overlap

    def split(self, line):
        line = line.rstrip("\n").split(" ")
        for i in range(0, math.ceil(len(line) / self.width)):
            pos_left = i * self.width
            pos_right = (i + 1) * self.width + self.overlap
            yield " ".join(line[pos_left:pos_right])


class SentenceSplitter:
    def __init__(self, width, overlap):
        self.width = width
        self.overlap = overlap

    def split(self, line):
        # naive sentence splitting
        line = line.rstrip("\n").split(". ")
        for i in range(0, math.ceil(len(line) / self.width)):
            pos_left = i * self.width
            pos_right = (i + 1) * self.width + self.overlap
            # lengths around 20 but some are 1..5 (todo: filter those)
            yield ". ".join(line[pos_left:pos_right])


def get_splitter(name, args):
    if name == "fixed":
        return FixedWidthSplitter(args.splitter_width)
    elif name == "overlap":
        return FixedOverlapSplitter(args.splitter_width, args.splitter_overlap)
    elif name in {"sent", "sentence"}:
        return SentenceSplitter(args.splitter_width, args.splitter_overlap)
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
