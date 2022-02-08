import math


class WordFilter:
    def __init__(self, count):
        self.count = count

    def filter(self, span):
        return len(span.split(" ")) >= self.count

class CharFilter:
    def __init__(self, count):
        self.count = count

    def filter(self, span):
        return len(span) >= self.count

class EntityFilter:
    def __init__(self, count):
        self.count = count

    def filter(self, span):
        raise NotImplementedError()

class NoFilter:
    def filter(self, _span):
        return True



def get_filter(name, args):
    if name is None or name == "non":
        return NoFilter()
    elif name == "word":
        return WordFilter(args.filter_count)
    elif name == "char":
        return CharFilter(args.filter_count)
    elif name in {"ner", "entity"}:
        return EntityFilter(args.filter_count)
    else:
        raise Exception(f"Unknown filter {name}")