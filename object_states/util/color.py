


class bc:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def color(*a, c=bc.FAIL):
    return f'{c}{" ".join(map(str, a))}{bc.END}'
def red(*a): return color(*a, c=bc.FAIL)
def green(*a): return color(*a, c=bc.GREEN)
def blue(*a): return color(*a, c=bc.BLUE)
def yellow(*a): return color(*a, c=bc.WARNING)