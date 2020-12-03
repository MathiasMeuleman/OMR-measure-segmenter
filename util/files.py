from pathlib import Path


# Gives a sorted list of pages. As long as the pages are numbered incrementally separated with a "-", this will work fine.
def get_sorted_page_paths(page_path):
    paths = [p for p in Path(page_path).iterdir() if p.is_file()]
    return [str(p.resolve()) for p in sorted(paths, key=lambda p: int(str(p.stem).split("-")[1]))]
