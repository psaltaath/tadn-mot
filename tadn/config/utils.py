import json
from typing import Callable

from ..data.mot_challenge import MOTChallengeCategorySet


def json_dumps_for_callables(v, *, default, **kwargs):
    """Utility function to output callables to json"""

    def _default(o):
        if isinstance(o, Callable):
            return o.__repr__().split(".")[0].split("function ")[-1]
        if isinstance(o, MOTChallengeCategorySet):
            return o.name
        else:
            return default(o)

    return json.dumps(v, default=_default, **kwargs)
