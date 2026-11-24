"""run/regions.py: the design point's regions, and the elements and values of a word in one."""

import json
import struct

from snax_forge.run import Region


def word(*values: int, fmt: str = "<q") -> bytes:
    return b"".join(struct.pack(fmt, v) for v in values)


def test_round_trip():
    r = Region("M", "l1", 0x200, (2, 3), (32, 8), "int64")
    assert Region.from_dict(json.loads(json.dumps(r.to_dict()))) == r
