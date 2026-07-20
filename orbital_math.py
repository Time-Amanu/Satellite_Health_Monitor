import numpy as np
import pandas as pd
from sgp4.api import Satrec
def parse_satellites(lines):
    satellites = []
    for i in range(0, len(lines), 3):
        if i + 2 < len(lines):
            try:
                s = Satrec.twoline2rv(lines[i+1], lines[i+2])
                satellites.append({'name': lines[i], 'satellite': s})
            except: pass
    return satellites