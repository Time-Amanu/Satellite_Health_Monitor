
import pandas as pd
import numpy as np
from sgp4.api import Satrec
import requests
from xgboost import XGBClassifier

def predict_satellite_decay_risk(sat_name, tle_lines, pipeline):
    try:
        name_idx = tle_lines.index(sat_name)
        line1 = tle_lines[name_idx + 1]
        line2 = tle_lines[name_idx + 2]
        s = Satrec.twoline2rv(line1, line2)
        mu_min = 398600.4418 * 60**2
        Re = 6378.0
        ecc, inc, mm, epoch = s.ecco, s.inclo, s.no_kozai, s.jdsatepoch
        sma = (mu_min / (mm**2))**(1/3)
        alt = sma - Re
        input_df = pd.DataFrame([{'Eccentricity': ecc, 'Inclination': inc, 'Mean Motion': mm, 'Epoch': epoch, 'Altitude': alt}])
        return pipeline.predict_proba(input_df)[0, 1]
    except Exception as e:
        return None
