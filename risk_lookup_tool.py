import pandas as pd
from sgp4.api import Satrec

def predict_satellite_decay_risk(sat_name, tle_lines, pipeline):
    try:
        name_idx = tle_lines.index(sat_name)
        line1 = tle_lines[name_idx + 1]
        line2 = tle_lines[name_idx + 2]
        s = Satrec.twoline2rv(line1, line2)
        mu_min, Re = 398600.4418 * 60**2, 6378.0
        sma = (mu_min / (s.no_kozai**2))**(1/3)
        input_df = pd.DataFrame([{
            'Eccentricity': s.ecco,
            'Inclination': s.inclo,
            'Mean Motion': s.no_kozai,
            'Epoch': s.jdsatepoch,
            'Altitude': sma - Re
        }])
        return pipeline.predict_proba(input_df)[0, 1]
    except: return None