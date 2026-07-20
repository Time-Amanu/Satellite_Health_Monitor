import pandas as pd
def prepare_features(df):
    mu_min = 398600.4418 * 60**2
    Re = 6378.0
    df['Semi-major Axis'] = (mu_min / (df['Mean Motion']**2))**(1/3)
    df['Altitude'] = df['Semi-major Axis'] - Re
    df['at risk'] = (df['bstar'] > 0.000463).astype(int)
    return df