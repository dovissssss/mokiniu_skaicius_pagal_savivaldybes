import pandas as pd
from config import MOKINIAI_PAGAL_SAVIVALDYBES


def read_data() -> pd.DataFrame:
    data = pd.read_excel(MOKINIAI_PAGAL_SAVIVALDYBES)
    data.drop(index=data.index[-1], axis=0, inplace=True)
    data["Metų pabaiga"] = data["BU Mokslo metai"].str[-4:]
    return data
