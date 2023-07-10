import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid

DATA_FILE_URL = "https://data.gov.lt/dataset/201/download/2853/05.csv"

MOKINIAI_PAGAL_SAVIVALDYBES = "Bendrojo ugdymo mokiniai pagal savivaldybes.xlsx"


def read_data() -> pd.DataFrame:
    data = pd.read_excel(MOKINIAI_PAGAL_SAVIVALDYBES)
    data.drop(index=data.index[-1], axis=0, inplace=True)
    data["Metų pabaiga"] = data["BU Mokslo metai"].str[-4:]
    return data


def prepare_school_data(data):
    X, y = (
        data[["BU Institucijos savivaldybė", "Metų pabaiga"]],
        data["BU Mokinių skaičius"],
    )
    return train_test_split(X, y, test_size=0.5)


def create_regression_model():
    cat_features = ["BU Institucijos savivaldybė"]
    cat_transformer = make_pipeline(OneHotEncoder(handle_unknown="ignore"))
    num_features = ["Metų pabaiga"]
    num_transformer = make_pipeline(SimpleImputer(strategy="most_frequent"))
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features),
        ]
    )
    model = make_pipeline(
        preprocessor, PolynomialFeatures(degree=2), LinearRegression()
    )
    return model


def predict_students_count(model, municipality: str, year: int):
    municipality_dict = {"Municipality": "Kauno m. sav."}
    data = {
        "BU Institucijos savivaldybė": municipality_dict[municipality],
        "Metų pabaiga": year,
    }
    z = pd.DataFrame([data])
    result = model.predict(z)
    return result


def create_testing_scenarios(study_year=[2016, 2030]):
    study_year_start, study_year_end = study_year
    param_grid = {
        "Metų pabaiga": range(study_year_start, study_year_end, 1),
        "BU Institucijos savivaldybė": ["Kauno m. sav.", "Marijampolės sav."],
    }
    return pd.DataFrame(ParameterGrid(param_grid))


if __name__ == "__main__":
    school_data = read_data()
    X_train, X_test, y_train, y_test = prepare_school_data(school_data)
    model = create_regression_model()
    print(model)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    score = model.score(X_test, y_test)
    print(score)
    scenarios = create_testing_scenarios()
    print(scenarios)
    predictions = model.predict(scenarios)
    r = scenarios.assign(predictions=predictions)
    print(r)
