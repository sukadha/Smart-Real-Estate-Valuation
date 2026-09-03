# ============================================================
# IMPORTS
# ============================================================

from flask import Flask, request, jsonify
from flask_cors import CORS

import pandas as pd
import numpy as np
import joblib
import os
import re

from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# Gemini
from google import genai


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

CORS(app)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv(
    os.path.join(
        BASE_DIR,
        ".env"
    )
)


# ============================================================
# GEMINI API
# ============================================================

GEMINI_API_KEY = os.getenv(
    "AQ.Ab8RN6J2vaIuw4ymPZhC5h4776PbHJuVS0-2LYcCRWFWMbpjkA"
)

gemini_client = None

if GEMINI_API_KEY:

    try:

        gemini_client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        print("Gemini AI connected successfully.")

    except Exception as e:

        print(
            "Gemini connection error:",
            str(e)
        )

else:

    print(
        "Warning: GEMINI_API_KEY not found."
    )


# ============================================================
# FILE PATHS
# ============================================================

DATASET_PATH = os.path.join(
    BASE_DIR,
    "Housing.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model.pkl"
)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "parking",
    "prefarea",
    "furnishingstatus"
]


TARGET = "price"


# ============================================================
# MAIN FRONTEND FEATURES
# ============================================================

MAIN_FEATURES = [
    "area",
    "bedrooms",
    "bathrooms",
    "mainroad",
    "basement"
]


# ============================================================
# BINARY FEATURES
# ============================================================

BINARY_FEATURES = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea"
]


# ============================================================
# FURNISHING MAPPING
# ============================================================

FURNISHING_MAPPING = {
    "unfurnished": 0,
    "semi-furnished": 1,
    "furnished": 2
}


# ============================================================
# LOAD AND PREPROCESS DATA
# ============================================================

def load_data():

    if not os.path.exists(DATASET_PATH):

        raise FileNotFoundError(
            f"Housing.csv not found at: {DATASET_PATH}"
        )

    df = pd.read_csv(
        DATASET_PATH
    )

    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    # --------------------------------------------------------
    # Binary columns
    # --------------------------------------------------------

    for col in BINARY_FEATURES:

        if col in df.columns:

            df[col] = (
                df[col]
                .astype(str)
                .str.lower()
                .str.strip()
                .map({
                    "yes": 1,
                    "no": 0,
                    "1": 1,
                    "0": 0
                })
            )

    # --------------------------------------------------------
    # Furnishing status
    # --------------------------------------------------------

    if "furnishingstatus" in df.columns:

        df["furnishingstatus"] = (
            df["furnishingstatus"]
            .astype(str)
            .str.lower()
            .str.strip()
            .map(
                FURNISHING_MAPPING
            )
        )

    # --------------------------------------------------------
    # Convert numeric columns
    # --------------------------------------------------------

    numeric_columns = [
        "price",
        "area",
        "bedrooms",
        "bathrooms",
        "stories",
        "parking"
    ]

    for col in numeric_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    # --------------------------------------------------------
    # Remove missing values
    # --------------------------------------------------------

    df = df.dropna(
        subset=FEATURES + [TARGET]
    ).copy()

    return df


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model():

    print("\nLoading dataset...")

    df = load_data()

    print(
        "Dataset shape:",
        df.shape
    )

    # --------------------------------------------------------
    # Input and target
    # --------------------------------------------------------

    X = df[FEATURES]

    y = df[TARGET]

    # --------------------------------------------------------
    # Train test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # --------------------------------------------------------
    # Gradient Boosting Model
    # --------------------------------------------------------

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    print("\n===================================")
    print("MODEL TRAINED")
    print("===================================")

    print(
        "R2 Score :",
        round(r2, 4)
    )

    print(
        "Score    :",
        round(r2 * 100, 2),
        "%"
    )

    print(
        "MAE      :",
        round(mae, 2)
    )

    print(
        "RMSE     :",
        round(rmse, 2)
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        "\nModel saved as:",
        MODEL_PATH
    )

    return (
        model,
        df,
        r2,
        mae,
        rmse
    )


# ============================================================
# LOAD MODEL
# ============================================================

try:

    if os.path.exists(MODEL_PATH):

        print(
            "Loading existing model..."
        )

        model = joblib.load(
            MODEL_PATH
        )

        df = load_data()

        r2 = None
        mae = None
        rmse = None

    else:

        model, df, r2, mae, rmse = train_model()

except Exception as e:

    print(
        "Existing model could not be loaded:"
    )

    print(
        str(e)
    )

    print(
        "Training a new model..."
    )

    model, df, r2, mae, rmse = train_model()


# ============================================================
# HELPER - FORMAT PRICE
# ============================================================

def format_price(value):

    try:

        return "₹{:,.0f}".format(
            float(value)
        )

    except:

        return "N/A"


# ============================================================
# HELPER - YES / NO
# ============================================================

def yes_no(value):

    try:

        return (
            "Yes"
            if int(value) == 1
            else "No"
        )

    except:

        return "Unknown"


# ============================================================
# HELPER - FURNISHING LABEL
# ============================================================

def furnishing_label(value):

    mapping = {
        0: "Unfurnished",
        1: "Semi-furnished",
        2: "Furnished"
    }

    try:

        return mapping.get(
            int(value),
            "Unknown"
        )

    except:

        return "Unknown"


# ============================================================
# DATASET STATISTICS
# ============================================================

def get_dataset_statistics():

    return {

        "total_houses":
            int(len(df)),

        "average_price":
            float(df["price"].mean()),

        "median_price":
            float(df["price"].median()),

        "minimum_price":
            float(df["price"].min()),

        "maximum_price":
            float(df["price"].max()),

        "average_area":
            float(df["area"].mean()),

        "minimum_area":
            float(df["area"].min()),

        "maximum_area":
            float(df["area"].max()),

        "average_bedrooms":
            float(df["bedrooms"].mean()),

        "average_bathrooms":
            float(df["bathrooms"].mean()),

        "average_stories":
            float(df["stories"].mean()),

        "average_parking":
            float(df["parking"].mean())
    }


# ============================================================
# CREATE COMPLETE DATASET CONTEXT
# ============================================================

def get_dataset_context():

    stats = get_dataset_statistics()

    context = f"""
Housing Dataset Information

Total houses: {stats['total_houses']}

PRICE:
Average price: {format_price(stats['average_price'])}
Median price: {format_price(stats['median_price'])}
Minimum price: {format_price(stats['minimum_price'])}
Maximum price: {format_price(stats['maximum_price'])}

AREA:
Average area: {stats['average_area']:.2f}
Minimum area: {stats['minimum_area']:.0f}
Maximum area: {stats['maximum_area']:.0f}

BEDROOMS:
Average bedrooms: {stats['average_bedrooms']:.2f}

BATHROOMS:
Average bathrooms: {stats['average_bathrooms']:.2f}

STORIES:
Average stories: {stats['average_stories']:.2f}

PARKING:
Average parking spaces: {stats['average_parking']:.2f}
"""

    # --------------------------------------------------------
    # Bedrooms summary
    # --------------------------------------------------------

    bedroom_summary = (
        df.groupby("bedrooms")["price"]
        .agg([
            "count",
            "mean",
            "median",
            "min",
            "max"
        ])
    )

    context += "\nBEDROOM-WISE PRICE INFORMATION:\n"

    for bedroom, row in bedroom_summary.iterrows():

        context += (
            f"\n{int(bedroom)} bedrooms:"
            f"\n Houses: {int(row['count'])}"
            f"\n Average price: {format_price(row['mean'])}"
            f"\n Median price: {format_price(row['median'])}"
            f"\n Minimum price: {format_price(row['min'])}"
            f"\n Maximum price: {format_price(row['max'])}"
            f"\n"
        )

    # --------------------------------------------------------
    # Bathroom summary
    # --------------------------------------------------------

    bathroom_summary = (
        df.groupby("bathrooms")["price"]
        .agg([
            "count",
            "mean",
            "median",
            "min",
            "max"
        ])
    )

    context += "\nBATHROOM-WISE PRICE INFORMATION:\n"

    for bathroom, row in bathroom_summary.iterrows():

        context += (
            f"\n{int(bathroom)} bathrooms:"
            f"\n Houses: {int(row['count'])}"
            f"\n Average price: {format_price(row['mean'])}"
            f"\n Median price: {format_price(row['median'])}"
            f"\n"
        )

    # --------------------------------------------------------
    # Mainroad
    # --------------------------------------------------------

    context += "\nMAIN ROAD INFORMATION:\n"

    for value in [0, 1]:

        subset = df[
            df["mainroad"] == value
        ]

        if not subset.empty:

            context += (
                f"\nMain road = {yes_no(value)}:"
                f"\n Houses: {len(subset)}"
                f"\n Average price: "
                f"{format_price(subset['price'].mean())}"
                f"\n"
            )

    # --------------------------------------------------------
    # Basement
    # --------------------------------------------------------

    context += "\nBASEMENT INFORMATION:\n"

    for value in [0, 1]:

        subset = df[
            df["basement"] == value
        ]

        if not subset.empty:

            context += (
                f"\nBasement = {yes_no(value)}:"
                f"\n Houses: {len(subset)}"
                f"\n Average price: "
                f"{format_price(subset['price'].mean())}"
                f"\n"
            )

    # --------------------------------------------------------
    # Furnishing
    # --------------------------------------------------------

    context += "\nFURNISHING INFORMATION:\n"

    for value in [0, 1, 2]:

        subset = df[
            df["furnishingstatus"] == value
        ]

        if not subset.empty:

            context += (
                f"\n{furnishing_label(value)}:"
                f"\n Houses: {len(subset)}"
                f"\n Average price: "
                f"{format_price(subset['price'].mean())}"
                f"\n Median price: "
                f"{format_price(subset['price'].median())}"
                f"\n"
            )

    return context


# ============================================================
# EXTRACT NUMBER
# ============================================================

def extract_number(
    question,
    keyword
):

    patterns = [

        rf"(\d+(?:\.\d+)?)\s*{keyword}",

        rf"{keyword}\s*(?:is|=|of|are|with)?\s*(\d+(?:\.\d+)?)"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            question,
            re.IGNORECASE
        )

        if match:

            try:

                return float(
                    match.group(1)
                )

            except:

                return None

    return None


# ============================================================
# EXTRACT FEATURES FROM QUESTION
# ============================================================

def extract_question_features(
    question
):

    q = question.lower()

    result = {}

    # --------------------------------------------------------
    # Bedrooms
    # --------------------------------------------------------

    bedroom_match = re.search(
        r"(\d+)\s*[- ]?bed(?:room)?s?",
        q
    )

    if bedroom_match:

        result["bedrooms"] = int(
            bedroom_match.group(1)
        )

    else:

        bedroom_match = re.search(
            r"bedrooms?\s*(?:is|=|of|are|with)?\s*(\d+)",
            q
        )

        if bedroom_match:

            result["bedrooms"] = int(
                bedroom_match.group(1)
            )

    # --------------------------------------------------------
    # Bathrooms
    # --------------------------------------------------------

    bathroom_match = re.search(
        r"(\d+)\s*[- ]?bath(?:room)?s?",
        q
    )

    if bathroom_match:

        result["bathrooms"] = int(
            bathroom_match.group(1)
        )

    else:

        bathroom_match = re.search(
            r"bathrooms?\s*(?:is|=|of|are|with)?\s*(\d+)",
            q
        )

        if bathroom_match:

            result["bathrooms"] = int(
                bathroom_match.group(1)
            )

    # --------------------------------------------------------
    # Area
    # --------------------------------------------------------

    area_patterns = [

        r"area\s*(?:is|=|of|around|about)?\s*(\d+(?:\.\d+)?)",

        r"(\d+(?:\.\d+)?)\s*(?:sq\.?\s*ft|sqft|square feet)"
    ]

    for pattern in area_patterns:

        match = re.search(
            pattern,
            q
        )

        if match:

            result["area"] = float(
                match.group(1)
            )

            break

    # --------------------------------------------------------
    # Main road
    # --------------------------------------------------------

    if (
        "mainroad yes" in q
        or "main road yes" in q
        or "with mainroad" in q
        or "with main road" in q
        or "on main road" in q
    ):

        result["mainroad"] = 1

    elif (
        "mainroad no" in q
        or "main road no" in q
        or "without mainroad" in q
        or "without main road" in q
    ):

        result["mainroad"] = 0

    # --------------------------------------------------------
    # Basement
    # --------------------------------------------------------

    if (
        "basement yes" in q
        or "with basement" in q
        or "has basement" in q
    ):

        result["basement"] = 1

    elif (
        "basement no" in q
        or "without basement" in q
        or "no basement" in q
    ):

        result["basement"] = 0

    # --------------------------------------------------------
    # Stories
    # --------------------------------------------------------

    stories_match = re.search(
        r"(\d+)\s*stor(?:y|ies)",
        q
    )

    if stories_match:

        result["stories"] = int(
            stories_match.group(1)
        )

    # --------------------------------------------------------
    # Parking
    # --------------------------------------------------------

    parking_match = re.search(
        r"(\d+)\s*parking",
        q
    )

    if parking_match:

        result["parking"] = int(
            parking_match.group(1)
        )

    # --------------------------------------------------------
    # Furnishing
    # --------------------------------------------------------

    if "semi-furnished" in q or "semi furnished" in q:

        result["furnishingstatus"] = 1

    elif "unfurnished" in q:

        result["furnishingstatus"] = 0

    elif "furnished" in q:

        result["furnishingstatus"] = 2

    # --------------------------------------------------------
    # Guest room
    # --------------------------------------------------------

    if (
        "guestroom yes" in q
        or "guest room yes" in q
        or "with guestroom" in q
        or "with guest room" in q
    ):

        result["guestroom"] = 1

    elif (
        "guestroom no" in q
        or "guest room no" in q
        or "without guestroom" in q
        or "without guest room" in q
    ):

        result["guestroom"] = 0

    # --------------------------------------------------------
    # Air conditioning
    # --------------------------------------------------------

    if (
        "airconditioning yes" in q
        or "air conditioning yes" in q
        or "with air conditioning" in q
        or "with ac" in q
    ):

        result["airconditioning"] = 1

    elif (
        "airconditioning no" in q
        or "without air conditioning" in q
        or "without ac" in q
    ):

        result["airconditioning"] = 0

    # --------------------------------------------------------
    # Preferred area
    # --------------------------------------------------------

    if (
        "prefarea yes" in q
        or "preferred area yes" in q
    ):

        result["prefarea"] = 1

    elif (
        "prefarea no" in q
        or "preferred area no" in q
    ):

        result["prefarea"] = 0

    # --------------------------------------------------------
    # Hot water heating
    # --------------------------------------------------------

    if (
        "hotwaterheating yes" in q
        or "hot water heating yes" in q
    ):

        result["hotwaterheating"] = 1

    elif (
        "hotwaterheating no" in q
        or "hot water heating no" in q
    ):

        result["hotwaterheating"] = 0

    return result


# ============================================================
# FILTER DATASET USING QUESTION
# ============================================================

def filter_dataset_from_question(
    question
):

    q = question.lower()

    extracted = extract_question_features(
        question
    )

    filtered = df.copy()

    applied_filters = []

    # --------------------------------------------------------
    # Exact feature filters
    # --------------------------------------------------------

    exact_columns = [
        "bedrooms",
        "bathrooms",
        "stories",
        "mainroad",
        "guestroom",
        "basement",
        "hotwaterheating",
        "airconditioning",
        "parking",
        "prefarea",
        "furnishingstatus"
    ]

    for column in exact_columns:

        if column in extracted:

            filtered = filtered[
                filtered[column]
                == extracted[column]
            ]

            applied_filters.append(
                f"{column}={extracted[column]}"
            )

    # --------------------------------------------------------
    # Area comparisons
    # --------------------------------------------------------

    area_above = re.search(
        r"area\s*(?:above|greater than|more than|over|>)\s*(\d+(?:\.\d+)?)",
        q
    )

    area_below = re.search(
        r"area\s*(?:below|less than|under|<)\s*(\d+(?:\.\d+)?)",
        q
    )

    if area_above:

        value = float(
            area_above.group(1)
        )

        filtered = filtered[
            filtered["area"] > value
        ]

        applied_filters.append(
            f"area>{value}"
        )

    elif area_below:

        value = float(
            area_below.group(1)
        )

        filtered = filtered[
            filtered["area"] < value
        ]

        applied_filters.append(
            f"area<{value}"
        )

    elif "area" in extracted:

        # Only use approximate area filtering for dataset comparison.
        area = extracted["area"]

        tolerance = max(
            area * 0.10,
            100
        )

        filtered = filtered[
            (
                filtered["area"]
                >= area - tolerance
            )
            &
            (
                filtered["area"]
                <= area + tolerance
            )
        ]

        applied_filters.append(
            f"area approximately {area}"
        )

    # --------------------------------------------------------
    # Price comparisons
    # --------------------------------------------------------

    price_above = re.search(
        r"price\s*(?:above|greater than|more than|over|>)\s*"
        r"(?:₹|rs\.?|rupees)?\s*([\d,]+)",
        q
    )

    price_below = re.search(
        r"price\s*(?:below|less than|under|<)\s*"
        r"(?:₹|rs\.?|rupees)?\s*([\d,]+)",
        q
    )

    if price_above:

        value = float(
            price_above
            .group(1)
            .replace(",", "")
        )

        filtered = filtered[
            filtered["price"] > value
        ]

        applied_filters.append(
            f"price>{value}"
        )

    if price_below:

        value = float(
            price_below
            .group(1)
            .replace(",", "")
        )

        filtered = filtered[
            filtered["price"] < value
        ]

        applied_filters.append(
            f"price<{value}"
        )

    return (
        filtered,
        applied_filters,
        extracted
    )


# ============================================================
# ANALYZE DATASET QUESTION
# ============================================================

def analyze_dataset_question(
    question
):

    q = question.lower()

    filtered, filters, extracted = (
        filter_dataset_from_question(
            question
        )
    )

    stats = get_dataset_statistics()

    result = {
        "filters": filters,
        "extracted_features": extracted
    }

    # ========================================================
    # NO FILTERS - COMPLETE DATASET
    # ========================================================

    if not filters:

        working_df = df

    else:

        working_df = filtered

    result["matching_houses"] = int(
        len(working_df)
    )

    # --------------------------------------------------------
    # No matching houses
    # --------------------------------------------------------

    if len(working_df) == 0:

        result["message"] = (
            "No houses in the dataset match "
            "the requested conditions."
        )

        return result

    # --------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------

    result["average_price"] = float(
        working_df["price"].mean()
    )

    result["median_price"] = float(
        working_df["price"].median()
    )

    result["minimum_price"] = float(
        working_df["price"].min()
    )

    result["maximum_price"] = float(
        working_df["price"].max()
    )

    result["average_area"] = float(
        working_df["area"].mean()
    )

    result["minimum_area"] = float(
        working_df["area"].min()
    )

    result["maximum_area"] = float(
        working_df["area"].max()
    )

    result["average_bedrooms"] = float(
        working_df["bedrooms"].mean()
    )

    result["average_bathrooms"] = float(
        working_df["bathrooms"].mean()
    )

    result["average_stories"] = float(
        working_df["stories"].mean()
    )

    result["average_parking"] = float(
        working_df["parking"].mean()
    )

    # --------------------------------------------------------
    # Overall dataset statistics
    # --------------------------------------------------------

    result["total_dataset_houses"] = (
        stats["total_houses"]
    )

    # --------------------------------------------------------
    # Most common bedroom count
    # --------------------------------------------------------

    if not working_df.empty:

        bedroom_mode = (
            working_df["bedrooms"]
            .mode()
        )

        if not bedroom_mode.empty:

            result["most_common_bedrooms"] = int(
                bedroom_mode.iloc[0]
            )

    # --------------------------------------------------------
    # Most common bathroom count
    # --------------------------------------------------------

    if not working_df.empty:

        bathroom_mode = (
            working_df["bathrooms"]
            .mode()
        )

        if not bathroom_mode.empty:

            result["most_common_bathrooms"] = int(
                bathroom_mode.iloc[0]
            )

    # --------------------------------------------------------
    # Main road count
    # --------------------------------------------------------

    result["mainroad_yes"] = int(
        (
            working_df["mainroad"] == 1
        ).sum()
    )

    result["mainroad_no"] = int(
        (
            working_df["mainroad"] == 0
        ).sum()
    )

    # --------------------------------------------------------
    # Basement count
    # --------------------------------------------------------

    result["basement_yes"] = int(
        (
            working_df["basement"] == 1
        ).sum()
    )

    result["basement_no"] = int(
        (
            working_df["basement"] == 0
        ).sum()
    )

    # --------------------------------------------------------
    # Furnishing counts
    # --------------------------------------------------------

    result["unfurnished_count"] = int(
        (
            working_df["furnishingstatus"] == 0
        ).sum()
    )

    result["semi_furnished_count"] = int(
        (
            working_df["furnishingstatus"] == 1
        ).sum()
    )

    result["furnished_count"] = int(
        (
            working_df["furnishingstatus"] == 2
        ).sum()
    )

    return result


# ============================================================
# NORMALIZE BINARY VALUE
# ============================================================

def normalize_binary(
    value,
    default=0
):

    if value is None:

        return default

    if isinstance(
        value,
        bool
    ):

        return int(value)

    if isinstance(
        value,
        str
    ):

        value_lower = (
            value
            .strip()
            .lower()
        )

        if value_lower in [
            "yes",
            "true",
            "1"
        ]:

            return 1

        if value_lower in [
            "no",
            "false",
            "0"
        ]:

            return 0

    try:

        return int(value)

    except:

        return default


# ============================================================
# NORMALIZE FURNISHING STATUS
# ============================================================

def normalize_furnishing(
    value
):

    if value is None:

        return 0

    if isinstance(
        value,
        str
    ):

        value = (
            value
            .strip()
            .lower()
        )

        return FURNISHING_MAPPING.get(
            value,
            0
        )

    try:

        value = int(value)

        if value in [
            0,
            1,
            2
        ]:

            return value

    except:

        pass

    return 0


# ============================================================
# CREATE MODEL INPUT
# ============================================================

def create_model_input(
    data
):

    area = float(
        data.get(
            "area",
            df["area"].median()
        )
    )

    bedrooms = float(
        data.get(
            "bedrooms",
            df["bedrooms"].median()
        )
    )

    bathrooms = float(
        data.get(
            "bathrooms",
            df["bathrooms"].median()
        )
    )

    stories = float(
        data.get(
            "stories",
            1
        )
    )

    mainroad = normalize_binary(
        data.get(
            "mainroad",
            0
        )
    )

    guestroom = normalize_binary(
        data.get(
            "guestroom",
            0
        )
    )

    basement = normalize_binary(
        data.get(
            "basement",
            0
        )
    )

    hotwaterheating = normalize_binary(
        data.get(
            "hotwaterheating",
            0
        )
    )

    airconditioning = normalize_binary(
        data.get(
            "airconditioning",
            0
        )
    )

    parking = float(
        data.get(
            "parking",
            0
        )
    )

    prefarea = normalize_binary(
        data.get(
            "prefarea",
            0
        )
    )

    furnishingstatus = normalize_furnishing(
        data.get(
            "furnishingstatus",
            "unfurnished"
        )
    )

    # ========================================================
    # IMPORTANT:
    # THIS ORDER MUST MATCH FEATURES EXACTLY
    # ========================================================

    input_data = pd.DataFrame(
        [[
            area,
            bedrooms,
            bathrooms,
            stories,
            mainroad,
            guestroom,
            basement,
            hotwaterheating,
            airconditioning,
            parking,
            prefarea,
            furnishingstatus
        ]],
        columns=FEATURES
    )

    return input_data


# ============================================================
# GET ML PREDICTION
# ============================================================

def get_model_prediction(
    data
):

    input_data = create_model_input(
        data
    )

    prediction = model.predict(
        input_data
    )[0]

    return float(
        prediction
    )


# ============================================================
# DETECT IF QUESTION IS PREDICTION QUESTION
# ============================================================

def is_prediction_question(
    question
):

    q = question.lower()

    prediction_words = [
        "predict",
        "prediction",
        "estimated price",
        "estimate price",
        "what will be the price",
        "what would be the price",
        "how much will",
        "how much would",
        "what if",
        "expected price",
        "house price for",
        "price for this house"
    ]

    return any(
        word in q
        for word in prediction_words
    )


# ============================================================
# DETECT IF ENOUGH FEATURES FOR ML MODEL
# ============================================================

def has_main_prediction_features(
    data
):

    required = [
        "area",
        "bedrooms",
        "bathrooms",
        "mainroad",
        "basement"
    ]

    return all(
        key in data
        and data[key] is not None
        for key in required
    )


# ============================================================
# BUILD SCENARIO INPUT
# ============================================================

def build_scenario_input(
    question,
    current_inputs
):

    scenario = {}

    # --------------------------------------------------------
    # Start from current React inputs
    # --------------------------------------------------------

    if isinstance(
        current_inputs,
        dict
    ):

        scenario.update(
            current_inputs
        )

    # --------------------------------------------------------
    # Override values mentioned in question
    # --------------------------------------------------------

    extracted = extract_question_features(
        question
    )

    scenario.update(
        extracted
    )

    return (
        scenario,
        extracted
    )


# ============================================================
# CREATE DETERMINISTIC FALLBACK RESPONSE
# ============================================================

def create_fallback_response(
    question,
    analysis,
    scenario_prediction=None,
    scenario_input=None
):

    lines = []

    lines.append(
        "SMART REAL ESTATE AI"
    )

    lines.append("")

    # --------------------------------------------------------
    # Scenario prediction
    # --------------------------------------------------------

    if scenario_prediction is not None:

        lines.append(
            "Based on the trained house-price prediction model:"
        )

        lines.append("")

        lines.append(
            f"Estimated house price: "
            f"{format_price(scenario_prediction)}"
        )

        lines.append(
            f"Approximately "
            f"{scenario_prediction / 100000:.2f} lakhs."
        )

        if scenario_input:

            lines.append("")

            lines.append(
                "Prediction details:"
            )

            for key in MAIN_FEATURES:

                if key in scenario_input:

                    value = scenario_input[key]

                    if key in [
                        "mainroad",
                        "basement"
                    ]:

                        value = yes_no(
                            normalize_binary(
                                value
                            )
                        )

                    lines.append(
                        f"- {key}: {value}"
                    )

        return "\n".join(
            lines
        )

    # --------------------------------------------------------
    # Dataset answer
    # --------------------------------------------------------

    matching = analysis.get(
        "matching_houses",
        0
    )

    if matching == 0:

        lines.append(
            "No houses in Housing.csv match "
            "those conditions."
        )

        return "\n".join(
            lines
        )

    filters = analysis.get(
        "filters",
        []
    )

    if filters:

        lines.append(
            f"Based on {matching} matching "
            f"house(s) in the dataset:"
        )

    else:

        lines.append(
            f"Based on all {matching} houses "
            f"in the dataset:"
        )

    lines.append("")

    q = question.lower()

    if (
        "average" in q
        and "area" in q
    ):

        lines.append(
            f"Average area: "
            f"{analysis['average_area']:,.2f}"
        )

    elif (
        "maximum" in q
        or "highest" in q
        or "max " in q
    ) and "area" in q:

        lines.append(
            f"Maximum area: "
            f"{analysis['maximum_area']:,.0f}"
        )

    elif (
        "minimum" in q
        or "lowest" in q
        or "min " in q
    ) and "area" in q:

        lines.append(
            f"Minimum area: "
            f"{analysis['minimum_area']:,.0f}"
        )

    elif (
        "how many" in q
        or "count" in q
        or "number of" in q
    ):

        lines.append(
            f"Number of matching houses: "
            f"{matching}"
        )

    else:

        lines.append(
            f"Average price: "
            f"{format_price(analysis['average_price'])}"
        )

        lines.append(
            f"Median price: "
            f"{format_price(analysis['median_price'])}"
        )

        lines.append(
            f"Price range: "
            f"{format_price(analysis['minimum_price'])}"
            f" to "
            f"{format_price(analysis['maximum_price'])}"
        )

    return "\n".join(
        lines
    )


# ============================================================
# ASK GEMINI
# ============================================================

def ask_gemini(
    question,
    analysis,
    current_inputs=None,
    current_prediction=None,
    scenario_input=None,
    scenario_prediction=None
):

    # --------------------------------------------------------
    # If Gemini is unavailable
    # --------------------------------------------------------

    if gemini_client is None:

        return create_fallback_response(
            question,
            analysis,
            scenario_prediction,
            scenario_input
        )

    dataset_context = get_dataset_context()

    prompt = f"""
You are SMART REAL ESTATE AI.

You are the chatbot inside a house-price prediction application.

The user is asking questions about the Housing.csv dataset and
the trained machine-learning house price prediction model.

VERY IMPORTANT RULES:

1. Never invent any number.
2. Never invent house prices.
3. Never invent dataset statistics.
4. Use ONLY the factual values supplied below.
5. If the question asks about the dataset, answer using DATASET ANALYSIS.
6. If the question asks for an ML prediction and SCENARIO PREDICTION
   is supplied, use exactly that prediction.
7. Do not calculate a new ML prediction yourself.
8. A bedroom-only question such as:
   "I need 3 bedrooms, what is the price?"
   should NOT be treated as an exact ML model prediction.
   Instead explain the statistics of matching houses in the dataset.
9. For a precise prediction, the main required user features are:
   area, bedrooms, bathrooms, mainroad and basement.
10. Keep the answer easy for a beginner to understand.
11. Use Indian Rupees.
12. When useful, also express prices in lakhs.
13. Do not mention that you are guessing.
14. If data is unavailable, clearly say it is unavailable.
15. Never claim an exact prediction came from the dataset.
16. Dataset statistics and ML predictions are different things.
17. If the user asks "how many", give the matching-house count.
18. If the user asks average/median/minimum/maximum, give the exact
    supplied value relevant to their question.
19. Do not change supplied numerical values.
20. Do not produce fictional examples unless explicitly requested.

============================================================
USER QUESTION
============================================================

{question}


============================================================
DATASET ANALYSIS FOR THIS QUESTION
============================================================

{analysis}


============================================================
GENERAL DATASET CONTEXT
============================================================

{dataset_context}


============================================================
CURRENT FORM INPUTS FROM REACT
============================================================

{current_inputs}


============================================================
CURRENT HOUSE PREDICTION DISPLAYED IN REACT
============================================================

{current_prediction}


============================================================
SCENARIO INPUT USED BY ML MODEL
============================================================

{scenario_input}


============================================================
ACTUAL SCENARIO ML PREDICTION
============================================================

{scenario_prediction}


============================================================
RESPONSE STYLE
============================================================

Start naturally.

If answering a bedroom-only price question, use a structure similar to:

"Based on the dataset of X comparable houses..."

Then give:
- Average price
- Median price
- Price range

Then explain that a precise ML prediction needs:
area, bathrooms, mainroad and basement in addition to bedrooms.

If an actual scenario ML prediction is supplied, clearly say:

"Based on the trained ML model, the estimated house price is..."

Do not call dataset averages "model predictions".

Keep the answer concise and useful.
"""

    try:

        response = (
            gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
        )

        if (
            response
            and response.text
        ):

            return response.text.strip()

    except Exception as e:

        print(
            "Gemini Error:",
            str(e)
        )

    # --------------------------------------------------------
    # Gemini failed -> deterministic fallback
    # --------------------------------------------------------

    return create_fallback_response(
        question,
        analysis,
        scenario_prediction,
        scenario_input
    )


# ============================================================
# HOME API
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def home():

    return jsonify({

        "message":
            "Smart Real Estate House Price Prediction API",

        "status":
            "running",

        "endpoints": [
            "/api/health",
            "/api/features",
            "/api/predict",
            "/api/chat",
            "/api/model-info",
            "/api/houses",
            "/api/statistics"
        ]
    })


# ============================================================
# HEALTH CHECK API
# ============================================================

@app.route(
    "/api/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status":
            "success",

        "message":
            "API is running successfully",

        "gemini_connected":
            gemini_client is not None
    })


# ============================================================
# FEATURES API
# ============================================================

@app.route(
    "/api/features",
    methods=["GET"]
)
def features():

    return jsonify({

        "status":
            "success",

        "target":
            TARGET,

        "features":
            FEATURES,

        "main_features":
            MAIN_FEATURES,

        "binary_features":
            BINARY_FEATURES,

        "furnishingstatus":
            FURNISHING_MAPPING
    })


# ============================================================
# PREDICTION API
# ============================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def predict():

    try:

        data = request.get_json(
            silent=True
        )

        if not data:

            return jsonify({
                "status": "error",
                "message": "No input data received"
            }), 400

        print(
            "==================================="
        )

        print(
            "Received prediction data:"
        )

        print(
            data
        )

        print(
            "==================================="
        )

        # ====================================================
        # REQUIRED FEATURES
        # ====================================================

        required_features = [
            "area",
            "bedrooms",
            "bathrooms",
            "mainroad",
            "basement"
        ]

        missing_features = [
            feature
            for feature
            in required_features
            if feature not in data
        ]

        if missing_features:

            return jsonify({

                "status":
                    "error",

                "message":
                    "Missing required features",

                "missing_features":
                    missing_features

            }), 400

        # ====================================================
        # VALIDATE VALUES
        # ====================================================

        area = float(
            data["area"]
        )

        bedrooms = float(
            data["bedrooms"]
        )

        bathrooms = float(
            data["bathrooms"]
        )

        if area <= 0:

            return jsonify({
                "status": "error",
                "message": "Area must be greater than 0"
            }), 400

        if bedrooms <= 0:

            return jsonify({
                "status": "error",
                "message": "Bedrooms must be greater than 0"
            }), 400

        if bathrooms <= 0:

            return jsonify({
                "status": "error",
                "message": "Bathrooms must be greater than 0"
            }), 400

        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        prediction = get_model_prediction(
            data
        )

        print(
            "Predicted price:",
            prediction
        )

        # ====================================================
        # RESPONSE
        # ====================================================

        return jsonify({

            "status":
                "success",

            "prediction":
                float(prediction),

            "predicted_price":
                float(prediction),

            "price_lakhs":
                float(
                    prediction / 100000
                )

        })

    except Exception as e:

        print(
            "Prediction Error:",
            str(e)
        )

        return jsonify({

            "status":
                "error",

            "message":
                str(e)

        }), 400


# ============================================================
# CHATBOT API
# ============================================================

@app.route(
    "/api/chat",
    methods=["POST"]
)
def chat():

    try:

        data = request.get_json(
            silent=True
        )

        if not data:

            return jsonify({
                "status": "error",
                "message": "No chatbot data received"
            }), 400

        # ----------------------------------------------------
        # Question
        # ----------------------------------------------------

        question = str(
            data.get(
                "question",
                ""
            )
        ).strip()

        if not question:

            return jsonify({
                "status": "error",
                "message": "Please enter a question."
            }), 400

        # ----------------------------------------------------
        # Current React form inputs
        # ----------------------------------------------------

        current_inputs = data.get(
            "inputs",
            {}
        )

        if not isinstance(
            current_inputs,
            dict
        ):

            current_inputs = {}

        # ----------------------------------------------------
        # Current prediction displayed in UI
        # ----------------------------------------------------

        current_prediction = data.get(
            "prediction"
        )

        # ====================================================
        # ANALYZE ACTUAL DATASET
        # ====================================================

        analysis = analyze_dataset_question(
            question
        )

        # ====================================================
        # CHECK WHETHER USER WANTS ML SCENARIO
        # ====================================================

        scenario_prediction = None
        scenario_input = None

        question_features = (
            extract_question_features(
                question
            )
        )

        # ----------------------------------------------------
        # Scenario such as:
        # "what if area is 2500?"
        #
        # We take the current UI values and only replace area.
        # ----------------------------------------------------

        if is_prediction_question(
            question
        ):

            scenario_input, extracted = (
                build_scenario_input(
                    question,
                    current_inputs
                )
            )

            if has_main_prediction_features(
                scenario_input
            ):

                try:

                    scenario_prediction = (
                        get_model_prediction(
                            scenario_input
                        )
                    )

                except Exception as e:

                    print(
                        "Scenario prediction error:",
                        str(e)
                    )

                    scenario_prediction = None

        # ----------------------------------------------------
        # If user supplied all 5 features directly in question
        # ----------------------------------------------------

        elif has_main_prediction_features(
            question_features
        ):

            scenario_input = (
                question_features
            )

            try:

                scenario_prediction = (
                    get_model_prediction(
                        scenario_input
                    )
                )

            except Exception as e:

                print(
                    "Direct question prediction error:",
                    str(e)
                )

        # ====================================================
        # ASK GEMINI USING REAL CALCULATED DATA
        # ====================================================

        answer = ask_gemini(

            question=
                question,

            analysis=
                analysis,

            current_inputs=
                current_inputs,

            current_prediction=
                current_prediction,

            scenario_input=
                scenario_input,

            scenario_prediction=
                scenario_prediction
        )

        # ====================================================
        # CHAT RESPONSE
        # ====================================================

        return jsonify({

            "status":
                "success",

            "answer":
                answer,

            "response":
                answer,

            "dataset_analysis":
                analysis,

            "scenario_prediction":
                scenario_prediction,

            "scenario_input":
                scenario_input

        })

    except Exception as e:

        print(
            "Chat Error:",
            str(e)
        )

        return jsonify({

            "status":
                "error",

            "message":
                str(e)

        }), 500


# ============================================================
# MODEL INFORMATION API
# ============================================================

@app.route(
    "/api/model-info",
    methods=["GET"]
)
def model_info():

    return jsonify({

        "status":
            "success",

        "model":
            "Gradient Boosting Regressor",

        "parameters": {

            "n_estimators":
                300,

            "learning_rate":
                0.03,

            "max_depth":
                3,

            "min_samples_split":
                5,

            "min_samples_leaf":
                2,

            "subsample":
                0.8,

            "random_state":
                42
        },

        "features":
            FEATURES,

        "target":
            TARGET,

        "evaluation": {

            "r2_score":
                r2,

            "mae":
                mae,

            "rmse":
                rmse
        }
    })


# ============================================================
# DATASET API
# ============================================================

@app.route(
    "/api/houses",
    methods=["GET"]
)
def houses():

    try:

        # ----------------------------------------------------
        # Number of records
        # ----------------------------------------------------

        limit = request.args.get(
            "limit",
            default=20,
            type=int
        )

        if limit < 1:

            limit = 1

        if limit > 100:

            limit = 100

        result = df.head(
            limit
        ).copy()

        # ----------------------------------------------------
        # Convert encoded values back for readability
        # ----------------------------------------------------

        for col in BINARY_FEATURES:

            if col in result.columns:

                result[col] = (
                    result[col]
                    .map({
                        1: "yes",
                        0: "no"
                    })
                )

        if "furnishingstatus" in result.columns:

            result[
                "furnishingstatus"
            ] = (
                result[
                    "furnishingstatus"
                ]
                .map({
                    0: "unfurnished",
                    1: "semi-furnished",
                    2: "furnished"
                })
            )

        records = result.to_dict(
            orient="records"
        )

        return jsonify({

            "status":
                "success",

            "count":
                len(records),

            "data":
                records
        })

    except Exception as e:

        return jsonify({

            "status":
                "error",

            "message":
                str(e)

        }), 500


# ============================================================
# DATASET STATISTICS API
# ============================================================

@app.route(
    "/api/statistics",
    methods=["GET"]
)
def statistics():

    try:

        statistics_data = (
            get_dataset_statistics()
        )

        return jsonify({

            "status":
                "success",

            "statistics":
                statistics_data
        })

    except Exception as e:

        return jsonify({

            "status":
                "error",

            "message":
                str(e)

        }), 500


# ============================================================
# ERROR HANDLER
# ============================================================

@app.errorhandler(404)
def not_found(
    error
):

    return jsonify({

        "status":
            "error",

        "message":
            "API endpoint not found"

    }), 404


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print(
        "\n=========================================="
    )

    print(
        "SMART REAL ESTATE API"
    )

    print(
        "=========================================="
    )

    print(
        "\nDataset:",
        DATASET_PATH
    )

    print(
        "Model:",
        MODEL_PATH
    )

    print(
        "Gemini:",
        (
            "Connected"
            if gemini_client
            else "Not Connected"
        )
    )

    print(
        "\nServer running at:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print(
        "\nAvailable APIs:"
    )

    print(
        "GET  /"
    )

    print(
        "GET  /api/health"
    )

    print(
        "GET  /api/features"
    )

    print(
        "POST /api/predict"
    )

    print(
        "POST /api/chat"
    )

    print(
        "GET  /api/model-info"
    )

    print(
        "GET  /api/houses"
    )

    print(
        "GET  /api/statistics"
    )

    print(
        "\n=========================================="
    )

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )