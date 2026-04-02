import pandas as pd
import numpy as np


def summarize_data():
    columns_to_keep = [
        "lei",
        "state_code",
        "county_code",
        "derived_ethnicity",
        "derived_race",
        "derived_sex",
        "action_taken",
        "loan_amount",
        # "interest_rate", Missing values on this column are related to the action taken
        # "lender_credits", Missing values on this column are related to the action taken
        "loan_term",
        "property_value",
        "income",
        "applicant_age",
    ]

    master = pd.read_csv(
        "../data/hmda_CA_2024.csv", na_values=["NA", "None", "Exempt", "", " "]
    )

    master = master[columns_to_keep]

    numeric_cols = [
        "loan_amount",
        "loan_term",
        "property_value",
        "income",
    ]

    for col in numeric_cols:
        master[col] = pd.to_numeric(master[col], errors="coerce")

    master = master.dropna(subset=columns_to_keep)

    action_map = {
        1: "Loan originated",
        2: "Application approved but not accepted",
        3: "Application denied",
    }

    master["action_taken"] = master["action_taken"].map(action_map)

    latina_female_mask = (master["derived_ethnicity"] == "Hispanic or Latino") & (
        master["derived_sex"] == "Female"
    )

    not_latina_female_mask = (
        master["derived_ethnicity"] == "Not Hispanic or Latino"
    ) & (master["derived_sex"] == "Female")

    white_female_mask = (
        (master["derived_ethnicity"] == "Not Hispanic or Latino")
        & (master["derived_race"] == "White")
        & (master["derived_sex"] == "Female")
    )

    latina_male_mask = (master["derived_ethnicity"] == "Hispanic or Latino") & (
        master["derived_sex"] == "Male"
    )

    not_latina_male_mask = (master["derived_ethnicity"] == "Not Hispanic or Latino") & (
        master["derived_sex"] == "Male"
    )

    white_male_mask = (
        (master["derived_ethnicity"] == "Not Hispanic or Latino")
        & (master["derived_race"] == "White")
        & (master["derived_sex"] == "Male")
    )

    summary = pd.DataFrame(
        {
            "group": [
                "Latina female",
                "Not Latina female",
                "White female",
                "Latina male",
                "Not Latina male",
                "White male",
            ],
            "total_apps": [
                latina_female_mask.sum(),
                not_latina_female_mask.sum(),
                white_female_mask.sum(),
                latina_male_mask.sum(),
                not_latina_male_mask.sum(),
                white_male_mask.sum(),
            ],
            "avg_loan_amount": [
                (master.loc[latina_female_mask, "loan_amount"].mean()),
                (master.loc[not_latina_female_mask, "loan_amount"].mean()),
                (master.loc[white_female_mask, "loan_amount"].mean()),
                (master.loc[latina_male_mask, "loan_amount"].mean()),
                (master.loc[not_latina_male_mask, "loan_amount"].mean()),
                (master.loc[white_male_mask, "loan_amount"].mean()),
            ],
            "std_loan_amount": [
                (master.loc[latina_female_mask, "loan_amount"].std()),
                (master.loc[not_latina_female_mask, "loan_amount"].std()),
                (master.loc[white_female_mask, "loan_amount"].std()),
                (master.loc[latina_male_mask, "loan_amount"].std()),
                (master.loc[not_latina_male_mask, "loan_amount"].std()),
                (master.loc[white_male_mask, "loan_amount"].std()),
            ],
            "std_loan_amount": [
                (master.loc[latina_female_mask, "loan_amount"].std()),
                (master.loc[not_latina_female_mask, "loan_amount"].std()),
                (master.loc[white_female_mask, "loan_amount"].std()),
                (master.loc[latina_male_mask, "loan_amount"].std()),
                (master.loc[not_latina_male_mask, "loan_amount"].std()),
                (master.loc[white_male_mask, "loan_amount"].std()),
            ],
            "Q1_loan_amount": [
                (np.percentile(master.loc[latina_female_mask, "loan_amount"], 25)),
                (np.percentile(master.loc[not_latina_female_mask, "loan_amount"], 25)),
                (np.percentile(master.loc[white_female_mask, "loan_amount"], 25)),
                (np.percentile(master.loc[latina_male_mask, "loan_amount"], 25)),
                (np.percentile(master.loc[not_latina_male_mask, "loan_amount"], 25)),
                (np.percentile(master.loc[white_male_mask, "loan_amount"], 25)),
            ],
            "Q3_loan_amount": [
                (np.percentile(master.loc[latina_female_mask, "loan_amount"], 75)),
                (np.percentile(master.loc[not_latina_female_mask, "loan_amount"], 75)),
                (np.percentile(master.loc[white_female_mask, "loan_amount"], 75)),
                (np.percentile(master.loc[latina_male_mask, "loan_amount"], 75)),
                (np.percentile(master.loc[not_latina_male_mask, "loan_amount"], 75)),
                (np.percentile(master.loc[white_male_mask, "loan_amount"], 75)),
            ],
            "iqr_loan_amount": [
                (
                    np.percentile(master.loc[latina_female_mask, "loan_amount"], 75)
                    - np.percentile(master.loc[latina_female_mask, "loan_amount"], 25)
                ),
                (
                    np.percentile(master.loc[not_latina_female_mask, "loan_amount"], 75)
                    - np.percentile(
                        master.loc[not_latina_female_mask, "loan_amount"], 25
                    )
                ),
                (
                    np.percentile(master.loc[white_female_mask, "loan_amount"], 75)
                    - np.percentile(master.loc[white_female_mask, "loan_amount"], 25)
                ),
                (
                    np.percentile(master.loc[latina_male_mask, "loan_amount"], 75)
                    - np.percentile(master.loc[latina_male_mask, "loan_amount"], 25)
                ),
                (
                    np.percentile(master.loc[not_latina_male_mask, "loan_amount"], 75)
                    - np.percentile(master.loc[not_latina_male_mask, "loan_amount"], 25)
                ),
                (
                    np.percentile(master.loc[white_male_mask, "loan_amount"], 75)
                    - np.percentile(master.loc[white_male_mask, "loan_amount"], 25)
                ),
            ],
            "avg_loan_term": [
                (master.loc[latina_female_mask, "loan_term"].mean()),
                (master.loc[not_latina_female_mask, "loan_term"].mean()),
                (master.loc[white_female_mask, "loan_term"].mean()),
                (master.loc[latina_male_mask, "loan_term"].mean()),
                (master.loc[not_latina_male_mask, "loan_term"].mean()),
                (master.loc[white_male_mask, "loan_term"].mean()),
            ],
            "std_loan_term": [
                (master.loc[latina_female_mask, "loan_term"].std()),
                (master.loc[not_latina_female_mask, "loan_term"].std()),
                (master.loc[white_female_mask, "loan_term"].std()),
                (master.loc[latina_male_mask, "loan_term"].std()),
                (master.loc[not_latina_male_mask, "loan_term"].std()),
                (master.loc[white_male_mask, "loan_term"].std()),
            ],
            "Q1_loan_term": [
                (np.percentile(master.loc[latina_female_mask, "loan_term"], 25)),
                (np.percentile(master.loc[not_latina_female_mask, "loan_term"], 25)),
                (np.percentile(master.loc[white_female_mask, "loan_term"], 25)),
                (np.percentile(master.loc[latina_male_mask, "loan_term"], 25)),
                (np.percentile(master.loc[not_latina_male_mask, "loan_term"], 25)),
                (np.percentile(master.loc[white_male_mask, "loan_term"], 25)),
            ],
            "Q3_loan_term": [
                (np.percentile(master.loc[latina_female_mask, "loan_term"], 75)),
                (np.percentile(master.loc[not_latina_female_mask, "loan_term"], 75)),
                (np.percentile(master.loc[white_female_mask, "loan_term"], 75)),
                (np.percentile(master.loc[latina_male_mask, "loan_term"], 75)),
                (np.percentile(master.loc[not_latina_male_mask, "loan_term"], 75)),
                (np.percentile(master.loc[white_male_mask, "loan_term"], 75)),
            ],
            "iqr_loan_term": [
                (
                    np.percentile(master.loc[latina_female_mask, "loan_term"], 75)
                    - np.percentile(master.loc[latina_female_mask, "loan_term"], 25)
                ),
                (
                    np.percentile(master.loc[not_latina_female_mask, "loan_term"], 75)
                    - np.percentile(master.loc[not_latina_female_mask, "loan_term"], 25)
                ),
                (
                    np.percentile(master.loc[white_female_mask, "loan_term"], 75)
                    - np.percentile(master.loc[white_female_mask, "loan_term"], 25)
                ),
                (
                    np.percentile(master.loc[latina_male_mask, "loan_term"], 75)
                    - np.percentile(master.loc[latina_male_mask, "loan_term"], 25)
                ),
                (
                    np.percentile(master.loc[not_latina_male_mask, "loan_term"], 75)
                    - np.percentile(master.loc[not_latina_male_mask, "loan_term"], 25)
                ),
                (
                    np.percentile(master.loc[white_male_mask, "loan_term"], 75)
                    - np.percentile(master.loc[white_male_mask, "loan_term"], 25)
                ),
            ],
            "avg_property_value": [
                (master.loc[latina_female_mask, "property_value"].mean()),
                (master.loc[not_latina_female_mask, "property_value"].mean()),
                (master.loc[white_female_mask, "property_value"].mean()),
                (master.loc[latina_male_mask, "property_value"].mean()),
                (master.loc[not_latina_male_mask, "property_value"].mean()),
                (master.loc[white_male_mask, "property_value"].mean()),
            ],
            "std_property_value": [
                (master.loc[latina_female_mask, "property_value"].std()),
                (master.loc[not_latina_female_mask, "property_value"].std()),
                (master.loc[white_female_mask, "property_value"].std()),
                (master.loc[latina_male_mask, "property_value"].std()),
                (master.loc[not_latina_male_mask, "property_value"].std()),
                (master.loc[white_male_mask, "property_value"].std()),
            ],
            "Q1_property_value": [
                (np.percentile(master.loc[latina_female_mask, "property_value"], 25)),
                (
                    np.percentile(
                        master.loc[not_latina_female_mask, "property_value"], 25
                    )
                ),
                (np.percentile(master.loc[white_female_mask, "property_value"], 25)),
                (np.percentile(master.loc[latina_male_mask, "property_value"], 25)),
                (np.percentile(master.loc[not_latina_male_mask, "property_value"], 25)),
                (np.percentile(master.loc[white_male_mask, "property_value"], 25)),
            ],
            "Q3_property_value": [
                (np.percentile(master.loc[latina_female_mask, "property_value"], 75)),
                (
                    np.percentile(
                        master.loc[not_latina_female_mask, "property_value"], 75
                    )
                ),
                (np.percentile(master.loc[white_female_mask, "property_value"], 75)),
                (np.percentile(master.loc[latina_male_mask, "property_value"], 75)),
                (np.percentile(master.loc[not_latina_male_mask, "property_value"], 75)),
                (np.percentile(master.loc[white_male_mask, "property_value"], 75)),
            ],
            "iqr_property_value": [
                (
                    np.percentile(master.loc[latina_female_mask, "property_value"], 75)
                    - np.percentile(
                        master.loc[latina_female_mask, "property_value"], 25
                    )
                ),
                (
                    np.percentile(
                        master.loc[not_latina_female_mask, "property_value"], 75
                    )
                    - np.percentile(
                        master.loc[not_latina_female_mask, "property_value"], 25
                    )
                ),
                (
                    np.percentile(master.loc[white_female_mask, "property_value"], 75)
                    - np.percentile(master.loc[white_female_mask, "property_value"], 25)
                ),
                (
                    np.percentile(master.loc[latina_male_mask, "property_value"], 75)
                    - np.percentile(master.loc[latina_male_mask, "property_value"], 25)
                ),
                (
                    np.percentile(
                        master.loc[not_latina_male_mask, "property_value"], 75
                    )
                    - np.percentile(
                        master.loc[not_latina_male_mask, "property_value"], 25
                    )
                ),
                (
                    np.percentile(master.loc[white_male_mask, "property_value"], 75)
                    - np.percentile(master.loc[white_male_mask, "property_value"], 25)
                ),
            ],
            "avg_income": [
                (master.loc[latina_female_mask, "income"].mean()),
                (master.loc[not_latina_female_mask, "income"].mean()),
                (master.loc[white_female_mask, "income"].mean()),
                (master.loc[latina_male_mask, "income"].mean()),
                (master.loc[not_latina_male_mask, "income"].mean()),
                (master.loc[white_male_mask, "income"].mean()),
            ],
            "std_income": [
                (master.loc[latina_female_mask, "income"].std()),
                (master.loc[not_latina_female_mask, "income"].std()),
                (master.loc[white_female_mask, "income"].std()),
                (master.loc[latina_male_mask, "income"].std()),
                (master.loc[not_latina_male_mask, "income"].std()),
                (master.loc[white_male_mask, "income"].std()),
            ],
            "Q1_income": [
                (np.percentile(master.loc[latina_female_mask, "income"], 25)),
                (np.percentile(master.loc[not_latina_female_mask, "income"], 25)),
                (np.percentile(master.loc[white_female_mask, "income"], 25)),
                (np.percentile(master.loc[latina_male_mask, "income"], 25)),
                (np.percentile(master.loc[not_latina_male_mask, "income"], 25)),
                (np.percentile(master.loc[white_male_mask, "income"], 25)),
            ],
            "Q3_income": [
                (np.percentile(master.loc[latina_female_mask, "income"], 75)),
                (np.percentile(master.loc[not_latina_female_mask, "income"], 75)),
                (np.percentile(master.loc[white_female_mask, "income"], 75)),
                (np.percentile(master.loc[latina_male_mask, "income"], 75)),
                (np.percentile(master.loc[not_latina_male_mask, "income"], 75)),
                (np.percentile(master.loc[white_male_mask, "income"], 75)),
            ],
            "iqr_income": [
                (
                    np.percentile(master.loc[latina_female_mask, "income"], 75)
                    - np.percentile(master.loc[latina_female_mask, "income"], 25)
                ),
                (
                    np.percentile(master.loc[not_latina_female_mask, "income"], 75)
                    - np.percentile(master.loc[not_latina_female_mask, "income"], 25)
                ),
                (
                    np.percentile(master.loc[white_female_mask, "income"], 75)
                    - np.percentile(master.loc[white_female_mask, "income"], 25)
                ),
                (
                    np.percentile(master.loc[latina_male_mask, "income"], 75)
                    - np.percentile(master.loc[latina_male_mask, "income"], 25)
                ),
                (
                    np.percentile(master.loc[not_latina_male_mask, "income"], 75)
                    - np.percentile(master.loc[not_latina_male_mask, "income"], 25)
                ),
                (
                    np.percentile(master.loc[white_male_mask, "income"], 75)
                    - np.percentile(master.loc[white_male_mask, "income"], 25)
                ),
            ],
            "accepted": [
                (
                    master.loc[latina_female_mask, "action_taken"] == "Loan originated"
                ).sum(),
                (
                    master.loc[not_latina_female_mask, "action_taken"]
                    == "Loan originated"
                ).sum(),
                (
                    master.loc[white_female_mask, "action_taken"] == "Loan originated"
                ).sum(),
                (
                    master.loc[latina_male_mask, "action_taken"] == "Loan originated"
                ).sum(),
                (
                    master.loc[not_latina_male_mask, "action_taken"]
                    == "Loan originated"
                ).sum(),
                (
                    master.loc[white_male_mask, "action_taken"] == "Loan originated"
                ).sum(),
            ],
            "approved_not_accepted": [
                (
                    master.loc[latina_female_mask, "action_taken"]
                    == "Application approved but not accepted"
                ).sum(),
                (
                    master.loc[not_latina_female_mask, "action_taken"]
                    == "Application approved but not accepted"
                ).sum(),
                (
                    master.loc[white_female_mask, "action_taken"]
                    == "Application approved but not accepted"
                ).sum(),
                (
                    master.loc[latina_male_mask, "action_taken"]
                    == "Application approved but not accepted"
                ).sum(),
                (
                    master.loc[not_latina_male_mask, "action_taken"]
                    == "Application approved but not accepted"
                ).sum(),
                (
                    master.loc[white_male_mask, "action_taken"]
                    == "Application approved but not accepted"
                ).sum(),
            ],
            "denied": [
                (
                    master.loc[latina_female_mask, "action_taken"]
                    == "Application denied"
                ).sum(),
                (
                    master.loc[not_latina_female_mask, "action_taken"]
                    == "Application denied"
                ).sum(),
                (
                    master.loc[white_female_mask, "action_taken"]
                    == "Application denied"
                ).sum(),
                (
                    master.loc[latina_male_mask, "action_taken"] == "Application denied"
                ).sum(),
                (
                    master.loc[not_latina_male_mask, "action_taken"]
                    == "Application denied"
                ).sum(),
                (
                    master.loc[white_male_mask, "action_taken"] == "Application denied"
                ).sum(),
            ],
        }
    )

    summary["acceptance_rate"] = summary["accepted"] / summary["total_apps"]
    summary["acceptance_rate_pct"] = summary["acceptance_rate"] * 100

    with open("../data/summary.txt", "w") as f:
        f.write(summary.to_string())
    # summary.to_csv("../data/summary.csv", index=False) For csv file

    print("Summary saved as summary.txt")


if __name__ == "__main__":
    summarize_data()
