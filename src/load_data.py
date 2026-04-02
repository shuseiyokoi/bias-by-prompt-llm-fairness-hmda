import requests


def load_data():
    url = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"
    params = {"states": "CA", "years": "2024", "action_taken": "1,2,3"}

    response = requests.get(url, params=params, allow_redirects=True)
    response.raise_for_status()

    with open("../data/hmda_CA_2024.csv", "wb") as f:
        f.write(response.content)

    print("Saved as hmda_CA_2024.csv")


if __name__ == "__main__":
    load_data()
