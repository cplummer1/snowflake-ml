import os
import pandas as pd
import tempfile
import time
import logging
import jwt
import requests
from utils.snowflake.KeyManager import KeyManager
import json


# === SNOWFLAKE CONNECTION CLASS ===

class SnowflakeConnection:
    def __init__(self, user='harkinsadmin', role='ANALYTICSADMIN', warehouse="TRANSFORM_WH", database="PC_DB", schema="SAFETY_APP"):
        self.account = "harkins.east-us-2.azure.snowflakecomputing.com"
        self.user = user
        self.role = role
        self.warehouse = warehouse
        self.database = database
        self.schema = schema
        self.jwt_token = None
        self.token_expiry = 0

        logging.info("Initializing SnowflakeConnection with key-pair authentication.")
        self.key_manager = KeyManager()
        self.private_key_pem = self.key_manager.get_private_key_from_azure()
        self.private_key, self.public_key_fp = self.key_manager.load_private_key_and_generate_fingerprint(self.private_key_pem)
        self._refresh_session()

    def _refresh_session(self):
        if self.is_token_expired():
            self.jwt_token = self.generate_jwt()

    def is_token_expired(self):
        return self.jwt_token is None or time.time() >= (self.token_expiry - 300)

    def generate_jwt(self):
        now = int(time.time())
        exp = now + 3600
        self.token_expiry = exp

        account_name = self.account.split('.')[0]
        qualified_user = f"{account_name.upper()}.{self.user.upper()}"
        payload = {
            "iss": f"{qualified_user}.{self.public_key_fp}",
            "sub": qualified_user,
            "iat": now,
            "exp": exp
        }
        return jwt.encode(payload, self.private_key, algorithm="RS256")
   
#Runs a SQL query in Snowflake via REST API

    def execute_query(self, sql_statement, csv_path=None):
        self._refresh_session()
        api_url = f"https://{self.account}/api/v2/statements"
        headers = {
            "Authorization": f"Bearer {self.jwt_token}",
            "X-Snowflake-Authorization-Token-Type": "KEYPAIR_JWT",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        request_body = {
            "statement": sql_statement,
            "timeout": 120,
            "warehouse": self.warehouse,
            "database": self.database,
            "schema": self.schema,
            "role": self.role
        }

        response = requests.post(api_url, headers=headers, json=request_body)
        if response.status_code != 200:
            raise Exception(f"Error {response.status_code}: {response.text}")

        result_data = response.json()
        statement_status_url = f"https://{self.account}{result_data.get('statementStatusUrl')}"
        partition_info = result_data['resultSetMetaData'].get('partitionInfo', [])
        all_rows = []
        columns = [col['name'] for col in result_data['resultSetMetaData']['rowType']]

        for partition_id, _ in enumerate(partition_info):
            partition_url = f"{statement_status_url}&partition={partition_id}"
            partition_response = requests.get(partition_url, headers=headers)
            all_rows.extend(partition_response.json().get('data', []))

        df = pd.DataFrame(all_rows, columns=columns)
        if csv_path:
            df.to_csv(csv_path, index=False)

        return df, csv_path
    
#Sends a prompt to a Large Language Model (LLM) inside Snowflake Cortex

    def execute_cortex_inference(self, model, messages, temperature=0.7, max_tokens=4096, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0, stop=None):
        self._refresh_session()
        api_url = f"https://{self.account}/api/v2/cortex/inference:complete"
        headers = {
            "Authorization": f"Bearer {self.jwt_token}",
            "X-Snowflake-Authorization-Token-Type": "KEYPAIR_JWT",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        request_body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_output_tokens": max_tokens
        }
        if top_p != 1.0:
            request_body["top_p"] = top_p
        if frequency_penalty != 0.0:
            request_body["frequency_penalty"] = frequency_penalty
        if presence_penalty != 0.0:
            request_body["presence_penalty"] = presence_penalty
        if stop is not None:
            request_body["stop"] = stop

        with requests.post(api_url, headers=headers, json=request_body, stream=True) as response:
            if response.status_code != 200:
                raise Exception(f"API Error: {response.status_code}: {response.text}")

            full_response = ""
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        json_data = decoded_line.replace("data: ", "")
                        try:
                            json_obj = json.loads(json_data)
                            full_response += json_obj.get('choices', [{}])[0].get('delta', {}).get('content', '')
                        except json.JSONDecodeError:
                            continue

            if not full_response:
                raise Exception("Cortex API returned an empty response.")
            return full_response

# === INCIDENT CLASSIFIER ===

class IncidentClassifier:
    def __init__(self, model="mistral-large2", temperature=0.25, max_tokens=256):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def classify_single_incident(self, sf, incident, labeled_examples):
        prompt = self._build_prompt(incident, labeled_examples)

        response_text = sf.execute_cortex_inference(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        predicted_type = self._parse_llm_response(response_text)
        return {
            "INCIDENT_ID": incident.get("INCIDENT_ID"),
            "TITLE": incident.get("TITLE"),
            "PREDICTED_TYPE": predicted_type
        }

    def _build_prompt(self, new_incident, labeled_examples):
        few_shot = "\n".join(
            f"Title: {ex['TITLE']}\nDescription: {ex['INCIDENT_DESCRIPTION']}\nType: {ex['INCIDENT_TYPE']}\n"
            for ex in labeled_examples
        )
        new_input = f"Title: {new_incident['TITLE']}\nDescription: {new_incident['INCIDENT_DESCRIPTION']}\nType:"
        prompt = f"""
You are a construction safety expert. 
Your task is to classify safety incidents into one of the following three categories based on the title and description.


Categories of Incidents
1. Injury: The incident involves a person getting hurt on the job site in some capacity.
Look for keywords like: injured, hurt, fell, cut, burn, fracture, sprain, hit by object, struck, accident, medical attention.
2. Near Miss: Someone was almost hurt, or a potential hazard was identified.
It is common to see a gas, sewage, electrical, water, or other type of pipe hit during excavation or digging. This is an example of a Near Miss
Look for keywords like: almost, close call, near miss, avoided, potential hazard, gas leak, pipe hit, struck utility, excavation issue.
3. Other: The description is very short or unclear (e.g., single words, empty values, placeholders like "TEST" or "TESTING"). Other examples I want in this category include
theft: Someone stole equipment, tools, or materials from the job site. Damage: Equipment, vehicles, or property was damaged on-site.
Also include anything that does not fit into "Injury" or "Near Miss" into Other.
Look for keywords like: stolen, theft, vandalized, broken, damaged, missing, test, null, empty, parentheses only.


Your output should be just one of the following: Injury, Near Miss, or Other.

Use the examples below to learn the pattern:

{few_shot}

Now classify this new incident:

{new_input}
""".strip()
        return prompt
    
# Makes sure one of the catagories is chosen for the final output

    def _parse_llm_response(self, response_text):
        cleaned = response_text.strip().lower()
        if "injury" in cleaned:
            return "Injury"
        elif "near miss" in cleaned:
            return "Near Miss"
        else:
            return "Other"

# === DATA FETCHING + DRIVER ===

def fetch_data(sf):
    tmp_dir = tempfile.gettempdir()

    labeled_sql = """
        
    WITH labeled AS (
        SELECT * FROM PC_DB.SAFETY_APP.INCIDENTS_VIEW
        WHERE INCIDENT_TYPE IS NOT NULL
        AND INCIDENT_DATE <= '2025-03-01'
    ),
    injury AS (
        SELECT INCIDENT_ID, TITLE, INCIDENT_DESCRIPTION, INCIDENT_TYPE
        FROM labeled
        WHERE INCIDENT_TYPE = 'Injury'
        LIMIT 3
    ),
    near_miss AS (
        SELECT INCIDENT_ID, TITLE, INCIDENT_DESCRIPTION, INCIDENT_TYPE
        FROM labeled
        WHERE INCIDENT_TYPE = 'Near Miss'
        LIMIT 3
    ),
    other AS (
        SELECT INCIDENT_ID, TITLE, INCIDENT_DESCRIPTION, INCIDENT_TYPE
        FROM labeled
        WHERE INCIDENT_TYPE = 'Other'
        LIMIT 3
    )
    SELECT * FROM injury
    UNION ALL
    SELECT * FROM near_miss
    UNION ALL
    SELECT * FROM other

    """
    labeled_path = os.path.join(tmp_dir, "labeled.csv")
    _, labeled_csv = sf.execute_query(labeled_sql, csv_path=labeled_path)
    labeled_df = pd.read_csv(labeled_csv)
    labeled_data = labeled_df.to_dict(orient="records")

    new_sql = """
        SELECT INCIDENT_ID, TITLE, INCIDENT_DESCRIPTION
        FROM PC_DB.SAFETY_APP.INCIDENTS_VIEW
        WHERE INCIDENT_DATE >= '2025-03-01'
        ORDER BY INCIDENT_DATE DESC
        LIMIT 1
    """
    new_path = os.path.join(tmp_dir, "newest.csv")
    _, new_csv = sf.execute_query(new_sql, csv_path=new_path)
    new_df = pd.read_csv(new_csv)
    new_data = new_df.to_dict(orient="records")

    return labeled_data, new_data

def driver():
    print("🔗 Connecting to Snowflake...")
    sf = SnowflakeConnection()

    print("📦 Fetching training data + latest incident...")
    labeled_examples, new_incidents = fetch_data(sf)

    if not new_incidents:
        print("✅ No new incidents since 2025-03-01.")
        return

    latest = new_incidents[0]
    classifier = IncidentClassifier()

    print("🧐 Classifying latest incident...")
    result = classifier.classify_single_incident(sf, latest, labeled_examples)

    print("🔍 Result:")
    print(f"ID: {result['INCIDENT_ID']} | Type: {result['PREDICTED_TYPE']} | Title: {result['TITLE']}")

    print("📄 Final JSON Output:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    driver()