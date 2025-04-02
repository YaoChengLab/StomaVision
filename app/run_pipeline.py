import requests
import json


def run_stomata_detection(api_token, input, org_uid=None):
    """
    Calls the stomata detection pipeline and returns the response.

    Parameters:
    - api_token (str): The API token for authorization.
    - input (str): base64 encoded image
    - org_uid (str): The organization UID.

    Returns:
    - str: The detected objects from the API.
    """

    # Define the headers
    headers = {"Content-Type": "application/json"}

    # Add the API token to the headers
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    # Add the organization UID to the headers if provided
    if org_uid:
        headers["Instill-Requester-Uid"] = org_uid

    url = "http://localhost:8080/v1beta/users/admin/pipelines/stomavision/trigger"

    # Define the payload
    payload = {"inputs": [{"input": input}]}

    # Make the POST request
    response = requests.post(
        url, headers=headers, data=json.dumps(payload), timeout=600
    )

    # Check if the request was successful
    if response.status_code == 200:
        response_json = response.json()
        if "objects" in response_json["outputs"][0]:
            return response_json["outputs"][0]["objects"]
        else:
            print("The response JSON does not contain the 'objects' field.")
            return {}
    else:
        print(response.json())
        return {}


def run_stomata_detection_async(api_token, input, org_uid=None):
    """
    Calls the stomata detection pipeline and returns the response.

    Parameters:
    - api_token (str): The API token for authorization.
    - input (str): base64 encoded image
    - org_uid (str): The organization UID.

    Returns:
    - json (or None): The response from the API.
    """

    # Define the headers
    headers = {"Content-Type": "application/json"}

    # Add the API token to the headers
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    # Add the organization UID to the headers if provided
    if org_uid:
        headers["Instill-Requester-Uid"] = org_uid

    url = "http://localhost:8080/v1beta/users/admin/pipelines/stomavision/triggerAsync"

    # Define the payload
    payload = {"inputs": [{"input": input}]}

    # Make the POST request
    response = requests.post(
        url, headers=headers, data=json.dumps(payload), timeout=600
    )

    # Check if the request was successful
    if response.status_code == 200:
        response_json = response.json()
        return response_json
    else:
        print(response.json())
        return None


# Retrieve a long-running operation


def get_operation(api_token, op_name, org_uid=None):
    """
    Retrieves the status of a long-running operation.

    Parameters:
    - api_token (str): The API token for authorization.
    - op_name (str): The name of the operation. It should be a resource name ending with `operations/{unique_id}`.
    - org_uid (str): The organization UID.

    Returns:
    - json (or None): The response from the API.

    """
    # Define the headers
    headers = {"Content-Type": "application/json"}

    # Add the API token to the headers
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    # Add the organization UID to the headers if provided
    if org_uid:
        headers["Instill-Requester-Uid"] = org_uid

    url = f"http://localhost:8080/v1beta/{op_name}"

    # Make the POST request
    response = requests.get(url, headers=headers, timeout=60)

    # Check if the request was successful
    if response.status_code == 200:
        response_json = response.json()
        return response_json
    else:
        print(response.json())
        return None
