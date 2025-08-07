import requests
import os
import json
import time
import pandas as pd

BEARER_TOKEN = os.environ.get("BEARER_TOKEN")


def create_url(user_id):
    return f"https://api.twitter.com/2/users/{user_id}/following"
    # For followers instead: return f"https://api.twitter.com/2/users/{user_id}/followers"

def get_params():
    return {"user.fields": "created_at", "max_results": "1000"}

def get_params_more(next_token):
    return {"user.fields": "created_at", "max_results": "1000", "pagination_token": next_token}

def create_headers(token):
    return {"Authorization": f"Bearer {token}"}

def connect_to_endpoint(url, headers, params):
    response = requests.get(url, headers=headers, params=params)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(f"Request returned an error: {response.status_code} {response.text}")
    return response.json()

def main():
    headers = create_headers(BEARER_TOKEN)

    users = pd.read_csv('users_df.csv', header=None)
    users.columns = ['ind', 'id', 'name']
    users = users.drop_duplicates(subset=['id'])['id'].tolist()

    ns_users = pd.read_csv('fol_ns_df.csv', header=None)
    ns_users.columns = ['id', 'ns']
    resume_user = ns_users['id'].iloc[-1]
    resume_ns = ns_users['ns'].iloc[-1]
    j = users.index(resume_user)
    i = 0
    params = get_params_more(resume_ns)

    print(f"Resuming at index {j}")

    while True:
        url = create_url(resume_user)
        json_response = connect_to_endpoint(url, headers, params)
        js2 = json_response.get('meta', {})

        flist = [[resume_user, ud['id']] for ud in json_response.get('data', [])]
        pd.DataFrame(flist, columns=['user', 'fol_id']).to_csv('following_df.csv', mode='a', header=False, index=False)

        i += 1
        if 'next_token' in js2:
            next_t = js2['next_token']
            pd.DataFrame([[resume_user, next_t]], columns=['user', 'token']).to_csv('fol_ns_df.csv', mode='a', header=False, index=False)
            params = get_params_more(next_t)
            print(json.dumps(json_response, indent=2))
            print(f"{i} continue - user {j}")
        else:
            print(json.dumps(json_response, indent=2))
            print(f"End of user {j}")
            break

        time.sleep(60)

    j += 1
    users = users[j:]

    for k, user in enumerate(users):
        params = get_params()
        i = 0
        notend = True

        while notend:
            url = create_url(user)
            json_response = connect_to_endpoint(url, headers, params)

            if 'data' not in json_response:
                pd.DataFrame([[user, 0]], columns=['user', 'fol_id']).to_csv('following_df.csv', mode='a', header=False, index=False)
                print(json.dumps(json_response, indent=2))
                notend = False
                print(f"End of user {k + j}")
            else:
                js2 = json_response['meta']
                flist = [[user, ud['id']] for ud in json_response['data']]
                pd.DataFrame(flist, columns=['user', 'fol_id']).to_csv('following_df.csv', mode='a', index=False)

                i += 1
                if 'next_token' in js2:
                    next_t = js2['next_token']
                    pd.DataFrame([[user, next_t]], columns=['user', 'token']).to_csv('fol_ns_df.csv', mode='a', header=False, index=False)
                    params = get_params_more(next_t)
                    print(json.dumps(json_response, indent=2))
                    print(f"{i} continue - user {k + j}")
                else:
                    print(json.dumps(json_response, indent=2))
                    notend = False
                    print(f"End of user {k + j}")

            time.sleep(60)

        print(f"Start next user {k + j + 1}")

    print("Finished")

if __name__ == "__main__":
    main()
