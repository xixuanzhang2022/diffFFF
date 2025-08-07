import requests
import os
import json
import time
from collections import defaultdict
import pandas as pd
from bs4 import BeautifulSoup

BEARER_TOKEN = os.environ.get("BEARER_TOKEN")

search_url = "https://api.twitter.com/2/tweets/search/all"

# Time range and search configuration
st = '2019-03-18T00:00:00Z'
et = '2019-03-19T00:00:00Z'
mx = '500'

query_params0 = {
    'query': '#FridaysForFuture lang:de',
    'expansions': 'author_id',
    'max_results': mx,
    'tweet.fields': 'id,created_at,author_id,text,referenced_tweets',
    'user.fields': 'username',
    'start_time': st,
    'end_time': et
}

def create_headers(token):
    return {"Authorization": f"Bearer {token}"}

def connect_to_endpoint(url, headers, params):
    response = requests.get(url, headers=headers, params=params)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response

def main():
    headers = create_headers(BEARER_TOKEN)
    query_params = query_params0
    i = 0

    while True:
        ns = []
        tweet_lst = []
        user_dict = defaultdict()

        json_response = connect_to_endpoint(search_url, headers, query_params)
        soup = BeautifulSoup(json_response.text, 'html.parser')
        js = json.loads(str(soup))
        js2 = js['meta']

        for tweet in js.get('data', []):
            if 'referenced_tweets' in tweet:
                for ref in tweet['referenced_tweets']:
                    reftype = ref.get('type', '')
                    refid = ref.get('id', '')
                tweet_lst.append([tweet['created_at'], tweet['id'], tweet['author_id'], tweet['text'], reftype, refid])
            else:
                tweet_lst.append([tweet['created_at'], tweet['id'], tweet['author_id'], tweet['text'], '', ''])

        for user in js.get('includes', {}).get('users', []):
            user_dict[user['id']] = user['username']

        time.sleep(3)

        users_df = pd.DataFrame(user_dict.items(), columns=['id', 'username'])
        tweet_df = pd.DataFrame(tweet_lst, columns=['time', 'id', 'author_id', 'text', 'ref_type', 'ref_id'])

        tweet_df.to_csv('tweet_df.csv', mode='a', header=False, index=False)
        users_df.to_csv('users_df.csv', mode='a', header=False, index=False)

        i += 1
        if 'next_token' in js2:
            next_t = js2['next_token']
            pd.DataFrame([next_t]).to_csv('ns_df.csv', mode='a', header=False, index=False)
            query_params = {
                'query': '#FridaysForFuture lang:de',
                'expansions': 'author_id',
                'next_token': next_t,
                'max_results': mx,
                'tweet.fields': 'id,created_at,author_id,text,referenced_tweets',
                'user.fields': 'username',
                'start_time': st,
                'end_time': et
            }
            print(i, 'continue')
            print(json.dumps(js, indent=4, sort_keys=True))
        else:
            print(json.dumps(js, indent=4, sort_keys=True))
            print("end")
            break

if __name__ == "__main__":
    main()
