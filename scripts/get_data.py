import requests
import json
from pyyoutube import Api
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import os


os.environ["MLFLOW_REGISTRY_URI"] = "/home/ml-srv-admin/xflow_project/mlflow/"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("get_data")

key = 'AIzaSyCHgIuogiRoolZaQ9AEm82DeB_04qveUM4'

api = Api(api_key=key)

query = "'rickroll'"
video = api.search_by_keywords(q=query, search_type=["video"], count=100, limit=30)

maxResults = 100
nextPageToken = ""
s = 0

with mlflow.start_run():
    dct = {}
    for id_ in [x.id.videoId for x in video.items]:
        uri = "https://www.googleapis.com/youtube/v3/commentThreads?" + \
                "key={}&textFormat=plainText&" + \
                "part=snippet&" + \
                "videoId={}&" + \
                "maxResults={}&" + \
                "pageToken={}"
        uri = uri.format(key, id_, maxResults, nextPageToken)
        content = requests.get(uri).text
        data = json.loads(content)
        c = 0
        for item in data['items']:
            c += 1
            channel_id = item['snippet']['topLevelComment']['snippet']['channelId']
            num_likes = item['snippet']['topLevelComment']['snippet']['likeCount']
            if channel_id in dct.keys():
                dct[channel_id] += num_likes
            else:
                dct[channel_id] = num_likes

    df = pd.DataFrame.from_dict(dct, orient='index')
    df = df.reset_index()
    df.rename(columns = {0:'counts', 'index':'id'}, inplace = True)

    mlflow.log_artifact(local_path="/home/ml-srv-admin/xflow_project/scripts/get_data.py",
                        artifact_path="get_data code")
    mlflow.end_run()

df.to_csv('/home/ml-srv-admin/xflow_project/datasets/data.csv', index=False)
