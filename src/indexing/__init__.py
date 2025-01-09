import os

from opensearchpy import OpenSearch

host = os.getenv('OPENSEARCH_HOST')
auth = (os.getenv('OPENSEARCH_USER'), os.getenv('OPENSEARCH_PASSWORD'))

client = OpenSearch(
    hosts=host,
    http_auth=auth,
    use_ssl=True,
    verify_certs=True
)

info = client.info()
print(f"Welcome to {info['cluster_name']} {info['version']['number']}!")
print(f"Searching for image captions from Uncle Sam...")

index_name = "hackaton_gmkg_image_caption"
query = {
	"query": {
		"match": {
			"captions": "uncle sam"
		}
	},
	"highlight": {
		"fields": {
			"captions": {
				"fragment_size": 3,
				"pre_tags": [
					""
				],
				"post_tags": [
					""
				]
			}
		}
	}
}

response = client.search(
    body = query,
    index = index_name
)

print(f"Found {response['hits']['total']['value']} record(s)!")
print(response)