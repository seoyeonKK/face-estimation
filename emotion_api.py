# Python script to analyze
# emotion of image
import http.client
import urllib.parse, urllib.error
import simplejson as json
from PIL import Image
import operator
import cv2
import numpy as np
import base64

def get_emotion(imgNum):
	# 서연 key
	subscription_key = 'f4d3237b49004c02b59f7c1c1ec82861'
	headers = {
		'Content-Type': 'application/octet-stream',
		'Ocp-Apim-Subscription-Key': subscription_key,
	}
	params = urllib.parse.urlencode({
		'returnFaceId': True,
		'returnFaceLandmarks': False,
		'returnFaceAttributes': 'emotion'
	})
	# Replace the URL
	# below with the
	# URL of the image
	# you want to analyze.
	with open("thumbnail" + "0" + ".jpg", 'rb') as f:
		image = f.read()

	try:
		# NOTE: You must use the same region in your REST call as you used to obtain your subscription keys.
		# For example, if you obtained your subscription keys from westcentralus, replace "westus" in the
		# URL below with "westcentralus".
		conn = http.client.HTTPSConnection('westus.api.cognitive.microsoft.com')
		conn.request("POST", "/face/v1.0/detect?%s" % params, image, headers)
		response = conn.getresponse()
		data = response.read()
		parsed = json.loads(data)
		dict = parsed[0]['faceAttributes']['emotion']
		print(dict)
		# 가장 높은 감정 출력
		negative = float(dict['anger']) + float(dict['sadness']) + float(dict['contempt']) + float(dict['disgust']) + float(dict['fear'])
		neutral = float(dict['neutral'])
		positive = float(dict['happiness']) + float(dict['surprise'])
		dict_final = {'negative': negative, 'neutral': neutral, 'positive': positive}
		sortedArr = sorted(dict_final.items(), key=operator.itemgetter(1))
		conn.close()
		return sortedArr
	except Exception as e:
		print(e.args)
