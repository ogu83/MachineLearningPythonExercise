import os
import threading
import binascii
import subprocess
import commands
import hashlib
import numpy as np
from sklearn import preprocessing, model_selection, neighbors, svm
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.linear_model import LinearRegression
style.use('fivethirtyeight')

alphabet = '123456789abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ'
base_count = len(alphabet)

def decode(s):
	""" Decodes the base58-encoded string s into an integer """
	decoded = 0
	multi = 1
	s = s[::-1]
	for char in s:
		decoded += multi * alphabet.index(char)
		multi = multi * base_count
		
	return decoded

def encode(num):
	""" Returns num in a base58-encoded string """
	encode = ''
	
	if (num < 0):
		return ''
	
	while (num >= base_count):	
		mod = num % base_count
		encode = alphabet[mod] + encode
		num = num / base_count

	if (num):
		encode = alphabet[num] + encode

	return "1" + encode

N = 10000
n = 0
Xa = []
while n < N:
    n += 1

    def MyThread1():
        sha256Key = binascii.hexlify(os.urandom(32))
        cmd = '/root/bitcoin-tool/bitcoin-tool --input-type private-key --input-format hex --output-type address --output-format base58check --input ' + sha256Key + ' --public-key-compression uncompressed --network bitcoin'  
        result = commands.getstatusoutput(cmd)
        decodedResult = decode(result[1])
        sha256Int = int(sha256Key, 16)    
        Xa.append([decodedResult,sha256Int])

    t1 = threading.Thread(target=MyThread1, args=[])
    t1.start()    
    t1.join()
    if (n % 100 == 0):
        print(n)    
    #print(sha256Int, decodedResult)
    # if n % 100 == 0:
    #     print(sha256Int, decodedResult)

X = np.array(Xa)
#print(X.head())

#clf = svm.SVC()
clf = KMeans(n_clusters=2)
clf.fit(X)
print(clf.score(X))

# centroids = clf.cluster_centers_
# labels = clf.labels_

# colors = 10*["g.","r.","c.","b.","k."]

# for i in range(len(X)):
#     plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 2)
# plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=15, linewidths=2)

# plt.show()

# plt.scatter(x, y, s=5)
# plt.xlabel('BitCoin Public Keys')
# plt.ylabel('SHA256 Keys')
# plt.show()

#print(binascii.hexlify(os.urandom(32)))
# c = hashlib.sha256("1")
# ch = c.hexdigest()
# shaCmd = '/root/bitcoin-tool/bitcoin-tool --input-type private-key --input-format hex --output-type address --output-format base58check --input ' + ch + ' --public-key-compression uncompressed --network bitcoin'
# shaResult = commands.getstatusoutput(shaCmd)
# dShaResult = decode(shaResult[1])
# chi = int(ch, 16)
# print(chi, dShaResult)
# plt.scatter(chi, dShaResult)
# plt.show()

# print(decode("1EHNa6Q4Jz2uvNExL497mE43ikXhwF6kZm"))
# print(encode(10414491441948160499424555720329880979107174345751581803042))

# print(decode("1LagHJk2FyCV2VzrNHVqg3gYG4TSYwDV4m"))
# print(encode(11877453449768650282840037944548959951253217739798088679134))

# print(decode("1Jkwi4Ybq1dt6KWnrE6WN76s4MYnTngDka"))
# print(encode(11387062010840212440145433949667716374549167325640421289555))