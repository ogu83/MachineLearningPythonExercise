import requests
import json
import prettytable
headers = {'Content-type': 'application/json'}
data = json.dumps({"seriesid": ['CUUR0000SA0', 'CUSR0000SA0', 'CUUR0000SA0L1E'], "startyear":"2001", "endyear":"2018"})
p = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data=data, headers=headers)
json_data = json.loads(p.text)
#print(json_data)

for series in json_data['Results']['series']:
    x=prettytable.PrettyTable(["series id","year","period","value"])
    seriesId = series['seriesID']
    for item in series['data']:
        year = item['year']
        period = item['period']
        value = item['value']
                
        if 'M01' <= period <= 'M12':
            x.add_row([seriesId,year,period,value])
                
    output = open(seriesId + '.txt','w')
    output.write (x.get_string())
    output.close()
