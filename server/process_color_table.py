from bs4 import BeautifulSoup

import json

html = open('c.html', 'r').read()

s=BeautifulSoup(html, 'html.parser')
table=s.find('table')
#print 'TTTT', table

#print 'BBBB', tbody

count = 0

data={}
data['colours'] = []
for row in table.find_all('tr'):
    #print count, row
    if count > 1:
        #print 'CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC'
        cols = row.find_all('td')
        #print cols
        cols = [ele.text.strip() for ele in cols]
        data['colours'].append({'number': cols[0], 'name': cols[1], 'R': cols[7], 'G': cols[8], 'B': cols[9]})
        #print '{"number"'cols[0], cols[1], cols[7], cols[8], cols[9]
    count = count + 1
#    print 'YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY ', row
#    cols = r.find_all('td')
#    print 'YYYY', cols
print json.dumps(data)