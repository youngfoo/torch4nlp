import requests


rsp = requests.get('https://baike.baidu.com/starmap/view?lemmaTitle=%E5%88%98%E5%BE%B7%E5%8D%8E&lemmaId=114923&pageType=relation&starmapFrom=kg_card_relation')
text = rsp.text

res = text
print(res)