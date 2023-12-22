import requests

url="https://www.eboo.ir/api/ocr/getway"

params = (
    ("Token", ""),
    ("Command", ""),
    ("FileLink", ""),
    ("Token", ""),
)

data = {
    "token": "",
    "command": "",
}

def addfile(filelink):
    data['filelink'] = filelink
    data['command'] = 'addfile'

    res = requests.post(url, data=data)
    print("Addfile Res:", res.json(), flush=True)
    return res.json()

def convert(filetoken):
    data['filetoken'] = filetoken
    data['command'] = 'convert'
    data['method'] = 4
    data['output'] = 'txtraw'

    res = requests.post(url, data=data)
    print("Convert Res:", res.json(), flush=True)
    return res.json()
