import requests

url="https://www.eboo.ir/api/ocr/getway"

params = (
    ("Token", ""),
    ("Command", ""),
    ("FileLink", ""),
    ("Token", ""),
)

data = {
    "token": "eboo_token"
}

def addfile(filelink):
    data['filelink'] = filelink
    data['command'] = 'addfile'

    res = requests.post(url, data=data, timeout=60)
    print("Addfile Res:", res.json(), flush=True)
    return res.json()

def convert(filetoken):
    data['filetoken'] = filetoken
    data['command'] = 'convert'
    data['method'] = 4
    data['output'] = 'txtraw'

    res = requests.post(url, data=data, timeout=60)
    print("Convert Res Status_Code:", res.status_code, flush=True)
    return res.text
