from typing import Literal

import requests
from bs4 import BeautifulSoup    
    
def _translate(origin_text: str) -> str:
    translated = google_translate(origin_text, "auto", "en")
    return translated

def google_translate(
        text: str,
        source: Literal["auto", "en", "ko"],
        target: Literal["en", "ko"],
):
    text = text.strip()
    if not text:
        return ""

    # 크롤링할 주소
    endpoint_url = "https://translate.google.com/m"

    # 요청할 때 전달할 parameter
    params = {
        "h1": source,
        "sl": source,
        "tl": target,
        "q" : text,
        "ie": "UTF-8",
        "prev": "_m",
    }

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/86.0.4240.183 Mobile Safari/537.36"
        ),
    }

    res = requests.get(
        endpoint_url,
        params=params,
        headers=headers,
        timeout=5,
    )
    #응답 상태코드가 200OK가 아니라면, 예외 발생
    res.raise_for_status()

    #응답에서 원하는 값 파싱
    soup = BeautifulSoup(res.text, "html.parser")
    translated_text = soup.select_one(".result-container").text.strip()

    return translated_text