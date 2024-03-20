# -*- coding; utf-8 -*-

import json
import sys

sys.path.append("/raid/data/stva02/qgen_demo")

import identifyphrases


class PhraseResource:
    
    def on_post(self, req, resp):
        body = req.stream.read()
        body = json.loads(body)
        text = body["text"]
        data, starts, ends = identifyphrases.getphrases(text, 100)
        phrases = {
            "words": data["words"], "phrases": data["phrases"], 
            "starts": starts, "ends": ends
        }
       	resp.body = json.dumps(phrases, indent=2, ensure_ascii=False)
