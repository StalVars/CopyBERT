# -*- coding: utf-8 -*-

import falcon

from waitress import serve
from qgen_resource import QGenResource
from phrase_resource import PhraseResource
from qgen_n_phrase_resource import QGenResource as QGenNPhraseResource

api = application = falcon.API()
api.req_options.auto_parse_form_urlencoded = True
#model_name = "models/yahoo-nonfactoid-qg-bert-large-cased"
model_name = "models/baseline-bert-large-cased"

qgen_resource = QGenResource(model_name)

qgen_n_phrase_resource = QGenNPhraseResource(model_name)


api.add_route('/qgen', qgen_resource)

api.add_route('/qgennphr', qgen_n_phrase_resource)

phrase_resource = PhraseResource()
api.add_route('/phrases', phrase_resource)


if __name__=="__main__":
    serve(api, host='localhost', port='1223')
