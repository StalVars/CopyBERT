# -*- coding: utf-8 -*-

import falcon

from waitress import serve
from qgen_resource import QGenResource
#from phrase_resource import PhraseResource

api = application = falcon.API()
api.req_options.auto_parse_form_urlencoded = True
model_name = "models/yahoo-nonfactoid-qg-bert-large-cased"
qgen_resource = QGenResource(model_name)
api.add_route('/qgen', qgen_resource)


if __name__=="__main__":
    serve(api, host='localhost', port='1223')
