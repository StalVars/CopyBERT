# -*- coding: utf-8 -*-

import falcon

from waitress import serve
from qgen_resource import QGenResource
from phrase_resource import PhraseResource

api = application = falcon.API()
api.req_options.auto_parse_form_urlencoded = True
qgen_resource = QGenResource()
api.add_route('/qgen', qgen_resource)
phrase_resource = PhraseResource()
api.add_route('/phrases', phrase_resource)


if __name__=="__main__":
    serve(api, host='exs-91203.sb.dfki.de', port='1223')
