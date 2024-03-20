#!/usr/bin/env pythlon3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool to get dense results and validate them
"""

import requests
from io import BytesIO
import pickle
import argparse


def main():


    # create ArgumentParser object
    parser = argparse.ArgumentParser(description='argparse script')

    # add arguments
    parser.add_argument('--hostname', type=str, required=True, help='hostname')

    # parse arguments
    args = parser.parse_args()

    ''' Initialize Dense Retriever '''
    context="We already discussed this at some point so sometimes we want to override positions of the code and we brought this up in the context of decoding for neural models so why do we want to do this I think the most common one is terminology..People have terminology lists often translate as work with terminology lists that says in our corpus we use this word and that's the word you're gonna use..For example maybe you have equal probability to produce output like liberty and freedom and they're both really mean the same thing but maybe one client says no no we have to talk about freedom that's the word we want to use..So I have a German word like fire height which might translate to either of these don't let the new machine translation system make its decision always go with freedom so you want to override the new machine translation decisions how to translate words..So what I also have is a rule based component that deals with numbers and quantities that for instance translets 2.5 cm into one inch as part of the translation process and we just have that's very easy to do with the rule based component and then we just have to make sure that the new machine translation system follows that so we want to want to use that as well..So we talked about this as I said in the decoding session so once method that was developed for statistical machine translation was a smartgap scheme which says can you translate this word here.."
    post_dict = {"context": context,  "doc_ids":[], "n_docs":5}
    post_dict_pkl = pickle.dumps(post_dict)

    headers = {"content-type": "application/json"}
    #http://192.168.33.214:1755
    url = "http://" +args.hostname+":1755"

    #          headers=headers

    resp = requests.post(
                f'{url}/generate',
                data=post_dict_pkl
            )

    resp_content = pickle.loads(resp.content)

    print(resp_content)

if __name__ == '__main__':
    main()
