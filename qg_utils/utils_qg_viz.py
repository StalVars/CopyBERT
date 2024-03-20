
#from transformers.tokenization.bert import whitespace_tokenize
from transformers.models.bert.tokenization_bert import whitespace_tokenize
import collections
import logging
logger = logging.getLogger(__name__)
import json
from transformers import LayoutLMv3FeatureExtractor
from PIL import Image
import os
import pickle


class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 doc_bboxes,
                 image,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.doc_bboxes = doc_bboxes
        self.image = image
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 bboxes=None,
                 pixel_values=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

        self.bboxes = bboxes
        self.pixel_values = pixel_values 

def read_enc2dec_examples(input_file, is_training, version_2_with_negative):

    if is_training:
      enc_file = input_file+"/from.txt"
      dec_file = input_file+"/to.txt"
    else:
      enc_file = input_file+"/from.dev.txt"
      dec_file = input_file+"/to.dev.txt"

    input_lines=[]
    output_lines=[]
    with open(enc_file, "r", encoding='utf-8') as reader:
        for line in reader:
             line=line.strip()
             input_lines.append(line)
    with open(dec_file, "r", encoding='utf-8') as reader:
        for line in reader:
             line=line.strip()
             output_lines.append(line)


 
    examples=[]
    for idx in range(len(input_lines)):

        start_position = 0 
        end_position = 0 
        orig_answer_text = ""
        is_impossible=False

        if len(output_lines[idx])!=0 and len(input_lines[idx])!=0: #Ensure the input and output are not empty strings
          example = SquadExample(
                    qas_id=idx,
                    question_text=input_lines[idx],
                    doc_tokens=output_lines[idx],
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible)
          examples.append(example)
    return examples

def read_squad_examples_in_txt(input_file, is_training):
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False
    """Read a SQuAD json file into a list of SquadExample."""
    paragraphs=[]
    questions=[]
    positions=[]
    with open(input_file+".source.txt", "r") as f:
         for line in f:
             line=line.strip()
             paragraphs.append(line)
    with open(input_file+".target.txt", "r") as f:
         for line in f:
             line=line.strip()
             questions.append(line)
    with open(input_file+".positions.txt", "r") as f:
         for line in f:
             line=line.strip()
             values=line.split()
             assert len(values) == 2
             values=[int(i) for i in values]
             positions.append(values)

    assert len(paragraphs) == len(questions)
    assert len(paragraphs) == len(positions)
    print("data length:",len(paragraphs))
    examples=[]
    for pi in range(len(paragraphs)):
        paragraph_text=paragraphs[pi]
        question_text=questions[pi]
        answer_offset, end_pos_offset=positions[pi]
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

        
        qas_id = pi
        start_position = None
        end_position = None
        orig_answer_text = None
        is_impossible = False
        if answer_offset != -1:
            orig_answer_text = paragraph_text[answer_offset:end_pos_offset+1]
            answer_length = len(orig_answer_text)
            start_position = char_to_word_offset[answer_offset]
            end_position = char_to_word_offset[answer_offset + answer_length - 1]
            actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
            cleaned_answer_text = " ".join(
                            whitespace_tokenize(orig_answer_text))
            if actual_text.find(cleaned_answer_text) == -1:
                 logger.warning("Could not find answer: '%s' vs. '%s'",
                                    actual_text, cleaned_answer_text)
                 continue

            example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible)
            examples.append(example)


    return examples

def read_squad_examples_with_image_n_layout(input_file, is_training, version_2_with_negative):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    # If there exists a dumped examples file for dev and test separately, load them and ignore further processing:

    
    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            words = paragraph["words"]
            layout = paragraph["layout"]
            image = paragraph["image"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True

            #for c in paragraph_text:
            doc_tokens = []
            paragraph_text = ""
            for wi in range(len(words)):
                word = words[wi]
                doc_tokens.append(word)
                for c in word:
                  char_to_word_offset.append(wi)
                  paragraph_text += c

                paragraph_text += " " 
                char_to_word_offset.append(wi)

            print("Lengths: #char_to_word_offset: %s #paragraph: %s:" %(len(char_to_word_offset), len(paragraph_text)) )

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                print("qas_id:  %s\n#context:%s"%(qas_id, len(paragraph_text)) )
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                if True: #is_training:
                    if version_2_with_negative:
                        is_impossible = qa["is_impossible"]

                    '''
                    if (len(qa["answers"]) != 1) and (not is_impossible):
                        raise ValueError(
                            "For training, each question should have exactly 1 answer.")
                    '''
                    if not is_impossible:
                        #print(qa)
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        reconstructed_answer = paragraph_text[answer_offset:answer_offset+len(orig_answer_text)]
                        print("Original     :%s\nReconstructed:%s" %(orig_answer_text, reconstructed_answer) )
                        print("##############")
                        if answer_offset > len(char_to_word_offset): 
                          print(
                                 "qas-", qas_id,
                                  "offset-", answer_offset, 
                                  "ans length-",answer_length,  
                                  "chars' length < offset",len(char_to_word_offset)
                                  )


                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(
                            whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            logger.warning("Could not find answer: '%s' vs. '%s'",
                                           actual_text, cleaned_answer_text)
                            continue
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    doc_bboxes=layout,
                    image=image,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible)
                examples.append(example)

    #Shuffle 
    if not is_training:
        '''
        if not os.path.exists("dev_examples.pt"): 
          random.shuffle(examples)
          dev_length=len(examples)//2
          dev_examples=examples[:dev_length]
          test_examples=examples[dev_length:]
          with open("dev_examples.pt","wb") as writer:
              pickle.dump((dev_examples,test_examples),writer)
        else:
          with open("dev_examples.pt", "rb") as reader:
                examples = pickle.load(reader)
          dev_examples, test_examples = examples
        print("Dev dataset length:",len(dev_examples))
        print("Test dataset length:",len(test_examples))
        log_file.write("Dev dataset length:"+str(len(dev_examples))+"\n")
        log_file.write("Test dataset length:"+str(len(test_examples))+"\n")
        return dev_examples, test_examples
        '''
        return examples # both dev and test are (same for now ;) )

    # Take half and save 2 files dev and test
    # return dev

    return examples


def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]

def convert_examples_to_features(examples, 
        tokenizer, 
        max_seq_length,
        doc_stride, 
        max_query_length, 
        is_training,
        no_question=False,
        is_generate=False, 
        args=None,
        mode="train",
        cache_file="./cache",
        bucket_size=1000):
    """Loads a data file into a list of `InputBatch`s."""

    feature_extractor = LayoutLMv3FeatureExtractor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    unique_id = 1000000000
    num_out_of_span = 0

    start_token = tokenizer.cls_token 
    sep_token = tokenizer.sep_token 
    mask_token = tokenizer.mask_token
    print("start_token:%s\n, sep_token:%s\n, mask_token:%s\n"%(start_token, sep_token, mask_token))
    #input("")

    #mask_token = "[MASK]"



    # Load image sizes:
    image_sizes_file = os.path.dirname(args.train_file) + "/"+ mode +"/documents/sizes.pkl"
    with open(image_sizes_file,"rb") as f:
        image_sizes = pickle.load(f)



    # split examples into (threshold) buckets each and store them as features

    idx=0
    example_buckets=[]

    while idx <len(examples):
      '''
      # len(examples)=3560
      # idx=0;     1000 (idx+bucket_size=1000)
      # idx=1000:  2000(idx+bucket_size=2000)
      # idx=2000:  3000(idx+bucket_size=3000)
      # idx=3000:  4000(idx+bucket_size=4000)
      # 
      '''

      #print("idx:%s\nidx+bucket_size:%s\n#len:%s"%(idx,idx+bucket_size,len(examples)))
      #input("-")

      example_buckets.append(examples[idx:idx+bucket_size])
      idx=idx+bucket_size

    for bi in range(len(example_buckets)):

     feature_file = cache_file+"."+str(bi)
     print("feature_file:", feature_file)

     if os.path.isfile(feature_file):

          with open(feature_file,"rb") as f:
            features = pickle.load(f)
          yield features
     else:

      examples_bucket=example_buckets[bi]

      print("length of examples:", len(examples_bucket))

      features = []
      for (example_index, example) in enumerate(examples_bucket):

        query_tokens = tokenizer.tokenize(example.question_text)
        #print(query_tokens)
        #input("query tokens")

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        all_doc_bboxes = []

        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            bbox = example.doc_bboxes[i]

            if token != mask_token:
              sub_tokens = tokenizer.tokenize(token)
            else:
              sub_tokens = [token]

            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)
                all_doc_bboxes.append(bbox)

        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        if no_question:
          #max_tokens_for_doc = max_seq_length - max_query_length - 3 # fixing max question length to 20 tokens
          max_tokens_for_doc = max_seq_length - 30 - 3 # fixing max question length to 30 tokens
        else:
          max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        ans_flag = False
        if is_generate:


         # Move start_offset slightly if start/end position fall in 2 different partitions
         change_offset_flag=False
         if tok_start_position // max_tokens_for_doc  != tok_end_position // max_tokens_for_doc:
             end_position_offset = tok_end_position % max_tokens_for_doc
             start_offset = start_offset + end_position_offset + 1 
             #print("Changing start offset from 0 to ", start_offset)
             change_offset_flag=True
             #input("-")


         while start_offset < len(all_doc_tokens):

            length = len(all_doc_tokens) - start_offset

            if length > max_tokens_for_doc:
                length = max_tokens_for_doc

            #print(start_offset, length, len(all_doc_tokens))
            #if start_offset + length == len(all_doc_tokens):
            #    break

            #end_offset = min(length, doc_stride)
            end_offset = length

            if start_offset != 0 or True:

              #print("##")
              #print( "example_index: %s\nstart_offset:%s\nlength left:%s\nmax_tokens:%s\nend_off_set:%s\n" % ( example_index, start_offset, 
              #    len(all_doc_tokens)-start_offset, 
              #    max_tokens_for_doc,  
              #    start_offset + end_offset) )
              #print("start_pos:%s, end_pos:%s"%(tok_start_position, tok_end_position))

              pass

            if tok_start_position >=start_offset and tok_end_position < start_offset + end_offset:
              doc_spans.append(_DocSpan(start=start_offset, length=length))
              ans_flag=True
              #input("-")
              break
            start_offset += min(length, doc_stride)

         if not ans_flag:
            print("ans_flag:", ans_flag)
            input("")
            #pass

         '''
         length = len(all_doc_tokens) - start_offset
         print("start_offset:", start_offset, 
                 "#all_doc_tokens:", len(all_doc_tokens), 
                 "#max_tokens_for_doc:", max_tokens_for_doc)
         out_of_span_in_gen = False
         if length > max_tokens_for_doc:
                print("length > max_tokens_for_doc")
                length = max_tokens_for_doc
                if not (tok_start_position >=start_offset and tok_end_position < start_offset + length):
                   out_of_span_in_gen = True
                   num_out_of_span += 1
                   start_offset = len(all_doc_tokens) - length
         doc_spans.append(_DocSpan(start=start_offset, length=length))
         '''

        else:

         while start_offset < len(all_doc_tokens):

            length = len(all_doc_tokens) - start_offset

            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))

            if start_offset + length == len(all_doc_tokens):
                break

            start_offset += min(length, doc_stride)

        imagename = example.image.split("/")[-1]

        if imagename in image_sizes.keys():
           #print("imagename:", imagename)
           width=image_sizes[imagename][0]
           height=image_sizes[imagename][1]
           #print("lno 573:", width, height)
        else:
           print("Can't find ", imagename, " in the keys of sizes.pkl file")
           print("image_sizes file:", image_sizes_file)
           input("-")

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            bboxes = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            #tokens.append("[CLS]")
            tokens.append(start_token) 
            segment_ids.append(0)
            bboxes.append([0,0,0,0])

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])

                #normalize

                normalized_bbox=normalize_bbox(all_doc_bboxes[split_token_index], width, height)
                #bboxes.append(all_doc_bboxes[split_token_index])

                bboxes.append(normalized_bbox) 
                segment_ids.append(0)

            #tokens.append("[MASK]")
            tokens.append(mask_token) 
            bboxes.append([0,0,0,0])
            segment_ids.append(1) 

            # Question to come after the context
            if not no_question:
             for token in query_tokens:
                tokens.append(token)
                segment_ids.append(1)
                bboxes.append([0,0,0,0])
             #tokens.append("[SEP]")
             tokens.append(sep_token)
             bboxes.append([0,0,0,0])
             segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            if no_question:
             for token in query_tokens:
                tokens.append(token)
                #bboxes.append([0,0,0,0])
             #tokens.append("[SEP]")
             tokens.append(sep_token)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                bboxes.append([0,0,0,0])

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(bboxes) == max_seq_length

            #if len(bboxes) != max_seq_length:
            #  print("#bboxes:", len(bboxes), max_seq_length)

            start_position = None
            end_position = None
            if not example.is_impossible and is_training:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                    num_out_of_span += 1
                else:
                    #doc_offset = len(query_tokens) + 2
                    doc_offset =  1 #Just [CLS] token
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset
            if is_training and example.is_impossible:
                start_position = 0
                end_position = 0
            if example_index < 20:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % " ".join(tokens))
                logger.info("token_to_orig_map: %s" % " ".join([
                    "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                logger.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                ]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training and example.is_impossible:
                    logger.info("impossible example")
                if is_training and not example.is_impossible:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    logger.info("start_position: %d" % (start_position))
                    logger.info("end_position: %d" % (end_position))
                    logger.info(
                        "answer: %s" % (answer_text))

             

            image_path = os.path.dirname(args.train_file) + "/"+ mode +"/"+ example.image
            image = Image.open(image_path).convert("RGB")
            vis_features = feature_extractor(images=image, return_tensors='np')["pixel_values"][0]
            pixel_values = vis_features.tolist()


            if (ans_flag and is_generate) or (not out_of_span and not is_generate):
             features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_position=start_position,
                    end_position=end_position,
                    bboxes = bboxes,
                    pixel_values = pixel_values,
                    is_impossible=example.is_impossible))

             #features.append(
             #print("Length of features:", len(features))


            unique_id += 1

      print("Number of out of spans:", num_out_of_span)

      # Save features:
      with open(feature_file,"wb") as f:
          pickle.dump(features,f)
      yield features


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index
