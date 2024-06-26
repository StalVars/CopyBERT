import torch.nn as nn
import torch
from typing import Optional, Union, List, Tuple
from transformers.models.bart.modeling_bart import shift_tokens_right, BartDecoder, BartConfig
#from transformers.models.layoutlmv3.modeling_layoutlmv3 import (
from qg_modules.modeling_layoutlmv3 import (
    LayoutLMv3PreTrainedModel,
    LayoutLMv3Encoder,
    LayoutLMv3Layer,
    LayoutLMv3Model,
    LayoutLMv3PatchEmbeddings,
    LayoutLMv3Config)

from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaLayer, RobertaConfig
from transformers.modeling_outputs import Seq2SeqLMOutput, Seq2SeqModelOutput, BaseModelOutputWithPastAndCrossAttentions
from transformers.utils import logging
from torch.nn import CrossEntropyLoss
from torch.nn import CrossEntropyLoss, NLLLoss
from transformers import AutoTokenizer

import sys

logger = logging.get_logger(__name__)
#sys.path.append("./")


class LayoutLMv3TransformerModel(LayoutLMv3PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.encoder = LayoutLMv3Model(config)
        self.embeddings = self.encoder.embeddings
        # roberta_config = RobertaConfig.from_pretrained('roberta-base')
        # self.decoder = RobertaDecoder(roberta_config, self.encoder.embeddings)
        bart_config = BartConfig.from_pretrained('facebook/bart-base')
        self.decoder = BartDecoder(bart_config, self.encoder.embeddings.word_embeddings)
        # self.init_weights()

    def get_input_embeddings(self):
        return self.encoder.embeddings.word_embeddings

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def set_input_embeddings(self, value):
        self.encoder.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        past_key_values=None,
        cross_attn_head_mask=None,
        decoder_inputs_embeds=None,
        inputs_embeds=None,
        pixel_values=None,
        output_attentions=None,
        output_hidden_states=None,
        use_cache: Optional[bool] = None,
        return_dict=None,
        **kwargs
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                bbox=bbox,
                pixel_values=pixel_values,
            )
        encoder_hidden_states = encoder_outputs[0]
        batch_size, seq_len, _ = encoder_hidden_states.size()
        visual_attention_mask = torch.ones(
            (batch_size, seq_len - attention_mask.size(1)), dtype=torch.long, device=encoder_hidden_states.device
        )
        updated_attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=1)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=updated_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs
        
        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class LayoutLMv3ForQuestionGeneration(LayoutLMv3PreTrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head.weight"]

    def __init__(self, config: LayoutLMv3Config):
        super().__init__(config)
        self.layoutlmv3 = LayoutLMv3Model(config) #TransformerModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.layoutlmv3.embeddings.word_embeddings.num_embeddings)))
        self.lm_head = nn.Linear(config.hidden_size, self.layoutlmv3.embeddings.word_embeddings.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        self.attentionsquare = nn.Linear(config.hidden_size, config.num_attention_heads * config.num_hidden_layers ) # config.n_heads * config.n_layers
        self.attentionsquare_lateral = nn.Linear(config.hidden_size, config.num_attention_heads ) # config.n_heads * config.n_layers
        self.attention_over_attention = nn.Linear(config.hidden_size, 3 ) # 
        self.attention_over_attention2 = nn.Linear(config.hidden_size, 2 ) # 
        self.linear_copy_probs = nn.Linear(config.hidden_size, config.hidden_size ) # config.n_heads * config.n_layers

        self.p_value = nn.Linear(config.hidden_size, 1)

    def get_copy_probs_with_bilinearmat(self,sequence_output, self_attentions,src_seq, attention):
        copy_probs = torch.zeros(src_seq.size(0), src_seq.size(1), self.config.vocab_size).to(sequence_output.device)

        #copy_probs.to(self.dev)
        # attention: b x seq x seq
        # copy_probs b x seq x vocab
        #print("attn sum:", attention.sum(-1))

        copy_probs.scatter_(2,src_seq.unsqueeze(2), 1) #scatters vocab
        copy_probs = torch.bmm(attention, copy_probs) # copy scores

        # [0.2(a) 0.6(boy) 0.2(a)] 
        # b x seq x seq  * b x seq x vocab

        #print("copy probs:", copy_probs.sum(-1))

        return copy_probs

    def get_copy_probs(self,sequence_output, self_attentions,src_seq):


        seq_len = src_seq.size(1)
        #print(seq_len, sequence_output.size())
        #print("attn size:", self_attentions[0].size())

        #sequence_output = sequence_output[:,:seq_len,:]

        # Use sequence_output to choose the self_attentions wisely
        attention_square = self.attentionsquare(sequence_output) # batch_size x seq_size x (12*12)
        attention_square = torch.softmax(attention_square,dim=-1)



        attention_square = attention_square.unsqueeze(2) #batch_size x seq_size x 1 x (12*12)

        print("attn square:  sum ", attention_square.sum(-1))

        #print("attention square:", attention_square.size())
        self_attentions = torch.cat(self_attentions,dim=1) # batch_size x (12*12) x seq_size x seq_size
        #print("self attentions size:", self_attentions.size())


        # cut the visual attentions
        self_attentions = self_attentions[:,:,:seq_len, :seq_len]
        #print("self attentions size:", self_attentions.size())

        #print("self attentions:", self_attentions.size())
        #input("")

        self_attentions = self_attentions.transpose(1,2) # batch_size x seq_size x (12*12) x seq_size

        # The below does weighted average over all the attentions of all the layers and heads
        attention = torch.matmul(attention_square,self_attentions).squeeze(2) # batch_size x seq_size x seq_size

        #print("attention:", attention.sum(-1))

        # attention value doesn't sum up to 1? why?
        # src_seq has the source vocab ids

        copy_probs = torch.zeros(src_seq.size(0), src_seq.size(1), self.config.vocab_size).to(sequence_output.device)
        

        #copy_probs.to(self.dev)
        #print("src seq:", src_seq)
        print("attn: sum ", attention.sum(-1))

        copy_probs.scatter_(2,src_seq.unsqueeze(2), 1) #scatters vocab
        copy_probs = torch.bmm(attention, copy_probs) # copy scores

        print("copy probs: sum ", copy_probs.sum(-1))

        return copy_probs


    def get_copy_probs_corrupted(self,sequence_output, self_attentions,src_seq):


        seq_len = src_seq.size(1)

        #sequence_output = sequence_output[:,:seq_len,:]

        ''' 
        '''

        # Uncomment for attention square of all layers:
        # Use sequence_output to choose the self_attentions wisely
        attention_square = self.attentionsquare(sequence_output) # batch_size x seq_size x (12*12)
        attention_square = torch.softmax(attention_square,dim=-1)

        #print("Attention square size:", attention_square.size())
        #print("attention_square max:", torch.max(attention_square))


        attention_square = attention_square.unsqueeze(2) #batch_size x seq_size x 1 x (12*12)

        self_attentions = torch.cat(self_attentions,dim=1) # batch_size x (12*12) x seq_size x seq_size

        # cut the visual attentions
        self_attentions = self_attentions[:,:,:seq_len, :seq_len]
        #self_attentions = torch.cat(self_attentions,dim=1) # batch_size x (12*12) x seq_size x seq_size
        self_attentions = self_attentions.transpose(1,2) # batch_size x seq_size x (12*12) x seq_size
        # The below does weighted average over all the attentions of all the layers and heads
        attention = torch.matmul(attention_square,self_attentions).squeeze(2) # batch_size x seq_size x seq_size

        '''
        # Uncomment for attention square last layer:
        attention_square = self.attentionsquare_lateral(sequence_output) # batch_size x seq_size x (12)
        attention_square = torch.softmax(attention_square,dim=-1)

        #attention_square = attention_square + 1e-4 # to avoid nan with vanishing values
        #attention_square.masked_fill((1-extra_mask).bool(), -1e4) #= copy_scores + (1-extra_mask) * -1e9

        last_layer_attention = self_attentions[-1] # b x 12 x seq x seq
        last_layer_attention = last_layer_attention[:,:,:seq_len, :seq_len]
        last_layer_attention = last_layer_attention.transpose(-2,-3) # b x seq x 12 x seq 

        attention_square = attention_square.unsqueeze(2) #batch_size x seq_size x 1 x (12)
        attention = torch.matmul(attention_square,last_layer_attention).squeeze(2) # batch_size x seq_size x seq_size
        '''


        # attention value doesn't sum up to 1? why?
        # src_seq has the source vocab ids

        copy_probs = torch.zeros(src_seq.size(0), src_seq.size(1), self.config.vocab_size).to(sequence_output.device)

        #copy_probs.to(self.dev)

        # attention: b x seq x seq
        # copy_probs b x seq x vocab

        copy_probs.scatter_(2,src_seq.unsqueeze(2), 1) #scatters vocab
        copy_probs = torch.bmm(attention, copy_probs) # copy scores
        #print("copy probs:", copy_probs.sum(-1))

        return copy_probs

    def prepare_diag_mask_n_target_ids(self, input_ids, token_type_ids, attention_mask):
        # When the order of the sequence is paragraph + question


        question_mask_ids = (token_type_ids == 1).float()

        # For Question Generation use below line
        
        prediction_ids = (input_ids * question_mask_ids.long())
        prediction_ids[:,0]=0 #self.PADID
        prediction_ids = torch.cat((prediction_ids[:,1:],prediction_ids[:,0:1]),-1) # batch_size x seq_length

        question_mask_ids = question_mask_ids.unsqueeze(-1)

        #return None, prediction_ids
        # Is below necessary?

        #Mask begin of question
        input_mask_ids = attention_mask.unsqueeze(-1).float().to(input_ids.device)
        #para_mask_ids = (1-question_mask_ids)*input_mask_ids

        # Question 2D mask
        # question_mask = torch.bmm(question_mask_ids, question_mask_ids.transpose(1,2)) * torch.tril(torch.ones(question_mask_ids.size(1),question_mask_ids.size(1))).unsqueeze(0)#.cuda()
        question_mask = torch.bmm(question_mask_ids, question_mask_ids.transpose(1,2)) * torch.tril(torch.ones(question_mask_ids.size(1),question_mask_ids.size(1))).unsqueeze(0).to(input_ids.device)

        para_mask = torch.bmm((1-question_mask_ids)*input_mask_ids,
                ((1-question_mask_ids)*input_mask_ids).transpose(1,2))

        question2para_mask = torch.bmm(input_mask_ids,
                ((1-question_mask_ids)*input_mask_ids).transpose(1,2))

        extra_mask = torch.bmm(input_mask_ids,
                input_mask_ids.transpose(1,2))
        extra_mask = extra_mask * torch.tril(torch.zeros_like(extra_mask[0]) + 1 )

        extra_mask = extra_mask + para_mask
        extra_mask = (extra_mask != 0).long()

        return extra_mask, prediction_ids
        '''
        '''
    
    def get_encoder(self):
        return self.layoutlmv3.get_encoder()

    def get_decoder(self):
        return self.layoutlmv3.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_input_embeddings(self):
        return self.layoutlmv3.get_input_embeddings()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        token_type_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        bbox: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        is_train: bool = True,
        start_positions:int = 0,
        end_positions:int = 0,
        consider_ans_pos = False,
        copy_probs = False,
        **kwargs,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        if not is_train:
            return self.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values,
                use_cache=True,
                return_dict=return_dict,
                **kwargs,
            )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        extra_mask, prediction_ids = self.prepare_diag_mask_n_target_ids(input_ids, token_type_ids, attention_mask)
        #print("prediction ids", prediction_ids)

        # answer phrase ids
        #print("start, end:", start_positions, end_positions)
        start_positions = start_positions.long()
        end_positions = end_positions.long()
        #print(start_positions, end_positions)
        para_len = (attention_mask * (1-token_type_ids)).sum().item()
        #print("para_len:", para_len, end_positions)

        if (end_positions > para_len).sum().item() <= 0 and consider_ans_pos:

          answer_phrase_ids = torch.zeros_like(input_ids)

          #if end_positions.item() > answer_phrase_ids.size(-1) :
          #   end_positions.fill_(answer_phrase_ids.size(-1))

          end_positions = end_positions.clamp(max=answer_phrase_ids.size(-1)-2)

          #print("modifying token_type_ids", start_positions, end_positions, answer_phrase_ids.size())
          answer_phrase_ids = answer_phrase_ids.scatter(-1,start_positions.unsqueeze(-1),1)
          answer_phrase_ids = answer_phrase_ids.scatter(-1,(end_positions+1).unsqueeze(-1),-1)

          answer_phrase_ids = answer_phrase_ids.float().matmul(torch.triu(torch.ones(answer_phrase_ids.size(1), answer_phrase_ids.size(1)) ).to(input_ids.device) ).long() # segment_id 1 for answer phrase
          token_type_ids = torch.logical_or(token_type_ids,answer_phrase_ids).long()

        #self.print(input_ids=input_ids,prediction_ids=prediction_ids)

        #position_ids=position_ids,
        #print("attention mask size:", extra_mask.size())
        #print("token_type_ids:", token_type_ids, token_type_ids.sum(-1))


        #print("token_type_ids:", token_type_ids, token_type_ids.sum(-1))
        #input("-")

        # attention_mask has to be 2d, it will be made 3d in modeling_layoutlmv3.py
        # We can't get semi-diagnol mask without visual embeddings

        #print(pixel_values.size())
        #print(torch.max(pixel_values))
        #print(torch.min(pixel_values))
        #input("-pixel values-")
        #print("bbox size:", bbox.size())
        #print("bbox max:", torch.max(bbox))
        #print("bbox min:", torch.min(bbox))

        


        #print("token type ids:", torch.max(token_type_ids), torch.max(input_ids), torch.max(bbox))

        outputs = self.layoutlmv3(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids, 
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=True, 
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            bbox=bbox,
            pixel_values=pixel_values,
        )




        sequence_output = outputs.last_hidden_state
        attns = outputs.attentions
        if torch.isnan(sequence_output).any():
            print("00Nan in sequence_output")

        #print("input_ids size:", input_ids.size())
        #print("sequence output size:", sequence_output.size())
        #input("-")




        text_len = input_ids.size(1)
        sequence_output = sequence_output[:,:text_len]
        #print("sequence_output nan:", torch.isnan(sequence_output).any())




        #lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        qgen_logits = torch.softmax(self.lm_head(sequence_output) + self.final_logits_bias,dim=-1)

        #temp#
        #qgen_logits = torch.softmax(self.lm_head(sequence_output) + self.final_logits_bias,dim=-1)
        
        # bilinear attention
        # copy_seq_output = self.linear_copy_probs(sequence_output) # batch_size x seq_size x (12)
        # copy_scores = copy_seq_output.matmul(sequence_output.transpose(-1,-2)) # b x seq x seq

        #print("Extra mask:", extra_mask.size(), extra_mask.sum(-1), extra_mask.sum(-2)) 

        #copy_scores = copy_scores + (1-extra_mask) * -1e9

        #copy_scores.masked_fill((1-extra_mask).bool(), -1e4) #= copy_scores + (1-extra_mask) * -1e9

        # copy_attention =  torch.softmax(copy_scores, dim=-1) 
        # copy_qgen_logits =  self.get_copy_probs_with_bilinearmat(sequence_output,attns,input_ids, attention=copy_attention)

        '''
        '''

        

        #Use a, obtain a by combining (head*layer) self-attentions

       
        #qgen_logits = (p+1e-9) * torch.softmax(self.lm_head(sequence_output) + self.final_logits_bias,dim=-1)
        #qgen_logits = torch.softmax(self.lm_head(sequence_output),dim=-1) 

        # For copy probs:
        if torch.isnan(qgen_logits).any():
            print("11Nan in qgen_logits")


        if copy_probs:

          copy_seq_output = self.linear_copy_probs(sequence_output) # batch_size x seq_size x (12)
          copy_scores = copy_seq_output.matmul(sequence_output.transpose(-1,-2)) # b x seq x seq

          #print("Extra mask:", extra_mask.size(), extra_mask.sum(-1), extra_mask.sum(-2)) 
          copy_scores = copy_scores + (1-extra_mask) * -1e9
          copy_scores.masked_fill((1-extra_mask).bool(), -1e4) #= copy_scores + (1-extra_mask) * -1e9
          copy_attention =  torch.softmax(copy_scores, dim=-1) 
          copy_qgen_logits =  self.get_copy_probs_with_bilinearmat(sequence_output,attns,input_ids, attention=copy_attention)


          p = torch.sigmoid(self.p_value(sequence_output))

          #copy_qgen_logits =  self.get_copy_probs(sequence_output,attns,input_ids)

          print("copy probs")
          if torch.isnan(copy_qgen_logits).any():
              print("22Nan in copy probs..")


          print("prob p :", p, qgen_logits.size())
          qgen_logits = (p) * qgen_logits 
          print("copy probs:", copy_qgen_logits.sum(-1))

          #Backward pass Error here:
          #RuntimeError: Function 'MulBackward0' returned nan values in its 1th output.

          #copy_qgen_logits = copy_qgen_logits * (1-p)

          qgen_logits = qgen_logits + copy_qgen_logits

          qgen_logits  = qgen_logits + 1e-9

        '''
        '''

        lm_logits = qgen_logits

        #print("qgen logits:", torch.min(qgen_logits), torch.max(qgen_logits))

        # It becomes LogSoftMax after this step
        qgen_logits = torch.log(qgen_logits)


        # print(qgen_logits)
        #Question prediction
        if torch.isnan(qgen_logits).any():
            print("qgen logits isnan")
            print("lm_logits isnan", torch.min(lm_logits), torch.min(lm_logits))
            print("lm logits:", lm_logits)
            input("-")
        '''
        '''

        '''
        if evaluate: #start_positions is not None and end_positions is not None:
            loss_fct = NLLLoss(ignore_index=0)
            loss = loss_fct(qgen_logits.view(-1,self.vocab_size), prediction_ids.view(-1))
            return loss
        else:
            return qgen_logits
        '''

        loss_fct = NLLLoss(ignore_index=0)

        if False: #debug:
          self.print(input_ids, prediction_ids=prediction_ids, 
                phr_start_positions=start_positions.unsqueeze(1), phr_end_positions=end_positions.unsqueeze(1),
                qgen_logits=qgen_logits)  
          input("")
                
        masked_lm_loss = loss_fct(qgen_logits.view(-1,qgen_logits.size(-1)), prediction_ids.view(-1))
        #print("loss:", masked_lm_loss)

        #print("loss:", masked_lm_loss)
        #input("")
        #print("pred ids:", prediction_ids)
        #input("")

        #print(torch.min(qgen_logits))
        if torch.isnan(masked_lm_loss).any() and False:
           print("loss:", masked_lm_loss)
           print("seq output isnan:",torch.isnan(sequence_output).any())
           print("qgen logits isnan:",torch.isnan(qgen_logits).any())
           print("qgen_logits", torch.min(qgen_logits), torch.max(qgen_logits))
           sys.exit(1)
           #input("-")


        #print("Loss = ", loss)
        '''
        '''

        #masked_lm_loss = None

        #if labels is not None:
        #    loss_fct = CrossEntropyLoss()
        #    masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=None, 
            decoder_hidden_states=None, 
            decoder_attentions=None, 
            cross_attentions=None, 
            encoder_last_hidden_state=outputs.last_hidden_state,
            encoder_hidden_states=None, 
            encoder_attentions=outputs.attentions,
        )


    def print(self, input_ids, phr_start_positions=None, phr_end_positions=None, prediction_ids=None,qgen_logits=None, prefix="input_ids", token_type_ids=None, only_return=False):

         model_path = "microsoft/layoutlmv3-base"
         tokenizer = AutoTokenizer.from_pretrained(model_path)

         b=0
         if not only_return:
             for b in range(input_ids.size(0)):

              print("##### %s ######"%(prefix))

              #print(prefix,":"," ".join(tokenizer.convert_ids_to_tokens([ input_ids[b][i].item() for i in range(input_ids[b].size(0)) if input_ids[b][i].item() !=0 ])))
              print(prefix,":"," ".join(tokenizer.convert_ids_to_tokens([ input_ids[b][i].item() for i in range(input_ids[b].size(0)) ]))) 

              #if token_type_ids is not None:
              #  print("only segment 1s:"," ".join(tokenizer.convert_ids_to_tokens([ input_ids[b][i].item() for i in range(input_ids[b].size(0)) if token_type_ids[b][i].item() ==1 ])))

              if prediction_ids is not None:
                print("#source "," ".join(tokenizer.convert_ids_to_tokens([ input_ids[b][i].item() for i in range(prediction_ids[b].size(0)) if prediction_ids[b][i].item() !=0 ])))
                print("#target "," ".join(tokenizer.convert_ids_to_tokens([ prediction_ids[b][i].item() for i in range(prediction_ids[b].size(0)) if prediction_ids[b][i].item() !=0 ])))

              if qgen_logits is not None:
               val, ind = qgen_logits.max(-1) 
               print("#predicted"," ".join(tokenizer.convert_ids_to_tokens([ ind[b][i].item() for i in range(prediction_ids[b].size(0)) if prediction_ids[b][i].item() !=0 ])))

              if phr_start_positions is not None:
                for p in range(phr_start_positions.size(1)):
                  if phr_start_positions[b][p] == 0:
                      continue
                  print("#phrase (%s):%s" %(p, " ".join(tokenizer.convert_ids_to_tokens([ input_ids[b][i].item() for i in range(input_ids[b].size(0)) 
                      if i>=phr_start_positions[b][p] and i <=phr_end_positions[b][p] ] ))) ) 

              '''
              '''
              #input("-")
              #print(soft_segment_ids[b,:,0].detach().tolist())
        
         if token_type_ids is not None:
           return  tokenizer.convert_ids_to_tokens([ input_ids[b][i].item() for i in range(input_ids[b].size(0)) if token_type_ids[b][i].item() !=0 ])
         else:
           return  tokenizer.convert_ids_to_tokens([ input_ids[b][i].item() for i in range(input_ids[b].size(0)) ] )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        print("labels:", labels)
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past



from transformers import PreTrainedModel, PretrainedConfig
from transformers import EncoderDecoderConfig, AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.encoder_decoder.modeling_encoder_decoder import DEPRECATION_WARNING
import warnings

class CustomizedEncoderDecoderModel(PreTrainedModel):
    r"""
    [`EncoderDecoderModel`] is a generic model class that will be instantiated as a transformer architecture with one
    of the base model classes of the library as encoder and another one as decoder when created with the
    :meth*~transformers.AutoModel.from_pretrained* class method for the encoder and
    :meth*~transformers.AutoModelForCausalLM.from_pretrained* class method for the decoder.
    """
    config_class = EncoderDecoderConfig
    base_model_prefix = "encoder_decoder"

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
    ):
        if config is None and (encoder is None or decoder is None):
            raise ValueError("Either a configuration or an encoder and a decoder has to be provided.")
        if config is None:
            config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"Config: {config} has to be of type {self.config_class}")

        if config.decoder.cross_attention_hidden_size is not None:
            if config.decoder.cross_attention_hidden_size != config.encoder.hidden_size:
                raise ValueError(
                    "If `cross_attention_hidden_size` is specified in the decoder's configuration, it has to be equal"
                    f" to the encoder's `hidden_size`. Got {config.decoder.cross_attention_hidden_size} for"
                    f" `config.decoder.cross_attention_hidden_size` and {config.encoder.hidden_size} for"
                    " `config.encoder.hidden_size`."
                )

        # initialize with config
        super().__init__(config)

        if encoder is None:
            from transformers.models.auto.modeling_auto import AutoModel

            encoder = AutoModel.from_config(config.encoder)

        if decoder is None:
            from transformers.models.auto.modeling_auto import AutoModelForCausalLM

            decoder = AutoModelForCausalLM.from_config(config.decoder)

        self.encoder = encoder
        self.decoder = decoder

        if self.encoder.config.to_dict() != self.config.encoder.to_dict():
            logger.warning(
                f"Config of the encoder: {self.encoder.__class__} is overwritten by shared encoder config:"
                f" {self.config.encoder}"
            )
        if self.decoder.config.to_dict() != self.config.decoder.to_dict():
            logger.warning(
                f"Config of the decoder: {self.decoder.__class__} is overwritten by shared decoder config:"
                f" {self.config.decoder}"
            )

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.encoder.config = self.config.encoder
        self.decoder.config = self.config.decoder

        # encoder outputs might need to be projected to different dimension for decoder
        if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            self.enc_to_dec_proj = nn.Linear(self.encoder.config.hidden_size, self.decoder.config.hidden_size)

        if self.encoder.get_output_embeddings() is not None:
            raise ValueError(
                f"The encoder {self.encoder} should not have a LM Head. Please use a model without LM Head"
            )

        # tie encoder, decoder weights if config set accordingly
        self.tie_weights()

    def tie_weights(self):
        # tie encoder & decoder if needed
        if self.config.tie_encoder_decoder:
            # tie encoder and decoder base model
            decoder_base_model_prefix = self.decoder.base_model_prefix
            self._tie_encoder_decoder_weights(
                self.encoder, self.decoder._modules[decoder_base_model_prefix], self.decoder.base_model_prefix
            )

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.decoder.set_output_embeddings(new_embeddings)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # At the moment fast initialization is not supported for composite models
        if kwargs.get("_fast_init", False):
            logger.warning(
                "Fast initialization is currently not supported for EncoderDecoderModel. "
                "Falling back to slow initialization..."
            )
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)

    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs
    ) -> PreTrainedModel:
        r"""
        Instantiate an encoder and a decoder from one or two base classes of the library from pretrained model
        checkpoints.


        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you need to first set it back in training mode with `model.train()`.

        Params:
            encoder_pretrained_model_name_or_path (`str`, *optional*):
                Information necessary to initiate the encoder. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                      user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                      this case, `from_tf` should be set to `True` and a configuration object should be provided as
                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            decoder_pretrained_model_name_or_path (`str`, *optional*, defaults to `None`):
                Information necessary to initiate the decoder. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                      user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                      this case, `from_tf` should be set to `True` and a configuration object should be provided as
                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args (remaining positional arguments, *optional*):
                All remaining positional arguments will be passed to the underlying model's `__init__` method.

            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`).

                - To update the encoder configuration, use the prefix *encoder_* for each configuration parameter.
                - To update the decoder configuration, use the prefix *decoder_* for each configuration parameter.
                - To update the parent model configuration, do not use a prefix for each configuration parameter.

                Behaves differently depending on whether a `config` is provided or automatically loaded.

        Example:

        ```python
        >>> from transformers import EncoderDecoderModel

        >>> # initialize a bert2bert from two pretrained BERT models. Note that the cross-attention layers will be randomly initialized
        >>> model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")
        >>> # saving model after fine-tuning
        >>> model.save_pretrained("./bert2bert")
        >>> # load fine-tuned model
        >>> model = EncoderDecoderModel.from_pretrained("./bert2bert")
        ```"""

        kwargs_encoder = {
            argument[len("encoder_") :]: value for argument, value in kwargs.items() if argument.startswith("encoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        # remove encoder, decoder kwargs from kwargs
        for key in kwargs_encoder.keys():
            del kwargs["encoder_" + key]
        for key in kwargs_decoder.keys():
            del kwargs["decoder_" + key]

        # Load and initialize the encoder and decoder
        # The distinction between encoder and decoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        encoder = kwargs_encoder.pop("model", None)
        if encoder is None:
            if encoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `encoder_model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_encoder:
                encoder_config, kwargs_encoder = AutoConfig.from_pretrained(
                    encoder_pretrained_model_name_or_path, **kwargs_encoder, return_unused_kwargs=True
                )

                if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:
                    logger.info(
                        f"Initializing {encoder_pretrained_model_name_or_path} as a encoder model "
                        "from a decoder model. Cross-attention and casual mask are disabled."
                    )
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False

                kwargs_encoder["config"] = encoder_config

            encoder = AutoModel.from_pretrained(encoder_pretrained_model_name_or_path, *model_args, **kwargs_encoder)

        decoder = kwargs_decoder.pop("model", None)
        if decoder is None:
            if decoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_decoder:
                decoder_config, kwargs_decoder = AutoConfig.from_pretrained(
                    decoder_pretrained_model_name_or_path, **kwargs_decoder, return_unused_kwargs=True
                )

                if decoder_config.is_decoder is False or decoder_config.add_cross_attention is False:
                    logger.info(
                        f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. Cross attention"
                        f" layers are added to {decoder_pretrained_model_name_or_path} and randomly initialized if"
                        f" {decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers."
                    )
                    decoder_config.is_decoder = True
                    decoder_config.add_cross_attention = True

                kwargs_decoder["config"] = decoder_config

            if kwargs_decoder["config"].is_decoder is False or kwargs_decoder["config"].add_cross_attention is False:
                logger.warning(
                    f"Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. "
                    f"In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, "
                    "make sure that the attributes `is_decoder` and `add_cross_attention` of `decoder_config` "
                    "passed to `.from_encoder_decoder_pretrained(...)` are set to `True` or do not pass a "
                    "`decoder_config` to `.from_encoder_decoder_pretrained(...)`"
                )

            decoder = AutoModelForCausalLM.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)

        # instantiate config with corresponding kwargs
        config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config, **kwargs)
        return cls(encoder=encoder, decoder=decoder, config=config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        bbox: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        is_train:bool=True,
        **kwargs,
    ) -> Union[Tuple, Seq2SeqLMOutput]:

        if not is_train:
            return self.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values,
                use_cache=True,
                return_dict=return_dict,
                **kwargs,
            )

        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import EncoderDecoderModel, BertTokenizer
        >>> import torch

        >>> tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        >>> model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        ...     "bert-base-uncased", "bert-base-uncased"
        ... )  # initialize Bert2Bert from pre-trained checkpoints

        >>> # training
        >>> model.config.decoder_start_token_id = tokenizer.cls_token_id
        >>> model.config.pad_token_id = tokenizer.pad_token_id
        >>> model.config.vocab_size = model.config.decoder.vocab_size

        >>> input_ids = tokenizer("This is a really long text", return_tensors="pt").input_ids
        >>> labels = tokenizer("This is the corresponding summary", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=input_ids)
        >>> loss, logits = outputs.loss, outputs.logits

        >>> # save and load from pretrained
        >>> model.save_pretrained("bert2bert")
        >>> model = EncoderDecoderModel.from_pretrained("bert2bert")

        >>> # generation
        >>> generated = model.generate(input_ids)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                bbox=bbox,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        encoder_hidden_states = encoder_outputs[0]
        batch_size, seq_len, _ = encoder_hidden_states.size()
        # optionally project encoder_hidden_states
        if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        visual_attention_mask = torch.ones(
            (batch_size, seq_len - attention_mask.size(1)), dtype=torch.long, device=encoder_hidden_states.device
        )
        updated_attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=1)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=updated_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        # Compute loss independent from decoder (as some shift the logits inside them)
        loss = None
        if labels is not None:
            warnings.warn(DEPRECATION_WARNING, FutureWarning)
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.view(-1))

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def prepare_inputs_for_generation(
        self, input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids, past=past)
        decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
        batch_size, seq_len, _ = encoder_outputs[0].size()
        visual_attention_mask = torch.ones(
            (batch_size, seq_len - attention_mask.size(1)), dtype=torch.long, device=encoder_outputs[0].device
        )
        updated_attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=1)
        input_dict = {
            "attention_mask": updated_attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "encoder_outputs": encoder_outputs,
            "past_key_values": decoder_inputs["past_key_values"],
            "use_cache": use_cache,
        }
        return input_dict

    def resize_token_embeddings(self, *args, **kwargs):
        raise NotImplementedError(
            "Resizing the embedding layers via the EncoderDecoderModel directly is not supported. Please use the"
            " respective methods of the wrapped objects (model.encoder.resize_token_embeddings(...) or"
            " model.decoder.resize_token_embeddings(...))"
        )

    def _reorder_cache(self, past, beam_idx):
        # apply decoder cache reordering here
        return self.decoder._reorder_cache(past, beam_idx)


if __name__ == '__main__':
    from transformers import RobertaModel, RobertaConfig
    from transformers import LayoutLMv3TokenizerFast, LayoutLMv3Tokenizer, LayoutLMv3FeatureExtractor, \
        LayoutLMv3Processor
    # old = RobertaModel.from_pretrained('roberta-base')
    #
    # new_model = RobertaDecoder(RobertaConfig.from_pretrained('roberta-base'))
    #
    # new_model.layer.load_state_dict(old.encoder.layer.state_dict(), strict=False)

    model = LayoutLMv3ForConditionalGeneration(LayoutLMv3Config.from_pretrained('microsoft/layoutlmv3-base'))
    model.config.decoder_start_token_id = model.config.eos_token_id
    model.config.is_encoder_decoder = True
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained('microsoft/layoutlmv3-base')
    # res = tokenizer.encoder_plus(['Hello', 'world'], boxes = [[0,0,0,0] for _ in range(2)], return_tensors='pt')
    # print(res)
    # input_ids = torch.tensor(([]))

    input_ids =  torch.tensor([[0, 20920, 232, 2, 1], [0, 20920, 232, 100, 2]])
    attention_mask = torch.tensor([[1,1,1,1,0], [1,1,1,1,1]])
    bbox = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],  [0, 0, 0, 0],  [0, 0, 0, 0]],
                         [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],  [0, 0, 0, 0],  [0, 0, 0, 0]]
                         ])
    result = model(input_ids = input_ids, attention_mask = attention_mask, bbox = bbox, pixel_values=torch.randn(2, 3, 224, 224), is_train=False)
    print(result)

    # from transformers import RobertaTokenizerFast
    # roberta_tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    # print(tokenizer.decode(result[0]))
    # res_str = tokenizer.decode(result[0], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
    # print(res_str)


    # from transformers import BartForConditionalGeneration, BartTokenizerFast
    # bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    # bar_tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')
    # bart_res = bar_tokenizer.batch_encode_plus(["how are you doing"], return_tensors='pt')
    # bart_result = bart_model.generate(input_ids=bart_res['input_ids'], attention_mask=bart_res['attention_mask'], num_beams=1)
