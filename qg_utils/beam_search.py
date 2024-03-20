import torch

#from onmt.translate.decode_strategy import DecodeStrategy



class DecodeStrategy(object):
    """Base class for generation strategies.

    Args:
        pad (int): Magic integer in output vocab.
        bos (int): Magic integer in output vocab.
        eos (int): Magic integer in output vocab.
        batch_size (int): Current batch size.
        device (torch.device or str): Device for memory bank (encoder).
        parallel_paths (int): Decoding strategies like beam search
            use parallel paths. Each batch is repeated ``parallel_paths``
            times in relevant state tensors.
        min_length (int): Shortest acceptable generation, not counting
            begin-of-sentence or end-of-sentence.
        max_length (int): Longest acceptable sequence, not counting
            begin-of-sentence (presumably there has been no EOS
            yet if max_length is used as a cutoff).
        block_ngram_repeat (int): Block beams where
            ``block_ngram_repeat``-grams repeat.
        exclusion_tokens (set[int]): If a gram contains any of these
            tokens, it may repeat.
        return_attention (bool): Whether to work with attention too. If this
            is true, it is assumed that the decoder is attentional.

    Attributes:
        pad (int): See above.
        bos (int): See above.
        eos (int): See above.
        predictions (list[list[LongTensor]]): For each batch, holds a
            list of beam prediction sequences.
        scores (list[list[FloatTensor]]): For each batch, holds a
            list of scores.
        attention (list[list[FloatTensor or list[]]]): For each
            batch, holds a list of attention sequence tensors
            (or empty lists) having shape ``(step, inp_seq_len)`` where
            ``inp_seq_len`` is the length of the sample (not the max
            length of all inp seqs).
        alive_seq (LongTensor): Shape ``(B x parallel_paths, step)``.
            This sequence grows in the ``step`` axis on each call to
            :func:`advance()`.
        is_finished (ByteTensor or NoneType): Shape
            ``(B, parallel_paths)``. Initialized to ``None``.
        alive_attn (FloatTensor or NoneType): If tensor, shape is
            ``(step, B x parallel_paths, inp_seq_len)``, where ``inp_seq_len``
            is the (max) length of the input sequence.
        min_length (int): See above.
        max_length (int): See above.
        block_ngram_repeat (int): See above.
        exclusion_tokens (set[int]): See above.
        return_attention (bool): See above.
        done (bool): See above.
    """

    def __init__(self, pad, bos, eos, batch_size, device, parallel_paths,
                 min_length, block_ngram_repeat, exclusion_tokens,
                 return_attention, max_length):

        # magic indices
        self.pad = pad
        self.bos = bos
        self.eos = eos

        # result caching
        self.predictions = [[] for _ in range(batch_size)]
        self.scores = [[] for _ in range(batch_size)]
        self.attention = [[] for _ in range(batch_size)]

        self.alive_seq = torch.full(
            [batch_size * parallel_paths, 1], self.bos,
            dtype=torch.long, device=device)
        self.is_finished = torch.zeros(
            [batch_size, parallel_paths],
            dtype=torch.uint8, device=device)
        self.alive_attn = None

        self.min_length = min_length
        self.max_length = max_length
        self.block_ngram_repeat = block_ngram_repeat
        self.exclusion_tokens = exclusion_tokens
        self.return_attention = return_attention

        self.done = False

    def __len__(self):
        return self.alive_seq.shape[1]

    def ensure_min_length(self, log_probs):
        if len(self) <= self.min_length:
            log_probs[:, self.eos] = -1e20

    def ensure_max_length(self):
        # add one to account for BOS. Don't account for EOS because hitting
        # this implies it hasn't been found.
        if len(self) == self.max_length + 1:
            self.is_finished.fill_(1)

    def block_ngram_repeats(self, log_probs):
        cur_len = len(self)
        if self.block_ngram_repeat > 0 and cur_len > 1:
            for path_idx in range(self.alive_seq.shape[0]):
                # skip BOS
                hyp = self.alive_seq[path_idx, 1:]
                ngrams = set()
                fail = False
                gram = []
                for i in range(cur_len - 1):
                    # Last n tokens, n = block_ngram_repeat
                    gram = (gram + [hyp[i].item()])[-self.block_ngram_repeat:]
                    # skip the blocking if any token in gram is excluded
                    if set(gram) & self.exclusion_tokens:
                        continue
                    if tuple(gram) in ngrams:
                        fail = True
                    ngrams.add(tuple(gram))
                if fail:
                    log_probs[path_idx] = -10e20

    def advance(self, log_probs, attn):
        """DecodeStrategy subclasses should override :func:`advance()`.

        Advance is used to update ``self.alive_seq``, ``self.is_finished``,
        and, when appropriate, ``self.alive_attn``.
        """

        raise NotImplementedError()

    def update_finished(self):
        """DecodeStrategy subclasses should override :func:`update_finished()`.

        ``update_finished`` is used to update ``self.predictions``,
        ``self.scores``, and other "output" attributes.
        ``update_finished`` is used to update ``self.predictions``,
        ``self.scores``, and other "output" attributes.
        """

        raise NotImplementedError()


class BeamSearch:
    """Generation beam search.

    Note that the attributes list is not exhaustive. Rather, it highlights
    tensors to document their shape. (Since the state variables' "batch"
    size decreases as beams finish, we denote this axis with a B rather than
    ``batch_size``).

    Args:
        beam_size (int): Number of beams to use (see base ``parallel_paths``).
        batch_size (int): See base.
        pad (int): See base.
        bos (int): See base.
        eos (int): See base.
        n_best (int): Don't stop until at least this many beams have
            reached EOS.
        mb_device (torch.device or str): See base ``device``.
        global_scorer (onmt.translate.GNMTGlobalScorer): Scorer instance.
        min_length (int): See base.
        max_length (int): See base.
        return_attention (bool): See base.
        block_ngram_repeat (int): See base.
        exclusion_tokens (set[int]): See base.
        memory_lengths (LongTensor): Lengths of encodings. Used for
            masking attentions.

    Attributes:
        top_beam_finished (ByteTensor): Shape ``(B,)``.
        _batch_offset (LongTensor): Shape ``(B,)``.
        _beam_offset (LongTensor): Shape ``(batch_size x beam_size,)``.
        alive_seq (LongTensor): See base.
        topk_log_probs (FloatTensor): Shape ``(B x beam_size,)``. These
            are the scores used for the topk operation.
        select_indices (LongTensor or NoneType): Shape
            ``(B x beam_size,)``. This is just a flat view of the
            ``_batch_index``.
        topk_scores (FloatTensor): Shape
            ``(B, beam_size)``. These are the
            scores a sequence will receive if it finishes.
        topk_ids (LongTensor): Shape ``(B, beam_size)``. These are the
            word indices of the topk predictions.
        _batch_index (LongTensor): Shape ``(B, beam_size)``.
        _prev_penalty (FloatTensor or NoneType): Shape
            ``(B, beam_size)``. Initialized to ``None``.
        _coverage (FloatTensor or NoneType): Shape
            ``(1, B x beam_size, inp_seq_len)``.
        hypotheses (list[list[Tuple[Tensor]]]): Contains a tuple
            of score (float), sequence (long), and attention (float or None).
    """

    def __init__( self,beam_size, tokenizer, batch_size, n_best, min_length, max_length, block_ngram_repeat,bos, eos, device=0):
        mb_device=0
        return_attention=True
        exclusion_tokens=[]
        self.bos=bos #103
        self.eos=eos #102 # [SEP]
        self.pad=0
        self.predictions = [[] for _ in range(batch_size)]
        self.segments = [[] for _ in range(batch_size)]
        self.scores = [[] for _ in range(batch_size)]
        self.attention = [[] for _ in range(batch_size)]
        self.exclusion_tokens = set()
        self.question_id = tokenizer.convert_tokens_to_ids(["?"])[0]

        #super(BeamSearch, self).__init__(
        #    pad, bos, eos, batch_size, mb_device, beam_size, min_length,
        #    block_ngram_repeat, exclusion_tokens, return_attention,
        #    max_length)
        # beam parameters

        parallel_paths = beam_size

        self.beam_size = beam_size
        self.n_best = n_best
        self.batch_size = batch_size

        self.min_length = min_length
        self.max_length = max_length
        self.block_ngram_repeat = block_ngram_repeat

        self.alive_seq = torch.full(
            [batch_size * parallel_paths, 1], self.bos,
            dtype=torch.long, device=device)

        self.alive_segment = torch.full(
            [batch_size * parallel_paths, 1], 0,
            dtype=torch.long, device=device)

        self.alive_input_mask = torch.full(
            [batch_size * parallel_paths, 1], 1,
            dtype=torch.long, device=device)

        self.alive_start_positions = torch.full(
            [batch_size * parallel_paths ], 0,
            dtype=torch.long, device=device)

        self.alive_end_positions = torch.full(
            [batch_size * parallel_paths ], 0,
            dtype=torch.long, device=device)

        self.is_finished = torch.zeros(
            [batch_size, parallel_paths],
            dtype=torch.uint8, device=device)
        self.alive_attn = None

        # result caching
        self.hypotheses = [[] for _ in range(batch_size)]

        # beam state
        self.top_beam_finished = torch.zeros([batch_size], dtype=torch.uint8)
        self.best_scores = torch.full([batch_size], -1e10, dtype=torch.float, device=device)

        self._batch_offset = torch.arange(batch_size, dtype=torch.long)
        self._beam_offset = torch.arange(
            0, batch_size * beam_size, step=beam_size, dtype=torch.long, device=device)
        self.topk_log_probs = torch.tensor(
            [0.0] + [float("-inf")] * (beam_size - 1), device=device).repeat(batch_size) 
        #).repeat(batch_size)
        self.select_indices = None
        #self._memory_lengths = memory_lengths

        # buffers for the topk scores and 'backpointer'
        self.topk_scores = torch.empty((batch_size, beam_size),
                                       dtype=torch.float, device=device)
        self.topk_ids = torch.empty((batch_size, beam_size), dtype=torch.long,
                                    device=device)
        self._batch_index = torch.empty([batch_size, beam_size],
                                        dtype=torch.long, device=device)
        self.done = False
        # "global state" of the old beam
        self._prev_penalty = None
        self._coverage = None
        self.alpha = 1
        self.length_exceeded=False

    def __len__(self):
        question_length = self.alive_segment.sum(-1)
        #print(question_length)
        return question_length[0].item()

    def ensure_max_length(self):
        # add one to account for BOS. Don't account for EOS because hitting
        # this implies it hasn't been found.
        #print("length of self",len(self),self.max_length)
        #input("length")
        if len(self) == self.max_length + 1:
            print("--Ensuring max length")
            self.is_finished.fill_(1)
            self.length_exceeded = True
            print("is finished",self.is_finished)
        #print("Ensuring max length")

    def length_penalty(self,steps):
        #return steps * self.alpha 
        return 1 

    def ensure_min_length(self, log_probs):
        if len(self) <= self.min_length:
            log_probs[:, self.eos] = -1e20

    def remove_repeated_same_words(self, log_probs):
        #print("question id",self.question_id)
        previous_question_id_check = (self.current_predictions == self.question_id).float()
        #print(previous_question_id_check)
        #print(log_probs.size(), previous_question_id_check*-1e20, log_probs[:,self.question_id ])
        #input("previous question id check")
        log_probs[:,self.question_id ] += previous_question_id_check*-1e20
       
    def assign_whats_alive(self,input_ids, segment_ids, input_mask, start_positions, end_positions):
        self.alive_seq = input_ids
        self.alive_segment = segment_ids
        self.alive_input_mask = input_mask
        self.alive_start_positions = start_positions
        self.alive_end_positions = end_positions

    def get_whats_alive(self):
        return (self.alive_seq,self.alive_segment, self.alive_input_mask, self.alive_start_positions, self.alive_end_positions)

    def block_ngram_repeats(self, log_probs):
        cur_len = len(self)
        if self.block_ngram_repeat > 0 and cur_len > 1:
            for path_idx in range(self.alive_seq.shape[0]):
                # skip BOS
                hyp = self.alive_seq[path_idx, 1:]
                ngrams = set()
                fail = False
                gram = []
                for i in range(cur_len - 1):
                    # Last n tokens, n = block_ngram_repeat
                    gram = (gram + [hyp[i].item()])[-self.block_ngram_repeat:]
                    # skip the blocking if any token in gram is excluded
                    if set(gram) & self.exclusion_tokens:
                        continue
                    if tuple(gram) in ngrams:
                        fail = True
                    ngrams.add(tuple(gram))
                if fail:
                    log_probs[path_idx] = -10e20

    @property
    def current_predictions(self):
        return self.alive_seq[:, -1]

    @property
    def current_origin(self):
        return self.select_indices

    @property
    def current_backptr(self):
        # for testing
        return self.select_indices.view(self.batch_size, self.beam_size)\
            .fmod(self.beam_size)

    def advance(self, log_probs, attn):

        #log_probs size = (batch_size * beam_size) x vocab_size 
        #obtained from applying the model: model(alive_seq[:])
        # For the first time, it will be (batch_size x vocab_size) it is made into (batch_size*beam_size) x vocab_size 
        # by filling 0's and giving that as input to this function

        vocab_size = log_probs.size(-1)

        # using integer division to get an integer _B without casting : _B is batch_size
        _B = log_probs.shape[0] // self.beam_size

        # 

        '''
        if self._stepwise_cov_pen and self._prev_penalty is not None:
            self.topk_log_probs += self._prev_penalty
            self.topk_log_probs -= self.global_scorer.cov_penalty(
                self._coverage + attn, self.global_scorer.beta).view(
                _B, self.beam_size)
        '''

        '''
        # force the output to be longer than self.min_length
        step = len(self)
        self.ensure_min_length(log_probs)
        '''

        # Multiply probs by the beam probability.

        log_probs += self.topk_log_probs.view(_B * self.beam_size, 1)
        # topk_log_probs.size() = (batch_size , beam_size)
        # log_probs.size() = (batch_size * beam_size) x vocab_size

        #stop repeating ngrams
        #self.block_ngram_repeats(log_probs)
        #self.remove_repeated_same_words(log_probs)
        '''
        '''

        # if the sequence ends now, then the penalty is the current
        # length + 1, to include the EOS token

        '''
        #Penalty
        length_penalty = self.length_penalty(step + 1, alpha=self.alpha)

        # Flatten probs into a list of possibilities.
        curr_scores = log_probs / length_penalty
        '''
        curr_scores = log_probs
        curr_scores = curr_scores.reshape(_B, self.beam_size * vocab_size)
        torch.topk(curr_scores,  self.beam_size, dim=-1,
                   out=(self.topk_scores, self.topk_ids))

        # Recover log probs.
        # Length penalty is just a scalar. It doesn't matter if it's applied
        # before or after the topk.

        length_penalty=1
        torch.mul(self.topk_scores, length_penalty, out=self.topk_log_probs)

        # Resolve beam origin and map to batch index flat representation.

        self._batch_index = self.topk_ids // vocab_size
        #torch.div(self.topk_ids, vocab_size, out=self._batch_index) # per batch we get the corresponding beam for topk seq
                                                                    # example: 0 : 1,1,2,3 3
                                                                    #          1 : 2,2,1,0 0
                                                                    #          ..: ..
                                                                    #          ..: ..

        self._batch_index += self._beam_offset[:_B].unsqueeze(1)  # Add [0, beam_size, 2*beam_size,..] 
                                                                  # so that the index makes sense when stretched
        self.select_indices = self._batch_index.view(_B * self.beam_size)

        self.topk_ids.fmod_(vocab_size)  # resolve true word ids

        # Append last prediction.
        self.alive_seq = torch.cat(
            [self.alive_seq.index_select(0, self.select_indices),
             self.topk_ids.view(_B * self.beam_size, 1)], -1)

        #Adjust segment_ids, input_mask
        self.alive_segment = torch.cat(
            [self.alive_segment.index_select(0, self.select_indices),
             torch.ones_like(self.topk_ids.view(_B * self.beam_size, 1))], -1)
        self.alive_input_mask = (self.alive_seq  != 0 )

        # start_positions
        self.alive_start_positions = self.alive_start_positions.index_select(0, self.select_indices)
        self.alive_end_positions = self.alive_end_positions.index_select(0, self.select_indices)


        #if self.return_attention or self._cov_pen:

        #if self._vanilla_cov_pen:
            # shape: (batch_size x beam_size, 1)
            #cov_penalty = self.global_scorer.cov_penalty(
            #    self._coverage,
            #    beta=self.global_scorer.beta)
            #self.topk_scores -= cov_penalty.view(_B, self.beam_size)

        self.is_finished = self.topk_ids.eq(self.eos)
        self.ensure_max_length()

    def update_finished(self):
        # Penalize beams that finished.
        _B_old = self.topk_log_probs.shape[0]
        step = self.alive_seq.shape[-1]  # 1 greater than the step in advance
        self.topk_log_probs.masked_fill_(self.is_finished, -1e10)
        # on real data (newstest2017) with the pretrained transformer,
        # it's faster to not move this back to the original device
        self.is_finished = self.is_finished.to('cpu')
        self.top_beam_finished |= self.is_finished[:, 0].eq(1)
        predictions = self.alive_seq.view(_B_old, self.beam_size, step)
        segments = self.alive_segment.view(_B_old, self.beam_size, step)
        input_mask = self.alive_input_mask.view(_B_old, self.beam_size, step)
        start_positions = self.alive_start_positions.view(_B_old, self.beam_size,1)
        end_positions = self.alive_end_positions.view(_B_old, self.beam_size,1 )
        attention = None

        non_finished_batch = []
        #print(self.is_finished.size(0))
        #input("is finished size")

        for i in range(self.is_finished.size(0)):
            b = self._batch_offset[i]
            finished_hyp = self.is_finished[i].nonzero().view(-1)
            # Store finished hypotheses for this batch.
            for j in finished_hyp:
                if True: #self.ratio > 0:
                    s = self.topk_scores[i, j] / (step + 1)
                    if self.best_scores[b] < s:
                        self.best_scores[b] = s
                self.hypotheses[b].append((
                    self.topk_scores[i, j],
                    predictions[i, j, 1:],  # Ignore start_token.
                    segments[i, j, 1:],  # Ignore start_token.
                    attention[:, i, j, :] #self._memory_lengths[i]]
                    if attention is not None else None))
            # End condition is the top beam finished and we can return
            # n_best hypotheses.
            if False: #self.ratio > 0:
                #pred_len = self._memory_lengths[i] * self.ratio
                pred_len = 10 #self._memory_lengths[i] * self.ratio
                finish_flag = ((self.topk_scores[i, 0] / pred_len)
                               <= self.best_scores[b]) or \
                    self.is_finished[i].all()
            else:
                finish_flag = self.top_beam_finished[i] != 0
            if finish_flag and (len(self.hypotheses[b]) >= self.n_best or self.length_exceeded):
                best_hyp = sorted(
                    self.hypotheses[b], key=lambda x: x[0], reverse=True)
                #print("best hyp",len(best_hyp))

                for n, (score, pred,segment, attn) in enumerate(best_hyp):

                    if n >= self.n_best:
                        #print("n>n_best",n,self.n_best)
                        break
                    #print(pred,pred.size())
                    #print(score)

                    self.scores[b].append(score)
                    self.predictions[b].append(pred)
                    self.segments[b].append(segment)
            else:
                non_finished_batch.append(i)

        non_finished = torch.tensor(non_finished_batch)
        # If all sentences are translated, no need to go further.
        if len(non_finished) == 0:
            self.done = True
            return

        _B_new = non_finished.shape[0]

        ''' <Remove finished batches for the next step. '''
        self.top_beam_finished = self.top_beam_finished.index_select(
            0, non_finished)

        self._batch_offset = self._batch_offset.index_select(0, non_finished)
        non_finished = non_finished.to(self.topk_ids.device)
        self.topk_log_probs = self.topk_log_probs.index_select(0,
                                                               non_finished)
        self._batch_index = self._batch_index.index_select(0, non_finished)
        self.select_indices = self._batch_index.view(_B_new * self.beam_size)

        #sequence
        self.alive_seq = predictions.index_select(0, non_finished) \
            .view(-1, self.alive_seq.size(-1))

        #segment
        self.alive_segment = segments.index_select(0, non_finished) \
            .view(-1, self.alive_seq.size(-1))

        #input_mask
        self.alive_input_mask = (self.alive_seq != 0)

        #start_positions
        self.alive_start_positions = start_positions.index_select(0, non_finished) \
            .view(-1)

        #end_positions
        self.alive_end_positions = end_positions.index_select(0, non_finished) \
            .view(-1)
        ''' /Remove finished batches for the next step.> '''

        self.topk_scores = self.topk_scores.index_select(0, non_finished)
        self.topk_ids = self.topk_ids.index_select(0, non_finished)
