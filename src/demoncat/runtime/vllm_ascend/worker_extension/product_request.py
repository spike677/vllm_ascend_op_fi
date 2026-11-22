"""Observe an actual user request; shared sites, no pre-registered user input.

The Worker observer supplies real scheduler IDs and completion evidence. Range decisions use Worker request/scheduler state before
any site is installed. No inference is started by this observer.
"""
from .continuous_binding import ContinuousRequest, digest
from .serving_context import require
from fault_injection.product_contract import request_range, summarize_request


class ProductRequest(ContinuousRequest):
    def __init__(self, binding, ids, *, inject=True, forced_skip=None):
        super().__init__(binding, ids, 'fault' if inject else 'normal')
        self.prompts = {}
        self.skip = [forced_skip] if forced_skip else []
        self.checked = False
        self.finished = False
        self.inject = inject
        self.external_request_id = None
        self.visual = binding.mechanism == "17.spatial_padding"
        self.encoder_calls = self.embedding_calls = 0
        self.encoder_output = None
        self.in_encoder = False

    def install(self, *, managed=False):
        import torch
        from fault_injection.sites.mla28 import fingerprint
        self.binding.check()
        require(len(self.ids) == 1 and type(self.ids[0]) is str and self.ids[0], 'PRODUCT_INTERNAL_ID_REQUIRED')

        def patch(obj, name, fn):
            require(name not in vars(obj), 'PRODUCT_METHOD_ALREADY_OWNED:' + name)
            original = getattr(obj, name)
            def wrapped(*args, **kwargs):
                self.inside += 1
                try:
                    return fn(original, *args, **kwargs)
                except BaseException as error:
                    self.error = repr(error)
                    raise
                finally:
                    self.inside -= 1
            setattr(obj, name, wrapped)
            self.patches.append((obj, name, wrapped))

        def execute(original, scheduler, intermediate_tensors=None):
            self.scheduler = scheduler
            if scheduler.total_num_scheduled_tokens == 0:
                return original(scheduler, intermediate_tensors)
            self.model_output = None
            result = original(scheduler, intermediate_tensors)
            state = self.runner.execute_model_state
            if not self.skip:
                require(result is None and state is not None and state.scheduler_output is scheduler
                        and state.hidden_states is self.model_output and state.logits.shape[0] == 1,
                        'PRODUCT_ACTUAL_MODEL_RESULT_UNPROVEN')
                self.rows.append(dict(hidden=fingerprint(state.hidden_states), logits=fingerprint(state.logits),
                    finite=bool(torch.isfinite(state.hidden_states).all().item() and torch.isfinite(state.logits).all().item())))
            return result

        def preprocess(original, scheduler, total, intermediate_tensors=None):
            rid = self.ids[0]
            req = self.runner.requests.get(rid)
            batch = self.runner.input_batch
            if not self.checked:
                self.checked = True
                prompt = getattr(req, 'prompt_token_ids', None)
                valid = type(prompt) is list and all(type(t) is int and t >= 0 for t in prompt)
                index = batch.req_id_to_index.get(rid)
                computed = int(batch.num_computed_tokens_cpu[index]) if type(index) is int else None
                features = getattr(req, 'mm_features', None)
                facts = dict(request_count=len(batch.req_ids), prompt_length=len(prompt) if valid else None,
                    max_tokens=getattr(getattr(req, 'sampling_params', None), 'max_tokens', None),
                    single_chunk=bool(valid and batch.req_ids == [rid] and scheduler.num_scheduled_tokens == {rid: len(prompt)}
                        and total == scheduler.total_num_scheduled_tokens == len(prompt) and computed == 0),
                    fresh_cache=computed == 0, speculative=bool(scheduler.scheduled_spec_decode_tokens),
                    lora=getattr(req, 'lora_request', None) is not None,
                    prompt_embeds=getattr(req, 'prompt_embeds', None) is not None,
                    image_count=sum(f.modality == 'image' for f in features) if isinstance(features, list) else None,
                    video_count=sum(f.modality == 'video' for f in features) if isinstance(features, list) else None)
                self.range_facts = facts
                external=getattr(req, 'external_req_id', None)
                if type(external) is str: self.external_request_id=external
                if self.visual and isinstance(features, list) and len(features) == 1:
                    feature = features[0]
                    try:
                        grid = feature.data['image_grid_thw'].data
                        if grid.device.type == 'cpu' and grid.numel() == 3 and grid.reshape(-1).tolist() == [1,4,4]:
                            facts['image_size'] = [64,64]
                    except (AttributeError, KeyError, TypeError):
                        pass
                    if (scheduler.scheduled_encoder_inputs != {rid:[0]} or feature.identifier in self.runner.encoder_cache
                            or not 0 <= feature.mm_position.offset < feature.mm_position.offset + feature.mm_position.length <= len(prompt)):
                        self.skip.append('IMAGE_ENCODER_MAPPING_OR_CACHE_UNPROVEN')
                    self.feature = feature
                self.skip.extend(request_range(facts, visual=self.visual)['reasons'])
                if intermediate_tensors is not None:
                    self.skip.append('INTERMEDIATE_PIPELINE_INPUT_UNSUPPORTED')
                if not self.skip:
                    self.prompts = {rid: tuple(prompt)}  # live snapshot, never compared with a canary
                    if self.visual: self.install_visual_site(patch)
                    elif self.inject:
                        self.install_site()
                        if self.binding.mechanism=='18.api_half':
                            def logits(original,*args,**kwargs):
                                value=original(*args,**kwargs)
                                require(self.site.active and not self.site.pending and not self.site.parents,
                                        'PRODUCT_NORM_FINAL_LOGITS_CONSUMER_PENDING')
                                self.site.mode.__exit__(None,None,None)
                                self.site.active=False
                                return value
                            patch(self.model,'compute_logits',logits)
            elif not self.skip:
                require(batch.req_ids == [rid] and req.prompt_token_ids == list(self.prompts[rid])
                    and scheduler.num_scheduled_tokens == {rid: 1} and total == 1
                    and int(batch.num_computed_tokens_cpu[0]) == len(self.prompts[rid]) + self.samples - 1
                    and not scheduler.scheduled_encoder_inputs and not scheduler.scheduled_spec_decode_tokens,
                    'PRODUCT_ACTIVE_REQUEST_MAPPING_CHANGED')
            result = original(scheduler, total, intermediate_tensors)
            if not self.skip:
                require(len(result) == 6 and result[3] is None and result[4] == {} and result[5] is None,
                        'PRODUCT_PREPROCESS_EXTRA_INPUT')
                self.preprocessed = result
            return result

        def enter(module, args, kwargs):
            if self.skip:
                return
            require(self.preprocessed is not None and not args, 'PRODUCT_MODEL_INPUT_ORDER')
            expected = dict(zip(('input_ids', 'inputs_embeds', 'positions', 'intermediate_tensors'), self.preprocessed[:4]))
            require(set(kwargs) == set(expected) and all(kwargs[k] is v for k, v in expected.items()),
                    'PRODUCT_MODEL_INPUT_ASSOCIATION')

        def output(module, args, value):
            self.model_output = value

        def sample(original, *args, **kwargs):
            result = original(*args, **kwargs)
            if not self.skip:
                require(result.req_ids == self.ids and self.runner.execute_model_state is None,
                        'PRODUCT_SAMPLE_REQUEST_ASSOCIATION')
            self.samples += 1
            return result

        def visual(*unused, **kwargs):
            require(bool(self.skip) or self.visual and self.in_encoder, 'PRODUCT_TEXT_VISUAL_PATH_EXECUTED')

        try:
            self.managed_execute = execute
            self.managed_sample = sample
            if not managed: patch(self.runner, 'execute_model', execute)
            patch(self.runner, '_preprocess', preprocess)
            if not managed: patch(self.runner, 'sample_tokens', sample)
            self.hooks.extend([self.model.register_forward_pre_hook(enter, with_kwargs=True),
                               self.model.register_forward_hook(output)])
            if getattr(self.model, 'visual', None) is not None:
                self.hooks.append(self.model.visual.register_forward_pre_hook(visual))
        except BaseException:
            self.close()
            raise

    def install_visual_site(self, patch):
        import torch
        from fault_injection.sites.vision17 import Vision17Site
        def encoder(original, scheduler):
            require(scheduler is self.scheduler, 'PRODUCT_VISUAL_SCHEDULE_CHANGED')
            if not scheduler.scheduled_encoder_inputs:
                return original(scheduler)
            hashes, kwargs, refs = self.runner._batch_mm_inputs_from_scheduler(scheduler)
            require(self.encoder_calls == 0 and hashes == [self.feature.identifier]
                    and len(kwargs) == 1 and kwargs[0][0] == 'image' and kwargs[0][1] is self.feature.data
                    and len(refs) == 1 and refs[0][0] == self.ids[0]
                    and refs[0][1] is self.feature.mm_position, 'PRODUCT_VISUAL_REAL_FEATURE_JOIN')
            self.in_encoder = True
            try:
                result = original(scheduler)
            finally:
                self.in_encoder = False
            cached = self.runner.encoder_cache[self.feature.identifier]
            require(self.encoder_output is not None and cached.untyped_storage()._cdata == self.encoder_output.untyped_storage()._cdata,
                    'PRODUCT_VISUAL_CACHE_CONSUMER_CHANGED')
            return result
        def output(module, args, value):
            require(self.in_encoder, 'PRODUCT_VISUAL_ENCODER_NOT_CURRENT_REQUEST')
            self.encoder_calls += 1
            self.encoder_output = value
        def embedding(original, input_ids, *args, **kwargs):
            if self.samples == 0:
                embeds = kwargs.get('multimodal_embeddings')
                require(embeds is not None and len(embeds) == 1 and self.encoder_output is not None
                        and embeds[0].untyped_storage()._cdata == self.encoder_output.untyped_storage()._cdata,
                        'PRODUCT_VISUAL_EMBEDDING_CONSUMER_CHANGED')
                self.embedding_calls += 1
            return original(input_ids, *args, **kwargs)
        def complete(stream):
            event = torch.npu.Event(); event.record(stream); event.synchronize()
        patch(self.runner, '_execute_mm_encoder', encoder)
        patch(self.model, 'embed_input_ids', embedding)
        self.hooks.append(self.model.visual.register_forward_hook(output))
        if self.inject:
            self.site = Vision17Site(self.model.visual, fault=True, references={}, product_mode=True,
                request_ids=lambda:self.runner.input_batch.req_ids, stream=torch.npu.current_stream, complete=complete)
            self.site.install()

    def collect(self):
        require(self.inside == 0 and self.finished and self.error is None and self.samples > 0,
                'PRODUCT_REQUEST_COMPLETION_REQUIRED')
        if self.visual and not self.skip:
            require(self.encoder_calls == 1 and self.embedding_calls >= 1, 'PRODUCT_VISUAL_CONSUMER_UNPROVEN')
        events = self.site.events if self.site is not None else []
        if self.site is not None:
            require(not self.site.pending and not getattr(self.site, 'uncovered', []), 'PRODUCT_UNRESTORED_OR_UNCOVERED_SITE')
            require(events, 'PRODUCT_NO_ACTUAL_SITE_CALLS')
            calls=getattr(self.site,'forward_calls',None)
            if calls is not None:
                expected_forwards=self.encoder_calls if self.visual else self.samples
                require(len(calls)==expected_forwards and all(calls)
                    and sum(sum(c.values()) for c in calls)==len(events),'PRODUCT_ACTUAL_FORWARD_COVERAGE_INCOMPLETE')
            elif not self.visual:
                require(len(events)==self.samples*len(self.site.sites),'PRODUCT_ACTUAL_SITE_COVERAGE_INCOMPLETE')
        report = summarize_request(self.ids[0], self.binding.identity['rank'], events,
                                   skipped=self.skip, completed=True)
        report.update(external_request_id=self.external_request_id,external_request_id_status='OBSERVED' if self.external_request_id is not None else 'UNKNOWN',
                      prompt_sha256=digest(list(self.prompts[self.ids[0]])) if self.prompts else None,
                      range_facts=self.range_facts, observed_model_steps=self.samples,
                      restore_kind='CALL_ARGUMENT_ONLY_NO_STORAGE_OVERWRITE' if self.visual else 'LIVE_REGION_BYTES')
        require(report['coverage']['failed'] == 0, 'PRODUCT_ELIGIBLE_CALL_FAILED')
        if not self.inject:
            report.update(output=list(self.rows),all_finite=bool(self.rows) and all(r['finite'] for r in self.rows))
        self.close()
        report['resources'] = dict(leases=0, hooks=0, inflight=0)
        return report
