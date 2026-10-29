"""Hermetic helper tests: no model loads, network requests, or audio devices."""
import io
import json
import types
import unittest
from unittest.mock import patch

import numpy as np
import read_aloud as tts


def single(pipeline):
    """A Voices stand-in that serves one fake pipeline for every language."""
    return types.SimpleNamespace(pipeline=lambda lang: pipeline)


class ReadAloudTests(unittest.TestCase):
    def test_mlx_cache_is_bounded_before_model_load(self):
        from unittest.mock import Mock
        mx, loader = Mock(), Mock()
        modules = {'mlx': types.ModuleType('mlx'), 'mlx.core': mx,
                   'mlx_audio': types.ModuleType('mlx_audio'),
                   'mlx_audio.tts': types.ModuleType('mlx_audio.tts'),
                   'mlx_audio.tts.utils': types.SimpleNamespace(load_model=loader)}
        loader.side_effect = lambda model: mx.set_cache_limit.assert_called_once_with(64 * 1024 * 1024)
        with patch.dict('sys.modules', modules), patch.object(tts, 'use_mlx', return_value=True):
            tts.load_model()
        loader.assert_called_once_with(tts.MODEL)

    def test_synthesis_releases_cache_on_success_and_failure(self):
        from unittest.mock import Mock
        for fail in (False, True):
            control = tts.Control()
            control.jobs.put({'id': control.ident})
            voices = Mock()
            def generate(*args):
                control.closed.set()
                if fail:
                    raise RuntimeError('test failure')
            with patch.object(tts, 'Voices', return_value=voices), patch.object(tts, 'generate_job', side_effect=generate):
                tts.synthesize(control)
            voices.release_cache.assert_called_once_with()

    def test_cache_release_preserves_warm_model_and_skips_other_backends(self):
        from unittest.mock import Mock
        voices, mx = tts.Voices(), Mock()
        with patch.dict('sys.modules', {'mlx': types.ModuleType('mlx'), 'mlx.core': mx}), patch.object(tts, 'use_mlx', return_value=True):
            voices.release_cache()
            mx.clear_cache.assert_not_called()
            voices.model = model = object()
            voices.release_cache()
            mx.clear_cache.assert_called_once_with()
            self.assertIs(voices.model, model)
        with patch.object(tts, 'use_mlx', return_value=False):
            voices.release_cache()
        mx.clear_cache.assert_called_once_with()

    def test_cuda_cache_release_only_touches_an_initialized_device(self):
        from unittest.mock import Mock
        voices, torch = tts.Voices(), Mock()
        with patch.dict('sys.modules', {'torch': torch}), patch.object(tts, 'use_mlx', return_value=False):
            voices.release_cache()  # no model yet
            torch.cuda.is_initialized.return_value = False
            voices.model = model = object()
            voices.release_cache()  # CPU helper: no CUDA context
            torch.cuda.empty_cache.assert_not_called()
            torch.cuda.is_initialized.return_value = True
            voices.release_cache()
            torch.cuda.empty_cache.assert_called_once_with()
            self.assertIs(voices.model, model)

    def test_cuda_allocator_uses_growable_segments_unless_user_set(self):
        from unittest.mock import Mock
        for preset, expected in ((None, tts.CUDA_ALLOC_CONF), ('backend:cudaMallocAsync', 'backend:cudaMallocAsync')):
            seen = []
            kmodel = Mock(side_effect=lambda **kw: seen.append(tts.os.environ.get('PYTORCH_CUDA_ALLOC_CONF')) or Mock())
            env = {} if preset is None else {'PYTORCH_CUDA_ALLOC_CONF': preset}
            modules = {'torch': Mock(), 'kokoro': types.SimpleNamespace(KModel=kmodel)}
            with patch.dict('sys.modules', modules), patch.dict(tts.os.environ, env, clear=False), patch.object(tts, 'use_mlx', return_value=False):
                if preset is None:
                    tts.os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)
                tts.load_model()
            self.assertEqual(seen, [expected])

    def test_utf8_word_mapping_and_repeated_words(self):
        tokens = [types.SimpleNamespace(text=text, start_ts=i, end_ts=i+0.5)
                  for i, text in enumerate(["Olá", "go", "go", "normalized-missing"])]
        self.assertEqual(tts.word_marks("Olá go go", tokens),
                         [(0,0.5,0,4),(1,1.5,5,7),(2,2.5,8,10)])

    def test_duration_marks_time_words_including_expanded_numbers(self):
        vocab={c:1 for c in 'abcdefghijklmnopqrstuvwxyz '}
        phonemes='ola em mil nove'   # "Olá em 1990" with the number spoken as two words
        durations=[5]+[2]*len(phonemes)+[0]
        g2p=lambda word:({'Olá':'ola','em':'em','1990':'mil nove'}[word],None)
        marks=tts.duration_marks('Olá em 1990',phonemes,durations,vocab,g2p)
        # Words keep UTF-8 offsets; 1990 spans both of its spoken words.
        self.assertEqual([m[2:] for m in marks],[(0,4),(5,7),(8,12)])
        self.assertAlmostEqual(marks[0][0],2/40)          # leading-silence convention
        self.assertAlmostEqual(marks[0][1],(2+6)/40)
        self.assertAlmostEqual(marks[2][1],(2+2*len(phonemes))/40)
        self.assertTrue(all(a<b for a,b,_,_ in marks))

    def test_numbers_are_spelled_out_for_espeak_languages_only(self):
        self.assertEqual(tts.speakable_numbers('pt','versão 3.5 e 3,5'),'versão 3 ponto 5 e 3 vírgula 5')
        self.assertEqual(tts.speakable_numbers('pt','1.000 e 1.000.000 e 1.000,50'),'1000 e 1000000 e 1000 vírgula 50')
        self.assertEqual(tts.speakable_numbers('es','release 1.18.0.'),'release 1 punto 18 punto 0.')
        self.assertEqual(tts.speakable_numbers('fr','3,5 %'),'3 virgule 5 %')
        self.assertEqual(tts.speakable_numbers('it','3.5'),'3 punto 5')
        self.assertEqual(tts.speakable_numbers('pt','Fim. Outra frase, 3 itens.'),'Fim. Outra frase, 3 itens.','prose punctuation untouched')
        self.assertEqual(tts.speakable_numbers('en','3.5 and 1,000'),'3.5 and 1,000','Misaki reads English numbers')

    def test_rewritten_numbers_still_highlight_the_original_word(self):
        vocab={c:1 for c in 'abcdefghijklmnopqrstuvwxyz '}
        phon={'versão':'versao','3':'tres','ponto':'ponto','5':'sinko'}
        phonemize=lambda t:(' '.join(phon[w] for w in t.split()),None)
        sent=[]
        class Pipeline:
            model=types.SimpleNamespace(vocab=vocab)
            g2p=staticmethod(phonemize)
            def __call__(self,text,**kwargs):
                sent.append(text)
                ps=phonemize(text)[0]
                yield types.SimpleNamespace(audio=np.ones(480,dtype=np.float32),graphemes=text,tokens=None,
                    phonemes=ps,output=types.SimpleNamespace(pred_dur=[5]+[2]*len(ps)+[0]))
        control=tts.Control()
        job=dict(cmd='speak',id=1,document=10,offset=0,paused=False,speed=1.0,
                 chunks=[{'text':'versão 3.5','lang':'pt'}])
        tts.apply_command(control,job)
        tts.generate_job(control,control.jobs.get_nowait(),single(Pipeline()),np)
        self.assertEqual(sent,['versão 3 ponto 5'])
        marks=control.results.get_nowait().marks
        # "3.5" (bytes 8..11 of the original) spans all three spoken words.
        self.assertEqual([m[2:] for m in marks],[(0,7),(8,11)])

    def test_duration_marks_refuse_to_guess_on_any_mismatch(self):
        vocab={c:1 for c in 'abcdefghijklmnopqrstuvwxyz '}
        g2p=lambda word:(word.lower(),None)
        good=[5]+[2]*len('ola em')+[0]
        self.assertNotEqual(tts.duration_marks('ola em',  'ola em',good,vocab,g2p),[])
        self.assertEqual(tts.duration_marks('ola em',  'ola em',good[:-1],vocab,g2p),[],'durations out of step')
        self.assertEqual(tts.duration_marks('ola em x','ola em',good,vocab,g2p),[],'word counts disagree')
        self.assertEqual(tts.duration_marks('ola em',  'ola em',None,vocab,g2p),[],'backend gave no durations')

    def test_missing_timestamps_do_not_invent_alignment(self):
        token = types.SimpleNamespace(text="hello", start_ts=None, end_ts=None)
        self.assertEqual(tts.word_marks("hello", [token]), [])

    def test_callback_progress_is_dac_clock_based(self):
        player = tts.Player(np)
        player.load(np.ones(24000,dtype=np.float32),1.5)
        out=np.zeros((480,1),dtype=np.float32)
        player.callback(out,480,types.SimpleNamespace(outputBufferDacTime=10.0),None)
        self.assertIsNone(player.time_in_phrase(9.99))
        self.assertAlmostEqual(player.time_in_phrase(10.01),0.015)
        self.assertAlmostEqual(player.time_in_phrase(11),0.03)  # clamps to queued samples
        self.assertEqual(float(out.sum()),480)
        self.assertAlmostEqual(player.next_source_time(),0.03)

    def test_pause_outputs_silence_without_advancing_and_resume_keeps_position(self):
        player = tts.Player(np)
        player.load(np.ones(960,dtype=np.float32),1.0)
        out=np.zeros((480,1),dtype=np.float32)
        timing=types.SimpleNamespace(outputBufferDacTime=1.0)
        player.paused=True
        player.callback(out,480,timing,None)
        self.assertEqual(player.position,0)
        self.assertEqual(out.sum(),0)
        player.paused=False
        player.callback(out,480,timing,None)
        self.assertEqual(player.position,480)
        self.assertFalse(player.done(1.02))
        timing.outputBufferDacTime=1.02
        player.callback(out,480,timing,None)
        self.assertFalse(player.done(1.03))
        self.assertTrue(player.done(1.04))

    def test_load_and_clear_reset_timing(self):
        player=tts.Player(np)
        player.load(np.ones(480,dtype=np.float32),2.0,origin=3.0)
        self.assertAlmostEqual(player.next_source_time(),3.0)
        player.clear()
        self.assertIsNone(player.time_in_phrase(100))
        self.assertEqual(len(player.audio),0)
        self.assertTrue(player.done(100))  # speed change at the very end must not hang

    def test_progress_tracks_audible_buffer_when_callbacks_run_ahead_of_dac(self):
        # ALSA/PipeWire: each callback fills a buffer ~2 blocks before it is audible,
        # so the newest buffer's DAC time is always in the future.
        player=tts.Player(np)
        player.load(np.ones(2400,dtype=np.float32),1.0)
        out=np.zeros((480,1),dtype=np.float32)
        for dac in (10.04,10.06,10.08):
            player.callback(out,480,types.SimpleNamespace(outputBufferDacTime=dac),None)
        self.assertIsNone(player.time_in_phrase(10.03))
        self.assertAlmostEqual(player.time_in_phrase(10.05),0.01)
        self.assertAlmostEqual(player.time_in_phrase(10.07),0.03)
        self.assertAlmostEqual(player.elapsed_in_phrase(10.07),0.03)
        player.load(np.ones(480,dtype=np.float32),1.0)
        self.assertIsNone(player.time_in_phrase(10.07))  # a seek drops queued clocks

    def test_backend_and_device_selection(self):
        cuda=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
        cpu=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
        with patch.dict(tts.os.environ,{},clear=True):
            self.assertEqual(tts.torch_device(cuda),'cuda')
            self.assertEqual(tts.torch_device(cpu),'cpu')
        with patch.dict(tts.os.environ,{'HERDR_TTS_DEVICE':'cpu'}):
            self.assertEqual(tts.torch_device(cuda),'cpu')
        with patch.object(tts.sys,'platform','darwin'), patch.object(tts.platform,'machine',return_value='arm64'):
            self.assertTrue(tts.use_mlx())
        with patch.object(tts.sys,'platform','linux'), patch.object(tts.platform,'machine',return_value='x86_64'):
            self.assertFalse(tts.use_mlx())

    def test_speed_is_bounded_and_nan_is_safe(self):
        for value,expected in [(99,2.5),(-1,0.5),(1.2,1.2),(float('nan'),1.0)]:
            self.assertEqual(tts.speed_value(value),expected)

    def test_time_stretch_uses_pipes_pitch_preserving_filters_and_timeout(self):
        audio=np.ones(100,dtype=np.float32)
        with patch.object(tts.subprocess,'run',return_value=types.SimpleNamespace(stdout=audio.tobytes())) as run:
            tts.stretch(audio,2.5,np)
        args,kwargs=run.call_args
        self.assertIn('atempo=2.0,atempo=1.25',args[0])
        self.assertEqual(kwargs['timeout'],5)
        self.assertEqual(kwargs['input'],audio.tobytes())
        self.assertNotIn('shell',kwargs)
        self.assertIs(tts.stretch(audio,1.0,np),audio)

    def test_protocol_stdout_is_independent_of_library_stdout_redirection(self):
        original, diagnostics=io.StringIO(),io.StringIO()
        with patch.object(tts,'PROTOCOL_OUT',original):
            with tts.contextlib.redirect_stdout(diagnostics):
                print('library diagnostics')
                tts.emit('done',42)
        self.assertEqual(json.loads(original.getvalue()),dict(event='done',id=42))
        self.assertEqual(diagnostics.getvalue(),'library diagnostics\n')


class NavigationTests(unittest.TestCase):
    def job(self, ident=1, document=10, offset=4, paused=False, text='Hello.'):
        return dict(cmd='speak', id=ident, document=document, offset=offset,
                    paused=paused, speed=1.0, chunks=[dict(text=text)])

    def test_lru_limits_pcm_bytes_promotes_hits_and_rejects_oversized_audio(self):
        cache=tts.AudioCache(max_bytes=16,max_phrases=2)
        cache.reset(10)
        audio=np.ones(2,dtype=np.float32)
        cache.put(10,0,'zero',audio,[])
        cache.put(10,1,'one',audio,[])
        self.assertIsNotNone(cache.get(10,0,'zero'))  # zero is now most recent
        cache.put(10,2,'two',audio,[])
        self.assertIsNone(cache.get(10,1,'one'))
        self.assertIsNotNone(cache.get(10,0,'zero'))
        self.assertEqual(cache.bytes,16)
        cache.put(10,3,'huge',np.ones(10,dtype=np.float32),[])
        self.assertIsNone(cache.get(10,3,'huge'))
        self.assertLessEqual(cache.bytes,16)
        self.assertEqual(len(cache.entries),2)

    def test_entry_limit_is_enforced_independently_of_byte_limit(self):
        cache=tts.AudioCache(max_bytes=1000,max_phrases=2)
        cache.reset(10)
        for i in range(5): cache.put(10,i,str(i),np.ones(1,dtype=np.float32),[])
        self.assertEqual(list(cache.entries),[3,4])
        self.assertEqual(cache.bytes,8)

    def test_cache_is_text_checked_document_scoped_and_stale_generation_cannot_reinsert(self):
        cache=tts.AudioCache()
        cache.reset(10);audio=np.ones(5,dtype=np.float32)
        cache.put(10,0,'Hello.',audio,[])
        self.assertIsNone(cache.get(10,0,'Changed text.'))
        cache.reset(11)
        cache.put(10,0,'Hello.',audio,[])
        self.assertIsNone(cache.get(10,0,'Hello.'))
        self.assertEqual(cache.bytes,0)
        cache.put(11,0,'New.',audio,[])
        cache.reset(None)
        cache.put(11,0,'New.',audio,[])
        self.assertEqual(cache.bytes,0)

    def test_paused_seek_clears_old_playback_and_retains_speed_and_cached_synthesis(self):
        control=tts.Control();player=tts.Player(np)
        tts.apply_command(control,self.job(),player)
        audio=np.ones(480,dtype=np.float32)
        control.cache.put(10,4,'Hello.',audio,[])
        player.load(audio,1.0)
        control.results.put(tts.Generated(1,0,audio,[]))
        job=self.job(ident=2,paused=True);job['speed']=1.7
        self.assertEqual(tts.apply_command(control,job,player),'speak')
        self.assertTrue(control.paused);self.assertEqual(control.speed,1.7)
        self.assertEqual(len(player.audio),0);self.assertTrue(control.results.empty())
        self.assertIsNotNone(control.cache.get(10,4,'Hello.'))
        self.assertTrue(control.jobs.get_nowait()['paused'])
        out=np.ones((480,1),dtype=np.float32)
        player.callback(out,480,types.SimpleNamespace(outputBufferDacTime=1.0),None)
        self.assertEqual(out.sum(),0)
        self.assertEqual(player.position,0)

    def test_cached_revisit_retags_new_playback_id_and_uses_absolute_phrase_offset(self):
        control=tts.Control();calls=[]
        def pipeline(text,**kwargs):
            calls.append(text)
            yield types.SimpleNamespace(audio=np.ones(480,dtype=np.float32),graphemes=text,tokens=[])
        def run(job):
            tts.apply_command(control,job)
            tts.generate_job(control,control.jobs.get_nowait(),single(pipeline),np)
            result=control.results.get_nowait()
            self.assertTrue(control.results.get_nowait().done)
            return result
        first=run(self.job())
        second=run(self.job(ident=2,paused=True))
        self.assertEqual(calls,['Hello.'])
        self.assertIs(first.audio,second.audio)
        self.assertEqual((second.ident,second.index),(2,0))
        run(self.job(ident=3,offset=5))
        self.assertEqual(len(calls),2,'different absolute phrase indices do not alias')
        run(self.job(ident=4,document=11))
        self.assertEqual(len(calls),3,'new document invalidates old cache')

    def test_canceled_multi_result_phrase_is_not_cached_as_partial_audio(self):
        control=tts.Control();tts.apply_command(control,self.job())
        job=control.jobs.get_nowait()
        def pipeline(text,**kwargs):
            yield types.SimpleNamespace(audio=np.ones(480,dtype=np.float32),graphemes='Hello',tokens=[])
            control.ident=2  # force cancellation between pipeline results
            yield types.SimpleNamespace(audio=np.ones(480,dtype=np.float32),graphemes='.',tokens=[])
        tts.generate_job(control,job,single(pipeline),np)
        self.assertIsNone(control.cache.get(10,4,('en','Hello.')))
        self.assertTrue(control.results.empty())

    def test_language_picks_voice_and_keys_the_cache(self):
        control=tts.Control();calls=[]
        def pipeline(text,voice,**kwargs):
            calls.append(voice)
            yield types.SimpleNamespace(audio=np.ones(480,dtype=np.float32),graphemes=text,tokens=[])
        job=self.job();job['chunks']=[{'text':'Hello.','lang':l} for l in ('pt','es','fr','it')]+[{'text':'Hello.'}]
        tts.apply_command(control,job)
        control.results=tts.queue.Queue()  # nothing drains here; the real queue holds two
        with patch.dict(tts.os.environ,{},clear=True):
            tts.generate_job(control,control.jobs.get_nowait(),single(pipeline),np)
        self.assertEqual(calls,['pf_dora','ef_dora','ff_siwis','if_sara','af_heart'])
        self.assertIsNotNone(control.cache.get(10,4,('pt','Hello.')))
        self.assertIsNone(control.cache.get(10,4,('en','Hello.')),'other language, other audio')

    def test_unknown_language_is_rejected_before_scheduling(self):
        control=tts.Control();job=self.job();job['chunks']=[{'text':'Hallo.','lang':'de'}]
        with patch.object(tts,'emit') as emit:
            self.assertIsNone(tts.apply_command(control,job))
        emit.assert_called_once()
        self.assertTrue(control.jobs.empty())

    def test_rapid_seeks_coalesce_and_old_stop_cannot_clear_new_session(self):
        control=tts.Control()
        for i in range(1,101): tts.apply_command(control,self.job(ident=i))
        self.assertEqual(control.jobs.qsize(),1)
        self.assertEqual(control.jobs.get_nowait()['id'],100)
        audio=np.ones(4,dtype=np.float32)
        control.cache.put(10,4,'Hello.',audio,[])
        self.assertIsNone(tts.apply_command(control,dict(cmd='stop',id=99)))
        self.assertEqual(control.ident,100)
        self.assertIsNotNone(control.cache.get(10,4,'Hello.'))
        control.ident=0  # completion/error does not prevent explicit cache release
        self.assertEqual(tts.apply_command(control,dict(cmd='stop',id=100)),'stop')
        self.assertEqual(control.cache.bytes,0)

    def test_legacy_speak_defaults_and_oversized_requests_are_rejected_before_scheduling(self):
        control=tts.Control()
        legacy=dict(cmd='speak',id=1,chunks=[dict(text='Hello.')],speed=1.0)
        self.assertEqual(tts.apply_command(control,legacy),'speak')
        job=control.jobs.get_nowait()
        self.assertEqual((job['document'],job['offset']),(1,0));self.assertFalse(control.paused)
        oversized=self.job(ident=2)
        oversized['chunks']=[dict(text='a'*1024)]*300
        out=io.StringIO()
        with patch.object(tts,'PROTOCOL_OUT',out):
            self.assertIsNone(tts.apply_command(control,oversized))
        self.assertEqual(json.loads(out.getvalue())['event'],'error')
        self.assertEqual(control.ident,1)
        self.assertTrue(control.jobs.empty())

    def test_audible_elapsed_excludes_pauses_and_survives_pitch_preserving_speed_changes(self):
        player=tts.Player(np)
        player.load(np.ones(48000,dtype=np.float32),1.5)
        out=np.zeros((24000,1),dtype=np.float32)
        player.callback(out,24000,types.SimpleNamespace(outputBufferDacTime=10.0),None)
        self.assertAlmostEqual(player.time_in_phrase(10.5),0.75)
        self.assertAlmostEqual(player.elapsed_in_phrase(10.5),0.5)
        source,elapsed=player.next_source_time(),player.next_elapsed_time()
        self.assertEqual((source,elapsed),(1.5,1.0))
        player.load(np.ones(960,dtype=np.float32),0.5,source,elapsed)
        out=np.zeros((480,1),dtype=np.float32)
        player.callback(out,480,types.SimpleNamespace(outputBufferDacTime=20.0),None)
        self.assertAlmostEqual(player.time_in_phrase(20.01),1.505)
        self.assertAlmostEqual(player.elapsed_in_phrase(20.01),1.01)
        player.paused=True
        player.callback(out,480,types.SimpleNamespace(outputBufferDacTime=100.0),None)
        self.assertEqual(out.sum(),0)
        self.assertAlmostEqual(player.elapsed_in_phrase(100.0),1.02)
        self.assertEqual(player.position,480)


if __name__=='__main__':
    unittest.main()
