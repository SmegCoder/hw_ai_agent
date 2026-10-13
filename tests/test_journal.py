from __future__ import annotations
import io,json,os,subprocess,sys,tempfile,unittest
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
from unittest.mock import patch

from hw_ai_agent.agent import CodingAgent
from hw_ai_agent.journal import ChangeJournal,JournalError,JournalConflict
from hw_ai_agent.session import SessionStore,SessionError,UncertainActionError
from hw_ai_agent.tools import ToolRuntime
from hw_ai_agent.policy import Policy,ApprovalDeferred
from hw_ai_agent.cli import main,verify_applied


def call(identifier,name,**args):
 return {'id':identifier,'function':{'name':name,'arguments':json.dumps(args)}}

class Client:
 def __init__(self,*responses):self.responses=iter(responses);self.requests=[]
 def complete(self,messages,tools):
  self.requests.append(list(messages));return next(self.responses)

class JournalTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve();self.workspace=self.root/'workspace';self.workspace.mkdir()
  (self.workspace/'order.py').write_bytes(b'original\r\n')
  (self.workspace/'test_order.py').write_text('# unchanged')
  self.path=self.root/'.agent/state.db';self.store=SessionStore(self.path)
 def tearDown(self):self.store.db.close();self.temp.cleanup()
 def runtime(self,**kwargs):return ToolRuntime(self.workspace,mode=kwargs.pop('mode','auto'),approve=kwargs.pop('approve',lambda r:True),trash_dir=self.root/'trash',**kwargs)
 def agent(self,client,runtime=None,**kwargs):return CodingAgent(client,runtime or self.runtime(),store=self.store,journal_mode=True,**kwargs)
 def run_actions(self,*actions,runtime=None):
  agent=self.agent(Client({'tool_calls':list(actions)},{'content':'done'}),runtime);agent.run('Task');return agent
 def write(self,identifier='w',path='order.py',content='fixed\n'):return call(identifier,'write_file',path=path,content=content)
 def journal(self,agent,runtime=None):return ChangeJournal(self.store,agent.session_id,runtime or self.runtime())

 def test_modify_diff_and_restore_exact_original_after_restart(self):
  agent=self.run_actions(self.write());identifier=agent.session_id
  self.assertIn('-original',agent.journal.diff());self.assertIn('+fixed',agent.journal.diff())
  self.store.db.close();self.store=SessionStore(self.path)
  journal=ChangeJournal(self.store,identifier,self.runtime());journal.rollback()
  self.assertEqual((self.workspace/'order.py').read_bytes(),b'original\r\n')
  self.assertEqual((self.workspace/'test_order.py').read_text(),'# unchanged')
  self.assertEqual(self.store.get(identifier).status,'rolled_back')
  self.assertIn('+fixed',journal.diff())

 def test_create_delete_and_modify_restore_only_owned_files(self):
  (self.workspace/'obsolete.bin').write_bytes(b'\x00\xffold')
  agent=self.run_actions(self.write(),self.write('new','notes.txt','note'),call('delete','delete_file',path='obsolete.bin'))
  (self.workspace/'external.txt').write_text('foreign')
  journal=self.journal(agent);self.assertEqual(set(journal.entries()),{'order.py','notes.txt','obsolete.bin'})
  journal.rollback()
  self.assertEqual((self.workspace/'obsolete.bin').read_bytes(),b'\x00\xffold');self.assertFalse((self.workspace/'notes.txt').exists())
  self.assertEqual((self.workspace/'external.txt').read_text(),'foreign')

 def test_multiple_writes_keep_first_before_and_last_after(self):
  agent=self.run_actions(self.write('one',content='first'),self.write('two',content='last'))
  entry=agent.journal.entries()['order.py']
  self.assertEqual(agent.journal.blob(entry['before']),b'original\r\n');self.assertEqual(agent.journal.blob(entry['after']),b'last')
  agent.journal.rollback();self.assertEqual((self.workspace/'order.py').read_bytes(),b'original\r\n')

 def test_external_change_in_one_file_refuses_all_rollback_writes(self):
  agent=self.run_actions(self.write(),self.write('new','notes.txt','note'))
  (self.workspace/'order.py').write_text('foreign')
  with self.assertRaises(JournalConflict):self.journal(agent).rollback()
  self.assertEqual((self.workspace/'order.py').read_text(),'foreign');self.assertEqual((self.workspace/'notes.txt').read_text(),'note')

 def test_external_creation_at_deleted_path_refuses_restore(self):
  agent=self.run_actions(call('delete','delete_file',path='order.py'));(self.workspace/'order.py').write_text('new foreign file')
  with self.assertRaises(JournalConflict):self.journal(agent).rollback()
  self.assertEqual((self.workspace/'order.py').read_text(),'new foreign file')

 def test_foreign_change_before_next_action_stops_before_modification(self):
  def edit(event):
   if event['event']=='tool_result' and event.get('tool_call_id')=='first':(self.workspace/'foreign.txt').write_text('external')
  agent=self.agent(Client({'tool_calls':[self.write('first'),self.write('second','notes.txt','note')]}),on_event=edit)
  with self.assertRaises(JournalConflict):agent.run('Task')
  self.assertFalse((self.workspace/'notes.txt').exists());self.assertEqual(self.store.calls(agent.session_id)[-1]['state'],'failed')
  self.assertEqual(self.store.get(agent.session_id).status,'journal_conflict')

 def test_noop_history_does_not_hide_external_conflict(self):
  agent=self.run_actions(self.write('one'),self.write('back',content='original\r\n'),self.write('new','notes.txt','note'))
  (self.workspace/'order.py').write_text('foreign after no-op')
  with self.assertRaises(JournalConflict):self.journal(agent).rollback()
  self.assertTrue((self.workspace/'notes.txt').exists())

 def test_all_policy_decisions_are_checked_before_any_restore(self):
  agent=self.run_actions(self.write(),self.write('new','notes.txt','note'))
  policy=Policy.from_dict({'rules':[{'tool':'write_file','path':'order.py','action':'deny'}]})
  with self.assertRaises(JournalError):self.journal(agent,self.runtime(policy=policy)).rollback()
  self.assertEqual((self.workspace/'order.py').read_text(),'fixed\n');self.assertTrue((self.workspace/'notes.txt').exists())

 def test_refusing_one_confirmation_changes_no_files(self):
  agent=self.run_actions(self.write(),self.write('new','notes.txt','note'))
  approvals=[]
  def approve(request):approvals.append(request.tool);return len(approvals)<2
  with self.assertRaises(JournalError):self.journal(agent,self.runtime(mode='safe',approve=approve)).rollback()
  self.assertEqual((self.workspace/'order.py').read_text(),'fixed\n');self.assertTrue((self.workspace/'notes.txt').exists())

 def test_delete_on_rollback_is_confirmed_in_both_modes(self):
  for mode in ['safe','auto']:
   with self.subTest(mode=mode):
    path='created_'+mode+'.txt';agent=self.run_actions(self.write(mode,path,'created'));approvals=[]
    self.journal(agent,self.runtime(mode=mode,approve=lambda r:approvals.append(r.tool) or True)).rollback()
    self.assertEqual(approvals,['delete_file']);self.assertFalse((self.workspace/path).exists())

 def test_changed_file_during_confirmation_refuses_all_writes(self):
  agent=self.run_actions(self.write(),self.write('new','notes.txt','note'))
  def approve(request):(self.workspace/'order.py').write_text('external during question');return True
  with self.assertRaises(JournalConflict):self.journal(agent,self.runtime(mode='safe',approve=approve)).rollback()
  self.assertTrue((self.workspace/'notes.txt').exists());self.assertEqual((self.workspace/'order.py').read_text(),'external during question')

 def test_policy_changed_during_confirmation_is_rechecked(self):
  agent=self.run_actions(self.write());runtime=self.runtime(mode='safe')
  def approve(request):
   runtime.policy=Policy.from_dict({'rules':[{'tool':'write_file','path':'order.py','action':'deny'}]});return True
  runtime.approve=approve
  with self.assertRaises(JournalError):self.journal(agent,runtime).rollback()
  self.assertEqual((self.workspace/'order.py').read_text(),'fixed\n')

 def test_changed_test_file_blocks_next_action_and_is_in_snapshot(self):
  def edit(event):
   if event['event']=='tool_result':(self.workspace/'test_order.py').write_text('foreign test edit')
  agent=self.agent(Client({'tool_calls':[self.write('one'),self.write('two','new.txt','new')]}),on_event=edit)
  with self.assertRaises(JournalConflict):agent.run('Task')
  self.assertFalse((self.workspace/'new.txt').exists())
  self.assertIn('test_order.py',agent.journal.image(agent.journal.head()['revision'])['files'])

 def test_pre_action_snapshot_failure_never_performs_mutation(self):
  from hw_ai_agent.journal import MAX_FILE_BYTES
  agent=self.run_actions(self.write())
  (self.workspace/'large.bin').write_bytes(b'x'*(MAX_FILE_BYTES+1))
  with self.assertRaises(JournalError):self.journal(agent).rollback()
  self.assertEqual((self.workspace/'order.py').read_text(),'fixed\n')

 def test_runtime_cannot_silently_claim_unexpected_extra_changes(self):
  runtime=self.runtime()
  runtime.bash=lambda **kwargs:((self.workspace/'unexpected.py').write_text('external') or 'exit_code=0')
  agent=self.agent(Client({'tool_calls':[call('test','Bash',command='pwd')]}),runtime)
  with self.assertRaises(JournalError):agent.run('Task')
  self.assertEqual(self.store.calls(agent.session_id)[0]['state'],'running')
  self.assertEqual(agent.journal.entries(),{})
  with self.assertRaises(JournalError):agent.journal.rollback()

 def test_legacy_schema_migration_does_not_reset_budget_or_add_journal(self):
  agent=CodingAgent(Client({'content':'done'}),self.runtime(),store=self.store);agent.run('Task')
  before=self.store.budget(agent.session_id);self.store.db.execute('PRAGMA user_version=4');self.store.db.close();self.store=SessionStore(self.path)
  self.assertEqual(self.store.budget(agent.session_id),before)
  self.assertFalse(self.store.get(agent.session_id).metadata['journal_mode'])
  self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM journals WHERE session_id=?',(agent.session_id,)).fetchone()[0],0)

 def test_journal_records_actual_structural_diff_with_hashes(self):
  agent=self.run_actions(self.write())
  event=next(e for e in self.store.get(agent.session_id).events if e['event']=='journal_action_completed')
  delta=event['changes']['order.py'];self.assertNotEqual(delta['before']['sha256'],delta['after']['sha256'])

 def test_symlink_swap_is_not_followed_during_rollback(self):
  agent=self.run_actions(self.write());outside=self.root/'private';outside.write_text('private')
  (self.workspace/'order.py').unlink();(self.workspace/'order.py').symlink_to(outside)
  with self.assertRaises((JournalError,SessionError)):self.journal(agent).rollback()
  self.assertEqual(outside.read_text(),'private')

 def test_secret_and_symlink_targets_are_not_captured(self):
  (self.workspace/'.env').write_text('SUPER_SECRET')
  outside=self.root/'secret';outside.write_text('OUTSIDE_SECRET');(self.workspace/'alias').symlink_to(outside)
  agent=self.run_actions(self.write())
  blobs=b''.join(bytes(r[0]) for r in self.store.db.execute('SELECT data FROM journal_blobs'))
  self.assertNotIn(b'SUPER_SECRET',blobs);self.assertNotIn(b'OUTSIDE_SECRET',blobs)
  self.assertEqual(agent.journal.image(agent.journal.head()['revision'])['files']['alias']['kind'],'symlink')

 def test_changed_workspace_root_blocks_rollback(self):
  agent=self.run_actions(self.write());self.workspace.rename(self.root/'old');self.workspace.mkdir();(self.workspace/'order.py').write_text('fixed\n')
  with self.assertRaises(JournalConflict):self.journal(agent).rollback()
  self.assertEqual((self.workspace/'order.py').read_text(),'fixed\n')

 def test_all_original_blobs_validate_before_any_write(self):
  agent=self.run_actions(self.write(),self.write('new','notes.txt','note'))
  sha=agent.journal.entries()['order.py']['before']['sha256'];self.store.db.execute('UPDATE journal_blobs SET data=? WHERE sha256=?',(b'bad',sha))
  with self.assertRaises(JournalError):self.journal(agent).rollback()
  self.assertTrue((self.workspace/'notes.txt').exists());self.assertEqual((self.workspace/'order.py').read_text(),'fixed\n')

 def test_original_mode_and_binary_content_are_restored(self):
  (self.workspace/'order.py').chmod(0o640);agent=self.run_actions(self.write());self.journal(agent).rollback()
  self.assertEqual((self.workspace/'order.py').stat().st_mode&0o777,0o640)

 def test_empty_file_creation_appears_in_diff_and_is_removed(self):
  agent=self.run_actions(self.write('empty','empty.txt',''));self.assertIn('empty.txt',agent.journal.diff())
  self.journal(agent).rollback();self.assertFalse((self.workspace/'empty.txt').exists())

 def test_snapshots_include_initial_before_after_and_path_lists(self):
  agent=self.run_actions(self.write('nested','pkg/sub/value.txt','value'))
  images=list(self.store.db.execute('SELECT tree FROM journal_images WHERE session_id=? ORDER BY revision',(agent.session_id,)))
  self.assertEqual(len(images),3);initial=json.loads(images[0][0]);last=json.loads(images[-1][0])
  self.assertNotIn('pkg/sub/value.txt',initial['files']);self.assertIn('pkg/sub/value.txt',last['files']);self.assertIn('pkg/sub',last['dirs'])
  self.journal(agent).rollback();self.assertFalse((self.workspace/'pkg/sub/value.txt').exists())

 def crash(self):
  runtime=self.runtime();original=runtime.execute
  def die(name,args):original(name,args);raise KeyboardInterrupt
  runtime.execute=die;agent=self.agent(Client({'tool_calls':[self.write()]}),runtime)
  with self.assertRaises(KeyboardInterrupt):agent.run('Task')
  return agent

 def test_uncertain_action_blocks_rollback_and_requires_ownership(self):
  agent=self.crash();journal=self.journal(agent)
  with self.assertRaises(JournalError):journal.rollback()
  item=self.store.calls(agent.session_id,True)[0];self.assertIn('+fixed',journal.recovery_view(item)['diff'])
  with self.assertRaises(JournalError):self.store.resolve(agent.session_id,item['key'],'applied',lambda i:True,journal=journal)
  with self.assertRaises(SessionError):self.store.resolve(agent.session_id,item['key'],'retry')

 def test_own_attribution_allows_rollback_without_refunding_budget(self):
  agent=self.crash();journal=self.journal(agent);item=self.store.calls(agent.session_id,True)[0];before=self.store.budget(agent.session_id)
  self.store.resolve(agent.session_id,item['key'],'applied',lambda i:verify_applied(journal.runtime,i),journal=journal,changes='own')
  self.assertEqual(self.store.budget(agent.session_id),before);journal.rollback()
  self.assertEqual((self.workspace/'order.py').read_bytes(),b'original\r\n');self.assertEqual(self.store.budget(agent.session_id),before)

 def test_external_attribution_excludes_changes_and_preserves_them(self):
  agent=self.crash();journal=self.journal(agent);item=self.store.calls(agent.session_id,True)[0]
  self.store.resolve(agent.session_id,item['key'],'skip',journal=journal,changes='external')
  self.assertEqual(journal.entries(),{});journal.rollback();self.assertEqual((self.workspace/'order.py').read_text(),'fixed\n')

 def test_attribution_cannot_claim_new_changes_after_preview(self):
  agent=self.crash();journal=self.journal(agent);item=self.store.calls(agent.session_id,True)[0]
  journal.reviewed_tree=journal.recovery_view(item)['_current']
  (self.workspace/'foreign.txt').write_text('unreviewed external change')
  with self.assertRaises(JournalConflict):self.store.resolve(agent.session_id,item['key'],'skip',journal=journal,changes='own')
  self.assertNotIn('foreign.txt',journal.entries())

 def test_abort_can_attribute_changes_then_rollback(self):
  agent=self.crash();journal=self.journal(agent);item=self.store.calls(agent.session_id,True)[0]
  self.store.resolve(agent.session_id,item['key'],'abort',journal=journal,changes='own')
  self.assertEqual(self.store.get(agent.session_id).status,'aborted');journal.rollback()
  self.assertEqual((self.workspace/'order.py').read_bytes(),b'original\r\n')

 def test_retry_records_new_attempt_and_keeps_initial_before(self):
  agent=self.crash();journal=self.journal(agent);item=self.store.calls(agent.session_id,True)[0]
  self.store.resolve(agent.session_id,item['key'],'retry',journal=journal,changes='own')
  resumed=self.agent(Client({'content':'done'}));resumed.resume(agent.session_id)
  self.assertEqual(self.store.budget(agent.session_id)['spent']['side_effects'],2)
  self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM journal_actions WHERE session_id=?',(agent.session_id,)).fetchone()[0],2)
  resumed.journal.rollback();self.assertEqual((self.workspace/'order.py').read_bytes(),b'original\r\n')

 def test_interrupted_rollback_resumes_by_expected_hashes_after_restart(self):
  agent=self.run_actions(self.write(),self.write('second','second.txt','created'))
  def die(path):raise KeyboardInterrupt
  with self.assertRaises(KeyboardInterrupt):self.journal(agent).rollback(after_file=die)
  self.store.db.close();self.store=SessionStore(self.path)
  journal=ChangeJournal(self.store,agent.session_id,self.runtime());self.assertEqual(journal.head()['status'],'rollback_pending')
  journal.rollback();self.assertEqual((self.workspace/'order.py').read_bytes(),b'original\r\n');self.assertFalse((self.workspace/'second.txt').exists())

 def test_rollback_is_available_after_budget_exhaustion(self):
  agent=self.agent(Client({'tool_calls':[self.write()]}),max_rounds=1)
  with self.assertRaises(Exception):agent.run('Task')
  self.assertEqual(self.store.get(agent.session_id).status,'budget_exhausted');self.journal(agent).rollback()
  self.assertEqual((self.workspace/'order.py').read_bytes(),b'original\r\n')

 def test_rolled_back_run_cannot_resume_or_spend_more(self):
  agent=self.run_actions(self.write());self.journal(agent).rollback();before=self.store.budget(agent.session_id);client=Client()
  with self.assertRaises(SessionError):self.agent(client).resume(agent.session_id)
  self.assertEqual(client.requests,[]);self.assertEqual(self.store.budget(agent.session_id),before)

 def test_cached_tool_call_is_not_reexecuted_or_logged_twice(self):
  action=self.write();agent=self.agent(Client({'tool_calls':[action]},{'tool_calls':[action]},{'content':'done'}));agent.run('Task')
  self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM journal_actions WHERE session_id=?',(agent.session_id,)).fetchone()[0],1)

 def test_same_size_foreign_edit_with_restored_mtime_is_detected(self):
  agent=self.run_actions(self.write());path=self.workspace/'order.py';info=path.stat();path.write_text('other\n');os.utime(path,ns=(info.st_atime_ns,info.st_mtime_ns))
  with self.assertRaises(JournalConflict):self.journal(agent).rollback()

 def test_policy_scope_excludes_denied_directories(self):
  (self.workspace/'private').mkdir();(self.workspace/'private/secret').write_text('DO_NOT_READ')
  policy=Policy.from_dict({'directories':{'allow':['.'],'deny':['private']}})
  agent=self.run_actions(self.write(),runtime=self.runtime(policy=policy))
  self.assertNotIn('private/secret',agent.journal.image(agent.journal.head()['revision'])['files'])

 def test_journal_cannot_be_enabled_without_store(self):
  with self.assertRaises(SessionError):CodingAgent(Client(),self.runtime(),journal_mode=True).run('Task')

 def test_cli_changes_and_rollback_require_no_model_key(self):
  agent=self.run_actions(self.write())
  with patch.object(sys,'argv',['hw-ai-agent','changes',agent.session_id,'--state-db',str(self.path)]),patch('hw_ai_agent.cli.load_env') as load,redirect_stdout(io.StringIO()) as output:
   self.assertEqual(main(),0);load.assert_not_called();self.assertIn('+fixed',output.getvalue())
  with patch.object(sys,'argv',['hw-ai-agent','rollback',agent.session_id,'--state-db',str(self.path)]),redirect_stdout(io.StringIO()):self.assertEqual(main(),0)
  self.assertEqual((self.workspace/'order.py').read_bytes(),b'original\r\n')

 def test_changes_can_select_completed_session_without_id(self):
  self.run_actions(self.write())
  with patch.object(sys,'argv',['hw-ai-agent','changes','--state-db',str(self.path)]),patch('sys.stdin.isatty',return_value=True),redirect_stdout(io.StringIO()) as output:
   self.assertEqual(main(),0);self.assertIn('+fixed',output.getvalue())

 def test_old_sessions_do_not_gain_fake_before_images(self):
  agent=CodingAgent(Client({'content':'done'}),self.runtime(),store=self.store);agent.run('Task')
  with patch.object(sys,'argv',['hw-ai-agent','rollback',agent.session_id,'--state-db',str(self.path)]),redirect_stderr(io.StringIO()):self.assertEqual(main(),1)


@unittest.skipUnless(os.getenv('REQUIRE_SANDBOX_TESTS')=='1','Requires native macOS sandbox')
class JournalIntegrationTests(unittest.TestCase):
 def test_new_demo_changes_and_rollback_in_native_runtime(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);workspace=root/'workspace';db=root/'state.db'
   with patch.object(sys,'argv',['hw-ai-agent','rollback-demo','--offline','--mode','auto','--workspace',str(workspace),'--state-db',str(db)]),patch('hw_ai_agent.cli.confirm_plan',return_value=True),redirect_stdout(io.StringIO()) as output:
    self.assertEqual(main(),0);self.assertIn('DEMO PASS',output.getvalue())
   with SessionStore(db) as store:
    identifier=store.list_sessions()[0]['id'];self.assertTrue(store.get(identifier).metadata['journal_mode'])
   with patch.object(sys,'argv',['hw-ai-agent','rollback',identifier,'--state-db',str(db)]),redirect_stdout(io.StringIO()):self.assertEqual(main(),0)
   from hw_ai_agent.demo import BROKEN_ORDER
   self.assertEqual((workspace/'order.py').read_bytes(),BROKEN_ORDER.encode())

 def test_hard_process_crash_requires_attribution_before_rollback(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);workspace=root/'workspace';workspace.mkdir();(workspace/'order.py').write_text('original');db=root/'state.db'
   script='''import sys,os,json
from hw_ai_agent.agent import CodingAgent
from hw_ai_agent.session import SessionStore
from hw_ai_agent.tools import ToolRuntime
class Client:
 def complete(self,messages,tools):return {'tool_calls':[{'id':'w','function':{'name':'write_file','arguments':json.dumps({'path':'order.py','content':'changed'})}}]}
class Runtime(ToolRuntime):
 def execute(self,name,args):super().execute(name,args);os._exit(86)
with SessionStore(sys.argv[2]) as store:CodingAgent(Client(),Runtime(sys.argv[1],mode='auto'),store=store,journal_mode=True).run('Task')
'''
   env=os.environ.copy();env['PYTHONPATH']=str(Path(__file__).resolve().parents[1]/'src')
   process=subprocess.run([sys.executable,'-c',script,str(workspace),str(db)],env=env,capture_output=True,timeout=10)
   self.assertEqual(process.returncode,86,process.stderr.decode())
   with SessionStore(db) as store:
    identifier=store.list_sessions()[0]['id'];runtime=ToolRuntime(workspace,mode='auto');journal=ChangeJournal(store,identifier,runtime)
    with self.assertRaises(JournalError):journal.rollback()
    store.resolve(identifier,'1:0','applied',lambda i:True,journal=journal,changes='own');journal.rollback()
   self.assertEqual((workspace/'order.py').read_text(),'original')

if __name__=='__main__':unittest.main()
