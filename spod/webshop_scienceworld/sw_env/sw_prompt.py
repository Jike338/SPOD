"""ScienceWorld student prompt: task, last 10 action/observation pairs, current observation and inventory."""
import os,sys
ENV_PY=os.environ.get('SCIENCEWORLD_PYTHON',sys.executable)  # interpreter with `scienceworld` installed
def gen_prompt(tok,messages):return tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
SYSTEM='You are controlling an agent in a science environment. Complete the task. Output exactly one action and nothing else.\nAction templates (OBJ denotes an object named in the environment): look around; inventory; go to OBJ; open OBJ; close OBJ; pick up OBJ; put down OBJ; move OBJ to OBJ; activate OBJ; deactivate OBJ; focus on OBJ; use OBJ on OBJ; pour OBJ in OBJ; dunk OBJ in OBJ; mix OBJ; connect OBJ to OBJ; disconnect OBJ; read OBJ; wait.\nRule: "focus on X" declares X to be the object your task is about. Focusing on any other object ends the task as a failure. Focus only on the exact object or substance named in the task, after you have found it.\nAlways name objects fully (for example "door to kitchen", not "door").'
def prompt(tok,task,state,history):
 prev=history[-10:];trim=0
 while True:
  content='Task:\n'+task+'\nRecent history (up to 10 action/observation pairs):\n'+'\n'.join('Action: '+h['action']+'\nObservation: '+h['observation'] for h in prev)+'\nCurrent observation:\n'+state['observation']+'\nInventory:\n'+state['inventory']
  messages=[dict(role='system',content=SYSTEM),dict(role='user',content=content)];ids=gen_prompt(tok,messages)
  if len(ids)+80<=12000:return messages,ids,trim
  assert prev,'Current observation/task/inventory exceeds context without history';prev=prev[1:];trim+=1
