"""Native chat-template prompts and exact command extraction (search[...] free text, click[...] from the list)."""
import re
from protocol import SEARCH, SEARCH_REGEX
def generation_prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
def teacher_prompt(teacher_tokenizer, student_tokenizer, sample):
    messages = sample['messages']
    if generation_prompt(student_tokenizer, messages) != sample['prompt']:
        raise ValueError('Saved messages do not reproduce the student prompt')
    return generation_prompt(teacher_tokenizer, messages)
def legal(action, commands):
    if SEARCH in commands and re.fullmatch(SEARCH_REGEX, action): return True
    return action in commands and action != SEARCH
def extract_action(tokenizer, tokens, commands, stop_token_ids):
    body = list(tokens)
    while body and body[-1] in stop_token_ids: body.pop()
    action = tokenizer.decode(body, skip_special_tokens=False).strip()
    if not legal(action, commands): raise ValueError(('Output is not an admissible command', action))
    return action
def grammar(commands):
    """Structured-output spec: regex when searching, exact choice otherwise."""
    if SEARCH in commands: return dict(regex=SEARCH_REGEX)
    return dict(choice=list(commands))
