"""Shared native-template prompt construction and exact grammar-action extraction."""
def generation_prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(messages, tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_dict=False)


def teacher_prompt(teacher_tokenizer, student_tokenizer, sample):
    messages = sample['messages']
    if generation_prompt(student_tokenizer, messages) != sample['prompt']:
        raise ValueError('Saved messages do not reproduce the student prompt')
    return generation_prompt(teacher_tokenizer, messages)


def extract_action(tokenizer, tokens, commands, stop_token_ids):
    # Accept only the exact grammar command, optionally followed by one configured stop token.
    # Never silently strip a thought channel, Action: prefix, or code fences.
    body = list(tokens)
    if body and body[-1] in stop_token_ids:
        body.pop()
    action = tokenizer.decode(body, skip_special_tokens=False).strip()
    if action not in commands:
        raise ValueError(('Output is not an admissible command', action))
    return action
